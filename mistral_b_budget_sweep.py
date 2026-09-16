from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mistral_remote_attention import (
    MistralAttentionPatch,
    advance_cache,
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    block_set_metrics,
    cache_signature,
    clone_dynamic_cache,
    complete_candidate_blocks,
    effects,
    model_forward,
    parse_max_memory,
    prepare_records,
    read_jsonl,
    run_branch,
    token_hash,
    top_blocks,
)

DEFAULT_MODEL = Path("/home/liuminglu/models/Mistral-7B-Instruct-v0.2")
DEFAULT_DATA = Path("/home/liuminglu/kvcache/datasets/defensivekv_dataset/longbench")
DEFAULT_SOURCE = Path("results/gov_report_remote_attention/predictions.jsonl")
DEFAULT_OUTPUT = Path("results/gov_report_b_budget_sweep")
SCHEMA_VERSION = 1
OUTPUT_FILES = ("metadata.jsonl", "block_metrics.jsonl", "a_only.jsonl", "interventions.jsonl", "journal.jsonl")
BRANCHES = ("prefill", "oracle", "random")


def effective_budget(requested: int, candidate_count: int) -> tuple[int, bool]:
    if requested < 0 or candidate_count < 0:
        raise ValueError("budgets and candidate counts must be nonnegative")
    effective = min(requested, candidate_count)
    return effective, effective < requested


def nested_random_selections(candidate_count: int, budgets: Iterable[int], seed: int) -> dict[int, list[int]]:
    pool = list(range(candidate_count))
    random.Random(seed).shuffle(pool)
    return {int(b): sorted(pool[:min(int(b), candidate_count)]) for b in budgets}


def kl_restoration(kl_a: float, kl_branch: float, threshold: float = 1e-12) -> tuple[float | None, bool]:
    sufficient = kl_a <= threshold
    return (None if sufficient else (kl_a - kl_branch) / kl_a), sufficient


def bootstrap(values: list[float], draws: int, seed: int) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95": [None, None], "n_samples": 0}
    rng = random.Random(seed)
    means = [float(np.mean([rng.choice(values) for _ in values])) for _ in range(draws)]
    return {
        "mean": float(np.mean(values)),
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "n_samples": len(values),
    }


def load_fixed_predictions(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise FileNotFoundError(f"no fixed predictions found at {path}")
    by_id: dict[str, dict[str, Any]] = {}
    required = {"source_id", "prompt_hash", "generated_token_ids", "trajectory_hash", "checkpoints", "important_layers"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"prediction row lacks {sorted(missing)}")
        source_id = str(row["source_id"])
        if source_id in by_id:
            raise ValueError(f"duplicate fixed prediction source_id={source_id}")
        trajectory = [int(x) for x in row["generated_token_ids"]]
        if token_hash(trajectory) != row["trajectory_hash"]:
            raise ValueError(f"trajectory hash mismatch in fixed prediction source_id={source_id}")
        by_id[source_id] = row
    return rows


def join_fixed_records(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_id = {str(row["source_id"]): row for row in predictions}
    joined = []
    for record in records:
        source_id = str(record["source_id"])
        if source_id not in by_id:
            raise ValueError(f"seed-42 record missing from fixed predictions: source_id={source_id}")
        fixed = by_id[source_id]
        if record["prompt_hash"] != fixed["prompt_hash"]:
            raise ValueError(f"prompt hash mismatch for source_id={source_id}")
        if int(fixed.get("prompt_tokens", len(record["prompt_ids"]))) != len(record["prompt_ids"]):
            raise ValueError(f"prompt length mismatch for source_id={source_id}")
        joined.append((record, fixed))
    if len(joined) != len(records):
        raise RuntimeError("fixed trajectory join changed sample count")
    return joined


def prepare_resume(output: Path, resume: bool) -> tuple[set[str], dict[str, Path]]:
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / name for name in OUTPUT_FILES}
    for path in paths.values():
        path.touch(exist_ok=True)
    if not resume and any(path.stat().st_size for path in paths.values()):
        raise FileExistsError("outputs exist; use --resume or choose a new output directory")
    journals = [row for row in read_jsonl(paths["journal.jsonl"]) if row.get("schema_version") == SCHEMA_VERSION]
    commits = [row for row in journals if row.get("state") == "committed"]
    by_source = {str(row["source_id"]): row for row in commits}
    if len(by_source) != len(commits):
        raise ValueError("duplicate committed journal source_id")
    data = {
        name: [row for row in read_jsonl(path) if row.get("schema_version") == SCHEMA_VERSION]
        for name, path in paths.items() if name != "journal.jsonl"
    }
    committed = set()
    for source_id, journal in by_source.items():
        expected = journal.get("counts", {})
        if all(sum(str(row["source_id"]) == source_id for row in data[name]) == int(expected.get(name, -1)) for name in data):
            committed.add(source_id)
    for name, path in paths.items():
        rows = journals if name == "journal.jsonl" else data[name]
        atomic_write_jsonl(path, (row for row in rows if str(row["source_id"]) in committed))
    return committed, paths


def stable_seed(base_seed: int, source_id: str, checkpoint: int, layer: int, repeat: int) -> int:
    text = f"{base_seed}:{source_id}:{checkpoint}:{layer}:{repeat}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) % (2**32)


def exact_a_tokens(key_length: int, sink: int, local: int) -> int:
    sink_end = min(sink, key_length)
    local_start = max(0, key_length - local)
    return sink_end + (key_length - max(sink_end, local_start))


def aggregate_sample_rows(rows: list[dict[str, Any]], value_keys: list[str], random_first: bool = False) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["source_id"]), int(row["checkpoint"]), int(row["layer_idx"]), int(row["requested_budget"]), str(row.get("branch", "block")))
        grouped[key].append(row)
    cells = []
    for key, group in grouped.items():
        branch = key[-1]
        if branch == "random" and random_first:
            cell = {name: float(np.mean([float(row[name]) for row in group if row.get(name) is not None])) for name in value_keys if any(row.get(name) is not None for row in group)}
        else:
            cell = {name: float(group[0][name]) for name in value_keys if group[0].get(name) is not None}
        cell.update({"source_id": key[0], "requested_budget": key[3], "branch": branch})
        cells.append(cell)
    sample_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        sample_groups[(cell["source_id"], cell["requested_budget"], cell["branch"])].append(cell)
    result = {}
    for (source_id, budget, branch), group in sample_groups.items():
        result[f"{source_id}|{budget}|{branch}"] = {
            name: float(np.mean([row[name] for row in group if name in row]))
            for name in value_keys if any(name in row for row in group)
        }
    return result


def summarize(paths: dict[str, Path], args: Any, model: Any, seconds: float) -> dict[str, Any]:
    predictions = read_jsonl(paths["metadata.jsonl"])
    block_rows = read_jsonl(paths["block_metrics.jsonl"])
    interventions = read_jsonl(paths["interventions.jsonl"])
    block_samples = aggregate_sample_rows(block_rows, ["recall", "jaccard", "coverage"])
    effect_samples = aggregate_sample_rows(interventions, ["kl_mean", "delta_nll_mean", "kl_restoration"], random_first=True)

    def effect_report(budget: int, branch: str, allowed_ids: set[str] | None = None) -> dict[str, Any]:
        entries = []
        for key, values in effect_samples.items():
            source_id, budget_text, row_branch = key.split("|")
            if int(budget_text) == budget and row_branch == branch and (allowed_ids is None or source_id in allowed_ids):
                entries.append({"source_id": source_id, **values})
        entries.sort(key=lambda row: row["source_id"])
        return {
            "per_sample": entries,
            "sample_macro": {
                name: bootstrap([float(row[name]) for row in entries if name in row], args.bootstrap_draws, args.seed + budget)
                for name in ("kl_mean", "delta_nll_mean", "kl_restoration")
            },
        }

    def block_report(budget: int, allowed_ids: set[str] | None = None) -> dict[str, Any]:
        entries = []
        for key, values in block_samples.items():
            source_id, budget_text, _ = key.split("|")
            if int(budget_text) == budget and (allowed_ids is None or source_id in allowed_ids):
                entries.append({"source_id": source_id, **values})
        entries.sort(key=lambda row: row["source_id"])
        return {
            "per_sample": entries,
            "sample_macro": {
                name: bootstrap([float(row[name]) for row in entries if name in row], args.bootstrap_draws, args.seed + budget)
                for name in ("recall", "jaccard", "coverage")
            },
        }

    saturated_ids = {
        budget: {str(row["source_id"]) for row in block_rows if int(row["requested_budget"]) == budget and row["saturated"]}
        for budget in args.budgets
    }
    all_ids = {str(row["source_id"]) for row in predictions}
    attended = {}
    for budget in args.budgets:
        per_sample = []
        for prediction in predictions:
            candidate_count = int(prediction["candidate_count"])
            effective, saturated = effective_budget(budget, candidate_count)
            ratios = []
            attended_tokens = []
            for checkpoint in prediction["checkpoints"]:
                key_length = int(prediction["prompt_tokens"]) + int(checkpoint) + 1
                a = exact_a_tokens(key_length, args.sink_tokens, args.local_tokens)
                important_count = len(prediction["important_layers"])
                total = (32 - important_count) * a + important_count * (a + args.block_size * effective)
                attended_tokens.append(total)
                ratios.append(total / (32 * key_length))
            per_sample.append({
                "source_id": str(prediction["source_id"]), "effective_budget": effective, "saturated": saturated,
                "mean_attended_tokens_across_32_layers": float(np.mean(attended_tokens)),
                "mean_attended_ratio": float(np.mean(ratios)),
            })
        attended[str(budget)] = {
            "per_sample": per_sample,
            "sample_macro_attended_ratio": float(np.mean([row["mean_attended_ratio"] for row in per_sample])),
        }

    return {
        "experiment": "mistral_b_budget_sweep", "schema_version": SCHEMA_VERSION,
        "model_path": str(args.model_path.resolve()), "model_type": model.config.model_type,
        "samples": len(predictions), "seed": args.seed, "fixed_predictions": str(args.source_predictions.resolve()),
        "budgets": args.budgets, "block_size": args.block_size, "sink_tokens": args.sink_tokens,
        "local_tokens": args.local_tokens, "checkpoint_count": args.checkpoint_count,
        "future_tokens": args.future_tokens, "important_layers": args.important_layers,
        "random_repeats": args.random_repeats, "smoke": args.smoke,
        "saturation": {
            str(budget): {
                "saturated_sample_ids": sorted(saturated_ids[budget]),
                "unsaturated_sample_ids": sorted(all_ids - saturated_ids[budget]),
                "equal_budget_across_all_samples": not saturated_ids[budget],
            } for budget in args.budgets
        },
        "effects": {
            str(budget): {branch: effect_report(budget, branch) for branch in BRANCHES}
            for budget in args.budgets
        },
        "block_metrics": {str(budget): block_report(budget) for budget in args.budgets},
        "unsaturated_sample_only": {
            str(budget): {
                "effects": {branch: effect_report(budget, branch, all_ids - saturated_ids[budget]) for branch in BRANCHES},
                "block_metrics": block_report(budget, all_ids - saturated_ids[budget]),
            } for budget in args.budgets
        },
        "theoretical_attention": {
            "definition": "Non-important layers use exact A; each stored important layer uses A plus block_size*effective_budget tokens, averaged over actual prompt/checkpoint key lengths.",
            "by_budget": attended,
        },
        "aggregation": "Random repeats are averaged within sample/checkpoint/layer cells, then checkpoints and layers within samples, then samples. Bootstrap resamples samples.",
        "kl_restoration_definition": "(KL_A-KL_branch)/KL_A; cells with KL_A<=1e-12 are excluded from the ratio.",
        "inference_note": "Five samples support descriptive sample-cluster bootstrap only.",
        "processing_seconds": seconds, "paths": {name: str(path.resolve()) for name, path in paths.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep remote-attention B budgets on exact fixed FullKV trajectories")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--longbench_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--source_predictions", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_context_length", type=int, default=32768)
    parser.add_argument("--checkpoint_count", type=int, default=8)
    parser.add_argument("--future_tokens", type=int, default=8)
    parser.add_argument("--important_layers", type=int, default=8)
    parser.add_argument("--budgets", nargs="+", type=int, default=[4, 8, 12, 16, 24])
    parser.add_argument("--random_repeats", type=int, default=5)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--sink_tokens", type=int, default=4)
    parser.add_argument("--local_tokens", type=int, default=512)
    parser.add_argument("--bootstrap_draws", type=int, default=1000)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--max_memory", nargs="*", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Hard-limit to 1 sample, 1 checkpoint, 2 important layers, future=2, B=4/8, random=1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.num_samples, args.checkpoint_count, args.future_tokens = 1, 1, 2
        args.important_layers, args.budgets, args.random_repeats = 2, [4, 8], 1
    if args.seed != 42:
        raise ValueError("this sweep is locked to seed 42")
    if not args.budgets or any(b <= 0 for b in args.budgets) or len(set(args.budgets)) != len(args.budgets):
        raise ValueError("budgets must be unique positive integers")
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache, MistralForCausalLM
    if args.device_map == "auto" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for device_map=auto")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_path)
    expected = (config.model_type, int(config.num_hidden_layers), int(config.num_attention_heads), int(config.num_key_value_heads))
    if expected != ("mistral", 32, 32, 8):
        raise TypeError(f"expected dense Mistral 32L/32Q/8KV, got {expected}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    records = prepare_records(args.longbench_root, tokenizer, args.seed, args.num_samples, args.max_context_length)
    fixed_predictions = load_fixed_predictions(args.source_predictions)
    joined = join_fixed_records(records, fixed_predictions)
    completed, paths = prepare_resume(args.output_dir, args.resume)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    kwargs: dict[str, Any] = {"dtype": dtype, "device_map": args.device_map, "low_cpu_mem_usage": True, "attn_implementation": args.attn_implementation}
    if (memory := parse_max_memory(args.max_memory)) is not None:
        kwargs["max_memory"] = memory
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs).eval()
    if not isinstance(model, MistralForCausalLM):
        raise TypeError("loaded model must be MistralForCausalLM")

    for ordinal, (record, fixed) in enumerate(joined, 1):
        source_id = str(record["source_id"])
        if source_id in completed:
            continue
        trajectory = [int(x) for x in fixed["generated_token_ids"]]
        checkpoints = [int(x) for x in fixed["checkpoints"]]
        important = [int(x) for x in fixed["important_layers"]]
        if args.smoke:
            checkpoints, important = checkpoints[:1], important[:2]
        else:
            if len(checkpoints) != args.checkpoint_count or len(important) != args.important_layers:
                raise ValueError(f"fixed metadata count mismatch for source_id={source_id}")
        if any(checkpoint + args.future_tokens >= len(trajectory) for checkpoint in checkpoints):
            raise ValueError(f"fixed trajectory lacks future targets for source_id={source_id}")
        blocks = complete_candidate_blocks(len(record["prompt_ids"]), args.block_size, args.sink_tokens, args.local_tokens)
        if not blocks:
            raise ValueError(f"sample has no candidate blocks: source_id={source_id}")
        patch = MistralAttentionPatch(model, blocks, args.sink_tokens, args.local_tokens)
        try:
            device = model.get_input_embeddings().weight.device
            base = DynamicCache(config=model.config)
            prompt = torch.tensor([record["prompt_ids"]], dtype=torch.long, device=device)
            with torch.inference_mode(), patch.mode(prefill_query=(record["question_span"]["query_start"], record["question_span"]["query_end"])):
                output = model_forward(model, prompt, base, 0)
            del output, prompt
            if set(patch.prefill_masses) != set(range(32)):
                raise RuntimeError("prefill capture did not cover every layer")
            prefill_rankings = {layer: top_blocks(patch.prefill_masses[layer]["block_mass"], len(blocks)) for layer in important}
            metadata_rows = [{
                "schema_version": SCHEMA_VERSION, "task": "gov_report", "source_id": source_id,
                "prompt_tokens": len(record["prompt_ids"]), "prompt_hash": record["prompt_hash"],
                "generated_token_ids": trajectory, "trajectory_hash": fixed["trajectory_hash"],
                "checkpoints": checkpoints, "important_layers": important, "candidate_count": len(blocks),
                "blocks": blocks, "fixed_source_row": True,
                "source_predictions": str(args.source_predictions.resolve()),
                "seed": args.seed, "budgets": args.budgets,
                "random_repeats": args.random_repeats, "future_tokens": args.future_tokens,
                "smoke": args.smoke,
            }]
            block_rows, a_rows, intervention_rows = [], [], []
            base_checkpoint = 0
            for checkpoint in checkpoints:
                with torch.inference_mode():
                    base_checkpoint = advance_cache(model, base, trajectory, len(record["prompt_ids"]), base_checkpoint, checkpoint)
                signature = cache_signature(base)
                with torch.inference_mode():
                    reference = run_branch(model, patch, base, trajectory, len(record["prompt_ids"]), checkpoint, args.future_tokens, capture=True)
                if cache_signature(base) != signature:
                    raise RuntimeError("reference branch polluted checkpoint cache")
                oracle_rankings = {layer: top_blocks(reference["attention"][layer]["block_mass"], len(blocks)) for layer in important}
                for layer in important:
                    with torch.inference_mode():
                        altered_a = run_branch(model, patch, base, trajectory, len(record["prompt_ids"]), checkpoint, args.future_tokens, layer, [])
                    a_effect = effects(reference, altered_a)
                    a_rows.append({
                        "schema_version": SCHEMA_VERSION, "task": "gov_report", "source_id": source_id,
                        "checkpoint": checkpoint, "layer_idx": layer, "branch": "A_only", **a_effect,
                    })
                    for budget in args.budgets:
                        effective, saturated = effective_budget(budget, len(blocks))
                        selected_prefill = prefill_rankings[layer][:effective]
                        selected_oracle = oracle_rankings[layer][:effective]
                        metrics = block_set_metrics(selected_prefill, selected_oracle, reference["attention"][layer]["block_mass"])
                        block_rows.append({
                            "schema_version": SCHEMA_VERSION, "task": "gov_report", "source_id": source_id,
                            "checkpoint": checkpoint, "layer_idx": layer, "requested_budget": budget,
                            "effective_budget": effective, "saturated": saturated, "candidate_count": len(blocks),
                            "prefill_b_blocks": selected_prefill, "decode_oracle_b_blocks": selected_oracle,
                            "decode_block_mass": reference["attention"][layer]["block_mass"], **metrics,
                        })
                        variants: list[tuple[str, int, int | None, list[int]]] = [
                            ("prefill", 0, None, selected_prefill), ("oracle", 0, None, selected_oracle)
                        ]
                        for repeat in range(args.random_repeats):
                            seed = stable_seed(args.seed, source_id, checkpoint, layer, repeat)
                            nested = nested_random_selections(len(blocks), args.budgets, seed)
                            variants.append(("random", repeat, seed, nested[budget]))
                        for branch, repeat, random_seed, selected in variants:
                            with torch.inference_mode():
                                altered = run_branch(model, patch, base, trajectory, len(record["prompt_ids"]), checkpoint, args.future_tokens, layer, selected)
                            result = effects(reference, altered)
                            restoration, sufficient = kl_restoration(a_effect["kl_mean"], result["kl_mean"])
                            intervention_rows.append({
                                "schema_version": SCHEMA_VERSION, "task": "gov_report", "source_id": source_id,
                                "checkpoint": checkpoint, "layer_idx": layer, "branch": branch, "repeat": repeat,
                                "requested_budget": budget, "effective_budget": effective, "saturated": saturated,
                                "candidate_count": len(blocks), "selected_b_blocks": selected, "random_seed": random_seed,
                                "a_only_kl_mean": a_effect["kl_mean"], "a_only_delta_nll_mean": a_effect["delta_nll_mean"],
                                "a_already_sufficient": sufficient, "kl_restoration": restoration, **result,
                            })
                if cache_signature(base) != signature:
                    raise RuntimeError("intervention branch polluted checkpoint cache")
                del reference, signature
            del base
            if token_hash(trajectory) != fixed["trajectory_hash"] or token_hash(record["prompt_ids"]) != fixed["prompt_hash"]:
                raise RuntimeError("fixed prompt or trajectory changed during processing")
            bundles = {
                "metadata.jsonl": metadata_rows, "block_metrics.jsonl": block_rows,
                "a_only.jsonl": a_rows, "interventions.jsonl": intervention_rows,
            }
            for name, rows in bundles.items():
                append_jsonl(paths[name], rows)
            append_jsonl(paths["journal.jsonl"], [{
                "schema_version": SCHEMA_VERSION, "task": "gov_report", "source_id": source_id,
                "state": "committed", "counts": {name: len(rows) for name, rows in bundles.items()},
                "prompt_hash": record["prompt_hash"], "trajectory_hash": fixed["trajectory_hash"],
            }])
            print(f"[{ordinal}/{len(joined)}] committed {source_id} candidates={len(blocks)}", flush=True)
        finally:
            patch.close(model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary = summarize(paths, args, model, time.perf_counter() - started)
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
