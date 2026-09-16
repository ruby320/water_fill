from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mistral_remote_attention import (
    MistralAttentionPatch,
    atomic_write_json,
    atomic_write_jsonl,
    clone_dynamic_cache,
    complete_candidate_blocks,
    locate_question_span,
    model_forward,
    parse_max_memory,
    token_hash,
)
from mistral_new_waterfill import new_waterfill_block_map
from mistral_task_quality import (
    generate,
)

DEFAULT_MODEL = Path("/home/liuminglu/models/Mistral-7B-Instruct-v0.2")
DEFAULT_RULER_ROOT = Path("/home/liuminglu/kvcache/datasets/defensivekv_dataset/ruler")
DEFAULT_REPO = Path("/home/liuminglu/kvcache/DefensiveKV")
DEFAULT_OUTPUT = Path("results/ruler_32k_new_waterfill_r20_n5_v1")
DEFAULT_TASKS = ("qa_1", "niah_multivalue", "niah_multikey_3", "cwe")
METHODS = ("new_waterfill",)
SCHEMA_VERSION = 1


def sample_records(tokenizer: Any, root: Path, context_length: int, tasks: list[str],
                   count: int, seed: int) -> list[dict[str, Any]]:
    import pandas as pd
    from datasets import load_from_disk

    frame = pd.DataFrame(load_from_disk(str(root / str(context_length))))
    records: list[dict[str, Any]] = []
    for task in tasks:
        subset = frame[frame["task"] == task]
        if len(subset) < count:
            raise ValueError(f"{task} has {len(subset)} rows, need {count}")
        selected = subset.sample(n=count, random_state=seed)
        for source_index, row in selected.iterrows():
            context = str(row["context"])
            question = str(row["question"])
            answer_prefix = str(row["answer_prefix"])
            raw = context + question + answer_prefix
            original_ids = [int(value) for value in tokenizer.encode(raw, add_special_tokens=True)]
            max_new_tokens = int(row["max_new_tokens"])
            prompt_budget = context_length - max_new_tokens
            prompt_ids = original_ids[-prompt_budget:]
            truncated = len(original_ids) - len(prompt_ids)
            full_span = locate_question_span(tokenizer, raw, context, question, raw)
            start = max(0, int(full_span["start"]) - truncated)
            end = max(start, int(full_span["end"]) - truncated)
            if end > len(prompt_ids) or end <= start:
                raise RuntimeError(f"invalid truncated question span for {task}:{source_index}")
            question_span = {
                **full_span,
                "start": start,
                "end": end,
                "query_start": max(start, end - 64),
                "query_end": end,
                "left_truncated_tokens": truncated,
            }
            answers = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
            records.append({
                "source_id": f"{task}:{int(source_index)}", "task": task,
                "answers": [str(value) for value in answers],
                "max_new_tokens": max_new_tokens, "prompt_ids": prompt_ids,
                "prompt_tokens": len(prompt_ids), "original_prompt_tokens": len(original_ids),
                "truncated_tokens": truncated, "prompt_hash": token_hash(prompt_ids),
                "question_span": question_span,
            })
    return records


def load_metric(repo: Path) -> Any:
    sys.path.insert(0, str(repo / "evaluation"))
    from ruler.calculate_metrics import calculate_metrics
    return calculate_metrics


def pending_methods(
    source_id: str, completed: set[tuple[str, str]], methods: tuple[str, ...] = METHODS
) -> tuple[str, ...]:
    """Return selected methods that still need one result for a sample."""
    return tuple(method for method in methods if (source_id, method) not in completed)


def summarize(rows: list[dict[str, Any]], args: Any, metric: Any) -> dict[str, Any]:
    import pandas as pd

    expected_count = len(args.tasks) * args.num_samples
    expected_ids = {str(row["source_id"]) for row in rows}
    if len(expected_ids) != expected_count:
        raise ValueError(
            f"summary requires {expected_count} unique samples, found {len(expected_ids)}"
        )
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        method_ids = [str(row["source_id"]) for row in method_rows]
        if len(method_rows) != expected_count or set(method_ids) != expected_ids:
            raise ValueError(
                f"summary requires exactly {expected_count} complete rows for {method}, "
                f"found {len(method_rows)} rows over {len(set(method_ids))} samples"
            )

    metrics: dict[str, Any] = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        frame = pd.DataFrame({
            "task": [row["task"] for row in method_rows],
            "answer": [row["answers"] for row in method_rows],
            "predicted_answer": [row["prediction"] for row in method_rows],
        })
        by_task = metric(frame)
        scores = [float(value["string_match"]) for value in by_task.values() if value is not None]
        metrics[method] = {
            "macro_average": float(np.mean(scores)), "tasks": by_task,
            "average_prompt_tokens": float(np.mean([row["prompt_tokens"] for row in method_rows])),
            "average_output_tokens": float(np.mean([row["output_tokens"] for row in method_rows])),
            "average_decode_seconds": float(np.mean([row["decode_seconds"] for row in method_rows])),
        }
    full = metrics.get("fullkv")
    if full is not None:
        for method, report in metrics.items():
            report["macro_retention_vs_fullkv"] = (
                report["macro_average"] / full["macro_average"] if full["macro_average"] else None
            )
            for task, value in report["tasks"].items():
                full_score = full["tasks"][task]["string_match"]
                value["retention_vs_fullkv"] = value["string_match"] / full_score if full_score else None
    return {
        "benchmark": "RULER", "schema_version": SCHEMA_VERSION,
        "model_path": str(args.model_path), "context_length": args.context_length,
        "tasks": args.tasks, "num_samples_per_task": args.num_samples,
        "num_total_samples": len({row["source_id"] for row in rows}), "seed": args.seed,
        "methods": list(METHODS), "target_prompt_kv_ratio": args.target_kv_ratio,
        "block_size": args.block_size, "sink_tokens": args.sink_tokens,
        "local_tokens": args.local_tokens,
        "prompt_policy": "RULER context + question + answer_prefix; left truncate to context minus generation budget",
        "generation": "greedy; task-specific official max_new_tokens",
        "metric": "official RULER string match; macro average across four tasks",
        "results": metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RULER quality benchmark for head-aware hierarchical new_waterfill")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--ruler_root", type=Path, default=DEFAULT_RULER_ROOT)
    parser.add_argument("--ruler_repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context_length", type=int, default=32768)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_kv_ratio", type=float, default=0.20)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--sink_tokens", type=int, default=4)
    parser.add_argument("--local_tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--max_memory", nargs="*", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.tasks, args.num_samples = [args.tasks[0]], 1
    if not 0.0 < args.target_kv_ratio <= 1.0:
        raise ValueError("target_kv_ratio must be in (0, 1]")

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache, MistralForCausalLM

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_path)
    expected = (config.model_type, int(config.num_hidden_layers), int(config.num_attention_heads), int(config.num_key_value_heads))
    if expected != ("mistral", 32, 32, 8):
        raise TypeError(f"expected dense Mistral 32L/32Q/8KV, got {expected}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    records = sample_records(tokenizer, args.ruler_root, args.context_length, args.tasks, args.num_samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "predictions.jsonl"
    rows: list[dict[str, Any]] = []
    if args.resume and output_path.exists():
        rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    elif output_path.exists() and output_path.stat().st_size:
        raise FileExistsError("outputs exist; use --resume or choose another output directory")
    completed = {(row["source_id"], row["method"]) for row in rows}

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    kwargs: dict[str, Any] = {"dtype": dtype, "device_map": args.device_map,
                              "low_cpu_mem_usage": True, "attn_implementation": args.attn_implementation}
    if (memory := parse_max_memory(args.max_memory)) is not None:
        kwargs["max_memory"] = memory
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs).eval()
    if not isinstance(model, MistralForCausalLM):
        raise TypeError("loaded model must be MistralForCausalLM")
    eos = model.generation_config.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in (eos or [])}
    device = model.get_input_embeddings().weight.device

    for ordinal, record in enumerate(records, 1):
        methods_for_record = pending_methods(record["source_id"], completed, METHODS)
        if not methods_for_record:
            continue
        blocks = complete_candidate_blocks(record["prompt_tokens"], args.block_size,
                                           args.sink_tokens, args.local_tokens)
        patch = MistralAttentionPatch(model, blocks, args.sink_tokens, args.local_tokens)
        try:
            cache = DynamicCache(config=model.config)
            prompt = torch.tensor([record["prompt_ids"]], dtype=torch.long, device=device)
            query = (record["question_span"]["query_start"], record["question_span"]["query_end"])
            with torch.inference_mode(), patch.mode(prefill_query=query):
                output = model_forward(model, prompt, cache, 0)
            initial_logits = output.logits[0, -1].detach()
            masses = {layer: patch.prefill_masses[layer]["block_mass"] for layer in range(32)}
            variants: dict[str, tuple[Any, Any, Any]] = {}
            if "new_waterfill" in methods_for_record:
                layer_map, details = new_waterfill_block_map(
                    32, record["prompt_tokens"], args.sink_tokens + args.local_tokens,
                    args.block_size, args.target_kv_ratio, masses,
                )
                variants["new_waterfill"] = (layer_map, None, details)
            del output, prompt
            for method, (layer_map, head_map, allocation) in variants.items():
                if (record["source_id"], method) in completed:
                    continue
                branch_cache = clone_dynamic_cache(cache)
                with torch.inference_mode():
                    generated, elapsed = generate(
                        model, patch, branch_cache, initial_logits, record["prompt_tokens"],
                        record["max_new_tokens"], eos_ids, layer_map, head_map,
                    )
                prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
                row = {key: value for key, value in record.items() if key != "prompt_ids"}
                row.update({
                    "schema_version": SCHEMA_VERSION, "method": method,
                    "allocation": allocation, "prediction": prediction,
                    "generated_token_ids": generated, "generation_hash": token_hash(generated),
                    "output_tokens": len(generated), "decode_seconds": elapsed,
                })
                rows.append(row)
                atomic_write_jsonl(output_path, rows)
                completed.add((record["source_id"], method))
                print(f"[{ordinal}/{len(records)}] {record['source_id']} {method} tokens={len(generated)}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del cache, initial_logits
        finally:
            patch.close(model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metric = load_metric(args.ruler_repo)
    summary = summarize(rows, args, metric)
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
