from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from mistral_remote_attention import (MistralAttentionPatch, atomic_write_json,
    atomic_write_jsonl, complete_candidate_blocks, model_forward, parse_max_memory,
    token_hash)
from mistral_ruler_quality import DEFAULT_RULER_ROOT, DEFAULT_TASKS, load_metric, sample_records
from mistral_task_quality import formula_block_map

DEFAULT_MODEL = Path("/home/liuminglu/models/Mistral-7B-Instruct-v0.2")
DEFAULT_REPO = Path("/home/liuminglu/kvcache/DefensiveKV")
DEFAULT_MOTIVATION2 = Path("/home/liuminglu/moe/add/motivation2")
DEFAULT_OUTPUT = Path("results/ruler_16k_waterfill_defensivekv_r20_v1")
METHOD = "waterfill_defensivekv_r20"
SCHEMA_VERSION = 1


def layer_block_counts_to_element_targets(layer_block_counts: dict[int, int],
        prompt_tokens: int, num_kv_heads: int, block_size: int,
        fixed_tokens: int) -> dict[int, int]:
    if not layer_block_counts or set(layer_block_counts) != set(range(len(layer_block_counts))):
        raise ValueError("layer counts must contain contiguous layer indices starting at zero")
    if min(prompt_tokens, num_kv_heads, block_size) < 1 or fixed_tokens < 0:
        raise ValueError("invalid KV budget dimensions")
    full = prompt_tokens * num_kv_heads
    fixed = min(fixed_tokens, prompt_tokens)
    targets = {}
    for layer, count in sorted(layer_block_counts.items()):
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"invalid block count for layer {layer}: {count}")
        targets[layer] = min(full, (fixed + count * block_size) * num_kv_heads)
    return targets


def exact_topk_indices(scores: list[float], n_kept: int) -> list[int]:
    if not 0 <= n_kept <= len(scores):
        raise ValueError("n_kept is outside score capacity")
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))[:n_kept]


def budget_metadata(layer_block_counts: dict[int, int], targets: dict[int, int],
        actuals: dict[int, int], prompt_tokens: int, num_kv_heads: int,
        requested_ratio: float) -> dict[str, Any]:
    if set(targets) != set(layer_block_counts) or set(actuals) != set(targets):
        raise ValueError("counts, targets, and actuals must contain the same layers")
    target_total, actual_total = sum(targets.values()), sum(actuals.values())
    if actual_total != target_total or any(actuals[layer] != targets[layer] for layer in targets):
        raise RuntimeError(f"exact KV budget mismatch: target={target_total}, actual={actual_total}")
    full_total = len(targets) * prompt_tokens * num_kv_heads
    return {
        "semantic": "Water-fill cross-layer budget with DefensiveKV flattened head-token selection",
        "requested_kv_ratio": requested_ratio,
        "waterfill_layer_block_counts": {str(k): v for k, v in sorted(layer_block_counts.items())},
        "target_kept_elements_by_layer": {str(k): v for k, v in sorted(targets.items())},
        "actual_kept_elements_by_layer": {str(k): v for k, v in sorted(actuals.items())},
        "global_full_elements": full_total, "global_target_kept_elements": target_total,
        "global_actual_kept_elements": actual_total, "global_target_ratio": target_total / full_total,
        "global_actual_ratio": actual_total / full_total, "budget_exact": True,
    }


def remove_instance_forward_wrappers(model: Any) -> None:
    for layer in model.model.layers:
        if "forward" in layer.self_attn.__dict__:
            delattr(layer.self_attn, "forward")
    shadowed = [i for i, layer in enumerate(model.model.layers)
                if "forward" in layer.self_attn.__dict__]
    if shadowed:
        raise RuntimeError(f"instance-level attention forward remains on layers {shadowed}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-prefill Water-fill plus DefensiveKV RULER benchmark")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--ruler_root", type=Path, default=DEFAULT_RULER_ROOT)
    parser.add_argument("--ruler_repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--defensivekv_repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--motivation2_dir", type=Path, default=DEFAULT_MOTIVATION2)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context_length", type=int, default=16384)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_kv_ratio", type=float, default=0.20)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--sink_tokens", type=int, default=4)
    parser.add_argument("--local_tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_memory", nargs="*", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def summarize(rows: list[dict[str, Any]], args: Any, metric: Any) -> dict[str, Any]:
    import pandas as pd
    expected = len(args.tasks) * args.num_samples
    if len(rows) != expected or len({row["source_id"] for row in rows}) != expected:
        raise ValueError(f"summary requires exactly {expected} unique predictions, found {len(rows)}")
    frame = pd.DataFrame({"task": [r["task"] for r in rows], "answer": [r["answers"] for r in rows],
                          "predicted_answer": [r["prediction"] for r in rows]})
    by_task = metric(frame)
    scores = [float(value["string_match"]) for value in by_task.values() if value is not None]
    return {"benchmark": "RULER", "schema_version": SCHEMA_VERSION, "method": METHOD,
        "model_path": str(args.model_path), "context_length": args.context_length,
        "tasks": args.tasks, "num_samples_per_task": args.num_samples,
        "num_total_samples": expected, "seed": args.seed,
        "target_prompt_kv_ratio": args.target_kv_ratio, "block_size": args.block_size,
        "sink_tokens": args.sink_tokens, "local_tokens": args.local_tokens,
        "prompt_policy": "RULER context + question + answer_prefix; left truncate to context minus generation budget",
        "generation": "greedy; task-specific official max_new_tokens",
        "metric": "official RULER string match; macro average across four tasks",
        "two_prefill_policy": "standard attention question-span Water-fill analysis, then exact-budget DefensiveKV prefill",
        "budget_validation": {"all_samples_exact": all(r["allocation"]["budget_exact"] for r in rows),
            "global_target_elements": sum(r["allocation"]["global_target_kept_elements"] for r in rows),
            "global_actual_elements": sum(r["allocation"]["global_actual_kept_elements"] for r in rows)},
        "result": {"macro_average": float(np.mean(scores)), "tasks": by_task,
            "average_prompt_tokens": float(np.mean([r["prompt_tokens"] for r in rows])),
            "average_output_tokens": float(np.mean([r["output_tokens"] for r in rows])),
            "average_prefill_seconds": float(np.mean([r["prefill_seconds"] for r in rows])),
            "average_decode_seconds": float(np.mean([r["decode_seconds"] for r in rows]))}}


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.tasks, args.num_samples = [args.tasks[0]], 1
    if not 0.0 < args.target_kv_ratio <= 1.0:
        raise ValueError("target_kv_ratio must be in (0, 1]")
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache, MistralForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DefensiveKV benchmark execution")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    config = AutoConfig.from_pretrained(args.model_path)
    expected = (config.model_type, int(config.num_hidden_layers), int(config.num_attention_heads), int(config.num_key_value_heads))
    if expected != ("mistral", 32, 32, 8):
        raise TypeError(f"expected dense Mistral 32L/32Q/8KV, got {expected}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    records = sample_records(tokenizer, args.ruler_root, args.context_length, args.tasks, args.num_samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "predictions.jsonl"
    rows = ([json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if args.resume and output_path.exists() else [])
    if not args.resume and output_path.exists() and output_path.stat().st_size:
        raise FileExistsError("outputs exist; use --resume or choose another output directory")
    if any(row.get("method") != METHOD for row in rows):
        raise ValueError("resume output contains a different method")
    completed = {str(row["source_id"]) for row in rows}
    pending = [record for record in records if record["source_id"] not in completed]
    if not pending:
        summary = summarize(rows, args, load_metric(args.ruler_repo)); atomic_write_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2)); return
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    kwargs: dict[str, Any] = {"dtype": dtype, "device_map": args.device_map,
                              "low_cpu_mem_usage": True, "attn_implementation": "sdpa"}
    if (memory := parse_max_memory(args.max_memory)) is not None: kwargs["max_memory"] = memory
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs).eval()
    if not isinstance(model, MistralForCausalLM): raise TypeError("loaded model must be MistralForCausalLM")
    offloaded = {name: str(placement) for name, placement in getattr(model, "hf_device_map", {}).items()
                 if str(placement).lower() in {"cpu", "disk"}}
    if offloaded:
        raise RuntimeError(f"DefensiveKV does not support CPU/disk offload: {dict(list(offloaded.items())[:8])}")
    for layer, decoder_layer in enumerate(model.model.layers):
        projection_devices = {str(getattr(decoder_layer.self_attn, name).weight.device)
                              for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
        if len(projection_devices) != 1 or not next(iter(projection_devices)).startswith("cuda:"):
            raise RuntimeError(f"attention layer {layer} is split or offloaded: {sorted(projection_devices)}")
    device = model.get_input_embeddings().weight.device
    allocations: dict[str, dict[str, Any]] = {}
    for ordinal, record in enumerate(pending, 1):
        blocks = complete_candidate_blocks(record["prompt_tokens"], args.block_size, args.sink_tokens, args.local_tokens)
        patch = MistralAttentionPatch(model, blocks, args.sink_tokens, args.local_tokens)
        try:
            cache = DynamicCache(config=model.config)
            prompt = torch.tensor([record["prompt_ids"]], dtype=torch.long, device=device)
            query = (record["question_span"]["query_start"], record["question_span"]["query_end"])
            with torch.inference_mode(), patch.mode(prefill_query=query): output = model_forward(model, prompt, cache, 0)
            masses = {layer: patch.prefill_masses[layer]["block_mass"] for layer in range(32)}
            _, waterfill = formula_block_map(32, record["prompt_tokens"], args.sink_tokens + args.local_tokens,
                                              args.block_size, args.target_kv_ratio, masses)
            counts = {int(k): int(v) for k, v in waterfill["layer_block_counts"].items()}
            targets = layer_block_counts_to_element_targets(counts, record["prompt_tokens"], 8,
                                                             args.block_size, args.sink_tokens + args.local_tokens)
            expected_total = (32 * min(args.sink_tokens + args.local_tokens, record["prompt_tokens"])
                              + args.block_size * int(waterfill["allocated_layer_blocks"])) * 8
            if sum(targets.values()) != expected_total: raise RuntimeError("Water-fill block and element budgets differ")
            allocations[record["source_id"]] = {"counts": counts, "targets": targets, "waterfill": waterfill}
            print(f"[analysis {ordinal}/{len(pending)}] {record['source_id']} blocks={sum(counts.values())}", flush=True)
            del output, prompt, cache
        finally:
            patch.close(model); remove_instance_forward_wrappers(model)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    sys.path[:0] = [str(args.motivation2_dir), str(args.defensivekv_repo), str(args.defensivekv_repo / "evaluation")]
    from kvpress.ada_cache import DynamicCacheSplitHeadFlatten
    from mistral_defensivekv_patch import patch_mistral_attention_for_defensivekv
    from mistral_waterfill_defensivekv_press import ExactLayerBudgetMistralDefensiveKVPress
    patch_mistral_attention_for_defensivekv()
    eos = model.generation_config.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in (eos or [])}
    for ordinal, record in enumerate(pending, 1):
        allocation = allocations[record["source_id"]]
        press = ExactLayerBudgetMistralDefensiveKVPress(allocation["targets"])
        cache = DynamicCacheSplitHeadFlatten()
        prompt = torch.tensor([record["prompt_ids"]], dtype=torch.long, device=device)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), press(model): output = model_forward(model, prompt, cache, 0)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        prefill_seconds = time.perf_counter() - start
        logits = output.logits[0, -1]
        actuals = {layer: int(cache.metadata_list[layer].head_lens.sum().item()) for layer in range(32)}
        if press.actual_kept != actuals: raise RuntimeError("press and cache metadata disagree")
        metadata = budget_metadata(allocation["counts"], allocation["targets"], actuals,
                                   record["prompt_tokens"], 8, args.target_kv_ratio)
        metadata["waterfill"] = allocation["waterfill"]
        metadata["actual_head_lengths_by_layer"] = {str(layer): [int(v) for v in cache.metadata_list[layer].head_lens.cpu().tolist()] for layer in range(32)}
        generated = []
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            for step in range(record["max_new_tokens"]):
                token_id = int(logits.argmax().item()); generated.append(token_id)
                if token_id in eos_ids or step + 1 == record["max_new_tokens"]: break
                token = torch.tensor([[token_id]], dtype=torch.long, device=device)
                next_output = model_forward(model, token, cache, record["prompt_tokens"] + step)
                logits = next_output.logits[0, -1]; del next_output, token
        if torch.cuda.is_available(): torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - start
        public = {key: value for key, value in record.items() if key != "prompt_ids"}
        row = {**public, "schema_version": SCHEMA_VERSION, "method": METHOD, "allocation": metadata,
               "prediction": tokenizer.decode(generated, skip_special_tokens=True).strip(),
               "generated_token_ids": generated, "generation_hash": token_hash(generated),
               "output_tokens": len(generated), "prefill_seconds": prefill_seconds,
               "decode_seconds": decode_seconds}
        rows.append(row); atomic_write_jsonl(output_path, rows)
        print(f"[generation {ordinal}/{len(pending)}] {record['source_id']} tokens={len(generated)}", flush=True)
        del output, prompt, cache, logits
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    summary = summarize(rows, args, load_metric(args.ruler_repo))
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
