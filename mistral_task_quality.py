from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import string
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mistral_remote_attention import (
    MistralAttentionPatch,
    atomic_write_json,
    atomic_write_jsonl,
    clone_dynamic_cache,
    complete_candidate_blocks,
    format_prompt,
    locate_question_span,
    model_forward,
    parse_max_memory,
    token_hash,
    top_blocks,
)

DEFAULT_MODEL = Path("/home/liuminglu/models/Mistral-7B-Instruct-v0.2")
DEFAULT_DATA = Path("/home/liuminglu/kvcache/datasets/defensivekv_dataset/longbench")
DEFAULT_SCORER = Path("/home/liuminglu/kvcache/DefensiveKV/evaluation/longbench/calculate_metrics.py")
DEFAULT_SITE_PACKAGES = Path("/home/liuminglu/miniconda3/envs/defensivekv/lib/python3.10/site-packages")
DEFAULT_OUTPUT = Path("results/gov_report_task_quality")
BUDGETS = (4, 8, 12, 16)
TASK_GENERATION_LENGTHS = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64,
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32,
    "trec": 64, "triviaqa": 32, "samsum": 128,
    "gov_report": 512, "qmsum": 512, "multi_news": 512,
    "passage_count": 32, "passage_retrieval_en": 32,
    "lcc": 64, "repobench-p": 64,
}
SCHEMA_VERSION = 7


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def bootstrap(values: list[float], draws: int, seed: int) -> dict[str, Any]:
    if not values:
        return {"mean": None, "median": None, "ci95": [None, None]}
    rng = random.Random(seed)
    means = [float(np.mean([rng.choice(values) for _ in values])) for _ in range(draws)]
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
    }


def load_scorer(path: Path, site_packages: Path, dataset: str = "gov_report") -> Any:
    if site_packages.is_dir() and str(site_packages) not in sys.path:
        sys.path.append(str(site_packages))
    spec = importlib.util.spec_from_file_location("longbench_calculate_metrics", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import scorer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if dataset not in module.dataset2metric:
        raise ValueError(f"LongBench scorer does not support task {dataset!r}")
    return module.dataset2metric[dataset]


def normalized_evidence_tokens(text: str) -> list[str]:
    lowered = text.lower()
    lowered = "".join(character for character in lowered if character not in string.punctuation)
    return [token for token in re.split(r"\s+", lowered) if token and token not in {"a", "an", "the"}]


def answer_has_lexical_evidence(context: str, answers: list[str]) -> bool:
    context_tokens = normalized_evidence_tokens(context)
    for answer in answers:
        answer_tokens = normalized_evidence_tokens(str(answer))
        width = len(answer_tokens)
        if width and any(context_tokens[index:index + width] == answer_tokens
                         for index in range(len(context_tokens) - width + 1)):
            return True
    return False


def prepare_records(root: Path, tokenizer: Any, task: str, count: int, seed: int, max_context: int,
                    max_new_tokens: int, min_prompt_tokens: int = 0,
                    require_answer_evidence: bool = False) -> list[dict[str, Any]]:
    from datasets import load_from_disk
    frame = load_from_disk(str(root)).to_pandas()
    subset = frame[frame["task"] == task]
    if len(subset) < count:
        raise ValueError(f"{task} has {len(subset)} rows, need {count}")
    candidates = subset.sample(frac=1.0, random_state=seed)
    records = []
    rejection_counts = {"too_short": 0, "too_long": 0, "missing_answer_evidence": 0}
    for row in candidates.to_dict("records"):
        context, question = str(row["context"]), str(row["question"])
        answers = [str(answer) for answer in row["answer"]]
        raw = context + question + str(row.get("answer_prefix", ""))
        prompt = format_prompt(tokenizer, task, raw)
        prompt_ids = [int(x) for x in tokenizer.encode(prompt, add_special_tokens=True)]
        if len(prompt_ids) < min_prompt_tokens:
            rejection_counts["too_short"] += 1
            continue
        if len(prompt_ids) + max_new_tokens > max_context:
            rejection_counts["too_long"] += 1
            continue
        evidence_present = answer_has_lexical_evidence(context, answers)
        if require_answer_evidence and not evidence_present:
            rejection_counts["missing_answer_evidence"] += 1
            continue
        records.append({
            "task": task, "source_id": str(row["_id"]), "prompt_ids": prompt_ids,
            "prompt_hash": token_hash(prompt_ids), "answer_evidence_present": evidence_present,
            "question_span": locate_question_span(tokenizer, prompt, context, question, raw),
            "answers": answers,
            "all_classes": list(row["all_classes"]) if row.get("all_classes") is not None else [],
        })
        if len(records) == count:
            break
    if len(records) < count:
        raise ValueError(
            f"{task} has only {len(records)} eligible rows, need {count}; rejections={rejection_counts}"
        )
    print(f"Selected {count} {task} rows; eligibility rejections before completion: {rejection_counts}", flush=True)
    return records


def random_block_maps(candidate_count: int, important: list[int], budget: int,
                      repeats: int, seed: int, source_id: str) -> list[dict[int, list[int]]]:
    effective = min(budget, candidate_count)
    source_seed = int(hashlib.sha256(source_id.encode()).hexdigest()[:16], 16)
    result = []
    for repeat in range(repeats):
        rng = random.Random(seed + source_seed + repeat)
        result.append({layer: sorted(rng.sample(range(candidate_count), effective)) for layer in important})
    return result

def stable_random_seed(seed: int, source_id: str, repeat: int) -> int:
    """Derive a process-independent random seed from run and sample identity."""
    payload = json.dumps([seed, source_id, repeat], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def random_pair_block_map(num_layers: int, candidate_count: int, total_budget: int,
                          seed: int, source_id: str,
                          repeat: int) -> tuple[dict[int, list[int]], int]:
    """Uniformly sample an exact budget over all feasible layer-block pairs."""
    if num_layers < 1 or candidate_count < 0 or total_budget < 0:
        raise ValueError("invalid random allocation dimensions")
    capacity = num_layers * candidate_count
    if total_budget > capacity:
        raise ValueError(f"random layer-block budget {total_budget} exceeds capacity {capacity}")
    random_seed = stable_random_seed(seed, source_id, repeat)
    rng = random.Random(random_seed)
    selected_pairs = rng.sample(range(capacity), total_budget)
    mapping = {layer: [] for layer in range(num_layers)}
    for pair_index in selected_pairs:
        layer, remote_block = divmod(pair_index, candidate_count)
        mapping[layer].append(remote_block)
    for blocks in mapping.values():
        blocks.sort()
    counts = {layer: len(mapping[layer]) for layer in range(num_layers)}
    allocated = sum(counts.values())
    if allocated != total_budget:
        raise RuntimeError("random allocation did not preserve the exact budget")
    return mapping, random_seed


def random_budget_block_map(num_layers: int, prompt_length: int, a_tokens: int,
                            block_size: int, target_kv_ratio: float,
                            candidate_count: int, seed: int, source_id: str,
                            repeat: int) -> tuple[dict[int, list[int]], dict[str, Any]]:
    budget = formula_total_budget(num_layers, prompt_length, a_tokens, block_size,
                                  target_kv_ratio, candidate_count)
    mapping, random_seed = random_pair_block_map(
        num_layers, candidate_count, budget, seed, source_id, repeat,
    )
    counts = {layer: len(mapping[layer]) for layer in range(num_layers)}
    allocated = sum(counts.values())
    achieved = (num_layers * min(a_tokens, prompt_length) + block_size * allocated) / (num_layers * prompt_length)
    return mapping, {
        "target_kv_ratio": target_kv_ratio,
        "total_layer_block_budget": budget,
        "allocated_layer_blocks": allocated,
        "layer_block_counts": counts,
        "achieved_decode_start_kv_ratio": achieved,
        "allocation_rule": "uniform_without_replacement_over_all_layer_block_pairs",
        "random_seed": random_seed,
    }


def coverage_count(masses: list[float], target: float = 0.5) -> int:
    if not 0.0 < target <= 1.0:
        raise ValueError("coverage target must be in (0, 1]")
    total = float(sum(masses))
    if total <= 0.0:
        return max(1, len(masses))
    cumulative = 0.0
    for count, mass in enumerate(sorted((float(x) for x in masses), reverse=True), 1):
        cumulative += mass
        if cumulative / total >= target:
            return count
    return len(masses)


def formula_total_budget(num_layers: int, prompt_length: int, a_tokens: int,
                         block_size: int, target_kv_ratio: float,
                         candidate_count: int) -> int:
    if not 0.0 < target_kv_ratio <= 1.0:
        raise ValueError("target KV ratio must be in (0, 1]")
    if min(num_layers, prompt_length, block_size) < 1 or a_tokens < 0 or candidate_count < 0:
        raise ValueError("invalid budget dimensions")
    available = num_layers * candidate_count
    token_budget = num_layers * (prompt_length * target_kv_ratio - min(a_tokens, prompt_length))
    return min(available, max(0, int(np.floor(token_budget / block_size))))


def uniform_block_counts(num_layers: int, candidate_count: int, total_budget: int) -> dict[int, int]:
    """Distribute a layer-block budget evenly, breaking remainder ties by layer index."""
    if num_layers < 1 or candidate_count < 0 or total_budget < 0:
        raise ValueError("invalid uniform allocation dimensions")
    allocated = min(total_budget, num_layers * candidate_count)
    quotient, remainder = divmod(allocated, num_layers)
    counts = {layer: quotient + int(layer < remainder) for layer in range(num_layers)}
    if any(count > candidate_count for count in counts.values()):
        raise RuntimeError("uniform allocation exceeded per-layer capacity")
    return counts


def ranked_fill_block_counts(num_layers: int, candidate_count: int, total_budget: int,
                             ranking: list[int]) -> dict[int, int]:
    """Fill layers in ranking order up to capacity while preserving the total budget."""
    if num_layers < 1 or candidate_count < 0 or total_budget < 0:
        raise ValueError("invalid ranked-fill allocation dimensions")
    if len(ranking) != num_layers or set(ranking) != set(range(num_layers)):
        raise ValueError("ranking must contain every layer exactly once")
    remaining = min(total_budget, num_layers * candidate_count)
    counts = {layer: 0 for layer in range(num_layers)}
    for layer in ranking:
        count = min(candidate_count, remaining)
        counts[layer] = count
        remaining -= count
        if remaining == 0:
            break
    return counts


def block_map_from_counts(masses: dict[int, list[float]], counts: dict[int, int]) -> dict[int, list[int]]:
    """Select each layer's highest-mass blocks according to an allocation count map."""
    if set(masses) != set(counts):
        raise ValueError("masses and counts must contain the same layers")
    mapping = {}
    for layer in sorted(counts):
        if not 0 <= counts[layer] <= len(masses[layer]):
            raise ValueError("layer block count is outside candidate capacity")
        mapping[layer] = top_blocks(masses[layer], counts[layer])
    return mapping


def waterfill_block_counts(masses: dict[int, list[float]], total_budget: int) -> tuple[dict[int, int], dict[int, dict[str, float]]]:
    if total_budget < 0:
        raise ValueError("total budget must be non-negative")
    parameters: dict[int, dict[str, float]] = {}
    marginal_values: list[tuple[float, int, int]] = []
    for layer, values in sorted(masses.items()):
        remote_mass = float(sum(values))
        k50 = coverage_count(values, 0.5)
        decay = float(np.log(2.0) / k50)
        parameters[layer] = {"remote_mass": remote_mass, "k50": float(k50), "lambda": decay}
        for count in range(1, len(values) + 1):
            marginal = remote_mass * (np.exp(-decay * (count - 1)) - np.exp(-decay * count))
            marginal_values.append((float(marginal), layer, count))
    marginal_values.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = marginal_values[:min(total_budget, len(marginal_values))]
    counts = {layer: 0 for layer in masses}
    for _, layer, count in selected:
        counts[layer] = max(counts[layer], count)
    if sum(counts.values()) != len(selected):
        raise RuntimeError("non-prefix water-filling allocation")
    return counts, parameters


def global_empirical_topk_block_map(
        num_layers: int, prompt_length: int, a_tokens: int, block_size: int,
        target_kv_ratio: float, masses: dict[int, list[float]],
) -> tuple[dict[int, list[int]], dict[str, Any]]:
    """Select the globally highest empirical block masses under the formula budget."""
    if set(masses) != set(range(num_layers)):
        raise ValueError("masses must contain every layer")
    candidate_count = len(next(iter(masses.values()), []))
    if any(len(values) != candidate_count for values in masses.values()):
        raise ValueError("all layers must share the same candidate block count")
    budget = formula_total_budget(
        num_layers, prompt_length, a_tokens, block_size,
        target_kv_ratio, candidate_count,
    )
    candidates = [
        (float(mass), layer, block_index)
        for layer in range(num_layers)
        for block_index, mass in enumerate(masses[layer])
    ]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    mapping = {layer: [] for layer in range(num_layers)}
    for _, layer, block_index in candidates[:budget]:
        mapping[layer].append(block_index)
    for blocks in mapping.values():
        blocks.sort()
    counts = {layer: len(mapping[layer]) for layer in range(num_layers)}
    allocated = sum(counts.values())
    if allocated != budget:
        raise RuntimeError("global empirical Top-K allocation did not preserve the exact budget")
    achieved = (
        num_layers * min(a_tokens, prompt_length) + block_size * allocated
    ) / (num_layers * prompt_length)
    return mapping, {
        "target_kv_ratio": target_kv_ratio,
        "total_layer_block_budget": budget,
        "allocated_layer_blocks": allocated,
        "layer_block_counts": counts,
        "achieved_decode_start_kv_ratio": achieved,
        "allocation_rule": "global_empirical_block_mass_desc_layer_asc_block_asc",
    }


def formula_block_map(num_layers: int, prompt_length: int, a_tokens: int, block_size: int,
                      target_kv_ratio: float, masses: dict[int, list[float]]) -> tuple[dict[int, list[int]], dict[str, Any]]:
    candidate_count = len(next(iter(masses.values()), []))
    if any(len(values) != candidate_count for values in masses.values()):
        raise ValueError("all layers must share the same candidate block count")
    budget = formula_total_budget(num_layers, prompt_length, a_tokens, block_size,
                                  target_kv_ratio, candidate_count)
    counts, parameters = waterfill_block_counts(masses, budget)
    mapping = block_map_from_counts(masses, counts)
    active = [layer for layer, count in counts.items() if count > 0]
    achieved = (num_layers * min(a_tokens, prompt_length) + block_size * sum(counts.values())) / (num_layers * prompt_length)
    details = {
        "target_kv_ratio": target_kv_ratio, "total_layer_block_budget": budget,
        "allocated_layer_blocks": sum(counts.values()), "layer_block_counts": counts,
        "active_layers": active, "a_fraction": len(active) / num_layers,
        "mean_b_fraction_active_layers": float(np.mean([counts[layer] / candidate_count for layer in active])) if active and candidate_count else 0.0,
        "achieved_decode_start_kv_ratio": achieved, "layer_parameters": parameters,
        "allocation_rule": "waterfill_by_remote_mass_and_k50_marginal_gain",
    }
    return mapping, details



def fixed_budget_block_map(num_layers: int, prompt_length: int, a_tokens: int, block_size: int,
                           target_kv_ratio: float, masses: dict[int, list[float]],
                           allocation: str, ranking: list[int] | None = None) -> tuple[dict[int, list[int]], dict[str, Any]]:
    candidate_count = len(next(iter(masses.values()), []))
    if set(masses) != set(range(num_layers)):
        raise ValueError("masses must contain every layer")
    if any(len(values) != candidate_count for values in masses.values()):
        raise ValueError("all layers must share the same candidate block count")
    budget = formula_total_budget(num_layers, prompt_length, a_tokens, block_size,
                                  target_kv_ratio, candidate_count)
    if allocation == "uniform":
        counts = uniform_block_counts(num_layers, candidate_count, budget)
        rule = "uniform_all_layers_equal_share_remainder_by_layer_index"
    elif allocation == "ranked_fill":
        if ranking is None:
            raise ValueError("ranking is required for ranked-fill allocation")
        counts = ranked_fill_block_counts(num_layers, candidate_count, budget, ranking)
        rule = "ranked_fill_by_descending_remote_mass_full_layers_to_capacity_then_partial_layer"
    else:
        raise ValueError(f"unknown fixed-budget allocation: {allocation}")
    mapping = block_map_from_counts(masses, counts)
    allocated = sum(counts.values())
    achieved = (num_layers * min(a_tokens, prompt_length) + block_size * allocated) / (num_layers * prompt_length)
    return mapping, {
        "target_kv_ratio": target_kv_ratio,
        "total_layer_block_budget": budget,
        "allocated_layer_blocks": allocated,
        "layer_block_counts": counts,
        "achieved_decode_start_kv_ratio": achieved,
        "allocation_rule": rule,
    }


def total_head_token_budget(num_layers: int, num_kv_heads: int, prompt_length: int,
                            target_ratio: float) -> int:
    if not 0.0 < target_ratio <= 1.0:
        raise ValueError("target ratio must be in (0, 1]")
    return int(np.floor(num_layers * num_kv_heads * prompt_length * target_ratio))


def adakv_token_map(scores: dict[int, list[list[float]]], target_ratio: float,
                    safeguard: float) -> tuple[dict[int, dict[int, list[int]]], dict[str, Any]]:
    layers = sorted(scores)
    if not layers or not scores[layers[0]]:
        raise ValueError("Ada-KV scores cannot be empty")
    num_heads, prompt_length = len(scores[layers[0]]), len(scores[layers[0]][0])
    layer_budget = int(np.floor(num_heads * prompt_length * target_ratio))
    nominal_per_head = int(np.floor(prompt_length * target_ratio))
    safe = int(np.floor(nominal_per_head * safeguard))
    mapping: dict[int, dict[int, list[int]]] = {}
    for layer in layers:
        selected = {head: set() for head in range(num_heads)}
        for head in range(num_heads):
            values = scores[layer][head]
            selected[head].update(sorted(range(prompt_length), key=lambda index: (-values[index], index))[:safe])
        remaining = layer_budget - sum(len(values) for values in selected.values())
        candidates = []
        for head in range(num_heads):
            for index, value in enumerate(scores[layer][head]):
                if index not in selected[head]:
                    candidates.append((float(value), head, index))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        for _, head, index in candidates[:remaining]:
            selected[head].add(index)
        mapping[layer] = {head: sorted(selected[head]) for head in range(num_heads)}
    actual = sum(len(indices) for heads in mapping.values() for indices in heads.values())
    full = len(layers) * num_heads * prompt_length
    return mapping, {
        "target_kv_ratio": target_ratio, "actual_kv_ratio": actual / full,
        "retained_kv_pairs": actual, "fullkv_pairs": full,
        "per_layer_retained_kv_pairs": layer_budget, "safeguard": safeguard,
        "observation_window": 32, "pooling_kernel": 7,
        "allocation_granularity": "within_layer_across_kv_heads_then_token",
    }



def streamingllm_token_map(num_layers: int, num_kv_heads: int, prompt_length: int,
                            target_ratio: float, sink_tokens: int = 4) -> tuple[dict[int, dict[int, list[int]]], dict[str, Any]]:
    if min(num_layers, num_kv_heads, prompt_length) < 1:
        raise ValueError("StreamingLLM dimensions must be positive")
    if not 0.0 < target_ratio <= 1.0:
        raise ValueError("target ratio must be in (0, 1]")
    capacity = min(prompt_length, max(1, int(np.floor(prompt_length * target_ratio))))
    sink = min(sink_tokens, capacity)
    recent = capacity - sink
    retained = list(range(sink))
    if recent:
        retained.extend(range(prompt_length - recent, prompt_length))
    retained = sorted(set(retained))
    if len(retained) != capacity:
        raise RuntimeError("StreamingLLM sink and recent window overlap unexpectedly")
    mapping = {
        layer: {head: retained.copy() for head in range(num_kv_heads)}
        for layer in range(num_layers)
    }
    actual = num_layers * num_kv_heads * len(retained)
    full = num_layers * num_kv_heads * prompt_length
    return mapping, {
        "target_kv_ratio": target_ratio, "actual_kv_ratio": actual / full,
        "retained_kv_pairs": actual, "fullkv_pairs": full,
        "per_head_retained_tokens": capacity, "sink_tokens": sink,
        "recent_tokens": recent, "allocation_granularity": "fixed_sink_plus_recent_window",
    }


def snapkv_token_map(scores: dict[int, list[list[float]]], target_ratio: float) -> tuple[dict[int, dict[int, list[int]]], dict[str, Any]]:
    layers = sorted(scores)
    if not layers or not scores[layers[0]]:
        raise ValueError("SnapKV scores cannot be empty")
    num_heads, prompt_length = len(scores[layers[0]]), len(scores[layers[0]][0])
    per_head = int(np.floor(prompt_length * target_ratio))
    mapping = {
        layer: {
            head: sorted(range(prompt_length), key=lambda index: (-scores[layer][head][index], index))[:per_head]
            for head in range(num_heads)
        }
        for layer in layers
    }
    actual = sum(len(indices) for heads in mapping.values() for indices in heads.values())
    full = len(layers) * num_heads * prompt_length
    return mapping, {
        "target_kv_ratio": target_ratio, "actual_kv_ratio": actual / full,
        "retained_kv_pairs": actual, "fullkv_pairs": full,
        "per_head_retained_tokens": per_head, "observation_window": 32,
        "pooling_kernel": 7, "allocation_granularity": "fixed_per_kv_head_then_token",
    }

def pyramid_layer_capacities(num_layers: int, prompt_length: int, target_ratio: float,
                             beta: int = 20) -> list[int]:
    total = int(np.floor(num_layers * prompt_length * target_ratio))
    average = total / num_layers
    minimum = average / beta
    maximum = 2.0 * average - minimum
    weights = np.linspace(maximum, minimum, num_layers)
    floors = np.floor(weights).astype(int)
    remainder = total - int(floors.sum())
    capacities = floors.tolist()
    fractions = [(float(weights[index] - floors[index]), index) for index in range(num_layers)]
    for _, index in sorted(fractions, key=lambda item: (-item[0], item[1]))[:remainder]:
        capacities[index] += 1
    if any(capacity < 1 or capacity > prompt_length for capacity in capacities):
        raise ValueError("PyramidKV capacity is outside valid range")
    if sum(capacities) != total:
        raise RuntimeError("PyramidKV capacity sum does not match target budget")
    return capacities


def pyramidkv_token_map(scores: dict[int, list[list[float]]], target_ratio: float,
                        beta: int = 20, window_size: int = 64) -> tuple[dict[int, dict[int, list[int]]], dict[str, Any]]:
    layers = sorted(scores)
    num_heads, prompt_length = len(scores[layers[0]]), len(scores[layers[0]][0])
    capacities = pyramid_layer_capacities(len(layers), prompt_length, target_ratio, beta)
    mapping: dict[int, dict[int, list[int]]] = {}
    for layer, capacity in zip(layers, capacities):
        recent = min(window_size, capacity)
        recent_indices = set(range(prompt_length - recent, prompt_length))
        old_budget = capacity - recent
        mapping[layer] = {}
        for head in range(num_heads):
            values = scores[layer][head]
            old = sorted(range(prompt_length - recent), key=lambda index: (-values[index], index))[:old_budget]
            mapping[layer][head] = sorted(set(old) | recent_indices)
    actual = sum(len(indices) for heads in mapping.values() for indices in heads.values())
    return mapping, {
        "target_kv_ratio": target_ratio, "actual_kv_ratio": actual / (len(layers) * num_heads * prompt_length),
        "retained_kv_pairs": actual, "fullkv_pairs": len(layers) * num_heads * prompt_length,
        "beta": beta, "observation_window": window_size, "pooling_kernel": 5,
        "layer_capacities": capacities, "allocation_granularity": "layer_kv_head_token",
    }

def method_maps(num_layers: int, important: list[int], masses: dict[int, list[float]],
                candidate_count: int, random_repeats: int, seed: int,
                source_id: str, budgets: tuple[int, ...], prompt_length: int | None = None,
                a_tokens: int = 516, block_size: int = 128,
                target_kv_ratio: float | None = None) -> list[tuple[str, dict[int, list[int]] | None, int | None]]:
    all_a = {layer: [] for layer in range(num_layers)}
    variants: list[tuple[str, dict[int, list[int]] | None, int | None]] = [("fullkv", None, None), ("all_a", all_a, None)]
    for budget in budgets:
        mapping = {layer: [] for layer in range(num_layers)}
        for layer in important:
            mapping[layer] = top_blocks(masses[layer], budget)
        variants.append((f"selected_top{budget}", mapping, None))
    for repeat, selected in enumerate(random_block_maps(candidate_count, important, 8, random_repeats, seed, source_id)):
        mapping = {layer: [] for layer in range(num_layers)}
        mapping.update(selected)
        variants.append(("selected_random8", mapping, repeat))
    if target_kv_ratio is not None:
        if prompt_length is None:
            raise ValueError("prompt length is required for formula allocation")
        mapping, _ = formula_block_map(num_layers, prompt_length, a_tokens, block_size,
                                       target_kv_ratio, masses)
        variants.append((f"formula_waterfill_r{int(round(target_kv_ratio * 100))}", mapping, None))
    return variants


def generate(model: Any, patch: MistralAttentionPatch, cache: Any, initial_logits: Any,
             prompt_length: int, max_new_tokens: int, eos_ids: set[int],
             layer_map: dict[int, list[int]] | None,
             head_token_map: dict[int, dict[int, list[int]]] | None = None) -> tuple[list[int], float]:
    import torch
    device = model.get_input_embeddings().weight.device
    logits = initial_logits
    generated: list[int] = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(max_new_tokens):
        token_id = int(logits.argmax())
        generated.append(token_id)
        if token_id in eos_ids or step + 1 == max_new_tokens:
            break
        token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        with patch.mode(layer_allowed_blocks=layer_map, head_allowed_tokens=head_token_map,
                        baseline_prompt_length=prompt_length if head_token_map is not None else None):
            output = model_forward(model, token, cache, prompt_length + step)
        logits = output.logits[0, -1]
        del output, token
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    del cache, logits
    return generated, elapsed


def preprocess_prediction(task: str, prediction: str) -> str:
    if task in {"trec", "triviaqa", "samsum", "lsht"}:
        return prediction.lstrip("\n").split("\n")[0]
    return prediction


def score_prediction(metric: Any, prediction: str, answers: list[str], all_classes: list[str],
                     task: str | None = None) -> float:
    prepared = preprocess_prediction(task, prediction) if task is not None else prediction
    return 100.0 * max(float(metric(prepared, answer, all_classes=all_classes)) for answer in answers)


def summarize(rows: list[dict[str, Any]], args: Any) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
    sample_ids = sorted({str(row["source_id"]) for row in rows})
    full_by_id = {str(row["source_id"]): float(row["score"]) for row in by_method.get("fullkv", [])}
    all_a_by_id = {str(row["source_id"]): float(row["score"]) for row in by_method.get("all_a", [])}
    reports = {}
    for method, method_rows in sorted(by_method.items()):
        by_sample: dict[str, list[float]] = defaultdict(list)
        for row in method_rows:
            by_sample[str(row["source_id"])].append(float(row["score"]))
        sample_scores = {source_id: float(np.mean(values)) for source_id, values in by_sample.items()}
        method_sample_ids = sorted(sample_scores)
        values = [sample_scores[source_id] for source_id in method_sample_ids]
        comparable_ids = sorted(set(method_sample_ids) & set(full_by_id))
        deltas = [sample_scores[source_id] - full_by_id[source_id] for source_id in comparable_ids]
        reports[method] = {
            "scores": bootstrap(values, args.bootstrap_draws, args.seed),
            "per_sample": sample_scores,
            "delta_vs_fullkv": bootstrap(deltas, args.bootstrap_draws, args.seed),
            "average_output_tokens": float(np.mean([row["output_tokens"] for row in method_rows])),
            "average_decode_seconds": float(np.mean([row["decode_seconds"] for row in method_rows])),
            "repeats_per_sample": len(method_rows) // len(method_sample_ids),
        }
    full = reports.get("fullkv", {}).get("scores", {}).get("mean")
    all_a = reports.get("all_a", {}).get("scores", {}).get("mean")
    for method, report in reports.items():
        score = report["scores"]["mean"]
        report["quality_retention_vs_fullkv"] = score / full if full else None
        report["quality_restoration_from_all_a"] = (
            (score - all_a) / (full - all_a)
            if full is not None and all_a is not None and abs(full - all_a) > 1e-12 else None
        )
    return {
        "experiment": "mistral_longbench_free_generation_quality", "schema_version": SCHEMA_VERSION,
        "task": args.task,
        "samples": len(sample_ids), "sample_ids": sample_ids, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens, "important_layers": args.important_layers,
        "budgets": list(args.budgets), "random8_repeats": args.random_repeats,
        "block_size": args.block_size, "sink_tokens": args.sink_tokens, "local_tokens": args.local_tokens,
        "formula_target_kv_ratio": args.formula_target_kv_ratio,
        "formula_target_kv_ratios": args.formula_target_kv_ratios,
        "min_prompt_tokens": args.min_prompt_tokens,
        "require_answer_evidence": args.require_answer_evidence,
        "external_baselines": ["streamingllm_r20", "snapkv_r20", "ada_snapkv_r20", "pyramidkv_r20"],
        "flux_attention_status": "not_run_official_no_mistral_support_or_router_checkpoint",
        "metric": f"LongBench {args.task} official metric x100; maximum across references, macro mean across samples",
        "methods": reports,
        "caveats": [
            "Five samples provide descriptive evidence only.",
            "The Python attention mask preserves mathematical sparsity but does not implement a sparse kernel; latency is not a production speed measurement.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Free-generation LongBench quality matrix for Mistral sparse decode")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--longbench_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--scorer_path", type=Path, default=DEFAULT_SCORER)
    parser.add_argument("--scorer_site_packages", type=Path, default=DEFAULT_SITE_PACKAGES)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", choices=sorted(TASK_GENERATION_LENGTHS), default="gov_report")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_context_length", type=int, default=32768)
    parser.add_argument("--min_prompt_tokens", type=int, default=0)
    parser.add_argument("--require_answer_evidence", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=-1,
                        help="Override LongBench task default; <=0 uses the official task setting")
    parser.add_argument("--important_layers", type=int, default=8)
    parser.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    parser.add_argument("--random_repeats", type=int, default=5)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--sink_tokens", type=int, default=4)
    parser.add_argument("--local_tokens", type=int, default=512)
    parser.add_argument("--formula_target_kv_ratio", type=float, default=0.20)
    parser.add_argument("--formula_target_kv_ratios", nargs="+", type=float, default=None)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Run only named methods; dependencies for their selection are still computed")
    parser.add_argument("--bootstrap_draws", type=int, default=1000)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--max_memory", nargs="*", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_new_tokens <= 0:
        args.max_new_tokens = TASK_GENERATION_LENGTHS[args.task]
    if args.smoke:
        args.num_samples, args.max_new_tokens = 1, 16
        args.budgets, args.random_repeats, args.important_layers = [4, 8], 1, 2
    if any(budget < 1 for budget in args.budgets):
        raise ValueError("budgets must be positive")
    target_ratios = args.formula_target_kv_ratios or [args.formula_target_kv_ratio]
    if any(not 0.0 < ratio <= 1.0 for ratio in target_ratios):
        raise ValueError("formula target KV ratios must be in (0, 1]")
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
    records = prepare_records(
        args.longbench_root, tokenizer, args.task, args.num_samples, args.seed,
        args.max_context_length, args.max_new_tokens, args.min_prompt_tokens,
        args.require_answer_evidence,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "predictions.jsonl"
    existing = []
    if args.resume and output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            existing = [json.loads(line) for line in handle if line.strip()]
    elif output_path.exists() and output_path.stat().st_size:
        raise FileExistsError("outputs exist; use --resume or choose another output directory")
    completed = {(str(row["source_id"]), str(row["method"]), row.get("repeat")) for row in existing}
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    kwargs: dict[str, Any] = {"dtype": dtype, "device_map": args.device_map, "low_cpu_mem_usage": True,
                              "attn_implementation": args.attn_implementation}
    if (memory := parse_max_memory(args.max_memory)) is not None:
        kwargs["max_memory"] = memory
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs).eval()
    if not isinstance(model, MistralForCausalLM):
        raise TypeError("loaded model must be MistralForCausalLM")
    metric = load_scorer(args.scorer_path, args.scorer_site_packages, args.task)
    eos = model.generation_config.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(x) for x in (eos or [])}
    device = model.get_input_embeddings().weight.device
    rows = list(existing)
    for ordinal, record in enumerate(records, 1):
        blocks = complete_candidate_blocks(len(record["prompt_ids"]), args.block_size, args.sink_tokens, args.local_tokens)
        patch = MistralAttentionPatch(model, blocks, args.sink_tokens, args.local_tokens)
        try:
            cache = DynamicCache(config=model.config)
            prompt = torch.tensor([record["prompt_ids"]], dtype=torch.long, device=device)
            query = (record["question_span"]["query_start"], record["question_span"]["query_end"])
            requested_methods = set(args.methods or [])
            need_snap_scores = args.methods is None or bool(requested_methods & {"snapkv_r20", "ada_snapkv_r20"})
            need_pyramid_scores = args.methods is None or "pyramidkv_r20" in requested_methods
            score_configs = {}
            if need_snap_scores:
                score_configs["adakv"] = (32, 7, "max")
            if need_pyramid_scores:
                score_configs["pyramidkv"] = (64, 5, "avg")
            with torch.inference_mode(), patch.mode(
                prefill_query=query, prompt_score_configs=score_configs or None
            ):
                output = model_forward(model, prompt, cache, 0)
            initial_logits = output.logits[0, -1].detach()
            masses = {layer: patch.prefill_masses[layer]["block_mass"] for layer in range(32)}
            layer_ranking = sorted(
                range(32), key=lambda layer: (-patch.prefill_masses[layer]["remote_mass"], layer)
            )
            important = layer_ranking[:args.important_layers]
            variants = method_maps(
                32, important, masses, len(blocks), args.random_repeats, args.seed,
                record["source_id"], tuple(args.budgets), len(record["prompt_ids"]),
                args.sink_tokens + args.local_tokens, args.block_size, None,
            )
            streamingllm_map, streamingllm_details = streamingllm_token_map(
                32, int(config.num_key_value_heads), len(record["prompt_ids"]),
                args.formula_target_kv_ratio, args.sink_tokens,
            )
            snapkv_map = adakv_map = pyramidkv_map = None
            snapkv_details = adakv_details = pyramidkv_details = None
            if need_snap_scores:
                snapkv_map, snapkv_details = snapkv_token_map(
                    patch.prompt_token_scores["adakv"], args.formula_target_kv_ratio,
                )
                adakv_map, adakv_details = adakv_token_map(
                    patch.prompt_token_scores["adakv"], args.formula_target_kv_ratio, 0.2,
                )
            if need_pyramid_scores:
                pyramidkv_map, pyramidkv_details = pyramidkv_token_map(
                    patch.prompt_token_scores["pyramidkv"], args.formula_target_kv_ratio, 20, 64,
                )
            allocation_details_by_variant = {}
            for ratio in target_ratios:
                ratio_suffix = int(round(ratio * 100))
                random_method = f"random_budget_r{ratio_suffix}"
                for repeat in range(args.random_repeats):
                    mapping, details = random_budget_block_map(
                        32, len(record["prompt_ids"]), args.sink_tokens + args.local_tokens,
                        args.block_size, ratio, len(blocks), args.seed,
                        record["source_id"], repeat,
                    )
                    variants.append((random_method, mapping, repeat))
                    allocation_details_by_variant[(random_method, repeat)] = details
                allocation_specs = (
                    (f"uniform_budget_r{ratio_suffix}", "uniform"),
                    (f"top_layer_budget_r{ratio_suffix}", "ranked_fill"),
                )
                for method_name, allocation in allocation_specs:
                    mapping, details = fixed_budget_block_map(
                        32, len(record["prompt_ids"]), args.sink_tokens + args.local_tokens,
                        args.block_size, ratio, masses, allocation, layer_ranking,
                    )
                    variants.append((method_name, mapping, None))
                    allocation_details_by_variant[(method_name, None)] = details
                method_name = f"formula_waterfill_r{ratio_suffix}"
                mapping, details = formula_block_map(
                    32, len(record["prompt_ids"]), args.sink_tokens + args.local_tokens,
                    args.block_size, ratio, masses,
                )
                variants.append((method_name, mapping, None))
                allocation_details_by_variant[(method_name, None)] = details
            variants.extend([("streamingllm_r20", None, None), ("snapkv_r20", None, None),
                             ("ada_snapkv_r20", None, None), ("pyramidkv_r20", None, None)])
            if args.methods is not None:
                requested = set(args.methods)
                available = {method for method, _, _ in variants}
                unknown = sorted(requested - available)
                if unknown:
                    raise ValueError(f"unknown requested methods: {unknown}; available={sorted(available)}")
                variants = [variant for variant in variants if variant[0] in requested]
            del output, prompt
            for method, layer_map, repeat in variants:
                key = (record["source_id"], method, repeat)
                if key in completed:
                    continue
                branch_cache = clone_dynamic_cache(cache)
                head_token_map = (streamingllm_map if method == "streamingllm_r20" else
                                  snapkv_map if method == "snapkv_r20" else
                                  adakv_map if method == "ada_snapkv_r20" else
                                  pyramidkv_map if method == "pyramidkv_r20" else None)
                with torch.inference_mode():
                    generated, elapsed = generate(model, patch, branch_cache, initial_logits,
                                                  len(record["prompt_ids"]), args.max_new_tokens,
                                                  eos_ids, layer_map, head_token_map)
                prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
                score = score_prediction(metric, prediction, record["answers"], record["all_classes"], args.task)
                selected = None if layer_map is None else {str(layer): values for layer, values in layer_map.items() if values}
                row = {
                    "schema_version": SCHEMA_VERSION, "task": args.task, "source_id": record["source_id"],
                    "prompt_hash": record["prompt_hash"], "prompt_tokens": len(record["prompt_ids"]),
                    "answer_evidence_present": record["answer_evidence_present"],
                    "candidate_blocks": len(blocks), "important_layers": important,
                    "question_span": record["question_span"],
                    "method": method, "repeat": repeat, "selected_blocks": selected,
                    "allocation": (allocation_details_by_variant.get((method, repeat)) if (method, repeat) in allocation_details_by_variant else
                                   streamingllm_details if method == "streamingllm_r20" else
                                   snapkv_details if method == "snapkv_r20" else
                                   adakv_details if method == "ada_snapkv_r20" else
                                   pyramidkv_details if method == "pyramidkv_r20" else None),
                    "generated_token_ids": generated, "generation_hash": token_hash(generated),
                    "prediction": prediction, "output_tokens": len(generated), "decode_seconds": elapsed,
                    "score": score,
                }
                rows.append(row)
                atomic_write_jsonl(output_path, rows)
                completed.add(key)
                print(f"[{ordinal}/{len(records)}] {record['source_id']} {method} repeat={repeat} tokens={len(generated)} score={score:.4f}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del cache, initial_logits
        finally:
            patch.close(model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary = summarize(rows, args)
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
