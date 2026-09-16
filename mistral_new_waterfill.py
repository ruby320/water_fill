from __future__ import annotations

from typing import Any

import numpy as np

from mistral_task_quality import coverage_count, waterfill_block_counts


def head_block_budget(
    num_layers: int,
    num_kv_heads: int,
    prompt_length: int,
    fixed_tokens: int,
    block_size: int,
    target_ratio: float,
    candidate_count: int,
) -> int:
    """Return the exact remote head-block budget under a global KV ratio."""
    if not 0.0 < target_ratio <= 1.0:
        raise ValueError("target_ratio must be in (0, 1]")
    if min(num_layers, num_kv_heads, prompt_length, block_size) < 1:
        raise ValueError("budget dimensions must be positive")
    if fixed_tokens < 0 or candidate_count < 0:
        raise ValueError("fixed_tokens and candidate_count must be non-negative")
    available = num_layers * num_kv_heads * candidate_count
    fixed = min(fixed_tokens, prompt_length)
    token_budget = num_layers * num_kv_heads * (prompt_length * target_ratio - fixed)
    return min(available, max(0, int(np.floor(token_budget / block_size))))


def _validate_masses(
    masses: dict[int, list[list[float]]],
) -> tuple[list[int], int, int]:
    layers = sorted(masses)
    if layers != list(range(len(layers))) or not layers:
        raise ValueError("masses must contain contiguous layers starting at zero")
    num_heads = len(masses[0])
    if num_heads < 1:
        raise ValueError("each layer must contain at least one KV head")
    candidate_count = len(masses[0][0])
    for layer in layers:
        if len(masses[layer]) != num_heads:
            raise ValueError("all layers must share the same KV-head count")
        if any(len(values) != candidate_count for values in masses[layer]):
            raise ValueError("all layer-heads must share the same block count")
    return layers, num_heads, candidate_count


def layer_masses_from_heads(
    masses: dict[int, list[list[float]]],
) -> dict[int, list[float]]:
    """Average KV-head block masses to obtain the layer allocation signal."""
    layers, num_heads, candidate_count = _validate_masses(masses)
    return {
        layer: [
            float(sum(masses[layer][head][block] for head in range(num_heads)) / num_heads)
            for block in range(candidate_count)
        ]
        for layer in layers
    }


def hierarchical_head_block_counts(
    masses: dict[int, list[list[float]]], total_budget: int
) -> tuple[dict[int, dict[int, int]], dict[str, Any]]:
    """Allocate head-blocks first across layers and then across heads per layer."""
    layers, num_heads, candidate_count = _validate_masses(masses)
    capacity_per_layer = num_heads * candidate_count
    if not 0 <= total_budget <= len(layers) * capacity_per_layer:
        raise ValueError("total_budget is outside global capacity")

    # Use the original layer-level Water-fill curve, expanded to head-block units.
    layer_signal = layer_masses_from_heads(masses)
    expanded_layer_signal = {
        layer: [mass / num_heads for mass in layer_signal[layer] for _ in range(num_heads)]
        for layer in layers
    }
    layer_counts, layer_parameters = waterfill_block_counts(expanded_layer_signal, total_budget)

    counts: dict[int, dict[int, int]] = {}
    head_parameters: dict[int, dict[int, dict[str, float]]] = {}
    for layer in layers:
        layer_budget = layer_counts[layer]
        head_values = {head: masses[layer][head] for head in range(num_heads)}
        head_counts, parameters = waterfill_block_counts(head_values, layer_budget)
        counts[layer] = head_counts
        head_parameters[layer] = parameters

    allocated = sum(count for heads in counts.values() for count in heads.values())
    if allocated != total_budget:
        raise RuntimeError("hierarchical allocation did not preserve the exact budget")
    return counts, {
        "layer_head_block_counts": counts,
        "layer_head_block_budgets": layer_counts,
        "layer_parameters": layer_parameters,
        "head_parameters": head_parameters,
    }


def new_waterfill_token_map(
    masses: dict[int, list[list[float]]],
    blocks: list[dict[str, int]],
    prompt_length: int,
    fixed_tokens: int,
    block_size: int,
    target_ratio: float,
) -> tuple[dict[int, dict[int, list[int]]], dict[str, Any]]:
    """Build a head-specific token map using hierarchical 32-token Water-fill."""
    layers, num_heads, candidate_count = _validate_masses(masses)
    if candidate_count != len(blocks):
        raise ValueError("block metadata does not match head-block masses")
    if any(block["end"] - block["start"] != block_size for block in blocks):
        raise ValueError("all candidate blocks must be complete blocks")

    budget = head_block_budget(
        len(layers), num_heads, prompt_length, fixed_tokens,
        block_size, target_ratio, candidate_count,
    )
    counts, details = hierarchical_head_block_counts(masses, budget)
    fixed = min(fixed_tokens, prompt_length)
    sink = min(4, fixed)
    local = fixed - sink
    fixed_indices = set(range(sink))
    if local:
        fixed_indices.update(range(prompt_length - local, prompt_length))

    mapping: dict[int, dict[int, list[int]]] = {}
    for layer in layers:
        mapping[layer] = {}
        for head in range(num_heads):
            values = masses[layer][head]
            selected = sorted(
                range(candidate_count), key=lambda index: (-float(values[index]), index)
            )[:counts[layer][head]]
            indices = set(fixed_indices)
            for block_index in selected:
                block = blocks[block_index]
                indices.update(range(block["start"], block["end"]))
            mapping[layer][head] = sorted(indices)

    retained = sum(len(indices) for heads in mapping.values() for indices in heads.values())
    full = len(layers) * num_heads * prompt_length
    details.update({
        "method": "new_waterfill",
        "target_kv_ratio": target_ratio,
        "actual_kv_ratio": retained / full,
        "retained_kv_pairs": retained,
        "fullkv_pairs": full,
        "total_head_block_budget": budget,
        "allocated_head_blocks": sum(
            count for heads in counts.values() for count in heads.values()
        ),
        "block_size": block_size,
        "sink_tokens": sink,
        "local_tokens": local,
        "allocation_granularity": "layer_then_kv_head_then_contiguous_block",
        "allocation_rule": "hierarchical_waterfill_remote_mass_k50",
    })
    return mapping, details


def new_waterfill_block_map(
    num_layers: int,
    prompt_length: int,
    fixed_tokens: int,
    block_size: int,
    target_ratio: float,
    masses: dict[int, list[float]],
) -> tuple[dict[int, list[int]], dict[str, Any]]:
    """Original layer-level Water-fill with a finer contiguous block size."""
    from mistral_task_quality import formula_block_map

    mapping, details = formula_block_map(
        num_layers, prompt_length, fixed_tokens, block_size, target_ratio, masses
    )
    details.update({
        "method": "new_waterfill",
        "head_aware": False,
        "allocation_granularity": "layer_then_shared_contiguous_block",
        "all_kv_heads_share_layer_blocks": True,
        "fixed_set_a_applied_to_every_layer": True,
    })
    return mapping, details
