from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

DEFAULT_INPUT = Path("results/gov_report_remote_attention")
DEFAULT_OUTPUT = Path("results/ab_remote_evidence")
DEFAULT_ALLOCATION_SUMMARY = Path(
    "results/gov_report_allocation_ablation_r20_v1/summary.json"
)
DEFAULT_ALLOCATION_PREDICTIONS = Path(
    "results/gov_report_allocation_ablation_r20_v1/predictions.jsonl"
)
ALLOCATION_METHODS = (
    ("all_a", "A-only"),
    ("random_budget_r20", "Random B"),
    ("top_layer_budget_r20", "Top-layer B"),
    ("formula_waterfill_r20", "Water-fill"),
)
B_ALLOCATION_METHODS = tuple(method for method, _ in ALLOCATION_METHODS[1:])
REQUIRED_FILES = (
    "predictions.jsonl",
    "prefill_scores.jsonl",
    "checkpoint_metrics.jsonl",
    "q1_interventions.jsonl",
    "q2_interventions.jsonl",
)
BRANCHES = ("A_only", "random", "prefill", "oracle")
BRANCH_LABELS = ("A-only", "A+Random", "A+Selected", "A+Oracle")
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "gray": "#7A7A7A",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"Required input is empty: {path}")
    return rows


def load_data(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    missing = [name for name in REQUIRED_FILES if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {input_dir}: {', '.join(missing)}")
    return {name: read_jsonl(input_dir / name) for name in REQUIRED_FILES}


def grouped_mean(rows: Iterable[dict[str, Any]], value_fn: Any) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_id"])].append(float(value_fn(row)))
    return {source: float(np.mean(values)) for source, values in sorted(grouped.items())}


def mean_sem(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    mean = float(array.mean())
    sem = float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0
    return mean, sem


def matched_important_kl(
    data: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[tuple[str, int, int], float]]:
    predictions = {str(row["source_id"]): row for row in data["predictions.jsonl"]}
    important = {
        source: set(map(int, row["important_layers"]))
        for source, row in predictions.items()
    }
    cells: dict[str, dict[tuple[str, int, int], float]] = {branch: {} for branch in BRANCHES}
    for row in data["q1_interventions.jsonl"]:
        source = str(row["source_id"])
        key = (source, int(row["checkpoint"]), int(row["layer_idx"]))
        if key[2] in important[source]:
            cells["A_only"][key] = float(row["kl_mean"])

    repeated: dict[tuple[str, tuple[str, int, int]], list[float]] = defaultdict(list)
    for row in data["q2_interventions.jsonl"]:
        branch = str(row["branch"])
        if branch not in BRANCHES[1:]:
            continue
        source = str(row["source_id"])
        key = (source, int(row["checkpoint"]), int(row["layer_idx"]))
        if key[2] in important[source]:
            repeated[(branch, key)].append(float(row["kl_mean"]))
    for (branch, key), values in repeated.items():
        cells[branch][key] = float(np.mean(values))

    common = set(cells["A_only"])
    for branch in BRANCHES[1:]:
        common &= set(cells[branch])
    if not common:
        raise ValueError("No matched important-layer/checkpoint intervention cells")
    return {branch: {key: cells[branch][key] for key in sorted(common)} for branch in BRANCHES}



def build_representative_heatmap(
    data: dict[str, list[dict[str, Any]]], position_bins: int | None = None
) -> dict[str, Any]:
    del position_bins  # Kept for compatibility with older callers.
    predictions = {str(row["source_id"]): row for row in data["predictions.jsonl"]}
    checkpoint_rows = data["checkpoint_metrics.jsonl"]
    source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in checkpoint_rows:
        source_rows[str(row["source_id"])].append(row)
    remote_dependence = {}
    for source, rows in sorted(source_rows.items()):
        important = set(map(int, predictions[source]["important_layers"]))
        observed = [float(row["decode_remote_mass"]) for row in rows if int(row["layer_idx"]) in important]
        if not observed:
            raise ValueError(f"No important-layer checkpoint rows for source {source}")
        remote_dependence[source] = float(np.mean(observed))
    median_value = float(np.median(list(remote_dependence.values())))
    source_id = min(remote_dependence, key=lambda source: (abs(remote_dependence[source] - median_value), source))
    prompt_tokens = int(predictions[source_id]["prompt_tokens"])
    important_layers = sorted(map(int, predictions[source_id]["important_layers"]))
    source_prefill = {int(row["layer_idx"]): row for row in data["prefill_scores.jsonl"] if str(row["source_id"]) == source_id}
    source_checkpoints = [row for row in checkpoint_rows if str(row["source_id"]) == source_id]
    layers = sorted(source_prefill)
    if layers != list(range(32)):
        raise ValueError(f"Expected all 32 layers for source {source_id}, got {layers}")
    blocks = source_prefill[0]["blocks"]
    block_indices = [int(block["block_idx"]) for block in blocks]
    if block_indices != list(range(len(blocks))):
        raise ValueError("Candidate blocks must use original contiguous index order")
    token_edges = [int(blocks[0]["start"])] + [int(block["end"]) for block in blocks]
    heatmap = np.empty((32, len(blocks)), dtype=float)
    checkpoints_per_layer = {}
    for output_row, layer in enumerate(layers):
        if source_prefill[layer]["blocks"] != blocks:
            raise ValueError(f"Candidate block geometry differs at layer {layer}")
        rows = [row for row in source_checkpoints if int(row["layer_idx"]) == layer]
        checkpoints_per_layer[str(layer)] = len(rows)
        if len(rows) != 8:
            raise ValueError(f"Expected 8 checkpoints at layer {layer}, got {len(rows)}")
        masses = np.asarray([row["decode_block_mass"] for row in rows], dtype=float)
        if masses.shape[1] != len(blocks):
            raise ValueError(f"Block-mass length mismatch at layer {layer}")
        heatmap[output_row] = masses.mean(axis=0)
    return {
        "selection_rule": "Choose the sample whose mean decode remote mass over its prefill-defined important layers and all checkpoints is nearest the median across the five samples.",
        "selection_metric": "sample-level mean important-layer decode_remote_mass",
        "selection_observations": remote_dependence, "selection_median": median_value,
        "source_id": source_id, "prompt_tokens": prompt_tokens,
        "important_layers": important_layers, "sink_start": 0, "sink_end": 4,
        "recent_boundary": prompt_tokens - 512, "recent_tokens": 512,
        "complete_remote_start": token_edges[0], "complete_remote_end": token_edges[-1],
        "raw_block_count": len(blocks), "block_indices": block_indices,
        "block_edges_tokens": token_edges,
        "block_edges_normalized": [edge / prompt_tokens for edge in token_edges],
        "layer_indices": layers, "checkpoints_per_layer": checkpoints_per_layer,
        "checkpoint_average": "Arithmetic mean of each raw decode_block_mass cell over 8 checkpoints within each layer; no selection, truncation, interpolation, or per-layer normalization.",
        "normalization": "Raw token positions divided by prompt_tokens on x; one row per model layer; one column per saved complete candidate block; global LogNorm over all positive cells.",
        "heatmap_mean_attention_mass": heatmap.tolist(),
        "display_scaling": "Global LogNorm using the minimum and maximum positive cell values.",
        "coverage_note": "Only saved complete remote blocks are colored; sink A is a not-to-scale gray annotation, recent-512 A is shown at true token-position scale, and uncovered remote boundary intervals remain blank.",
    }


def compute_statistics(data: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    predictions = {str(row["source_id"]): row for row in data["predictions.jsonl"]}
    source_ids = sorted(predictions)
    important = {
        source: set(map(int, row["important_layers"]))
        for source, row in predictions.items()
    }
    prefill_rows = data["prefill_scores.jsonl"]
    checkpoint_rows = data["checkpoint_metrics.jsonl"]

    panel_a_rows = {
        "prefill_remote_mass": prefill_rows,
        "decode_remote_mass": checkpoint_rows,
        "important_layer_decode_remote_mass": [
            row for row in checkpoint_rows
            if int(row["layer_idx"]) in important[str(row["source_id"])]
        ],
    }
    panel_a = {}
    for name, rows in panel_a_rows.items():
        field = "prefill_remote_mass" if name == "prefill_remote_mass" else "decode_remote_mass"
        per_sample = grouped_mean(rows, lambda row, field=field: row[field])
        mean, sem = mean_sem(per_sample.values())
        panel_a[name] = {
            "aggregate_mean": float(np.mean([float(row[field]) for row in rows])),
            "sample_mean": mean,
            "sample_sem": sem,
            "per_sample": per_sample,
            "sample_aggregation": "Mean over layers/checkpoints within each sample.",
        }

    cells = matched_important_kl(data)
    sums = {branch: float(sum(values.values())) for branch, values in cells.items()}
    aggregate_residual = {branch: 100.0 * sums[branch] / sums["A_only"] for branch in BRANCHES}
    sample_residual: dict[str, dict[str, float]] = {}
    for source in source_ids:
        sample_means = {branch: float(np.mean([value for key, value in values.items() if key[0] == source])) for branch, values in cells.items()}
        sample_residual[source] = {branch: 100.0 * sample_means[branch] / sample_means["A_only"] for branch in BRANCHES}
    panel_b = {
        "aggregate_residual_percent": aggregate_residual,
        "aggregate_recovery_percent": {branch: 100.0 - aggregate_residual[branch] for branch in BRANCHES},
        "per_sample_residual_percent": sample_residual,
        "matched_cell_count": len(cells["A_only"]),
        "aggregation": "Ratio of KL sums over cells jointly observed for all branches; random repeats are averaged within cell before matching.",
        "sample_aggregation": "For each sample, branch mean KL divided by A-only mean KL.",
        "sample_ratio_range": [min(v for sample in sample_residual.values() for v in sample.values()), max(v for sample in sample_residual.values() for v in sample.values())],
    }

    random_rows = [row for row in data["q2_interventions.jsonl"] if row.get("branch") == "random"]
    b_budget = int(round(np.mean([len(row["selected_b_blocks"]) for row in random_rows])))
    return {
        "title": "Important remote KV exists beyond set A",
        "task": next(iter(predictions.values())).get("task", "unknown"),
        "model": "Mistral-7B-Instruct-v0.2",
        "n_samples": len(source_ids),
        "source_ids": source_ids,
        "b_budget_blocks": b_budget,
        "a_definition": "sink + recent 512 tokens",
        "error_bars": "Sample standard error (SEM) across n=5 sample-level means.",
        "evidence_scope": "Descriptive evidence; no inferential significance claims.",
        "panel_a": panel_a,
        "panel_b_kl_recovery": panel_b,
        "representative_heatmap": build_representative_heatmap(data),
    }



def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_allocation_ablation(
    summary_path: Path, predictions_path: Path, representative_source: str
) -> dict[str, Any]:
    summary = read_json(summary_path)
    sample_ids = [str(value) for value in summary.get("sample_ids", [])]
    if summary.get("task") != "gov_report" or len(sample_ids) != 5 or len(set(sample_ids)) != 5:
        raise ValueError("Allocation ablation requires five unique GovReport samples")

    required_methods = [method for method, _ in ALLOCATION_METHODS] + ["fullkv"]
    methods = {
        label: {
            **_validated_method(summary, method, sample_ids, summary_path),
            "method_key": method,
            "label": label,
        }
        for method, label in ALLOCATION_METHODS
    }
    fullkv = _validated_method(summary, "fullkv", sample_ids, summary_path)
    expected_repeats = {
        method: int(summary["methods"][method].get("repeats_per_sample", 1))
        for method in required_methods
    }
    if expected_repeats["random_budget_r20"] != 5:
        raise ValueError("Random B requires exactly five repeats per sample")
    if any(expected_repeats[method] != 1 for method in required_methods if method != "random_budget_r20"):
        raise ValueError("Only Random B may contain repeated allocation rows")

    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    prompt_signatures: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(predictions_path):
        method = str(row.get("method"))
        source_id = str(row.get("source_id"))
        if method not in required_methods or source_id not in sample_ids:
            continue
        grouped_rows[(source_id, method)].append(row)
        signature = {
            "task": str(row.get("task")),
            "prompt_hash": str(row.get("prompt_hash")),
            "prompt_tokens": int(row.get("prompt_tokens")),
        }
        if signature["task"] != summary["task"]:
            raise ValueError(f"Task mismatch for {method}/{source_id}")
        previous = prompt_signatures.setdefault(source_id, signature)
        if previous != signature:
            raise ValueError(f"Methods do not share prompt metadata for {source_id}")

    expected = {(source_id, method) for source_id in sample_ids for method in required_methods}
    if set(grouped_rows) != expected:
        missing = sorted(expected - set(grouped_rows))
        raise ValueError(f"Incomplete matched allocation run: missing {missing}")

    allocations: dict[str, dict[str, list[float] | list[int]]] = {}
    budget_validation: dict[str, Any] = {}
    for source_id in sample_ids:
        source_allocations: dict[str, list[float] | list[int]] = {}
        totals: dict[str, int] = {}
        allocated: dict[str, int] = {}
        for method in required_methods:
            method_rows = grouped_rows[(source_id, method)]
            repeat_count = expected_repeats[method]
            if len(method_rows) != repeat_count:
                raise ValueError(
                    f"Expected {repeat_count} rows for {method}/{source_id}, got {len(method_rows)}"
                )
            if method == "random_budget_r20":
                repeat_ids = sorted(int(row.get("repeat")) for row in method_rows)
                if repeat_ids != list(range(repeat_count)):
                    raise ValueError(f"Random repeat IDs must be 0..{repeat_count - 1} for {source_id}")
            score_mean = float(np.mean([float(row["score"]) for row in method_rows]))
            if not np.isclose(score_mean, methods.get("Random B", fullkv)["per_sample"][source_id]
                              if method == "random_budget_r20"
                              else (fullkv if method == "fullkv" else next(
                                  value for value in methods.values() if value["method_key"] == method
                              ))["per_sample"][source_id], rtol=0.0, atol=1e-10):
                raise ValueError(f"Prediction/summary score mismatch for {method}/{source_id}")
            if method not in B_ALLOCATION_METHODS:
                continue

            repeat_counts: list[list[int]] = []
            for row in method_rows:
                allocation = row.get("allocation")
                if not isinstance(allocation, dict):
                    raise ValueError(f"Missing allocation for {method}/{source_id}")
                counts_raw = allocation.get("layer_block_counts")
                if not isinstance(counts_raw, dict):
                    raise ValueError(f"Missing layer_block_counts for {method}/{source_id}")
                counts = [int(counts_raw.get(str(layer), 0)) for layer in range(32)]
                if any(count < 0 for count in counts):
                    raise ValueError(f"Negative allocation for {method}/{source_id}")
                total = int(allocation["total_layer_block_budget"])
                used = int(allocation["allocated_layer_blocks"])
                if sum(counts) != used or used != total:
                    raise ValueError(f"Allocation sum/budget mismatch for {method}/{source_id}")

                selected = row.get("selected_blocks")
                if not isinstance(selected, dict):
                    raise ValueError(f"Missing selected_blocks for {method}/{source_id}")
                selected_total = 0
                candidate_blocks = int(row["candidate_blocks"])
                for layer in range(32):
                    values = selected.get(str(layer), [])
                    if not isinstance(values, list) or len(values) != len(set(values)):
                        raise ValueError(f"Duplicate or invalid selected blocks for {method}/{source_id}/L{layer}")
                    if any(not isinstance(value, int) or value < 0 or value >= candidate_blocks for value in values):
                        raise ValueError(f"Selected block out of range for {method}/{source_id}/L{layer}")
                    if len(values) != counts[layer]:
                        raise ValueError(f"Selected-block/count mismatch for {method}/{source_id}/L{layer}")
                    selected_total += len(values)
                if selected_total != total:
                    raise ValueError(f"Selected pair count mismatch for {method}/{source_id}")
                repeat_counts.append(counts)
                totals[f"{method}:{len(repeat_counts) - 1}"] = total
                allocated[f"{method}:{len(repeat_counts) - 1}"] = used

            source_allocations[method] = (
                np.mean(np.asarray(repeat_counts, dtype=float), axis=0).tolist()
                if method == "random_budget_r20"
                else repeat_counts[0]
            )
        if len(set(totals.values())) != 1 or len(set(allocated.values())) != 1:
            raise ValueError(f"B methods use unequal budgets for {source_id}")
        allocations[source_id] = source_allocations
        budget_validation[source_id] = {
            "total_layer_block_budget": next(iter(totals.values())),
            "allocated_layer_blocks": next(iter(allocated.values())),
            "methods_equal": True,
            "random_repeats": expected_repeats["random_budget_r20"],
        }

    if representative_source not in allocations:
        raise ValueError("Representative panel-(a) source is absent from allocation run")
    winner = max(methods, key=lambda label: methods[label]["mean"])
    water_delta = methods["Water-fill"]["mean"] - methods["Random B"]["mean"]
    return {
        "metric": "LongBench GovReport ROUGE-L x100",
        "aggregation": "Arithmetic mean of five matched sample-level official LongBench scores; Random B first averages five repeats within each sample.",
        "source_summary": str(summary_path.resolve()),
        "source_predictions": str(predictions_path.resolve()),
        "methods": methods,
        "fullkv": {**fullkv, "method_key": "fullkv", "label": "FullKV"},
        "winner": methods[winner]["method_key"],
        "winner_label": winner,
        "waterfill_vs_random_delta": water_delta,
        "validation": {
            "sample_ids": sample_ids,
            "n_samples": 5,
            "prompt_signatures": prompt_signatures,
            "all_five_methods_same_source_and_prompt": True,
            "budget_consistency": budget_validation,
            "all_b_methods_equal_budget_and_fully_allocated": True,
            "random_repeats_per_sample": expected_repeats["random_budget_r20"],
            "selected_block_pairs_validated": True,
        },
        "representative_allocation": {
            "source_id": representative_source,
            "total_layer_block_budget": budget_validation[representative_source]["total_layer_block_budget"],
            "layer_indices": list(range(32)),
            "layer_block_counts": allocations[representative_source],
            "allocation_method": {
                "random_budget_r20": "Per-layer arithmetic mean across the representative source's five random draws; counts may be fractional.",
                "top_layer_budget_r20": "Single run.",
                "formula_waterfill_r20": "Single run.",
            },
        },
    }

def _validated_method(
    summary: dict[str, Any], method: str, sample_ids: list[str], source_path: Path
) -> dict[str, Any]:
    if method not in summary.get("methods", {}):
        raise ValueError(f"Missing method {method!r} in {source_path}")
    raw = summary["methods"][method]
    per_sample = {str(key): float(value) for key, value in raw.get("per_sample", {}).items()}
    if set(per_sample) != set(sample_ids):
        raise ValueError(f"Method {method!r} in {source_path} is incomplete")
    ordered = {source_id: per_sample[source_id] for source_id in sample_ids}
    recomputed_mean = float(np.mean(list(ordered.values())))
    reported_mean = float(raw["scores"]["mean"])
    if not np.isclose(recomputed_mean, reported_mean, rtol=0.0, atol=1e-10):
        raise ValueError(
            f"Mean mismatch for {method!r} in {source_path}: "
            f"reported={reported_mean}, recomputed={recomputed_mean}"
        )
    return {"mean": recomputed_mean, "per_sample": ordered, "repeats": int(raw.get("repeats_per_sample", 1))}


def style_axis(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8.5)


def draw_plot(stats: dict[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(13.8, 5.3))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.0], wspace=0.30)
    ax_left = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[0.60, 0.40], hspace=0.78)
    ax_quality = fig.add_subplot(right[0, 0])
    ax_allocation = fig.add_subplot(right[1, 0])
    ax = ax_left; representative = stats["representative_heatmap"]
    heatmap = np.asarray(representative["heatmap_mean_attention_mass"], dtype=float)
    positive = heatmap[np.isfinite(heatmap) & (heatmap > 0.0)]
    norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    x_edges = np.asarray(representative["block_edges_normalized"], dtype=float)
    y_edges = np.arange(len(representative["layer_indices"]) + 1) - 0.5
    image = ax.pcolormesh(x_edges, y_edges, np.ma.masked_less_equal(heatmap, 0.0), cmap="magma", norm=norm, shading="flat", edgecolors=(1.0, 1.0, 1.0, 0.68), linewidth=0.10, antialiased=True, rasterized=True, zorder=2)
    prompt_tokens = representative["prompt_tokens"]
    recent_boundary = representative["recent_boundary"] / prompt_tokens
    ax.axvspan(recent_boundary, 1.0, color="#D3D3D3", alpha=0.92, zorder=3)
    ax.axvline(recent_boundary, color="#333333", linestyle=(0, (4, 3)), linewidth=1.0, zorder=5)
    sink_display_end = 0.020
    ax.axvspan(0.0, sink_display_end, color="#C8C8C8", alpha=0.96, zorder=4)
    ax.axvline(sink_display_end, color="#333333", linestyle=(0, (4, 3)), linewidth=1.0, zorder=5)
    ax.text(sink_display_end / 2, 0.50, "A: sink", rotation=90, transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=6.7, color="#333333", fontweight="bold", zorder=6)
    ax.annotate("width enlarged for visibility", xy=(sink_display_end, 0.91), xycoords=ax.get_xaxis_transform(), xytext=(0.085, 0.91), textcoords=ax.get_xaxis_transform(), fontsize=6.6, ha="left", va="center", color="#333333", arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.7}, zorder=6)
    ax.text((recent_boundary + 1.0) / 2, 0.50, "A: recent 512 tokens", rotation=90, transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=7.1, color="#333333", fontweight="bold", zorder=6)
    remote_center = (sink_display_end + recent_boundary) / 2
    ax.text(remote_center, 0.965, "Remote KV outside A", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.0, color="#222222", fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5}, zorder=7)
    ax.set_xlabel("KV position in prompt (0 = start, 1 = end)"); ax.set_ylabel("Layer")
    layers = representative["layer_indices"]
    ax.set_yticks(np.arange(len(layers)), [f"L{layer}" for layer in layers])
    important = set(representative["important_layers"])
    for tick, layer in zip(ax.get_yticklabels(), layers):
        if layer in important: tick.set_color(COLORS["blue"]); tick.set_fontweight("bold")
    important_rows = [i for i, layer in enumerate(layers) if layer in important]
    ax.scatter(np.full(len(important_rows), -0.014), important_rows, s=10, color=COLORS["blue"], marker="o", clip_on=False, zorder=7)
    ax.legend(handles=[Line2D([0], [0], marker="o", linestyle="none", markersize=4.0, color=COLORS["blue"], label="Remote-important layer")], loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.76, fontsize=6.8, handletextpad=0.30, borderpad=0.25)
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, len(layers) - 0.5)
    ax.tick_params(axis="y", labelsize=5.8, pad=2); ax.tick_params(axis="x", labelsize=8.0)
    ax.set_title("(a) Remote attention by layer", loc="left", fontweight="bold", pad=5)
    colorbar = fig.colorbar(image, ax=ax, pad=0.016, fraction=0.040)
    colorbar.set_label("Mean decode attention mass per block", fontsize=8.0); colorbar.ax.tick_params(labelsize=7.0)
    ax.text(0.005, -0.155, "Gray = set A; colored cells = remote KV outside A", transform=ax.transAxes, ha="left", va="top", fontsize=7.1, color="#333333")

    ax = ax_quality
    quality = stats["panel_b_allocation_ablation"]
    labels = [label for _, label in ALLOCATION_METHODS]
    methods = quality["methods"]
    means = [methods[label]["mean"] for label in labels]
    x = np.arange(len(labels))
    bar_colors = ["#AFAFAF", COLORS["blue"], COLORS["orange"], COLORS["green"]]
    ax.bar(x, means, width=0.66, color=bar_colors, edgecolor="white", linewidth=0.8, zorder=2)
    for index, mean in enumerate(means):
        ax.text(index, mean - 0.48, f"{mean:.2f}", ha="center", va="top",
                fontsize=8.0, fontweight="bold", color="white", zorder=4)
    fullkv_mean = quality["fullkv"]["mean"]
    ax.axhline(fullkv_mean, color="#333333", linestyle=(0, (5, 3)), linewidth=1.15, zorder=3)
    ax.text(0.025, fullkv_mean + 0.16, f"FullKV = {fullkv_mean:.2f}",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 0.8},
            zorder=5)
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylim(15.0, max(fullkv_mean, max(means)) + 2.8)
    ax.set_ylabel("GovReport ROUGE-L (↑)")
    ax.set_title("(b) Important B, not random B, restores quality", loc="left", fontweight="bold", pad=5)
    style_axis(ax)

    allocation = quality["representative_allocation"]
    ax = ax_allocation
    allocation_labels = ["Random B", "Top-layer B", "Water-fill"]
    allocation_keys = ["random_budget_r20", "top_layer_budget_r20", "formula_waterfill_r20"]
    all_counts = [count for key in allocation_keys for count in allocation["layer_block_counts"][key] if count > 0]
    color_norm = LogNorm(vmin=1, vmax=max(all_counts))
    for row_index, key in enumerate(allocation_keys):
        counts = np.asarray(allocation["layer_block_counts"][key], dtype=float)
        active = counts > 0
        sizes = 17.0 + 70.0 * counts[active] / max(all_counts)
        scatter = ax.scatter(np.arange(32)[active], np.full(active.sum(), row_index), c=counts[active],
                             s=sizes, marker="s", cmap="viridis", norm=color_norm,
                             edgecolor="white", linewidth=0.25)
    ax.set_xlim(-1, 32.4); ax.set_ylim(2.6, -0.6)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 31])
    ax.set_yticks(range(3), allocation_labels, fontsize=7.3)
    ax.set_xlabel("Layer index", labelpad=1)
    ax.tick_params(axis="x", labelsize=7.2, pad=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#E2E2E2", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.025, fraction=0.06)
    cbar.set_label("B blocks in layer", fontsize=7.2)
    cbar.ax.tick_params(labelsize=6.5)
    ax.set_title("(c) Same budget, different layer allocations", loc="left",
                 fontsize=10.5, fontweight="bold", pad=24)
    ax.text(0.0, 1.035,
            f"Same total K = {allocation['total_layer_block_budget']} · Random = mean of 5 draws",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2, color="#333333")

    fig.suptitle(stats["title"], fontsize=13.5, fontweight="bold", y=0.975)
    short_id = representative["source_id"][:10]
    caption = (f"Representative median-nearest sample ({short_id}…); panel (a): mean over 8 checkpoints; "
               "A = sink + recent 512 tokens. Mistral-7B-Instruct-v0.2 · GovReport · n=5. "
               "Random B averages 5 draws per sample; all B strategies use the same per-sample total budget.")
    fig.text(0.5, 0.025, caption, ha="center", va="bottom", fontsize=7.4, color="#333333")
    fig.subplots_adjust(left=0.068, right=0.987, bottom=0.225, top=0.875)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "motivation1_remote_context.png"; pdf_path = output_dir / "motivation1_remote_context.pdf"; stats_path = output_dir / "motivation1_remote_context_stats.json"
    fig.savefig(png_path, dpi=300, bbox_inches="tight"); fig.savefig(pdf_path, bbox_inches="tight"); plt.close(fig)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return png_path, pdf_path, stats_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the Motivation 1 remote-context evidence figure")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allocation-summary", type=Path, default=DEFAULT_ALLOCATION_SUMMARY)
    parser.add_argument("--allocation-predictions", type=Path, default=DEFAULT_ALLOCATION_PREDICTIONS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = load_data(args.input_dir)
    stats = compute_statistics(data)
    stats["panel_b_allocation_ablation"] = load_allocation_ablation(
        args.allocation_summary, args.allocation_predictions,
        stats["representative_heatmap"]["source_id"],
    )
    quality_ids = set(stats["panel_b_allocation_ablation"]["validation"]["sample_ids"])
    if set(stats["source_ids"]) != quality_ids:
        raise ValueError("Panel (a) and panel (b) do not use the same five sample IDs")
    stats["input_dir"] = str(args.input_dir.resolve())
    for path in draw_plot(stats, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
