from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_INPUT = Path("results/gov_report_remote_attention")
DEFAULT_OUTPUT = Path("results/ab_remote_evidence")
REQUIRED_FILES = (
    "predictions.jsonl",
    "prefill_scores.jsonl",
    "checkpoint_metrics.jsonl",
    "q1_interventions.jsonl",
)
N_LAYERS = 32
POSITION_BINS = 48
def importance_heat_values(relative_importance: Iterable[float]) -> tuple[np.ndarray, float, float]:
    """Map positive relative importance to log10 values for continuous color encoding."""
    relative = np.asarray(list(relative_importance), dtype=float)
    if relative.size == 0 or np.any(~np.isfinite(relative)) or np.any(relative < 1.0):
        raise ValueError("Relative importance must be finite, nonempty, and at least 1")
    transformed = np.log10(relative)
    return transformed, 0.0, float(transformed.max())


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


def mean_sem(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty sequence")
    sem = float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0
    return float(array.mean()), sem


def aggregate_layer_importance(
    rows: Iterable[dict[str, Any]], layers: Iterable[int] = range(N_LAYERS)
) -> dict[str, Any]:
    layers = list(layers)
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source_id"]), int(row["layer_idx"]))].append(float(row["kl_mean"]))
    source_ids = sorted({source for source, _ in grouped})
    if not source_ids:
        raise ValueError("No intervention rows")
    per_source: dict[str, dict[int, float]] = {source: {} for source in source_ids}
    for source in source_ids:
        for layer in layers:
            values = grouped.get((source, layer), [])
            if not values:
                raise ValueError(f"Missing q1 intervention values for {source}, layer {layer}")
            per_source[source][layer] = float(np.mean(values))
    importance, sem = [], []
    for layer in layers:
        layer_values = [per_source[source][layer] for source in source_ids]
        layer_mean, layer_sem = mean_sem(layer_values)
        importance.append(layer_mean)
        sem.append(layer_sem)
    return {
        "layers": layers,
        "source_ids": source_ids,
        "per_source": per_source,
        "importance": np.asarray(importance, dtype=float),
        "sem": np.asarray(sem, dtype=float),
    }


def distribute_blocks_to_bins(
    blocks: list[dict[str, Any]], masses: Iterable[float], prompt_tokens: int, position_bins: int
) -> np.ndarray:
    if prompt_tokens <= 0 or position_bins <= 0:
        raise ValueError("prompt_tokens and position_bins must be positive")
    masses_array = np.asarray(list(masses), dtype=float)
    if len(blocks) != masses_array.size:
        raise ValueError("Block metadata and decode_block_mass length differ")
    output = np.zeros(position_bins, dtype=float)
    for block, mass in zip(blocks, masses_array):
        start = max(0.0, min(1.0, float(block["start"]) / prompt_tokens))
        end = max(0.0, min(1.0, float(block["end"]) / prompt_tokens))
        if end <= start:
            raise ValueError(f"Invalid block interval: {block}")
        first = max(0, int(math.floor(start * position_bins)))
        last = min(position_bins - 1, int(math.ceil(end * position_bins) - 1))
        for bin_index in range(first, last + 1):
            bin_start = bin_index / position_bins
            bin_end = (bin_index + 1) / position_bins
            overlap = max(0.0, min(end, bin_end) - max(start, bin_start))
            if overlap:
                output[bin_index] += float(mass) * overlap / (end - start)
    return output


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    totals = matrix.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        bad = np.flatnonzero(totals[:, 0] <= 0).tolist()
        raise ValueError(f"Cannot normalize zero-mass heatmap rows: {bad}")
    return matrix / totals


def _attention_metadata(
    predictions_rows: Iterable[dict[str, Any]], prefill_rows: Iterable[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    predictions: dict[str, dict[str, Any]] = {}
    for row in predictions_rows:
        source = str(row["source_id"])
        if source in predictions:
            raise ValueError(f"Duplicate prediction row for {source}")
        predictions[source] = row
    prefill: dict[tuple[str, int], dict[str, Any]] = {}
    for row in prefill_rows:
        key = (str(row["source_id"]), int(row["layer_idx"]))
        if key in prefill:
            raise ValueError(f"Duplicate prefill row for {key}")
        prefill[key] = row
    return predictions, prefill


def aggregate_position_heatmap(
    predictions_rows: Iterable[dict[str, Any]],
    prefill_rows: Iterable[dict[str, Any]],
    checkpoint_rows: Iterable[dict[str, Any]],
    position_bins: int = POSITION_BINS,
    layers: Iterable[int] = range(N_LAYERS),
) -> dict[str, Any]:
    layers = list(layers)
    predictions, prefill = _attention_metadata(predictions_rows, prefill_rows)
    cells: dict[tuple[str, int, int], np.ndarray] = {}
    checkpoints: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in checkpoint_rows:
        source = str(row["source_id"])
        checkpoint, layer = int(row["checkpoint"]), int(row["layer_idx"])
        key = (source, checkpoint, layer)
        if key in cells:
            raise ValueError(f"Duplicate checkpoint cell for {key}")
        if source not in predictions or (source, layer) not in prefill:
            raise ValueError(f"Missing prediction or prefill metadata for {key}")
        cells[key] = distribute_blocks_to_bins(
            prefill[(source, layer)]["blocks"], row["decode_block_mass"],
            int(predictions[source]["prompt_tokens"]), position_bins,
        )
        checkpoints[(source, layer)].append(checkpoint)
    source_ids = sorted(predictions)
    source_heatmaps: dict[str, np.ndarray] = {}
    checkpoints_per_source_layer: dict[str, dict[str, int]] = {}
    for source in source_ids:
        matrix = np.zeros((len(layers), position_bins), dtype=float)
        checkpoints_per_source_layer[source] = {}
        for output_row, layer in enumerate(layers):
            checkpoint_ids = sorted(checkpoints.get((source, layer), []))
            if not checkpoint_ids:
                raise ValueError(f"Missing checkpoint metrics for {source}, layer {layer}")
            matrix[output_row] = np.mean(
                [cells[(source, checkpoint, layer)] for checkpoint in checkpoint_ids], axis=0
            )
            checkpoints_per_source_layer[source][str(layer)] = len(checkpoint_ids)
        source_heatmaps[source] = normalize_rows(matrix)
    heatmap = normalize_rows(np.mean(list(source_heatmaps.values()), axis=0))
    return {
        "layers": layers,
        "source_ids": source_ids,
        "heatmap": heatmap,
        "per_source_heatmaps": source_heatmaps,
        "checkpoints_per_source_layer": checkpoints_per_source_layer,
    }


def concentration(probabilities: Iterable[float], position_bins: int = POSITION_BINS) -> float:
    probabilities = np.asarray(list(probabilities), dtype=float)
    if probabilities.size != position_bins:
        raise ValueError(f"Expected {position_bins} probabilities, got {probabilities.size}")
    total = float(probabilities.sum())
    if total <= 0 or np.any(probabilities < 0):
        raise ValueError("Probabilities must be nonnegative with positive total")
    probabilities = probabilities / total
    positive = probabilities[probabilities > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return float(np.clip(1.0 - entropy / math.log(position_bins), 0.0, 1.0))


def bins_for_coverage(probabilities: Iterable[float], coverage: float) -> int:
    probabilities = np.asarray(list(probabilities), dtype=float)
    if not 0 < coverage <= 1 or probabilities.size == 0:
        raise ValueError("Coverage must be in (0, 1] and probabilities must be nonempty")
    total = float(probabilities.sum())
    if total <= 0 or np.any(probabilities < 0):
        raise ValueError("Probabilities must be nonnegative with positive total")
    cumulative = np.cumsum(np.sort(probabilities / total)[::-1])
    return int(np.searchsorted(cumulative, coverage - 1e-12) + 1)


def dispersion_axis_limits(
    observations: Iterable[float], position_bins: int
) -> tuple[int, int, int]:
    """Return an honest, rounded K50 axis that contains every observation."""
    values = np.asarray(list(observations), dtype=float)
    if position_bins <= 0:
        raise ValueError("position_bins must be positive")
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("K50 observations must be finite and nonempty")
    observed_max = float(values.max())
    if float(values.min()) < 0 or observed_max > position_bins:
        raise ValueError("K50 observations must be within the position-bin range")
    upper = min(position_bins, int(math.ceil(observed_max)) + 2)
    major_step = min(2 if upper <= 24 else 4, position_bins)
    return 0, upper, major_step


def aggregate_cell_level_dispersion(
    predictions_rows: Iterable[dict[str, Any]],
    prefill_rows: Iterable[dict[str, Any]],
    checkpoint_rows: Iterable[dict[str, Any]],
    position_bins: int = POSITION_BINS,
    layers: Iterable[int] = range(N_LAYERS),
) -> dict[str, Any]:
    """Compute one K50 observation for every source × checkpoint × layer cell."""
    layers = list(layers)
    predictions, prefill = _attention_metadata(predictions_rows, prefill_rows)
    source_ids = sorted(predictions)
    observed_checkpoints: dict[str, set[int]] = defaultdict(set)
    cells: dict[tuple[str, int, int], int] = {}
    for row in checkpoint_rows:
        source = str(row["source_id"])
        checkpoint, layer = int(row["checkpoint"]), int(row["layer_idx"])
        if layer not in layers:
            continue
        key = (source, checkpoint, layer)
        if key in cells:
            raise ValueError(f"Duplicate checkpoint cell for {key}")
        if source not in predictions or (source, layer) not in prefill:
            raise ValueError(f"Missing prediction or prefill metadata for {key}")
        distribution = distribute_blocks_to_bins(
            prefill[(source, layer)]["blocks"], row["decode_block_mass"],
            int(predictions[source]["prompt_tokens"]), position_bins,
        )
        distribution = normalize_rows(distribution[np.newaxis, :])[0]
        cells[key] = bins_for_coverage(distribution, 0.5)
        observed_checkpoints[source].add(checkpoint)

    expected_checkpoints = {
        source: sorted(map(int, predictions[source].get("checkpoints", observed_checkpoints[source])))
        for source in source_ids
    }
    missing = [
        (source, checkpoint, layer)
        for source in source_ids
        for checkpoint in expected_checkpoints[source]
        for layer in layers
        if (source, checkpoint, layer) not in cells
    ]
    if missing:
        raise ValueError(f"Missing {len(missing)} source/checkpoint/layer cells; first missing: {missing[0]}")

    per_layer_values: dict[int, np.ndarray] = {}
    per_layer_stats: dict[str, dict[str, Any]] = {}
    for layer in layers:
        values = np.asarray([
            cells[(source, checkpoint, layer)]
            for source in source_ids
            for checkpoint in expected_checkpoints[source]
        ], dtype=int)
        per_layer_values[layer] = values
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        per_layer_stats[str(layer)] = {
            "observations": values.tolist(), "count": int(values.size),
            "median": float(median), "q1": float(q1), "q3": float(q3),
            "min": int(values.min()), "max": int(values.max()),
        }
    actual_checkpoints = {source: sorted(observed_checkpoints[source]) for source in source_ids}
    return {
        "layers": layers,
        "per_layer_values": per_layer_values,
        "per_layer_stats": per_layer_stats,
        "structure": {
            "expected_source_ids": source_ids,
            "actual_source_ids": sorted(observed_checkpoints),
            "expected_checkpoints_by_source": expected_checkpoints,
            "actual_checkpoints_by_source": actual_checkpoints,
            "expected_cell_count_per_layer": int(sum(len(v) for v in expected_checkpoints.values())),
            "actual_cell_counts_by_layer": {str(layer): per_layer_stats[str(layer)]["count"] for layer in layers},
            "complete": True,
        },
    }


def select_representative_layers(
    importance: Iterable[float], layer_median_k50: Iterable[float]
) -> dict[str, Any]:
    importance = np.asarray(list(importance), dtype=float)
    medians = np.asarray(list(layer_median_k50), dtype=float)
    if importance.shape != medians.shape or importance.size == 0:
        raise ValueError("Importance and layer-median K50 arrays must have equal nonzero length")
    q75 = float(np.quantile(importance, 0.75))
    high = np.flatnonzero(importance > q75)
    if high.size < 2:
        raise ValueError("Need at least two layers strictly above importance Q75")
    return {
        "important_concentrated": int(high[np.argmin(medians[high])]),
        "important_dispersed": int(high[np.argmax(medians[high])]),
        "less_important": int(np.argmin(importance)),
        "importance_upper_quartile": q75,
        "high_importance_candidates": sorted(map(int, high)),
        "selection_rule": "Among layers strictly above source-first importance Q75, select the smallest cell-level K50 median as important+concentrated and the largest median as important+dispersed; select the global minimum-importance layer as less important. Ties use the lowest layer index.",
    }


def normalize_importance_to_minimum(
    importance: Iterable[float], sem: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    importance_array = np.asarray(list(importance), dtype=float)
    sem_array = np.asarray(list(sem), dtype=float)
    if importance_array.shape != sem_array.shape or importance_array.size == 0 or np.any(importance_array <= 0):
        raise ValueError("Importance and SEM must have equal nonzero shape, with positive importance")
    denominator = float(importance_array.min())
    return importance_array / denominator, sem_array / denominator


def compute_statistics(
    data: dict[str, list[dict[str, Any]]], input_dir: Path, position_bins: int
) -> dict[str, Any]:
    layers = list(range(N_LAYERS))
    ir = aggregate_layer_importance(data["q1_interventions.jsonl"], layers)
    hr = aggregate_position_heatmap(
        data["predictions.jsonl"], data["prefill_scores.jsonl"],
        data["checkpoint_metrics.jsonl"], position_bins, layers,
    )
    dr = aggregate_cell_level_dispersion(
        data["predictions.jsonl"], data["prefill_scores.jsonl"],
        data["checkpoint_metrics.jsonl"], position_bins, layers,
    )
    if ir["source_ids"] != hr["source_ids"]:
        raise ValueError("Intervention and attention inputs do not contain the same sources")
    heat, imp, sem = hr["heatmap"], ir["importance"], ir["sem"]
    if np.any(imp <= 0):
        raise ValueError("Log-scale importance requires positive means")
    relative, relative_sem = normalize_importance_to_minimum(imp, sem)
    heatmap_k50 = np.array([bins_for_coverage(row, 0.5) for row in heat])
    medians = np.array([dr["per_layer_stats"][str(layer)]["median"] for layer in layers])
    reps = select_representative_layers(imp, medians)
    per = {
        str(layer): {
            "importance_mean_forward_kl": float(imp[layer]),
            "importance_sem": float(sem[layer]),
            "relative_importance": float(relative[layer]),
            "relative_importance_sem": float(relative_sem[layer]),
            "aggregated_heatmap_k50_bins": int(heatmap_k50[layer]),
            "cell_level_k50_median": float(medians[layer]),
        }
        for layer in layers
    }
    reps["values"] = {
        name: {"layer": layer, **per[str(layer)], **dr["per_layer_stats"][str(layer)]}
        for name, layer in reps.items()
        if name in {"important_concentrated", "important_dispersed", "less_important"}
    }
    _, heat_vmin, heat_vmax = importance_heat_values(relative)
    all_dispersion_observations = [
        observation
        for layer in layers
        for observation in dr["per_layer_stats"][str(layer)]["observations"]
    ]
    dispersion_xmin, dispersion_xmax, dispersion_major_step = dispersion_axis_limits(
        all_dispersion_observations, position_bins
    )
    return {
        "title": "Layer importance and information dispersion are highly heterogeneous",
        "task": str(data["predictions.jsonl"][0].get("task", "unknown")),
        "n_samples": len(ir["source_ids"]),
        "source_ids": ir["source_ids"],
        "position_bins": position_bins,
        "layer_indices": layers,
        "input_paths": {name: str((input_dir / name).resolve()) for name in REQUIRED_FILES},
        "formulas": {
            "importance": "I_l = mean_s(mean_c(KL_{s,c,l})); SEM across sources.",
            "relative_importance": "I_l / min_j(I_j), using source-first global layer means; SEM uses the same global denominator.",
            "cell_level_dispersion": "For each source × checkpoint × layer, overlap-distribute decode_block_mass into 48 normalized-position bins, normalize that cell row, sort bins by attention, and define K50 as the minimum bin count whose cumulative mass reaches at least 50%.",
            "aggregated_heatmap_k50": "Supplementary only: K50 of the source-averaged, checkpoint-averaged final heatmap row; not used as a box-plot observation.",
        },
        "aggregation_order": {
            "importance": "checkpoint mean within source/layer -> mean and SEM across sources",
            "cell_level_dispersion": "block overlap -> normalize each source/checkpoint/layer cell -> K50 per cell -> descriptive layer distribution",
        },
        "dynamic_range": {
            "importance_max_over_min": float(relative.max()),
            "importance_min": float(imp.min()),
            "importance_max": float(imp.max()),
            "layer_median_k50_min": float(medians.min()),
            "layer_median_k50_max": float(medians.max()),
        },
        "relative_importance": {str(i): float(relative[i]) for i in layers},
        "importance_heat_strip": {
            "transform": "log10(relative_importance)",
            "normalization": {"vmin": heat_vmin, "vmax": heat_vmax},
            "colormap": "YlOrRd",
            "visual_ticks": [],
        },
        "per_layer": per,
        "cell_level_dispersion": {
            "definition": "K50 bins per source × checkpoint × layer cell after overlap distribution and cell-wise normalization.",
            "observed_range": {
                "min": int(min(all_dispersion_observations)),
                "max": int(max(all_dispersion_observations)),
            },
            "display_axis": {
                "xmin": dispersion_xmin,
                "xmax": dispersion_xmax,
                "major_step": dispersion_major_step,
            },
            "per_layer": dr["per_layer_stats"],
            "structure": dr["structure"],
        },
        "heatmap_row_probabilities": heat.tolist(),
        "heatmap_row_sums": heat.sum(1).tolist(),
        "aggregated_heatmap_k50_by_layer": {str(i): int(heatmap_k50[i]) for i in layers},
        "checkpoints_per_source_layer": hr["checkpoints_per_source_layer"],
        "representative_layers": reps,
        "plot_notes": {
            "composite_encoding": "A single shared layer axis overlays causal importance as full-row background color and cell-level K50 distributions as horizontal box plots.",
            "background": "Each layer row encodes log10(relative importance), normalized continuously from the least to most important layer with no numeric values shown.",
            "box": "Horizontal box plots show all source × checkpoint K50 observations per layer (median, IQR, 1.5×IQR whiskers, and outliers), using translucent near-white boxes so the importance background remains visible.",
            "descriptive_scope": "The 40 observations per layer in the formal data are correlated descriptive source × checkpoint measurements; no independent-sample inference is performed.",
            "representative_rule": reps["selection_rule"],
            "layer_direction": "Layer 0 is at bottom.",
        },
    }


def draw_plot(stats: dict[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
        "axes.labelsize": 9, "pdf.fonttype": 42,
    })
    layers = np.asarray(stats["layer_indices"])
    per = stats["per_layer"]
    dispersion = stats["cell_level_dispersion"]["per_layer"]
    relative = np.asarray([per[str(i)]["relative_importance"] for i in layers])
    heat_values, heat_vmin, heat_vmax = importance_heat_values(relative)
    observations = [dispersion[str(i)]["observations"] for i in layers]
    axis_config = stats["cell_level_dispersion"]["display_axis"]

    figure_size = (7.6, 6.6)
    background_alpha = 0.64
    background_cmap = "YlOrRd"
    fig, ax = plt.subplots(figsize=figure_size, facecolor="white")
    fig.subplots_adjust(left=0.105, right=0.975, bottom=0.115, top=0.82)

    heat_norm = matplotlib.colors.Normalize(vmin=heat_vmin, vmax=heat_vmax)
    x_min, x_max = axis_config["xmin"], axis_config["xmax"]
    background = np.repeat(heat_values[:, np.newaxis], 2, axis=1)
    ax.imshow(
        background, origin="lower", aspect="auto",
        extent=(x_min, x_max, -0.5, N_LAYERS - 0.5),
        cmap=background_cmap, norm=heat_norm, interpolation="none",
        alpha=background_alpha, zorder=0,
    )
    for boundary in np.arange(-0.5, N_LAYERS + 0.5, 1):
        ax.axhline(
            boundary, color="white", linewidth=0.50, alpha=0.78,
            zorder=1,
        )

    box_facecolor = (1.0, 1.0, 1.0, 0.76)
    box_edgecolor = "#343A40"
    box = ax.boxplot(
        observations, vert=False, positions=layers, widths=0.58, patch_artist=True,
        showfliers=True, whis=1.5,
        medianprops={"color": "#111111", "linewidth": 1.35, "zorder": 4},
        whiskerprops={"color": "#444A50", "linewidth": 0.78, "zorder": 3},
        capprops={"color": "#444A50", "linewidth": 0.78, "zorder": 3},
        flierprops={"marker": "o", "markersize": 1.8, "markerfacecolor": "#30343A",
                    "markeredgewidth": 0, "alpha": 0.38, "zorder": 4},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(box_facecolor)
        patch.set_edgecolor(box_edgecolor)
        patch.set_linewidth(0.82)
        patch.set_zorder(3)

    rng = np.random.default_rng(20260914)
    for layer, values in enumerate(observations):
        jitter = rng.uniform(-0.15, 0.15, len(values))
        ax.scatter(
            values, layer + jitter, s=4.5, color="#24292E", alpha=0.16,
            linewidths=0, zorder=3.5,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, N_LAYERS - 0.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(axis_config["major_step"]))
    ax.set_xlabel("K50: bins needed for 50% attention")
    ax.set_ylabel("Layer")
    layer_ticks = np.arange(N_LAYERS)
    ax.set_yticks(layer_ticks, labels=[f"L{layer}" for layer in layer_ticks])
    ax.tick_params(axis="y", labelsize=7.4, length=2.5, color="#737B83", pad=2)
    ax.annotate(
        "Concentrated ←", xy=(0.01, 1.014), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=7.2, color="#555B61",
    )
    ax.annotate(
        "→ Dispersed", xy=(0.99, 1.014), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=7.2, color="#555B61",
    )
    ax.grid(axis="x", color="#FFFFFF", linewidth=0.50, alpha=0.42, zorder=1.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#8B9298")
    ax.spines["bottom"].set_linewidth(0.65)

    legend_ax = ax.inset_axes([0.31, 1.075, 0.38, 0.020], transform=ax.transAxes)
    legend_ax.imshow(
        np.linspace(0, 1, 256)[np.newaxis, :], aspect="auto", cmap=background_cmap,
        extent=(0, 1, 0, 1), interpolation="bilinear", alpha=background_alpha,
    )
    legend_ax.set_title(
        "Causal importance: Less → More", fontsize=7.0, color="#4E555B", pad=2.0,
    )
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.spines[:].set_visible(False)

    checkpoint_counts = [
        len(checkpoints)
        for checkpoints in stats["cell_level_dispersion"]["structure"][
            "actual_checkpoints_by_source"
        ].values()
    ]
    checkpoint_text = (
        str(checkpoint_counts[0])
        if len(set(checkpoint_counts)) == 1
        else f"{min(checkpoint_counts)}–{max(checkpoint_counts)}"
    )
    fig.suptitle(
        "Layer importance and information dispersion are highly heterogeneous",
        fontsize=11.5, fontweight="semibold", y=0.965,
    )
    fig.text(
        0.5, 0.025,
        f"GovReport · {stats['n_samples']} sources × {checkpoint_text} checkpoints · "
        f"{stats['position_bins']} position bins",
        ha="center", fontsize=7.8, color="#50565C",
    )

    fig.canvas.draw()
    axes_bounds = ax.get_position().bounds
    stats["plot_config"] = {
        "single_panel_overlay": True,
        "figsize_inches": list(figure_size),
        "axes_bounds_figure_fraction": list(axes_bounds),
        "display_axis": dict(axis_config),
        "background": {
            "encoding": "log10(relative_importance)",
            "colormap": background_cmap,
            "alpha": background_alpha,
            "row_extent": [-0.5, N_LAYERS - 0.5],
            "x_extent": [x_min, x_max],
        },
        "box_style": {
            "encoding": "cell-level K50 distribution",
            "facecolor_rgba": list(box_facecolor),
            "edgecolor": box_edgecolor,
            "median_color": "#111111",
            "whisker_cap_color": "#444A50",
            "scatter_color": "#24292E",
            "scatter_alpha": 0.16,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "motivation2_layer_heterogeneity.png"
    pdf = output_dir / "motivation2_layer_heterogeneity.pdf"
    js = output_dir / "motivation2_layer_heterogeneity_stats.json"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    js.write_text(json.dumps(stats, indent=2) + "\n")
    return png, pdf, js


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the Motivation 2 layer-heterogeneity figure")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--position-bins", type=int, default=POSITION_BINS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.position_bins != POSITION_BINS:
        raise ValueError(f"This figure requires exactly {POSITION_BINS} position bins")
    stats = compute_statistics(load_data(args.input_dir), args.input_dir, args.position_bins)
    for path in draw_plot(stats, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
