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

REQUIRED_FILES = (
    "predictions.jsonl", "prefill_scores.jsonl", "checkpoint_metrics.jsonl",
    "q1_interventions.jsonl", "q2_interventions.jsonl",
)
BRANCH_LABELS = {
    "A_only": "A-only", "random": "A+Random-B", "prefill": "A+Selected-B (prefill)",
    "oracle": "A+Oracle-B",
}
COLORS = {"A_only": "#0072B2", "random": "#E69F00", "prefill": "#009E73", "oracle": "#CC79A7"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def load_directory(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    missing = [name for name in REQUIRED_FILES if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{input_dir}: missing required files: {', '.join(missing)}")
    data = {name: read_jsonl(input_dir / name) for name in REQUIRED_FILES}
    if any(not rows for rows in data.values()):
        empty = [name for name, rows in data.items() if not rows]
        raise ValueError(f"{input_dir}: empty required files: {', '.join(empty)}")
    return data


def mean_median(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=float)
    return {"mean": float(array.mean()), "median": float(np.median(array))} if array.size else {"mean": None, "median": None}


def aggregate_heatmap(data: dict[str, list[dict[str, Any]]], position_bins: int) -> tuple[np.ndarray, np.ndarray]:
    predictions = {str(r["source_id"]): r for r in data["predictions.jsonl"]}
    prefill = {(str(r["source_id"]), int(r["layer_idx"])): r for r in data["prefill_scores.jsonl"]}
    rows = data["checkpoint_metrics.jsonl"]
    layers = sorted({int(r["layer_idx"]) for r in rows})
    layer_to_index = {layer: index for index, layer in enumerate(layers)}
    grouped: dict[tuple[str, int, int], np.ndarray] = {}
    for row in rows:
        source, checkpoint, layer = str(row["source_id"]), int(row["checkpoint"]), int(row["layer_idx"])
        blocks = prefill[(source, layer)]["blocks"]
        masses = row["decode_block_mass"]
        if len(blocks) != len(masses):
            raise ValueError(f"{source} layer {layer}: block metadata/mass length mismatch")
        prompt_tokens = int(predictions[source]["prompt_tokens"])
        if prompt_tokens <= 0:
            raise ValueError(f"{source}: prompt_tokens must be positive")
        cell = grouped.setdefault((source, checkpoint, layer), np.zeros(position_bins, dtype=float))
        for block, mass in zip(blocks, masses):
            position = (float(block["start"]) + float(block["end"])) / (2.0 * prompt_tokens)
            bin_index = min(position_bins - 1, max(0, int(math.floor(position * position_bins))))
            cell[bin_index] += float(mass)
    heatmap = np.full((len(layers), position_bins), np.nan, dtype=float)
    for layer in layers:
        observations = [values for key, values in grouped.items() if key[2] == layer]
        if observations:
            heatmap[layer_to_index[layer]] = np.mean(observations, axis=0)
    normalized = np.nan_to_num(heatmap, nan=0.0)
    totals = normalized.sum(axis=1, keepdims=True)
    normalized = np.divide(normalized, totals, out=np.zeros_like(normalized), where=totals > 0)
    return heatmap, normalized


def cell_kl(data: dict[str, list[dict[str, Any]]]) -> dict[str, dict[tuple[str, int, int], float]]:
    predictions = {str(r["source_id"]): r for r in data["predictions.jsonl"]}
    important = {source: set(map(int, row["important_layers"])) for source, row in predictions.items()}
    result: dict[str, dict[tuple[str, int, int], float]] = {key: {} for key in BRANCH_LABELS}
    for row in data["q1_interventions.jsonl"]:
        source, layer = str(row["source_id"]), int(row["layer_idx"])
        if layer in important[source]:
            result["A_only"][(source, int(row["checkpoint"]), layer)] = float(row["kl_mean"])
    grouped: dict[tuple[str, tuple[str, int, int]], list[float]] = defaultdict(list)
    for row in data["q2_interventions.jsonl"]:
        branch = str(row["branch"])
        if branch not in ("random", "prefill", "oracle"):
            continue
        key = (str(row["source_id"]), int(row["checkpoint"]), int(row["layer_idx"]))
        if key[2] in important[key[0]]:
            grouped[(branch, key)].append(float(row["kl_mean"]))
    for (branch, key), values in grouped.items():
        result[branch][key] = float(np.mean(values))
    common = set(result["A_only"])
    for branch in ("random", "prefill", "oracle"):
        common &= set(result[branch])
    if not common:
        raise ValueError("no common important-layer/checkpoint intervention cells")
    return {branch: {key: values[key] for key in common} for branch, values in result.items()}


def sample_kl(cells: dict[str, dict[tuple[str, int, int], float]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    sources = sorted({key[0] for key in cells["A_only"]})
    for source in sources:
        output[source] = {
            branch: float(np.mean([value for key, value in values.items() if key[0] == source]))
            for branch, values in cells.items()
        }
    return output


def compute_stats(data: dict[str, list[dict[str, Any]]], cells: dict[str, dict[tuple[str, int, int], float]]) -> dict[str, Any]:
    predictions = {str(r["source_id"]): r for r in data["predictions.jsonl"]}
    important = {source: set(map(int, row["important_layers"])) for source, row in predictions.items()}
    prefill_rows, checkpoint_rows = data["prefill_scores.jsonl"], data["checkpoint_metrics.jsonl"]
    selected_concentration = []
    for row in checkpoint_rows:
        masses = list(map(float, row["decode_block_mass"]))
        selected = list(map(int, row["prefill_b_blocks"]))
        total = sum(masses)
        selected_concentration.append(sum(masses[i] for i in selected) / total if total else 0.0)
    b_counts = [len(r["selected_b_blocks"]) for r in data["q2_interventions.jsonl"] if r["branch"] == "random"]
    random_expected_coverage = []
    for row in checkpoint_rows:
        candidate_count = len(row["decode_block_mass"])
        if candidate_count:
            random_expected_coverage.append(min(len(row["prefill_b_blocks"]), candidate_count) / candidate_count)
    sums = {branch: float(sum(values.values())) for branch, values in cells.items()}
    means = {branch: float(np.mean(list(values.values()))) for branch, values in cells.items()}
    restoration = {branch: 1.0 - sums[branch] / sums["A_only"] for branch in ("random", "prefill", "oracle")}
    return {
        "task": next(iter(predictions.values())).get("task", "unknown"),
        "n_samples": len(predictions),
        "b_budget": int(round(float(np.mean(b_counts)))) if b_counts else None,
        "remote_candidate_definition": "Complete prompt blocks excluding sink/recent A.",
        "prefill_remote_mass": mean_median(float(r["prefill_remote_mass"]) for r in prefill_rows),
        "decode_remote_mass": mean_median(float(r["decode_remote_mass"]) for r in checkpoint_rows),
        "important_decode_remote_mass": mean_median(float(r["decode_remote_mass"]) for r in checkpoint_rows if int(r["layer_idx"]) in important[str(r["source_id"])]),
        "top_k_concentration_selected_blocks": mean_median(selected_concentration),
        "random_block_count_baseline": {
            **mean_median(b_counts),
            "candidate_count": mean_median(len(r["decode_block_mass"]) for r in checkpoint_rows),
            "budget_fraction": mean_median(random_expected_coverage),
        },
        "random_expected_remote_coverage_fraction": mean_median(random_expected_coverage),
        "important_cell_mean_kl": {BRANCH_LABELS[k]: v for k, v in means.items()},
        "ratio_of_sums_restoration": {BRANCH_LABELS[k]: v for k, v in restoration.items()},
        "selected_vs_random_residual_reduction": 1.0 - sums["prefill"] / sums["random"],
        "primary_statistic_note": "Restoration uses ratio of sums over matched cells; unstable per-cell restoration ratios are not averaged.",
    }


def plot_dataset(input_dir: Path, output_dir: Path, data: dict[str, list[dict[str, Any]]], position_bins: int, formats: Iterable[str] = ("png", "pdf")) -> tuple[Path, ...]:
    heatmap, normalized = aggregate_heatmap(data, position_bins)
    cells = cell_kl(data)
    samples = sample_kl(cells)
    stats = compute_stats(data, cells)
    task, n, b = stats["task"], stats["n_samples"], stats["b_budget"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("white")
    for ax, matrix, title, label in (
        (axes[0], heatmap, "(a) Absolute decode block attention", "Mean attention mass"),
        (axes[1], normalized, "(b) Layer-normalized decode attention", "Within-layer share"),
    ):
        image = ax.imshow(matrix, origin="lower", aspect="auto", extent=(0, 1, -0.5, matrix.shape[0] - 0.5), cmap=cmap)
        ax.set_title(title); ax.set_xlabel("Normalized remote block position"); ax.set_ylabel("Layer index")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=label)
    ax = axes[2]
    branches = ["A_only", "random", "prefill", "oracle"]
    x = np.arange(len(branches))
    for index, (source, values) in enumerate(sorted(samples.items())):
        ys = [values[b] for b in branches]
        ax.plot(x, ys, color="0.72", linewidth=0.8, alpha=0.75, zorder=1)
        ax.scatter(x, ys, s=18, color=[COLORS[b] for b in branches], alpha=0.8, zorder=2)
    means = [float(np.mean([v[b] for v in samples.values()])) for b in branches]
    ax.plot(x, means, color="black", linewidth=2.0, marker="D", markersize=5, label="Arithmetic mean", zorder=3)
    ax.set_yscale("log"); ax.set_xticks(x, [BRANCH_LABELS[b] for b in branches], rotation=22, ha="right")
    ax.set_ylabel("Forward KL (important layer/checkpoint mean)"); ax.set_title("(c) Paired intervention residual KL")
    ax.axhline(np.finfo(float).tiny, color="0.3", linestyle="--", linewidth=1)
    ax.text(0.02, 0.02, "FullKV: KL = 0 baseline (outside log scale)", transform=ax.transAxes, fontsize=8)
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle(f"Remote-attention evidence — {task} (n={n}, B={b})", fontsize=14)
    fig.text(0.5, 0.005, "Descriptive evidence. Remote candidates exclude sink/recent A; random repeats are averaged within matched cells.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_dir.name
    figure_paths = tuple(output_dir / f"{stem}_ab_remote_evidence.{fmt}" for fmt in formats)
    for path in figure_paths:
        save_kwargs = {"dpi": 300} if path.suffix == ".png" else {}
        fig.savefig(path, bbox_inches="tight", **save_kwargs)
    plt.close(fig)
    stats_path = output_dir / f"{stem}_ab_remote_evidence_stats.json"
    stats["input_dir"] = str(input_dir.resolve()); stats["position_bins"] = position_bins
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return (*figure_paths, stats_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot descriptive A/B remote-attention evidence")
    parser.add_argument("--input-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--position-bins", type=int, default=48)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.position_bins < 1:
        raise ValueError("position-bins must be positive")
    for input_dir in args.input_dir:
        outputs = plot_dataset(input_dir, args.output_dir, load_directory(input_dir), args.position_bins, args.formats)
        print("\n".join(map(str, outputs)))


if __name__ == "__main__":
    main()
