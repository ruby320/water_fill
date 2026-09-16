from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_INTERVENTIONS = Path("results/gov_report_b_budget_sweep/interventions.jsonl")
DEFAULT_METADATA = Path("results/gov_report_b_budget_sweep/metadata.jsonl")
DEFAULT_OUTPUT = Path("results/ab_remote_evidence")
SOURCE_ID = "1ab8b8752432e8ec7a89868dfe98842bf6c5ce6c96760fd8"
LAYERS = (11, 30, 31)
BUDGETS = (4, 8, 12, 16, 24)
EXPECTED_CHECKPOINTS = 8
COLORS = {11: "#D55E00", 30: "#0072B2", 31: "#009E73"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Required input is empty: {path}")
    return rows


def ratio_of_sums(rows: Iterable[dict[str, Any]]) -> tuple[float, float]:
    values = list(rows)
    if not values:
        raise ValueError("Cannot aggregate an empty intervention cell")
    kl_sum = float(sum(float(row["kl_mean"]) for row in values))
    baseline_sum = float(sum(float(row["a_only_kl_mean"]) for row in values))
    if not np.isfinite(kl_sum) or not np.isfinite(baseline_sum) or baseline_sum <= 0:
        raise ValueError("KL sums must be finite and baseline KL sum must be positive")
    residual = 100.0 * kl_sum / baseline_sum
    return residual, 100.0 - residual


def validate_recovery_series(recovery_by_budget: dict[int, float]) -> list[float]:
    expected = {0, *BUDGETS}
    if set(recovery_by_budget) != expected:
        raise ValueError(f"Recovery budgets must be {sorted(expected)}")
    values = [float(recovery_by_budget[budget]) for budget in (0, *BUDGETS)]
    if not np.all(np.diff(values) >= -1e-12):
        raise ValueError("Measured recovery must be monotonic non-decreasing")
    return values


def compute_marginals(recovery_by_budget: dict[int, float]) -> dict[str, float]:
    points = {0: 0.0, **recovery_by_budget}
    validate_recovery_series(points)
    return {
        f"({lo},{hi}]": (float(points[hi]) - float(points[lo])) / (hi - lo)
        for lo, hi in zip((0, *BUDGETS[:-1]), BUDGETS)
    }


def compute_statistics(interventions_path: Path, metadata_path: Path) -> dict[str, Any]:
    intervention_rows = read_jsonl(interventions_path)
    metadata_rows = read_jsonl(metadata_path)
    metadata_matches = [row for row in metadata_rows if str(row.get("source_id")) == SOURCE_ID]
    if len(metadata_matches) != 1:
        raise ValueError("Expected exactly one metadata row for representative source")
    metadata = metadata_matches[0]
    important = {int(layer) for layer in metadata.get("important_layers", [])}
    if not set(LAYERS).issubset(important):
        raise ValueError("All representative layers must be metadata important_layers")
    metadata_budgets = tuple(int(value) for value in metadata.get("budgets", []))
    if metadata_budgets != BUDGETS:
        raise ValueError(f"Metadata budgets must equal {list(BUDGETS)}")
    checkpoints = sorted(int(value) for value in metadata.get("checkpoints", []))
    if len(checkpoints) != EXPECTED_CHECKPOINTS or len(set(checkpoints)) != EXPECTED_CHECKPOINTS:
        raise ValueError("Metadata must define eight unique checkpoints")
    candidate_count = int(metadata["candidate_count"])
    if candidate_count <= max(BUDGETS):
        raise ValueError("Representative sample must be unsaturated at every measured budget")
    if int(metadata.get("future_tokens", -1)) != 8:
        raise ValueError("Expected forward KL over eight future tokens")

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    source_rows = 0
    branch_repeat_rows = 0
    for row in intervention_rows:
        if str(row.get("source_id")) != SOURCE_ID:
            continue
        source_rows += 1
        if row.get("branch") != "prefill" or int(row.get("repeat", -1)) != 0:
            continue
        branch_repeat_rows += 1
        layer = int(row["layer_idx"])
        budget = int(row["requested_budget"])
        if layer in LAYERS and budget in BUDGETS:
            grouped[(layer, budget)].append(row)

    expected_cells = {(layer, budget) for layer in LAYERS for budget in BUDGETS}
    if set(grouped) != expected_cells:
        raise ValueError(f"Incomplete target cells: {sorted(expected_cells - set(grouped))}")

    per_layer: dict[str, Any] = {}
    all_monotonic = True
    for layer in LAYERS:
        recovery: dict[int, float] = {0: 0.0}
        residual: dict[int, float] = {}
        counts: dict[str, int] = {}
        cell_sums: dict[str, Any] = {}
        for budget in BUDGETS:
            rows = grouped[(layer, budget)]
            observed_checkpoints = sorted(int(row["checkpoint"]) for row in rows)
            if len(rows) != EXPECTED_CHECKPOINTS or observed_checkpoints != checkpoints:
                raise ValueError(f"L{layer}/B{budget} must contain each metadata checkpoint exactly once")
            if any(bool(row.get("saturated")) for row in rows):
                raise ValueError(f"L{layer}/B{budget} unexpectedly saturated")
            if any(int(row.get("effective_budget", -1)) != budget for row in rows):
                raise ValueError(f"L{layer}/B{budget} effective budget mismatch")
            cell_residual, cell_recovery = ratio_of_sums(rows)
            residual[budget] = cell_residual
            recovery[budget] = cell_recovery
            counts[str(budget)] = len(rows)
            cell_sums[str(budget)] = {
                "kl_mean_sum": float(sum(float(row["kl_mean"]) for row in rows)),
                "a_only_kl_mean_sum": float(sum(float(row["a_only_kl_mean"]) for row in rows)),
            }
        try:
            validate_recovery_series(recovery)
        except ValueError as exc:
            raise ValueError(f"Measured recovery validation failed for L{layer}: {exc}") from exc
        monotonic = True
        all_monotonic &= monotonic
        per_layer[str(layer)] = {
            "checkpoints_per_budget": counts,
            "checkpoint_ids": checkpoints,
            "kl_sums": cell_sums,
            "kl_residual_percent": {"0": 100.0, **{str(k): v for k, v in residual.items()}},
            "causal_kl_recovery_percent": {str(k): v for k, v in recovery.items()},
            "marginal_recovery_percent_per_block": compute_marginals({k: recovery[k] for k in BUDGETS}),
            "monotonic_non_decreasing": monotonic,
        }

    return {
        "title": "Remote KV blocks have layer-dependent causal returns",
        "input_paths": {
            "interventions": str(interventions_path.resolve()),
            "metadata": str(metadata_path.resolve()),
        },
        "selection": {
            "source_id": SOURCE_ID,
            "source_rule": "Same representative source as Motivation 1.",
            "branch": "prefill",
            "repeat": 0,
            "layers": list(LAYERS),
            "budgets": list(BUDGETS),
            "added_baseline": {"budget": 0, "causal_kl_recovery_percent": 0.0},
            "representative_layer_rationale": {
                "11": "rapid saturation",
                "30": "earlier recovery followed by continued improvement",
                "31": "gradual improvement",
            },
            "all_layers_in_metadata_important_layers": True,
            "sample_unsaturated": True,
            "candidate_count": candidate_count,
        },
        "input_validation": {
            "intervention_rows_total": len(intervention_rows),
            "metadata_rows_total": len(metadata_rows),
            "metadata_rows_matching_source": len(metadata_matches),
            "intervention_rows_matching_source": source_rows,
            "rows_after_source_branch_repeat_filter": branch_repeat_rows,
            "target_rows_after_layer_budget_filter": sum(len(rows) for rows in grouped.values()),
            "expected_target_cells": len(expected_cells),
            "expected_checkpoints_per_cell": EXPECTED_CHECKPOINTS,
            "all_target_cells_complete": True,
            "all_effective_budgets_match_requested": True,
            "all_target_rows_unsaturated": True,
        },
        "aggregation": {
            "ratio_of_sums_formula": "KL residual % = 100 * sum(kl_mean) / sum(a_only_kl_mean) across 8 checkpoints within each layer x budget cell",
            "recovery_formula": "Causal KL recovery % = 100 - KL residual %",
            "marginal_formula": "(Recovery(B_hi) - Recovery(B_lo)) / (B_hi - B_lo)",
            "marginal_intervals": ["(0,4]", "(4,8]", "(8,12]", "(12,16]", "(16,24]"],
        },
        "measurement": "Measured intervention results; no surrogate fit",
        "monotonicity": {"checked_over_budgets": [0, *BUDGETS], "all_layers_passed": all_monotonic},
        "limitations": [
            "single representative sample",
            "per-layer intervention (not simultaneous full-model allocation)",
            "forward KL over 8 future tokens",
            "descriptive, no inference",
        ],
        "per_layer": per_layer,
    }


def draw_plot(stats: dict[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.0, "axes.titlesize": 10.5,
        "axes.labelsize": 9.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.7, 4.45), facecolor="white")
    budgets = np.asarray((0, *BUDGETS), dtype=float)
    for layer in LAYERS:
        values = stats["per_layer"][str(layer)]["causal_kl_recovery_percent"]
        recovery = [values[str(int(budget))] for budget in budgets]
        ax_a.plot(budgets, recovery, marker="o", markersize=5.2, linewidth=2.0,
                  color=COLORS[layer], label=f"L{layer}", zorder=3)
    ax_a.set_title("(a) Measured causal recovery", loc="left", fontweight="bold")
    ax_a.set_xlabel("Retained remote blocks B")
    ax_a.set_ylabel("Causal KL recovery (%)")
    ax_a.set_xticks(budgets)
    ax_a.set_ylim(0, 101)
    ax_a.legend(title="Layer", frameon=False, loc="lower right")

    interval_labels = ["(0,4]", "(4,8]", "(8,12]", "(12,16]", "(16,24]"]
    x = np.arange(len(interval_labels), dtype=float)
    width = 0.23
    for offset, layer in zip((-width, 0.0, width), LAYERS):
        marginal = stats["per_layer"][str(layer)]["marginal_recovery_percent_per_block"]
        ax_b.bar(x + offset, [marginal[label] for label in interval_labels], width=width,
                 color=COLORS[layer], label=f"L{layer}", edgecolor="white", linewidth=0.6)
    ax_b.set_title("(b) Measured marginal recovery", loc="left", fontweight="bold")
    ax_b.set_xlabel("Budget interval")
    ax_b.set_ylabel("% recovery per block")
    ax_b.set_xticks(x, interval_labels)
    ax_b.legend(title="Layer", frameon=False, loc="upper right")

    for ax in (ax_a, ax_b):
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8.2)
    fig.suptitle(stats["title"], fontsize=13.5, fontweight="bold", y=0.965)
    fig.text(0.5, 0.045, "Measured intervention results; no surrogate fit",
             ha="center", va="center", fontsize=9.2, fontweight="bold", color="#333333")
    fig.text(0.5, 0.014,
             f"GovReport · representative source {SOURCE_ID[:10]}… · prefill branch, repeat 0 · 8 checkpoints",
             ha="center", va="bottom", fontsize=7.4, color="#555555")
    fig.subplots_adjust(left=0.078, right=0.985, bottom=0.205, top=0.84, wspace=0.28)

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "motivation3_causal_returns.png"
    pdf = output_dir / "motivation3_causal_returns.pdf"
    stats_path = output_dir / "motivation3_causal_returns_stats.json"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return png, pdf, stats_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the measured Motivation 3 causal-returns figure")
    parser.add_argument("--interventions", type=Path, default=DEFAULT_INTERVENTIONS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = compute_statistics(args.interventions, args.metadata)
    for path in draw_plot(stats, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
