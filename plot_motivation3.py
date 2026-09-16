from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

DEFAULT_PREFILL = Path("results/gov_report_remote_attention/prefill_scores.jsonl")
DEFAULT_PREDICTIONS = Path("results/gov_report_allocation_ablation_r20_v1/predictions.jsonl")
DEFAULT_OUTPUT = Path("results/ab_remote_evidence")
SOURCE_ID = "1ab8b8752432e8ec7a89868dfe98842bf6c5ce6c96760fd8"
REPRESENTATIVE_LAYERS = (0, 16, 31, 3)
LAYER_ROLES = {
    0: "high-mass dispersed",
    16: "concentrated high marginal gain",
    31: "medium mass",
    3: "low mass",
}
N_LAYERS = 32
METHOD = "formula_waterfill_r20"
COLORS = {
    0: "#0072B2",
    16: "#D55E00",
    31: "#009E73",
    3: "#CC79A7",
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


def layer_utility(block_mass: Iterable[float]) -> dict[str, Any]:
    masses = np.asarray(list(block_mass), dtype=float)
    if masses.size == 0 or np.any(~np.isfinite(masses)) or np.any(masses < 0):
        raise ValueError("block_mass must be finite, nonnegative, and nonempty")
    total_mass = float(masses.sum())
    if total_mass <= 0:
        raise ValueError("block_mass must have positive total mass")
    sorted_mass = np.sort(masses)[::-1]
    empirical = np.concatenate(([0.0], np.cumsum(sorted_mass)))
    k50 = int(np.searchsorted(empirical, 0.5 * total_mass, side="left"))
    if k50 < 1:
        raise ValueError("K50 must be positive")
    decay = math.log(2.0) / k50
    k = np.arange(0, masses.size + 1, dtype=float)
    surrogate = total_mass * (1.0 - np.exp(-decay * k))
    marginal_k = np.arange(1, masses.size + 1, dtype=float)
    marginal = total_mass * (
        np.exp(-decay * (marginal_k - 1.0)) - np.exp(-decay * marginal_k)
    )
    return {
        "M": total_mass,
        "K50": k50,
        "lambda": decay,
        "empirical": empirical,
        "empirical_marginal": sorted_mass,
        "surrogate": surrogate,
        "marginal": marginal,
    }


def water_level(
    layer_utilities: dict[int, dict[str, Any]], total_budget: int
) -> tuple[float, list[int], dict[str, Any]]:
    candidates: list[tuple[float, int, int]] = []
    for layer in sorted(layer_utilities):
        marginal = np.asarray(layer_utilities[layer]["marginal"], dtype=float)
        candidates.extend((float(value), layer, index) for index, value in enumerate(marginal, 1))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    if total_budget <= 0 or total_budget > len(candidates):
        raise ValueError("total budget must be within the available marginal candidates")
    counts = [0] * N_LAYERS
    for _, layer, _ in candidates[:total_budget]:
        counts[layer] += 1
    last_inside = candidates[total_budget - 1]
    first_outside = candidates[total_budget] if total_budget < len(candidates) else None
    tau = (
        0.5 * (last_inside[0] + first_outside[0])
        if first_outside is not None
        else last_inside[0]
    )
    threshold = {
        "last_budget_marginal": {
            "value": last_inside[0], "layer": last_inside[1], "k": last_inside[2]
        },
        "first_outside_marginal": None if first_outside is None else {
            "value": first_outside[0], "layer": first_outside[1], "k": first_outside[2]
        },
        "candidate_count": len(candidates),
        "ranking_tie_break": "descending marginal, then ascending layer, then ascending k",
    }
    return float(tau), counts, threshold


def compute_statistics(prefill_path: Path, predictions_path: Path) -> dict[str, Any]:
    prefill_matches = [
        row for row in read_jsonl(prefill_path) if str(row.get("source_id")) == SOURCE_ID
    ]
    by_layer = {int(row["layer_idx"]): row for row in prefill_matches}
    if len(by_layer) != len(prefill_matches):
        raise ValueError("Duplicate prefill layer rows for representative sample")
    if sorted(by_layer) != list(range(N_LAYERS)):
        raise ValueError("Representative sample must contain exactly layers 0..31")
    block_counts = {len(row["block_mass"]) for row in by_layer.values()}
    if len(block_counts) != 1:
        raise ValueError("All layers must have the same number of block masses")
    candidate_blocks = next(iter(block_counts))

    prediction_matches = [
        row for row in read_jsonl(predictions_path)
        if str(row.get("source_id")) == SOURCE_ID and row.get("method") == METHOD
    ]
    if len(prediction_matches) != 1:
        raise ValueError("Expected exactly one representative formula-waterfill prediction")
    prediction = prediction_matches[0]
    if int(prediction["candidate_blocks"]) != candidate_blocks:
        raise ValueError("Prefill and allocation candidate-block counts differ")
    allocation = prediction.get("allocation")
    if not isinstance(allocation, dict):
        raise ValueError("Prediction is missing allocation metadata")
    raw_counts = allocation.get("layer_block_counts")
    if not isinstance(raw_counts, dict):
        raise ValueError("Prediction is missing layer_block_counts")
    allocated_counts = [int(raw_counts.get(str(layer), 0)) for layer in range(N_LAYERS)]
    total_budget = int(allocation["total_layer_block_budget"])
    reported_allocated = int(allocation["allocated_layer_blocks"])
    if any(count < 0 or count > candidate_blocks for count in allocated_counts):
        raise ValueError("Allocated layer count is outside candidate capacity")

    utilities = {layer: layer_utility(by_layer[layer]["block_mass"]) for layer in range(N_LAYERS)}
    tau, ranked_counts, threshold = water_level(utilities, total_budget)
    selected = prediction.get("selected_blocks")
    if not isinstance(selected, dict):
        raise ValueError("Prediction is missing selected_blocks")
    selected_counts = [len(selected.get(str(layer), [])) for layer in range(N_LAYERS)]
    checks = {
        "reported_budget_equals_reported_allocated": total_budget == reported_allocated,
        "sum_layer_counts_equals_budget": sum(allocated_counts) == total_budget,
        "selected_blocks_match_layer_counts": selected_counts == allocated_counts,
        "global_marginal_prefix_matches_layer_counts": ranked_counts == allocated_counts,
        "last_budget_marginal_at_or_above_tau": threshold["last_budget_marginal"]["value"] >= tau,
        "first_outside_marginal_at_or_below_tau": (
            threshold["first_outside_marginal"] is None
            or threshold["first_outside_marginal"]["value"] <= tau
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Allocation validation failed: {failed}")

    per_layer = {}
    for layer in range(N_LAYERS):
        utility = utilities[layer]
        allocated = allocated_counts[layer]
        per_layer[str(layer)] = {
            "M": utility["M"],
            "K50": utility["K50"],
            "lambda": utility["lambda"],
            "allocated_blocks": allocated,
            "representative": layer in REPRESENTATIVE_LAYERS,
            "representative_role": LAYER_ROLES.get(layer),
            "allocated_boundary_marginal": (
                float(utility["marginal"][allocated - 1]) if allocated else None
            ),
            "next_marginal": (
                float(utility["marginal"][allocated]) if allocated < candidate_blocks else None
            ),
        }

    return {
        "title": "Remote KV blocks exhibit heterogeneous per-block attention mass",
        "input_paths": {
            "prefill_scores": str(prefill_path.resolve()),
            "allocation_predictions": str(predictions_path.resolve()),
        },
        "sample": {
            "task": prediction.get("task"),
            "source_id": SOURCE_ID,
            "selection": "Same median-nearest representative used by Motivation 1.",
            "prompt_hash": prediction.get("prompt_hash"),
        },
        "representative_layers": list(REPRESENTATIVE_LAYERS),
        "representative_layer_roles": {str(key): value for key, value in LAYER_ROLES.items()},
        "candidate_blocks_per_layer": candidate_blocks,
        "formula": {
            "empirical_cumulative": "F_l(k) = sum of the k largest block_mass values in layer l",
            "empirical_marginal": "m_l(k) is the k-th largest block_mass value in layer l, k >= 1",
            "total_mass": "M_l = sum_i block_mass_{l,i}",
            "K50": "min k such that F_l(k) >= 0.5 M_l",
            "lambda": "lambda_l = ln(2) / K50_l",
            "surrogate": "Fhat_l(k) = M_l (1 - exp(-lambda_l k))",
            "marginal_gain": "delta_l(k) = M_l (exp(-lambda_l(k-1)) - exp(-lambda_l k)), k >= 1",
            "water_level": "Midpoint between the final budgeted and first unbudgeted globally ranked marginal; final budgeted marginal if no unbudgeted candidate exists.",
        },
        "allocation": {
            "method": METHOD,
            "total_budget": total_budget,
            "reported_allocated": reported_allocated,
            "tau": tau,
            "threshold_candidates": threshold,
            "reported_layer_block_counts": {str(i): allocated_counts[i] for i in range(N_LAYERS)},
            "recomputed_prefix_layer_block_counts": {str(i): ranked_counts[i] for i in range(N_LAYERS)},
            "budget_validation": {**checks, "all_checks_passed": all(checks.values())},
        },
        "per_layer": per_layer,
        "plot_notes": {
            "empirical_marginal": "At rank k, empirical marginal block mass is the k-th value after sorting each layer's block_mass values in descending order; values are absolute and not normalized per layer.",
            "surrogate_scope": "The solid exponential marginal curve is derived from the K50-calibrated allocation surrogate, not fitted to every empirical marginal value.",
            "allocation_boundary": "Each labeled vertical guide marks the reported layer allocation boundary k_l.",
            "zero_allocation_marker": "For L3 with k_l=0, the open marker is placed beside the first below-tau surrogate candidate at k=1 for visibility; it is not a selected marginal.",
        },
        "_utilities": utilities,
    }


def draw_plot(stats: dict[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.0, "axes.titlesize": 11.0,
        "axes.labelsize": 9.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    utilities = stats["_utilities"]
    candidate_blocks = int(stats["candidate_blocks_per_layer"])
    tau = float(stats["allocation"]["tau"])
    fig, ax = plt.subplots(figsize=(7.2, 5.05), facecolor="white")
    k_marginal = np.arange(1, candidate_blocks + 1)

    annotation_offsets = {
        0: (-39, 10),
        16: (5, 8),
        31: (5, 8),
    }
    for layer in REPRESENTATIVE_LAYERS:
        utility = utilities[layer]
        color = COLORS[layer]
        empirical_marginal = np.asarray(utility["empirical_marginal"], dtype=float)
        surrogate_marginal = np.asarray(utility["marginal"], dtype=float)
        ax.plot(
            k_marginal, empirical_marginal, color=color, linewidth=0.8,
            marker="o", markersize=2.5, markeredgewidth=0, alpha=0.38, zorder=2,
        )
        ax.plot(k_marginal, surrogate_marginal, color=color, linewidth=2.15, zorder=3)

        allocated = int(stats["per_layer"][str(layer)]["allocated_blocks"])
        if allocated:
            boundary_y = float(surrogate_marginal[allocated - 1])
            ax.vlines(
                allocated, 0.0, boundary_y, color=color, linewidth=1.0,
                linestyle=(0, (2, 2)), alpha=0.82, zorder=1,
            )
            ax.scatter(
                [allocated], [boundary_y], s=38, facecolor=color, edgecolor="white",
                linewidth=0.8, zorder=5,
            )
            ax.annotate(
                f"L{layer}: k={allocated}", xy=(allocated, boundary_y),
                xytext=annotation_offsets[layer], textcoords="offset points",
                fontsize=7.7, color=color,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.75},
            )
        else:
            first_y = float(surrogate_marginal[0])
            ax.scatter(
                [1], [first_y], s=46, facecolor="white", edgecolor=color,
                linewidth=1.4, zorder=6,
            )
            ax.annotate(
                f"L{layer}: k=0", xy=(1, first_y), xytext=(12, -10),
                textcoords="offset points", fontsize=7.7, color=color,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.75},
            )

    ax.axhline(tau, color="#222222", linewidth=1.25, linestyle=(0, (5, 3)), zorder=1)
    ax.text(
        0.985, tau, f"  Allocation threshold $\\tau$ = {tau:.4f}",
        transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8.0,
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.0},
    )

    ax.set_title(stats["title"], loc="left", fontweight="bold", pad=10)
    ax.set_xlabel("Block rank k")
    ax.set_ylabel("Attention mass per block")
    ax.set_xlim(0, candidate_blocks)
    ax.set_ylim(bottom=0)
    ax.set_facecolor("white")
    ax.grid(True, color="#D9D9D9", linewidth=0.65, alpha=0.68)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#777777")
    ax.tick_params(labelsize=8.3, color="#777777")

    layer_handles = [
        Line2D([0], [0], color=COLORS[layer], linewidth=2.2, label=f"L{layer}")
        for layer in REPRESENTATIVE_LAYERS
    ]
    style_handles = [
        Line2D(
            [0], [0], color="#666666", linewidth=0.8, marker="o", markersize=3.0,
            markeredgewidth=0, alpha=0.55, label="Empirical ranked block mass",
        ),
        Line2D(
            [0], [0], color="#444444", linewidth=2.15,
            label="K50-calibrated marginal proxy",
        ),
        Line2D(
            [0], [0], color="#222222", linewidth=1.25, linestyle=(0, (5, 3)),
            label="Global allocation threshold",
        ),
        Line2D(
            [0], [0], color="#666666", linewidth=1.0, linestyle=(0, (2, 2)),
            marker="o", markersize=4.0, label="Allocated block count $k_l$",
        ),
    ]
    layer_legend = ax.legend(
        handles=layer_handles, title="Layer", loc="upper right", ncol=4,
        frameon=True, facecolor="white", edgecolor="#D0D0D0", fontsize=7.7,
        title_fontsize=7.8, columnspacing=0.9, handlelength=1.8,
    )
    ax.add_artist(layer_legend)
    ax.legend(
        handles=style_handles, loc="upper right", bbox_to_anchor=(1.0, 0.865),
        frameon=True, facecolor="white", edgecolor="#D0D0D0", fontsize=7.4,
        handlelength=2.4,
    )

    fig.text(
        0.5, 0.018,
        f"GovReport representative sample {SOURCE_ID[:10]}… · global budget K = {stats['allocation']['total_budget']} across 32 layers",
        ha="center", va="bottom", fontsize=7.6, color="#444444",
    )
    fig.subplots_adjust(left=0.13, right=0.975, bottom=0.15, top=0.91)

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "motivation3_waterfill_utility.png"
    pdf = output_dir / "motivation3_waterfill_utility.pdf"
    js = output_dir / "motivation3_waterfill_utility_stats.json"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    serializable = {key: value for key, value in stats.items() if key != "_utilities"}
    js.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    return png, pdf, js


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the Motivation 3 water-filling utility figure")
    parser.add_argument("--prefill", type=Path, default=DEFAULT_PREFILL)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = compute_statistics(args.prefill, args.predictions)
    for path in draw_plot(stats, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
