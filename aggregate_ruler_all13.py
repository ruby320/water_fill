from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import pandas as pd

TASKS = (
    "cwe", "fwe", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue", "niah_single_1", "niah_single_2",
    "niah_single_3", "qa_1", "qa_2", "vt",
)
REGULAR_SOURCE_METHODS = (
    "fullkv", "formula_waterfill_r20", "global_empirical_topk_r20",
    "ada_snapkv_r20", "pyramidkv_r20", "streamingllm_r20", "snapkv_r20",
)
REGULAR_METHODS = tuple(method for method in REGULAR_SOURCE_METHODS
                        if method != "global_empirical_topk_r20")
METHOD_ORDER = (
    "fullkv", "waterfill_defensivekv_r20", "defensivekv_20", "layer_defensivekv_20",
    "formula_waterfill_r20", "snapkv_r20", "ada_snapkv_r20", "pyramidkv_r20",
    "streamingllm_r20",
)
DISPLAY_NAMES = {
    "fullkv": "FullKV",
    "formula_waterfill_r20": "Water-fill R20",
    "waterfill_defensivekv_r20": "Water-fill + DefensiveKV R20",
    "defensivekv_20": "DefensiveKV R20",
    "layer_defensivekv_20": "Layer-DefensiveKV R20",
    "ada_snapkv_r20": "Ada-SnapKV R20",
    "pyramidkv_r20": "PyramidKV R20",
    "streamingllm_r20": "StreamingLLM R20",
    "snapkv_r20": "SnapKV R20",
}
TARGET_RATIO = 0.2
EXPECTED_PER_GROUP = len(TASKS) * 5


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_calculate_metrics(path: Path) -> Callable[[pd.DataFrame], dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("official_ruler_calculate_metrics", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load metric module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.calculate_metrics


def task_counts(rows: list[dict[str, Any]], unique: bool = False) -> dict[str, int]:
    if unique:
        return {task: len({str(row["source_id"]) for row in rows if row["task"] == task}) for task in TASKS}
    return dict(Counter(str(row["task"]) for row in rows))


def ratio_range(values: list[float]) -> dict[str, float]:
    return {"min": min(values), "max": max(values)}


def score_rows(rows: list[dict[str, Any]], metric: Callable[[pd.DataFrame], dict[str, Any]],
               defensive: bool = False) -> dict[str, Any]:
    frame = pd.DataFrame({
        "task": [row["task"] for row in rows],
        "answer": [row["answer"] if defensive else row["answers"] for row in rows],
        "predicted_answer": [row["predicted_answer"] if defensive else row["prediction"] for row in rows],
    })
    measured = metric(frame)
    missing = [task for task in TASKS if task not in measured or measured[task] is None]
    if missing:
        raise ValueError(f"metric output is missing tasks: {missing}")
    tasks = {task: {"string_match": float(measured[task]["string_match"])} for task in TASKS}
    macro = sum(tasks[task]["string_match"] for task in TASKS) / len(TASKS)
    return {"display_name": "", "macro_average": macro, "tasks": tasks}


def validate_config(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    context = summary.get("context_length", summary.get("context"))
    generation = summary.get("generation")
    greedy = (isinstance(generation, str) and "greedy" in generation.lower()) or (
        isinstance(generation, dict) and generation.get("do_sample") is False
    )
    checks = {
        "context_length_16384": context == 16384,
        "seed_42": summary.get("seed") == 42,
        "num_samples_per_task_5": summary.get("num_samples_per_task") == 5,
        "num_total_samples_65": summary.get("num_total_samples") == EXPECTED_PER_GROUP,
        "tasks_in_fixed_order": summary.get("tasks") == list(TASKS),
        "greedy_generation": greedy,
    }
    return {"group": name, "checks": checks, "passed": all(checks.values())}


def validate_regular(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = {str(row["source_id"]) for row in rows}
    method_rows = {method: [row for row in rows if row["method"] == method]
                   for method in REGULAR_SOURCE_METHODS}
    count_checks = {
        method: {
            "rows": len(values),
            "unique_source_ids": len({str(row["source_id"]) for row in values}),
            "complete": len(values) == EXPECTED_PER_GROUP
                        and {str(row["source_id"]) for row in values} == ids,
        }
        for method, values in method_rows.items()
    }
    ratio_checks: dict[str, Any] = {}
    for method in REGULAR_SOURCE_METHODS[1:]:
        allocations = [row["allocation"] for row in method_rows[method]]
        targets = [float(allocation["target_kv_ratio"]) for allocation in allocations]
        actuals = [float(allocation.get("actual_kv_ratio", allocation.get("achieved_decode_start_kv_ratio")))
                   for allocation in allocations]
        ratio_checks[method] = {
            "target_ratio_range": ratio_range(targets),
            "actual_ratio_range": ratio_range(actuals),
            "all_targets_match_summary": all(value == float(summary["target_prompt_kv_ratio"]) for value in targets),
        }
    passed = (
        len(rows) == EXPECTED_PER_GROUP * len(REGULAR_SOURCE_METHODS)
        and len(ids) == EXPECTED_PER_GROUP
        and task_counts(rows, unique=True) == {task: 5 for task in TASKS}
        and set(summary.get("methods", [])) == set(REGULAR_SOURCE_METHODS)
        and float(summary.get("target_prompt_kv_ratio")) == TARGET_RATIO
        and all(value["complete"] for value in count_checks.values())
        and all(value["all_targets_match_summary"] for value in ratio_checks.values())
    )
    return {
        "rows": len(rows), "unique_source_ids": len(ids),
        "unique_source_ids_per_task": task_counts(rows, unique=True),
        "methods": count_checks, "summary_target_ratio": summary.get("target_prompt_kv_ratio"),
        "method_ratio_ranges": ratio_checks, "passed": passed,
    }


def validate_layer_defensive(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary_budgets = summary.get("actual_budgets", [])
    summary_record_matches = 0
    head_sum_checks = 0
    layer_total = 0
    global_exact_rows = 0
    layer_sum_exact_rows = 0
    ratios: list[float] = []
    for index, row in enumerate(rows):
        budget = row["actual_budget"]
        actuals = [int(value) for value in budget["actual_kept_kv_tokens_per_layer"]]
        heads = budget["actual_head_lengths_per_layer"]
        layer_total += len(actuals)
        head_sum_checks += sum(sum(head_lengths) == actual
                               for head_lengths, actual in zip(heads, actuals))
        expected_global = int(budget["global_expected_kept_kv_elements"])
        actual_global = int(budget["global_actual_kept_kv_elements"])
        global_exact_rows += int(expected_global == actual_global)
        layer_sum_exact_rows += int(sum(actuals) == actual_global)
        raw_global = int(budget["raw_kv_tokens_per_layer"]) * len(actuals)
        ratios.append(actual_global / raw_global)
        if index < len(summary_budgets) and summary_budgets[index] == budget:
            summary_record_matches += 1
    passed = (
        len(rows) == EXPECTED_PER_GROUP
        and len({str(row["source_id"]) for row in rows}) == EXPECTED_PER_GROUP
        and task_counts(rows) == {task: 5 for task in TASKS}
        and summary.get("arm") == "layer_defensivekv_20"
        and all(row.get("arm") == "layer_defensivekv_20" for row in rows)
        and layer_total == len(rows) * 32
        and head_sum_checks == layer_total
        and global_exact_rows == len(rows)
        and layer_sum_exact_rows == len(rows)
        and summary_record_matches == len(rows)
    )
    return {
        "rows": len(rows), "unique_source_ids": len({str(row["source_id"]) for row in rows}),
        "rows_per_task": task_counts(rows), "per_layer_head_sum_comparisons": layer_total,
        "head_lengths_sum_to_actual_comparisons": head_sum_checks,
        "global_expected_equals_actual_rows": global_exact_rows,
        "layer_actual_sum_equals_global_actual_rows": layer_sum_exact_rows,
        "summary_budget_records_equal_prediction_records": summary_record_matches,
        "global_actual_ratio_range": ratio_range(ratios),
        "declared_retention_ratio": summary.get("retention_ratio"), "passed": passed,
    }


def validate_defensive(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    layer_checks = 0
    head_sum_checks = 0
    summary_record_matches = 0
    realized: list[float] = []
    summary_budgets = summary.get("actual_budgets", [])
    for index, row in enumerate(rows):
        budget = row["actual_budget"]
        expected = int(budget["expected_kept_kv_tokens_per_layer"])
        actuals = [int(value) for value in budget["actual_kept_kv_tokens_per_layer"]]
        layer_checks += sum(value == expected for value in actuals)
        head_sum_checks += sum(sum(head_lengths) == expected for head_lengths in budget["actual_head_lengths_per_layer"])
        realized.extend(float(value) for value in budget["realized_retention_per_layer"])
        if index < len(summary_budgets) and summary_budgets[index] == budget:
            summary_record_matches += 1
    expected_layer_checks = len(rows) * 32
    passed = (
        len(rows) == EXPECTED_PER_GROUP
        and len({str(row["source_id"]) for row in rows}) == EXPECTED_PER_GROUP
        and task_counts(rows) == {task: 5 for task in TASKS}
        and summary.get("arm") == "defensivekv_20"
        and layer_checks == expected_layer_checks
        and head_sum_checks == expected_layer_checks
        and summary_record_matches == len(rows)
    )
    return {
        "rows": len(rows), "unique_source_ids": len({str(row["source_id"]) for row in rows}),
        "rows_per_task": task_counts(rows),
        "expected_kept_comparisons": expected_layer_checks,
        "actual_equals_expected_comparisons": layer_checks,
        "head_lengths_sum_to_expected_comparisons": head_sum_checks,
        "summary_budget_records_equal_prediction_records": summary_record_matches,
        "realized_retention_ratio_range": ratio_range(realized),
        "declared_retention_ratio": summary.get("retention_ratio"),
        "integer_budget_rule": "Use expected_kept_kv_tokens_per_layer from each record; do not recompute prompt_tokens * 0.2.",
        "passed": passed,
    }


def validate_mixed(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios: list[float] = []
    exact_rows = 0
    layer_exact = 0
    layer_total = 0
    target_total = 0
    actual_total = 0
    for row in rows:
        allocation = row["allocation"]
        targets = allocation["target_kept_elements_by_layer"]
        actuals = allocation["actual_kept_elements_by_layer"]
        layer_total += len(targets)
        layer_exact += sum(int(targets[layer]) == int(actuals[layer]) for layer in targets)
        target = int(allocation["global_target_kept_elements"])
        actual = int(allocation["global_actual_kept_elements"])
        target_total += target
        actual_total += actual
        exact_rows += int(bool(allocation["budget_exact"]) and target == actual)
        ratios.append(float(allocation["global_actual_ratio"]))
    summary_budget = summary.get("budget_validation", {})
    passed = (
        len(rows) == EXPECTED_PER_GROUP
        and len({str(row["source_id"]) for row in rows}) == EXPECTED_PER_GROUP
        and task_counts(rows) == {task: 5 for task in TASKS}
        and summary.get("method") == "waterfill_defensivekv_r20"
        and exact_rows == len(rows) and layer_exact == layer_total and target_total == actual_total
        and summary_budget.get("all_samples_exact") is True
        and summary_budget.get("global_target_elements") == target_total
        and summary_budget.get("global_actual_elements") == actual_total
    )
    return {
        "rows": len(rows), "unique_source_ids": len({str(row["source_id"]) for row in rows}),
        "rows_per_task": task_counts(rows), "budget_exact_rows": exact_rows,
        "per_layer_target_actual_equal": layer_exact, "per_layer_comparisons": layer_total,
        "global_target_elements": target_total, "global_actual_elements": actual_total,
        "global_actual_ratio_range": ratio_range(ratios), "passed": passed,
    }


def prompt_identity_validation(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    def hash_field(row: dict[str, Any]) -> str:
        return str(row.get("prompt_hash", row.get("prompt_token_sha256")))

    identities = {
        name: {(str(row["task"]), str(row["source_id"]), hash_field(row)) for row in rows}
        for name, rows in groups.items()
    }
    task_hashes = {
        name: {(str(row["task"]), hash_field(row)) for row in rows}
        for name, rows in groups.items()
    }
    exact = len({frozenset(value) for value in identities.values()}) == 1
    hash_aligned = len({frozenset(value) for value in task_hashes.values()}) == 1
    reference = identities.get("regular", set())
    pairwise_exact = {name: values == reference for name, values in identities.items() if name != "regular"}
    required_exact = all(pairwise_exact.get(name, False) for name in ("mixed", "layer_defensivekv"))
    defensive_known_mismatch = (
        "defensivekv" in identities
        and not pairwise_exact.get("defensivekv", True)
        and task_hashes["defensivekv"] == task_hashes.get("regular", set())
    )
    passed = required_exact and defensive_known_mismatch if "regular" in groups else exact
    return {
        "exact_task_source_id_prompt_hash_match": exact,
        "task_prompt_hash_sets_match": hash_aligned,
        "pairwise_exact_match_to_regular": pairwise_exact,
        "layer_strictly_matches_regular": pairwise_exact.get("layer_defensivekv"),
        "mixed_strictly_matches_regular": pairwise_exact.get("mixed"),
        "defensivekv_known_source_id_mismatch_with_hash_match": defensive_known_mismatch,
        "unique_task_source_hash_counts": {name: len(value) for name, value in identities.items()},
        "unique_task_hash_counts": {name: len(value) for name, value in task_hashes.items()},
        "note": ("DefensiveKV source_id values are sequential IDs rather than the original dataset indices; "
                 "its task/prompt hashes match one-to-one. Layer-DefensiveKV and mixed records strictly match "
                 "regular task/source_id/prompt-hash identities.") if defensive_known_mismatch else None,
        "passed": passed,
    }


def aggregate(base: Path, metric_path: Path, output_dir: Path) -> dict[str, Any]:
    paths = {
        "regular": base / "results/ruler_16k_all13_methods_r20_n5_v1",
        "defensivekv": base / "results/ruler_16k_all13_defensivekv_r20_n5_v1",
        "mixed": base / "results/ruler_16k_all13_waterfill_defensivekv_r20_n5_v1",
        "layer_defensivekv": base / "results/ruler_16k_all13_layer_defensivekv_r20_n5_v1",
    }
    summaries = {name: read_json(path / "summary.json") for name, path in paths.items()}
    rows = {name: read_jsonl(path / "predictions.jsonl") for name, path in paths.items()}
    metric = load_calculate_metrics(metric_path)

    scores: dict[str, Any] = {}
    for method in REGULAR_METHODS:
        scores[method] = score_rows([row for row in rows["regular"] if row["method"] == method], metric)
    scores["defensivekv_20"] = score_rows(rows["defensivekv"], metric, defensive=True)
    scores["layer_defensivekv_20"] = score_rows(rows["layer_defensivekv"], metric, defensive=True)
    scores["waterfill_defensivekv_r20"] = score_rows(rows["mixed"], metric)
    fullkv_macro = scores["fullkv"]["macro_average"]
    for score in scores.values():
        score["relative_to_fullkv_percent"] = score["macro_average"] / fullkv_macro * 100.0
    ordered_scores = {}
    for method in METHOD_ORDER:
        scores[method]["display_name"] = DISPLAY_NAMES[method]
        ordered_scores[method] = scores[method]

    validation = {
        "configuration": [validate_config(name, summaries[name]) for name in paths],
        "regular_group": validate_regular(summaries["regular"], rows["regular"]),
        "defensivekv_group": validate_defensive(summaries["defensivekv"], rows["defensivekv"]),
        "mixed_group": validate_mixed(summaries["mixed"], rows["mixed"]),
        "layer_defensivekv_group": validate_layer_defensive(
            summaries["layer_defensivekv"], rows["layer_defensivekv"]),
        "cross_group_prompt_identity": prompt_identity_validation({
            "regular": rows["regular"], "defensivekv": rows["defensivekv"],
            "mixed": rows["mixed"], "layer_defensivekv": rows["layer_defensivekv"],
        }),
    }
    validation["official_metric_recomputed"] = {
        "passed": True, "module": str(metric_path), "function": "calculate_metrics"
    }
    validation["all_required_checks_passed"] = (
        all(item["passed"] for item in validation["configuration"])
        and validation["regular_group"]["passed"]
        and validation["defensivekv_group"]["passed"]
        and validation["mixed_group"]["passed"]
        and validation["layer_defensivekv_group"]["passed"]
        and validation["cross_group_prompt_identity"]["passed"]
    )

    result = {
        "benchmark": "RULER", "schema_version": 1,
        "experiment_config": {
            "context_length": 16384, "seed": 42, "num_samples_per_task": 5,
            "num_tasks": len(TASKS), "tasks": list(TASKS),
            "metric": "official RULER string match", "macro_average": "unweighted mean across 13 tasks",
            "target_prompt_kv_ratio": TARGET_RATIO,
            "source_groups": {name: str(path) for name, path in paths.items()},
        },
        "validation": validation, "method_order": list(METHOD_ORDER), "scores": ordered_scores,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (output_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "macro_average", "relative_to_fullkv_percent", *TASKS])
        for method in METHOD_ORDER:
            score = ordered_scores[method]
            writer.writerow([score["display_name"], score["macro_average"],
                             score["relative_to_fullkv_percent"],
                             *[score["tasks"][task]["string_match"] for task in TASKS]])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and aggregate completed 13-task RULER runs")
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--metric", type=Path,
                        default=Path("/home/liuminglu/kvcache/DefensiveKV/evaluation/ruler/calculate_metrics.py"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.base / "results/ruler_16k_all13_r20_n5_comparison_v1"
    result = aggregate(args.base, args.metric, output_dir)
    print(json.dumps({
        "output_dir": str(output_dir),
        "all_required_checks_passed": result["validation"]["all_required_checks_passed"],
    }, indent=2))


if __name__ == "__main__":
    main()
