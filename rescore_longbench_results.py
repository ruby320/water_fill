from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "trec", "triviaqa", "samsum", "gov_report", "qmsum", "multi_news",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]
LEGACY_TASKS = {"gov_report", "hotpotqa", "lcc", "multi_news"}
METHODS = [
    "fullkv", "formula_waterfill_r20", "ada_snapkv_r20",
    "pyramidkv_r20", "snapkv_r20", "streamingllm_r20",
]
FIRST_LINE_TASKS = {"trec", "triviaqa", "samsum", "lsht"}


def load_scorer(path: Path, site_packages: Path) -> Any:
    sys.path.append(str(site_packages))
    spec = importlib.util.spec_from_file_location("longbench_metrics", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dataset2metric


def prediction_paths(root: Path, task: str) -> list[Path]:
    if task in LEGACY_TASKS:
        return [
            root / f"{task}_task_quality_baselines_r20_v1/predictions.jsonl",
            root / f"{task}_task_quality_new_baselines_r20_v1/predictions.jsonl",
        ]
    return [root / f"{task}_task_quality_all_methods_r20_v1/predictions.jsonl"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--dataset", type=Path, default=Path("/home/liuminglu/kvcache/datasets/defensivekv_dataset/longbench"))
    parser.add_argument("--scorer", type=Path, default=Path("/home/liuminglu/kvcache/DefensiveKV/evaluation/longbench/calculate_metrics.py"))
    parser.add_argument("--site_packages", type=Path, default=Path("/home/liuminglu/miniconda3/envs/defensivekv/lib/python3.10/site-packages"))
    parser.add_argument("--output", type=Path, default=Path("results/longbench_16tasks_corrected_r20_v1.json"))
    args = parser.parse_args()

    from datasets import load_from_disk
    dataset = load_from_disk(str(args.dataset))
    references = {str(row["_id"]): row for row in dataset}
    metrics = load_scorer(args.scorer, args.site_packages)
    tasks: dict[str, Any] = {}
    all_sample_deltas: list[float] = []

    for task in TASKS:
        rows = []
        for path in prediction_paths(args.results_root, task):
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        normalized = []
        for row in rows:
            method = "ada_snapkv_r20" if row["method"] == "adakv_snapkv_r20" else row["method"]
            if method not in METHODS:
                continue
            source = references[str(row["source_id"])]
            prediction = row["prediction"]
            if task in FIRST_LINE_TASKS:
                prediction = prediction.lstrip("\n").split("\n")[0]
            classes = list(source["all_classes"]) if source["all_classes"] is not None else []
            score = 100.0 * max(
                float(metrics[task](prediction, str(answer), all_classes=classes))
                for answer in source["answer"]
            )
            normalized.append({
                "source_id": str(row["source_id"]), "method": method, "score": score,
                "prompt_tokens": int(row["prompt_tokens"]), "output_tokens": int(row["output_tokens"]),
            })
        reports = {}
        for method in METHODS:
            method_rows = [row for row in normalized if row["method"] == method]
            by_sample = {row["source_id"]: row["score"] for row in method_rows}
            values = list(by_sample.values())
            reports[method] = {
                "mean": mean(values), "per_sample": by_sample,
                "average_prompt_tokens": mean(row["prompt_tokens"] for row in method_rows),
                "average_output_tokens": mean(row["output_tokens"] for row in method_rows),
            }
        full = reports["fullkv"]
        for method, report in reports.items():
            report["quality_retention_vs_fullkv"] = report["mean"] / full["mean"] if full["mean"] else None
        water = reports["formula_waterfill_r20"]
        sample_deltas = [water["per_sample"][key] - full["per_sample"][key] for key in full["per_sample"]]
        all_sample_deltas.extend(sample_deltas)
        tasks[task] = {"methods": reports, "waterfill_sample_deltas": sample_deltas}

    aggregate = {}
    for method in METHODS:
        retention = [tasks[task]["methods"][method]["quality_retention_vs_fullkv"] for task in TASKS]
        aggregate[method] = {
            "raw_score_macro_average": mean(tasks[task]["methods"][method]["mean"] for task in TASKS),
            "quality_retention_macro_average": mean(value for value in retention if value is not None),
            "quality_retention_median": median(value for value in retention if value is not None),
        }
    aggregate["formula_waterfill_r20"]["sample_delta_counts"] = {
        "wins": sum(value > 1e-9 for value in all_sample_deltas),
        "ties": sum(abs(value) <= 1e-9 for value in all_sample_deltas),
        "losses": sum(value < -1e-9 for value in all_sample_deltas),
        "total": len(all_sample_deltas),
    }
    output = {
        "experiment": "LongBench 16-task corrected scoring",
        "correction": "Official first-line preprocessing applied to trec, triviaqa, and samsum",
        "model": "Mistral-7B-Instruct-v0.2", "samples_per_task": 5, "seed": 42,
        "target_prompt_kv_ratio": 0.20, "tasks": tasks, "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
