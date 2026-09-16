import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from plot_motivation1 import build_representative_heatmap, load_allocation_ablation


class RepresentativeHeatmapTests(unittest.TestCase):
    def test_median_nearest_selection_preserves_raw_block_cells(self):
        blocks = [{"block_idx": 0, "start": 20, "end": 40}, {"block_idx": 1, "start": 40, "end": 60}]
        sources = (("low", 0.1, [0.05, 0.05]), ("typical", 0.5, [0.3, 0.2]), ("high", 0.9, [0.8, 0.1]))
        data = {
            "predictions.jsonl": [{"source_id": source, "prompt_tokens": 100, "important_layers": list(range(32))} for source, _, _ in sources],
            "prefill_scores.jsonl": [{"source_id": source, "layer_idx": layer, "blocks": blocks} for source, _, _ in sources for layer in range(32)],
            "checkpoint_metrics.jsonl": [{"source_id": source, "checkpoint": checkpoint, "layer_idx": layer, "decode_remote_mass": remote_mass, "decode_block_mass": block_mass} for source, remote_mass, block_mass in sources for checkpoint in range(8) for layer in range(32)],
        }
        result = build_representative_heatmap(data)
        heatmap = np.asarray(result["heatmap_mean_attention_mass"], dtype=float)
        self.assertEqual(result["source_id"], "typical")
        self.assertEqual(result["checkpoints_per_layer"], {str(i): 8 for i in range(32)})
        self.assertEqual(result["raw_block_count"], 2)
        np.testing.assert_allclose(heatmap, np.tile([0.3, 0.2], (32, 1)))
        self.assertNotIn("selection_scores", result)


class AllocationAblationTests(unittest.TestCase):
    @staticmethod
    def _method(values: list[float], repeats: int = 1) -> dict:
        return {
            "scores": {"mean": float(np.mean(values))},
            "per_sample": {f"sample-{index}": value for index, value in enumerate(values)},
            "repeats_per_sample": repeats,
        }

    def _write_run(self, root: Path, unequal_budget: bool = False, duplicate_random_block: bool = False) -> tuple[Path, Path]:
        sample_ids = [f"sample-{index}" for index in range(5)]
        methods = {
            "all_a": self._method([18, 19, 20, 21, 22]),
            "random_budget_r20": self._method([23, 24, 25, 26, 27], repeats=5),
            "top_layer_budget_r20": self._method([19, 20, 21, 22, 23]),
            "formula_waterfill_r20": self._method([26, 27, 28, 29, 30]),
            "fullkv": self._method([28, 29, 30, 31, 32]),
        }
        root.mkdir()
        summary_path = root / "summary.json"
        summary_path.write_text(json.dumps({
            "task": "gov_report", "samples": 5, "sample_ids": sample_ids, "methods": methods,
        }), encoding="utf-8")
        predictions_path = root / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            for index, source_id in enumerate(sample_ids):
                for method, report in methods.items():
                    repeats = report["repeats_per_sample"]
                    for repeat in range(repeats):
                        allocation = None
                        selected_blocks = {}
                        total = 32
                        if method not in {"all_a", "fullkv"}:
                            if unequal_budget and index == 0 and method == "top_layer_budget_r20":
                                total = 33
                            if method == "random_budget_r20":
                                active = (repeat + index) % 32
                                counts = {str(active): 32}
                            elif method == "top_layer_budget_r20":
                                counts = {"0": total}
                            else:
                                counts = {"0": 8, "8": 8, "16": 8, "31": total - 24}
                            selected_blocks = {
                                layer: list(range(count)) for layer, count in counts.items()
                            }
                            if duplicate_random_block and index == 0 and method == "random_budget_r20" and repeat == 0:
                                selected_blocks[str(active)][-1] = 0
                            allocation = {
                                "target_kv_ratio": 0.2,
                                "total_layer_block_budget": total,
                                "allocated_layer_blocks": total,
                                "layer_block_counts": counts,
                            }
                        handle.write(json.dumps({
                            "source_id": source_id, "task": "gov_report",
                            "prompt_hash": f"hash-{index}", "prompt_tokens": 3000 + index,
                            "candidate_blocks": 40, "method": method,
                            "repeat": repeat if repeats > 1 else None,
                            "score": report["per_sample"][source_id],
                            "selected_blocks": selected_blocks, "allocation": allocation,
                        }) + "\n")
        return summary_path, predictions_path

    def test_validates_repeats_budgets_and_extracts_mean_random_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary, predictions = self._write_run(Path(temporary) / "run")
            result = load_allocation_ablation(summary, predictions, "sample-0")
        self.assertEqual(result["winner"], "formula_waterfill_r20")
        self.assertEqual(result["waterfill_vs_random_delta"], 3.0)
        self.assertEqual(result["methods"]["Random B"]["repeats"], 5)
        random_counts = result["representative_allocation"]["layer_block_counts"]["random_budget_r20"]
        np.testing.assert_allclose(random_counts[:5], [6.4] * 5)
        self.assertEqual(sum(random_counts), 32)
        self.assertTrue(result["validation"]["selected_block_pairs_validated"])

    def test_rejects_unequal_budgets(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary, predictions = self._write_run(Path(temporary) / "run", unequal_budget=True)
            with self.assertRaisesRegex(ValueError, "unequal budgets"):
                load_allocation_ablation(summary, predictions, "sample-0")

    def test_rejects_duplicate_random_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary, predictions = self._write_run(Path(temporary) / "run", duplicate_random_block=True)
            with self.assertRaisesRegex(ValueError, "Duplicate or invalid selected blocks"):
                load_allocation_ablation(summary, predictions, "sample-0")


if __name__ == "__main__":
    unittest.main()
