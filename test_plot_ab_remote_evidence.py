import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from plot_ab_remote_evidence import (
    REQUIRED_FILES,
    aggregate_heatmap,
    cell_kl,
    compute_stats,
    load_directory,
)


class PlotABRemoteEvidenceTests(unittest.TestCase):
    def write_dataset(self, directory, data):
        for name in REQUIRED_FILES:
            rows = data[name]
            (directory / name).write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

    def base_data(self):
        return {
            "predictions.jsonl": [
                {"source_id": "s1", "prompt_tokens": 100, "important_layers": [0], "task": "synthetic"},
            ],
            "prefill_scores.jsonl": [
                {
                    "source_id": "s1",
                    "layer_idx": 0,
                    "blocks": [
                        {"block_idx": 0, "start": 0, "end": 20},
                        {"block_idx": 1, "start": 50, "end": 70},
                    ],
                    "prefill_remote_mass": 0.4,
                },
            ],
            "checkpoint_metrics.jsonl": [
                {
                    "source_id": "s1",
                    "checkpoint": 0,
                    "layer_idx": 0,
                    "decode_block_mass": [0.25, 0.75],
                    "prefill_b_blocks": [1],
                    "decode_remote_mass": 0.5,
                },
            ],
            "q1_interventions.jsonl": [
                {"source_id": "s1", "checkpoint": 0, "layer_idx": 0, "kl_mean": 10.0},
            ],
            "q2_interventions.jsonl": [
                {
                    "source_id": "s1",
                    "checkpoint": 0,
                    "layer_idx": 0,
                    "branch": "random",
                    "kl_mean": 2.0,
                    "selected_b_blocks": [0],
                },
                {
                    "source_id": "s1",
                    "checkpoint": 0,
                    "layer_idx": 0,
                    "branch": "random",
                    "kl_mean": 4.0,
                    "selected_b_blocks": [1],
                },
                {
                    "source_id": "s1",
                    "checkpoint": 0,
                    "layer_idx": 0,
                    "branch": "prefill",
                    "kl_mean": 1.0,
                    "selected_b_blocks": [1],
                },
                {
                    "source_id": "s1",
                    "checkpoint": 0,
                    "layer_idx": 0,
                    "branch": "oracle",
                    "kl_mean": 0.5,
                    "selected_b_blocks": [0],
                },
            ],
        }

    def test_load_directory_reports_missing_required_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            data = self.base_data()
            missing = "q2_interventions.jsonl"
            for name in REQUIRED_FILES:
                if name != missing:
                    (directory / name).write_text(
                        json.dumps(data[name][0]) + "\n", encoding="utf-8"
                    )
            with self.assertRaisesRegex(FileNotFoundError, missing):
                load_directory(directory)

    def test_cell_kl_averages_random_repeats_within_cell(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self.write_dataset(directory, self.base_data())
            cells = cell_kl(load_directory(directory))

        key = ("s1", 0, 0)
        self.assertEqual(set(cells["random"]), {key})
        self.assertAlmostEqual(cells["random"][key], 3.0)
        self.assertAlmostEqual(cells["A_only"][key], 10.0)

    def test_aggregate_heatmap_bins_averages_cells_and_normalizes_layers(self):
        data = self.base_data()
        data["predictions.jsonl"] = [
            {"source_id": "s1", "prompt_tokens": 100, "important_layers": [0], "task": "synthetic"},
            {"source_id": "s2", "prompt_tokens": 200, "important_layers": [0, 2], "task": "synthetic"},
        ]
        data["prefill_scores.jsonl"] = [
            {
                "source_id": "s1",
                "layer_idx": 0,
                "blocks": [{"start": 0, "end": 20}, {"start": 50, "end": 70}],
                "prefill_remote_mass": 0.4,
            },
            {
                "source_id": "s2",
                "layer_idx": 0,
                "blocks": [{"start": 40, "end": 60}, {"start": 140, "end": 160}],
                "prefill_remote_mass": 0.5,
            },
            {
                "source_id": "s2",
                "layer_idx": 2,
                "blocks": [{"start": 100, "end": 120}, {"start": 180, "end": 200}],
                "prefill_remote_mass": 0.6,
            },
        ]
        data["checkpoint_metrics.jsonl"] = [
            {"source_id": "s1", "checkpoint": 0, "layer_idx": 0, "decode_block_mass": [1, 3], "prefill_b_blocks": [1], "decode_remote_mass": 4},
            {"source_id": "s1", "checkpoint": 1, "layer_idx": 0, "decode_block_mass": [3, 1], "prefill_b_blocks": [0], "decode_remote_mass": 4},
            {"source_id": "s2", "checkpoint": 0, "layer_idx": 0, "decode_block_mass": [2, 2], "prefill_b_blocks": [0], "decode_remote_mass": 4},
            {"source_id": "s2", "checkpoint": 0, "layer_idx": 2, "decode_block_mass": [2, 6], "prefill_b_blocks": [1], "decode_remote_mass": 8},
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self.write_dataset(directory, data)
            heatmap, normalized = aggregate_heatmap(load_directory(directory), position_bins=4)

        np.testing.assert_allclose(heatmap, [[4 / 3, 2 / 3, 4 / 3, 2 / 3], [0, 0, 2, 6]])
        np.testing.assert_allclose(normalized, [[1 / 3, 1 / 6, 1 / 3, 1 / 6], [0, 0, 1 / 4, 3 / 4]])
        np.testing.assert_allclose(normalized.sum(axis=1), [1, 1])

    def test_compute_stats_uses_ratio_of_sums_restoration(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self.write_dataset(directory, self.base_data())
            data = load_directory(directory)

        keys = [("s1", 0, 0), ("s1", 1, 0)]
        cells = {
            "A_only": dict(zip(keys, [1.0, 9.0])),
            "random": dict(zip(keys, [0.5, 8.0])),
            "prefill": dict(zip(keys, [0.2, 1.8])),
            "oracle": dict(zip(keys, [0.1, 0.9])),
        }
        stats = compute_stats(data, cells)
        restoration = stats["ratio_of_sums_restoration"]

        self.assertAlmostEqual(restoration["A+Random-B"], 0.15)
        self.assertAlmostEqual(restoration["A+Selected-B (prefill)"], 0.8)
        self.assertAlmostEqual(restoration["A+Oracle-B"], 0.9)
        per_cell_average = ((1 - 0.5 / 1.0) + (1 - 8.0 / 9.0)) / 2
        self.assertNotAlmostEqual(restoration["A+Random-B"], per_cell_average)


if __name__ == "__main__":
    unittest.main()
