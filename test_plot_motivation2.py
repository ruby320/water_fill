import unittest

import numpy as np

from plot_motivation2 import (
    aggregate_cell_level_dispersion,
    aggregate_layer_importance,
    aggregate_position_heatmap,
    bins_for_coverage,
    dispersion_axis_limits,
    importance_heat_values,
    concentration,
    normalize_importance_to_minimum,
    select_representative_layers,
)


class ConcentrationTests(unittest.TestCase):
    def test_uniform_is_zero(self):
        self.assertAlmostEqual(concentration(np.ones(48) / 48), 0.0, places=12)

    def test_one_hot_is_one(self):
        probabilities = np.zeros(48)
        probabilities[17] = 1.0
        self.assertAlmostEqual(concentration(probabilities), 1.0, places=12)


class AttentionSummaryTests(unittest.TestCase):
    def test_k50_uniform_and_one_hot(self):
        self.assertEqual(bins_for_coverage(np.ones(48), 0.5), 24)
        self.assertEqual(bins_for_coverage([0, 0, 1, 0], 0.5), 1)

    def test_k50_small_vector_boundary(self):
        self.assertEqual(bins_for_coverage([0.5, 0.3, 0.2], 0.5), 1)
        self.assertEqual(bins_for_coverage([0.49, 0.26, 0.25], 0.5), 2)


class DispersionAxisTests(unittest.TestCase):
    def test_formal_observed_max_uses_rounded_dynamic_upper_bound(self):
        observations = [1, 4, 8, 12, 20]
        xmin, xmax, major_step = dispersion_axis_limits(observations, position_bins=48)
        self.assertEqual((xmin, xmax, major_step), (0, 22, 2))
        self.assertTrue(all(xmin <= value <= xmax for value in observations))

    def test_axis_never_exceeds_position_bins(self):
        observations = [2, 47, 48]
        xmin, xmax, major_step = dispersion_axis_limits(observations, position_bins=48)
        self.assertEqual((xmin, xmax, major_step), (0, 48, 4))
        self.assertLessEqual(xmax, 48)
        self.assertGreaterEqual(xmax, max(observations))


class RepresentativeLayerTests(unittest.TestCase):
    def test_high_importance_representatives_follow_median_k50(self):
        importance = [0, 1, 2, 3, 4, 5, 6, 7]
        medians = [9, 8, 7, 6, 5, 4, 12, 3]
        result = select_representative_layers(importance, medians)
        self.assertEqual(result["high_importance_candidates"], [6, 7])
        self.assertEqual(result["important_concentrated"], 7)
        self.assertEqual(result["important_dispersed"], 6)
        self.assertEqual(result["less_important"], 0)
        self.assertIn("cell-level K50 median", result["selection_rule"])


class PlotTransformationTests(unittest.TestCase):
    def test_relative_importance_uses_global_minimum(self):
        importance = np.array([2.0, 8.0, 5.0])
        relative, relative_sem = normalize_importance_to_minimum(importance, [0.2, 0.4, 0.1])
        self.assertAlmostEqual(float(relative.min()), 1.0)
        self.assertAlmostEqual(float(relative.max()), 4.0)
        np.testing.assert_allclose(relative_sem, [0.1, 0.2, 0.05])

    def test_importance_heat_values_use_log_relative_scale(self):
        transformed, vmin, vmax = importance_heat_values([1.0, 10.0, 1756.0])
        np.testing.assert_allclose(transformed, np.log10([1.0, 10.0, 1756.0]))
        self.assertEqual(vmin, 0.0)
        self.assertAlmostEqual(vmax, float(np.log10(1756.0)))


class ImportanceAggregationTests(unittest.TestCase):
    def test_source_first_checkpoint_aggregation(self):
        rows = [
            {"source_id": "a", "checkpoint": 0, "layer_idx": 0, "kl_mean": 0.0},
            {"source_id": "a", "checkpoint": 1, "layer_idx": 0, "kl_mean": 2.0},
            {"source_id": "a", "checkpoint": 2, "layer_idx": 0, "kl_mean": 4.0},
            {"source_id": "b", "checkpoint": 0, "layer_idx": 0, "kl_mean": 10.0},
        ]
        result = aggregate_layer_importance(rows, layers=[0])
        self.assertAlmostEqual(result["per_source"]["a"][0], 2.0)
        self.assertAlmostEqual(result["per_source"]["b"][0], 10.0)
        self.assertAlmostEqual(result["importance"][0], 6.0)
        self.assertAlmostEqual(result["sem"][0], 4.0)
        self.assertNotAlmostEqual(result["importance"][0], np.mean([0.0, 2.0, 4.0, 10.0]))


class CellLevelDispersionTests(unittest.TestCase):
    @staticmethod
    def fixture():
        predictions = [
            {"source_id": "a", "prompt_tokens": 4, "checkpoints": [0, 1]},
            {"source_id": "b", "prompt_tokens": 4, "checkpoints": [0, 1]},
        ]
        blocks = [
            {"block_idx": 0, "start": 0, "end": 1},
            {"block_idx": 1, "start": 1, "end": 2},
            {"block_idx": 2, "start": 2, "end": 3},
            {"block_idx": 3, "start": 3, "end": 4},
        ]
        prefill = [
            {"source_id": source, "layer_idx": layer, "blocks": blocks}
            for source in ("a", "b") for layer in (0, 1)
        ]
        masses = {
            ("a", 0): [1, 0, 0, 0],
            ("a", 1): [1, 1, 0, 0],
            ("b", 0): [1, 1, 1, 0],
            ("b", 1): [1, 1, 1, 1],
        }
        checkpoints = [
            {"source_id": source, "checkpoint": checkpoint, "layer_idx": layer,
             "decode_block_mass": masses[(source, checkpoint)] if layer == 0 else [1, 1, 1, 1]}
            for source in ("a", "b") for checkpoint in (0, 1) for layer in (0, 1)
        ]
        return predictions, prefill, checkpoints

    def test_constructs_one_k50_per_source_checkpoint_cell(self):
        predictions, prefill, checkpoints = self.fixture()
        result = aggregate_cell_level_dispersion(
            predictions, prefill, checkpoints, position_bins=4, layers=[0, 1]
        )
        np.testing.assert_array_equal(result["per_layer_values"][0], [1, 1, 2, 2])
        np.testing.assert_array_equal(result["per_layer_values"][1], [2, 2, 2, 2])
        self.assertEqual(result["structure"]["expected_cell_count_per_layer"], 4)
        self.assertEqual(result["structure"]["actual_cell_counts_by_layer"], {"0": 4, "1": 4})

    def test_box_statistics_use_all_cells(self):
        predictions, prefill, checkpoints = self.fixture()
        stats = aggregate_cell_level_dispersion(
            predictions, prefill, checkpoints, position_bins=4, layers=[0]
        )["per_layer_stats"]["0"]
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["observations"], [1, 1, 2, 2])
        self.assertEqual(stats["median"], 1.5)
        self.assertEqual(stats["q1"], 1.0)
        self.assertEqual(stats["q3"], 2.0)
        self.assertEqual((stats["min"], stats["max"]), (1, 2))


class HeatmapAggregationTests(unittest.TestCase):
    def test_final_and_source_rows_are_normalized(self):
        blocks_a = [{"start": 0, "end": 50}, {"start": 50, "end": 100}]
        blocks_b = [{"start": 0, "end": 100}, {"start": 100, "end": 200}]
        predictions = [
            {"source_id": "a", "prompt_tokens": 100},
            {"source_id": "b", "prompt_tokens": 200},
        ]
        prefill = [
            {"source_id": "a", "layer_idx": 0, "blocks": blocks_a},
            {"source_id": "b", "layer_idx": 0, "blocks": blocks_b},
        ]
        checkpoints = [
            {"source_id": "a", "checkpoint": 0, "layer_idx": 0, "decode_block_mass": [1.0, 3.0]},
            {"source_id": "a", "checkpoint": 1, "layer_idx": 0, "decode_block_mass": [2.0, 6.0]},
            {"source_id": "b", "checkpoint": 0, "layer_idx": 0, "decode_block_mass": [9.0, 1.0]},
        ]
        result = aggregate_position_heatmap(predictions, prefill, checkpoints, position_bins=48, layers=[0])
        self.assertAlmostEqual(float(result["heatmap"][0].sum()), 1.0, places=12)
        for matrix in result["per_source_heatmaps"].values():
            self.assertAlmostEqual(float(matrix[0].sum()), 1.0, places=12)
        self.assertTrue(np.all(result["heatmap"] >= 0))


if __name__ == "__main__":
    unittest.main()
