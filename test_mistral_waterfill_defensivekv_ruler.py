import unittest

from mistral_waterfill_defensivekv_ruler import (
    budget_metadata,
    exact_topk_indices,
    layer_block_counts_to_element_targets,
)


class WaterfillDefensiveKVRulerTests(unittest.TestCase):
    def test_counts_convert_to_exact_flattened_element_targets(self):
        targets = layer_block_counts_to_element_targets(
            {0: 0, 1: 2, 2: 5}, 2000, 8, 128, 516
        )
        self.assertEqual(targets, {0: 4128, 1: 6176, 2: 9248})

    def test_total_budget_matches_waterfill_token_formula(self):
        counts = {0: 0, 1: 3, 2: 1, 3: 2}
        targets = layer_block_counts_to_element_targets(counts, 4096, 8, 128, 516)
        expected = (4 * 516 + sum(counts.values()) * 128) * 8
        self.assertEqual(sum(targets.values()), expected)

    def test_targets_respect_prompt_upper_bound_and_fixed_lower_bound(self):
        targets = layer_block_counts_to_element_targets({0: 0, 1: 100}, 1000, 8, 128, 516)
        self.assertEqual(targets[0], 516 * 8)
        self.assertEqual(targets[1], 1000 * 8)
        with self.assertRaises(ValueError):
            layer_block_counts_to_element_targets({0: -1}, 1000, 8, 128, 516)

    def test_short_prompt_caps_fixed_budget_at_full_cache(self):
        self.assertEqual(
            layer_block_counts_to_element_targets({0: 0, 1: 1}, 128, 8, 128, 516),
            {0: 1024, 1: 1024},
        )

    def test_exact_topk_has_exact_size_and_deterministic_ties(self):
        self.assertEqual(exact_topk_indices([0.2, 0.9, 0.9, 0.1], 2), [1, 2])
        self.assertEqual(exact_topk_indices([0.2, 0.9], 0), [])
        with self.assertRaises(ValueError):
            exact_topk_indices([0.2], 2)

    def test_metadata_requires_exact_per_layer_budget(self):
        result = budget_metadata({0: 0, 1: 1}, {0: 16, 1: 24}, {0: 16, 1: 24}, 10, 2, 0.2)
        self.assertTrue(result["budget_exact"])
        self.assertEqual(result["global_target_kept_elements"], 40)
        self.assertEqual(result["global_actual_ratio"], 1.0)
        with self.assertRaisesRegex(RuntimeError, "budget mismatch"):
            budget_metadata({0: 0}, {0: 16}, {0: 15}, 10, 2, 0.2)


if __name__ == "__main__":
    unittest.main()
