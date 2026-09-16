import unittest
from argparse import Namespace

from mistral_new_waterfill import (
    head_block_budget,
    hierarchical_head_block_counts,
    layer_masses_from_heads,
    new_waterfill_token_map,
    new_waterfill_block_map,
)
from mistral_new_waterfill_ruler import METHODS, summarize


class NewWaterfillTests(unittest.TestCase):
    def test_head_block_budget_preserves_global_ratio(self):
        budget = head_block_budget(2, 2, 100, 20, 10, 0.5, 8)
        self.assertEqual(budget, 12)
        retained = 2 * 2 * 20 + budget * 10
        self.assertEqual(retained, 2 * 2 * 100 * 0.5)

    def test_layer_signal_averages_heads(self):
        masses = {0: [[2.0, 0.0], [0.0, 2.0]], 1: [[1.0, 1.0], [3.0, 1.0]]}
        self.assertEqual(layer_masses_from_heads(masses), {0: [1.0, 1.0], 1: [2.0, 1.0]})

    def test_hierarchical_counts_are_exact_and_bounded(self):
        masses = {
            0: [[9.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            1: [[2.0, 2.0, 2.0], [8.0, 1.0, 0.0]],
        }
        counts, details = hierarchical_head_block_counts(masses, 7)
        self.assertEqual(sum(v for heads in counts.values() for v in heads.values()), 7)
        self.assertTrue(all(0 <= v <= 3 for heads in counts.values() for v in heads.values()))
        self.assertEqual(sum(details["layer_head_block_budgets"].values()), 7)

    def test_token_map_is_head_specific_and_budget_exact(self):
        masses = {0: [[9.0, 1.0], [1.0, 9.0]]}
        blocks = [
            {"block_idx": 0, "start": 4, "end": 8},
            {"block_idx": 1, "start": 8, "end": 12},
        ]
        mapping, details = new_waterfill_token_map(
            masses, blocks, prompt_length=16, fixed_tokens=8,
            block_size=4, target_ratio=0.75,
        )
        self.assertNotEqual(mapping[0][0], mapping[0][1])
        self.assertAlmostEqual(details["actual_kv_ratio"], 0.75)
        self.assertEqual(details["allocated_head_blocks"], details["total_head_block_budget"])

    def test_formal_new_waterfill_is_layer_only_and_preserves_set_a(self):
        masses = {0: [9.0, 1.0], 1: [1.0, 9.0]}
        mapping, details = new_waterfill_block_map(
            2, prompt_length=16, fixed_tokens=8, block_size=4,
            target_ratio=0.75, masses=masses,
        )
        self.assertEqual(set(mapping), {0, 1})
        self.assertFalse(details["head_aware"])
        self.assertTrue(details["all_kv_heads_share_layer_blocks"])
        self.assertTrue(details["fixed_set_a_applied_to_every_layer"])
        self.assertAlmostEqual(details["achieved_decode_start_kv_ratio"], 0.75)

    def test_summary_supports_standalone_new_method(self):
        args = Namespace(
            tasks=["qa_1"], num_samples=1, model_path="model", context_length=32768,
            seed=42, target_kv_ratio=0.2, block_size=32, sink_tokens=4, local_tokens=512,
        )
        rows = [{
            "source_id": "qa_1:1", "method": "new_waterfill", "task": "qa_1",
            "answers": ["x"], "prediction": "x", "prompt_tokens": 100,
            "output_tokens": 1, "decode_seconds": 0.1,
        }]
        report = summarize(rows, args, lambda frame: {"qa_1": {"string_match": 100.0}})
        self.assertEqual(report["methods"], list(METHODS))
        self.assertEqual(report["results"]["new_waterfill"]["macro_average"], 100.0)


if __name__ == "__main__":
    unittest.main()
