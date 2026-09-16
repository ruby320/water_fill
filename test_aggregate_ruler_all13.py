import unittest

from aggregate_ruler_all13 import (
    METHOD_ORDER,
    TASKS,
    prompt_identity_validation,
    ratio_range,
    validate_layer_defensive,
)


class AggregateRulerAll13Tests(unittest.TestCase):
    def test_ratio_range(self):
        self.assertEqual(ratio_range([0.2, 0.19, 0.21]), {"min": 0.19, "max": 0.21})

    def test_prompt_identity_detects_source_id_mismatch_but_hash_alignment(self):
        groups = {
            "a": [{"task": TASKS[0], "source_id": "cwe:99", "prompt_hash": "abc"}],
            "b": [{"task": TASKS[0], "source_id": "cwe:0", "prompt_token_sha256": "abc"}],
        }
        result = prompt_identity_validation(groups)
        self.assertFalse(result["exact_task_source_id_prompt_hash_match"])
        self.assertTrue(result["task_prompt_hash_sets_match"])
        self.assertFalse(result["passed"])

    def test_expected_method_set_excludes_direct_empirical_topk(self):
        self.assertEqual(len(METHOD_ORDER), 9)
        self.assertIn("layer_defensivekv_20", METHOD_ORDER)
        self.assertNotIn("global_empirical_topk_r20", METHOD_ORDER)

    def test_layer_identity_strict_with_known_defensive_mismatch(self):
        regular = [{"task": TASKS[0], "source_id": "cwe:99", "prompt_hash": "abc"}]
        groups = {
            "regular": regular,
            "mixed": regular,
            "layer_defensivekv": [
                {"task": TASKS[0], "source_id": "cwe:99", "prompt_token_sha256": "abc"}
            ],
            "defensivekv": [
                {"task": TASKS[0], "source_id": "cwe:0", "prompt_token_sha256": "abc"}
            ],
        }
        result = prompt_identity_validation(groups)
        self.assertTrue(result["layer_strictly_matches_regular"])
        self.assertTrue(result["mixed_strictly_matches_regular"])
        self.assertTrue(result["defensivekv_known_source_id_mismatch_with_hash_match"])
        self.assertTrue(result["passed"])

    def test_layer_budget_validation(self):
        rows = []
        budgets = []
        for task in TASKS:
            for sample in range(5):
                actuals = [2] * 32
                budget = {
                    "actual_kept_kv_tokens_per_layer": actuals,
                    "actual_head_lengths_per_layer": [[1, 1] for _ in actuals],
                    "global_expected_kept_kv_elements": 64,
                    "global_actual_kept_kv_elements": 64,
                    "raw_kv_tokens_per_layer": 10,
                }
                budgets.append(budget)
                rows.append({
                    "task": task,
                    "source_id": f"{task}:{sample}",
                    "arm": "layer_defensivekv_20",
                    "actual_budget": budget,
                })
        summary = {
            "arm": "layer_defensivekv_20",
            "retention_ratio": 0.2,
            "actual_budgets": budgets,
        }
        result = validate_layer_defensive(summary, rows)
        self.assertTrue(result["passed"])
        self.assertEqual(result["global_expected_equals_actual_rows"], 65)
        self.assertEqual(result["global_actual_ratio_range"], {"min": 0.2, "max": 0.2})


if __name__ == "__main__":
    unittest.main()
