import unittest
from argparse import Namespace

from mistral_task_quality import (
    TASK_GENERATION_LENGTHS, answer_has_lexical_evidence, block_map_from_counts,
    fixed_budget_block_map, formula_block_map, formula_total_budget,
    global_empirical_topk_block_map,
    adakv_token_map, method_maps, normalized_evidence_tokens, pyramid_layer_capacities,
    preprocess_prediction, pyramidkv_token_map, random_block_maps, random_budget_block_map,
    random_pair_block_map, ranked_fill_block_counts, summarize,
    score_prediction, snapkv_token_map, streamingllm_token_map, uniform_block_counts,
    waterfill_block_counts,
)


class TaskQualityTests(unittest.TestCase):
    def test_method_matrix_masks_all_layers(self):
        masses = {layer: [0.1, 0.4, 0.2, 0.3] for layer in range(3)}
        variants = method_maps(3, [1], masses, 4, 2, 42, "sample", (2,))
        self.assertEqual([item[0] for item in variants], ["fullkv", "all_a", "selected_top2", "selected_random8", "selected_random8"])
        self.assertIsNone(variants[0][1])
        self.assertEqual(variants[1][1], {0: [], 1: [], 2: []})
        self.assertEqual(variants[2][1], {0: [], 1: [1, 3], 2: []})

    def test_random_is_deterministic_and_saturates(self):
        first = random_block_maps(3, [0, 2], 8, 2, 7, "x")
        second = random_block_maps(3, [0, 2], 8, 2, 7, "x")
        self.assertEqual(first, second)
        self.assertEqual(first[0][0], [0, 1, 2])

    def test_formula_budget_includes_fixed_a(self):
        self.assertEqual(formula_total_budget(4, 1000, 100, 100, 0.2, 10), 4)
        self.assertEqual(formula_total_budget(4, 400, 100, 100, 0.2, 10), 0)

    def test_uniform_budget_is_exact_and_balanced(self):
        counts = uniform_block_counts(32, 7, 101)
        self.assertEqual(sum(counts.values()), 101)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual([counts[layer] for layer in range(5)], [4, 4, 4, 4, 4])
        self.assertEqual(counts[5], 3)
        saturated = uniform_block_counts(32, 2, 64)
        self.assertEqual(sum(saturated.values()), 64)
        self.assertTrue(all(count == 2 for count in saturated.values()))

    def test_ranked_fill_preserves_budget_and_priority(self):
        ranking = [2, 0, 3, 1]
        counts = ranked_fill_block_counts(4, 3, 8, ranking)
        self.assertEqual(sum(counts.values()), 8)
        self.assertEqual(counts, {0: 3, 1: 0, 2: 3, 3: 2})

    def test_count_mapping_selects_top_mass_blocks(self):
        masses = {0: [0.1, 0.8, 0.4], 1: [0.9, 0.2, 0.7]}
        mapping = block_map_from_counts(masses, {0: 2, 1: 1})
        self.assertEqual(mapping, {0: [1, 2], 1: [0]})

    def test_fixed_baselines_share_formula_budget(self):
        masses = {layer: [0.1, 0.8, 0.4] for layer in range(4)}
        ranking = [3, 1, 2, 0]
        uniform_map, uniform_details = fixed_budget_block_map(
            4, 1000, 100, 100, 0.2, masses, "uniform", ranking
        )
        ranked_map, ranked_details = fixed_budget_block_map(
            4, 1000, 100, 100, 0.2, masses, "ranked_fill", ranking
        )
        expected = formula_total_budget(4, 1000, 100, 100, 0.2, 3)
        self.assertEqual(uniform_details["allocated_layer_blocks"], expected)
        self.assertEqual(ranked_details["allocated_layer_blocks"], expected)
        self.assertEqual(sum(map(len, uniform_map.values())), expected)
        self.assertEqual(sum(map(len, ranked_map.values())), expected)
        self.assertIn("allocation_rule", uniform_details)
        self.assertIn("allocation_rule", ranked_details)


    def test_random_budget_samples_unique_pairs_with_exact_formula_budget(self):
        mapping, details = random_budget_block_map(4, 1000, 100, 100, 0.2, 3, 42, "sample", 0)
        expected = formula_total_budget(4, 1000, 100, 100, 0.2, 3)
        pairs = [(layer, block) for layer, blocks in mapping.items() for block in blocks]
        self.assertEqual(len(pairs), expected)
        self.assertEqual(len(set(pairs)), expected)
        self.assertEqual(details["total_layer_block_budget"], expected)
        self.assertEqual(details["allocated_layer_blocks"], expected)
        self.assertEqual(sum(details["layer_block_counts"].values()), expected)
        self.assertAlmostEqual(details["achieved_decode_start_kv_ratio"], 0.2)
        self.assertEqual(details["allocation_rule"], "uniform_without_replacement_over_all_layer_block_pairs")

    def test_random_budget_is_repeatable_and_repeat_specific(self):
        first = random_budget_block_map(8, 2000, 100, 100, 0.2, 8, 7, "x", 0)
        second = random_budget_block_map(8, 2000, 100, 100, 0.2, 8, 7, "x", 0)
        other_repeat = random_budget_block_map(8, 2000, 100, 100, 0.2, 8, 7, "x", 1)
        self.assertEqual(first, second)
        self.assertNotEqual(first[0], other_repeat[0])
        self.assertNotEqual(first[1]["random_seed"], other_repeat[1]["random_seed"])

    def test_random_pair_budget_rejects_capacity_overflow(self):
        with self.assertRaisesRegex(ValueError, "exceeds capacity"):
            random_pair_block_map(2, 3, 7, 42, "sample", 0)

    def test_summary_averages_repeats_within_sample_first(self):
        rows = [
            {"method": "fullkv", "source_id": "a", "score": 10, "output_tokens": 1, "decode_seconds": 1},
            {"method": "fullkv", "source_id": "b", "score": 20, "output_tokens": 1, "decode_seconds": 1},
            {"method": "random_budget_r20", "source_id": "a", "score": 0, "output_tokens": 1, "decode_seconds": 1},
            {"method": "random_budget_r20", "source_id": "a", "score": 20, "output_tokens": 1, "decode_seconds": 1},
            {"method": "random_budget_r20", "source_id": "b", "score": 30, "output_tokens": 1, "decode_seconds": 1},
            {"method": "random_budget_r20", "source_id": "b", "score": 50, "output_tokens": 1, "decode_seconds": 1},
        ]
        args = Namespace(
            bootstrap_draws=10, seed=1, task="gov_report", max_new_tokens=1,
            important_layers=8, budgets=[8], random_repeats=2, block_size=128,
            sink_tokens=4, local_tokens=512, formula_target_kv_ratio=0.2,
            formula_target_kv_ratios=[0.2], min_prompt_tokens=0,
            require_answer_evidence=False,
        )
        report = summarize(rows, args)["methods"]["random_budget_r20"]
        self.assertEqual(report["per_sample"], {"a": 10.0, "b": 40.0})
        self.assertEqual(report["scores"]["mean"], 25.0)
        self.assertEqual(report["repeats_per_sample"], 2)

    def test_waterfill_obeys_budget_and_prefers_weight(self):
        masses = {0: [0.5, 0.3, 0.2], 1: [0.05, 0.03, 0.02]}
        counts, parameters = waterfill_block_counts(masses, 3)
        self.assertEqual(sum(counts.values()), 3)
        self.assertGreater(counts[0], counts[1])
        self.assertGreater(parameters[0]["remote_mass"], parameters[1]["remote_mass"])

    def test_formula_map_reports_emergent_a_and_b(self):
        masses = {0: [0.5, 0.3, 0.2], 1: [0.05, 0.03, 0.02]}
        mapping, details = formula_block_map(2, 1000, 100, 100, 0.2, masses)
        self.assertEqual(sum(map(len, mapping.values())), 2)
        self.assertAlmostEqual(details["achieved_decode_start_kv_ratio"], 0.2)
        self.assertEqual(details["allocated_layer_blocks"], 2)

    def test_global_empirical_topk_selects_highest_masses_with_exact_budget(self):
        masses = {0: [0.1, 0.8, 0.4], 1: [0.9, 0.2, 0.7]}
        mapping, details = global_empirical_topk_block_map(
            2, 1000, 100, 100, 0.25, masses,
        )
        self.assertEqual(mapping, {0: [1], 1: [0, 2]})
        self.assertEqual(details["total_layer_block_budget"], 3)
        self.assertEqual(details["allocated_layer_blocks"], 3)
        self.assertEqual(sum(details["layer_block_counts"].values()), 3)
        self.assertEqual(
            details["allocation_rule"],
            "global_empirical_block_mass_desc_layer_asc_block_asc",
        )

    def test_global_empirical_topk_uses_deterministic_tie_break(self):
        masses = {0: [1.0, 1.0], 1: [1.0, 1.0]}
        mapping, _ = global_empirical_topk_block_map(
            2, 1000, 100, 100, 0.25, masses,
        )
        self.assertEqual(mapping, {0: [0, 1], 1: [0]})

    def test_global_empirical_topk_can_differ_from_waterfill(self):
        masses = {0: [100.0, 0.0, 0.0], 1: [10.0, 9.0, 8.0]}
        global_map, global_details = global_empirical_topk_block_map(
            2, 1000, 100, 100, 0.2, masses,
        )
        waterfill_map, waterfill_details = formula_block_map(
            2, 1000, 100, 100, 0.2, masses,
        )
        self.assertEqual(global_details["total_layer_block_budget"],
                         waterfill_details["total_layer_block_budget"])
        self.assertEqual(global_map, {0: [0], 1: [0]})
        self.assertEqual(waterfill_map, {0: [0, 1], 1: []})
        self.assertNotEqual(global_map, waterfill_map)

    def test_global_empirical_topk_rejects_unequal_candidate_counts(self):
        with self.assertRaisesRegex(ValueError, "same candidate block count"):
            global_empirical_topk_block_map(
                2, 1000, 100, 100, 0.2, {0: [1.0], 1: [0.5, 0.4]},
            )

    def test_answer_evidence_uses_normalized_contiguous_tokens(self):
        self.assertTrue(answer_has_lexical_evidence("It has 7.2 million residents.", ["7.2 million"]))
        self.assertTrue(answer_has_lexical_evidence("At the University.", ["university"]))
        self.assertFalse(answer_has_lexical_evidence("It has 12,817 residents.", ["7.2 million"]))
        self.assertEqual(normalized_evidence_tokens("The University"), ["university"])

    def test_streamingllm_keeps_sink_and_recent_window(self):
        mapping, details = streamingllm_token_map(2, 3, 20, 0.2, 1)
        self.assertEqual(mapping[0][0], [0, 17, 18, 19])
        self.assertTrue(all(tokens == [0, 17, 18, 19] for heads in mapping.values() for tokens in heads.values()))
        self.assertAlmostEqual(details["actual_kv_ratio"], 0.2)

    def test_snapkv_fixed_budget_per_head(self):
        scores = {layer: [[float(index) for index in range(20)] for _ in range(2)] for layer in range(2)}
        mapping, details = snapkv_token_map(scores, 0.2)
        self.assertTrue(all(len(tokens) == 4 for heads in mapping.values() for tokens in heads.values()))
        self.assertAlmostEqual(details["actual_kv_ratio"], 0.2)

    def test_adakv_exact_global_budget_and_safeguard(self):
        scores = {layer: [[float(index + head) for index in range(20)] for head in range(2)] for layer in range(2)}
        mapping, details = adakv_token_map(scores, 0.2, 0.2)
        self.assertEqual(sum(len(tokens) for heads in mapping.values() for tokens in heads.values()), 16)
        self.assertAlmostEqual(details["actual_kv_ratio"], 0.2)
        self.assertTrue(all(len(tokens) >= 1 for heads in mapping.values() for tokens in heads.values()))

    def test_pyramid_exact_budget_and_decreasing_capacities(self):
        capacities = pyramid_layer_capacities(4, 100, 0.2, 20)
        self.assertEqual(sum(capacities), 80)
        self.assertEqual(capacities, sorted(capacities, reverse=True))
        scores = {layer: [[float(index) for index in range(100)] for _ in range(2)] for layer in range(4)}
        mapping, details = pyramidkv_token_map(scores, 0.2, 20, 8)
        self.assertEqual(sum(len(tokens) for heads in mapping.values() for tokens in heads.values()), 160)
        self.assertAlmostEqual(details["actual_kv_ratio"], 0.2)

    def test_task_generation_lengths(self):
        self.assertEqual(TASK_GENERATION_LENGTHS["hotpotqa"], 32)
        self.assertEqual(TASK_GENERATION_LENGTHS["lcc"], 64)
        self.assertEqual(TASK_GENERATION_LENGTHS["multi_news"], 512)
        self.assertEqual(TASK_GENERATION_LENGTHS["repobench-p"], 64)
        self.assertEqual(len(TASK_GENERATION_LENGTHS), 16)

    def test_score_uses_best_reference(self):
        metric = lambda prediction, answer, **kwargs: 1.0 if prediction == answer else 0.25
        self.assertEqual(score_prediction(metric, "right", ["wrong", "right"], []), 100.0)

    def test_official_first_line_prediction_preprocessing(self):
        self.assertEqual(preprocess_prediction("trec", "\nCity\nQuestion: extra"), "City")
        self.assertEqual(preprocess_prediction("triviaqa", "answer\nextra"), "answer")
        self.assertEqual(preprocess_prediction("samsum", "summary\nDialogue: extra"), "summary")
        self.assertEqual(preprocess_prediction("gov_report", "line one\nline two"), "line one\nline two")


if __name__ == "__main__":
    unittest.main()
