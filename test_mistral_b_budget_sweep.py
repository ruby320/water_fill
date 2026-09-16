import unittest

from mistral_b_budget_sweep import effective_budget, kl_restoration, nested_random_selections
from mistral_remote_attention import MistralAttentionPatch, complete_candidate_blocks, model_forward


class BudgetSweepPureTests(unittest.TestCase):
    def test_budget_saturation(self):
        self.assertEqual(effective_budget(24, 18), (18, True))
        self.assertEqual(effective_budget(16, 18), (16, False))

    def test_nested_random(self):
        selected = nested_random_selections(18, [4, 8, 12, 16, 24], 123)
        self.assertEqual(len(selected[24]), 18)
        self.assertTrue(set(selected[4]) < set(selected[8]) < set(selected[12]) < set(selected[16]) < set(selected[24]))
        self.assertEqual(selected, nested_random_selections(18, [4, 8, 12, 16, 24], 123))

    def test_kl_restoration(self):
        self.assertEqual(kl_restoration(2.0, 0.5), (0.75, False))
        self.assertEqual(kl_restoration(1e-12, 0.0), (None, True))
        self.assertEqual(kl_restoration(0.0, 0.0), (None, True))


class TinyMistralIntegrationTest(unittest.TestCase):
    def test_prefill_capture_without_gpu(self):
        import torch
        from transformers import DynamicCache, MistralConfig, MistralForCausalLM

        config = MistralConfig(
            vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        )
        config._attn_implementation = "sdpa"
        model = MistralForCausalLM(config).eval()
        blocks = complete_candidate_blocks(24, 4, 2, 8)
        patch = MistralAttentionPatch(model, blocks, 2, 8)
        try:
            cache = DynamicCache(config=config)
            with torch.inference_mode(), patch.mode(prefill_query=(16, 20)):
                model_forward(model, torch.tensor([list(range(1, 25))]), cache, 0)
            self.assertEqual(set(patch.prefill_masses), {0, 1})
            self.assertEqual(len(patch.prefill_masses[0]["block_mass"]), len(blocks))
        finally:
            patch.close(model)


if __name__ == "__main__":
    unittest.main()
