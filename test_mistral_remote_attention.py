import tempfile
import unittest
from pathlib import Path

import numpy as np

from mistral_remote_attention import (
    MistralAttentionPatch, block_set_metrics, cache_signature, clone_dynamic_cache,
    complete_candidate_blocks, effects, model_forward, recovery, run_branch, stable_kl,
    stable_nll, top_blocks, uniform_checkpoints,
)


class PureTests(unittest.TestCase):
    def test_uniform_checkpoints(self):
        self.assertEqual(uniform_checkpoints(99, 8, 8), [0, 13, 26, 39, 51, 64, 77, 90])
        with self.assertRaises(ValueError):
            uniform_checkpoints(8, 8, 8)

    def test_blocks_exclude_a(self):
        blocks = complete_candidate_blocks(1024, 128, 4, 512)
        self.assertEqual(blocks, [
            {"block_idx": 0, "start": 128, "end": 256},
            {"block_idx": 1, "start": 256, "end": 384},
            {"block_idx": 2, "start": 384, "end": 512},
        ])

    def test_set_metrics_and_top(self):
        self.assertEqual(top_blocks([0.1, 0.5, 0.2], 2), [1, 2])
        result = block_set_metrics([0, 1], [1, 2], [0.2, 0.3, 0.5])
        self.assertEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["coverage"], 0.5)

    def test_recovery_small_denominator(self):
        self.assertIsNone(recovery(1e-6, 0.0))
        self.assertIsNone(recovery(-1e-7, 0.0))
        self.assertAlmostEqual(recovery(2e-6, 1e-6), 0.5)

    def test_stable_losses(self):
        import torch
        logits = torch.tensor([1000.0, 999.0, -1000.0])
        self.assertTrue(np.isfinite(stable_nll(logits, 1)))
        self.assertEqual(stable_kl(logits, logits), 0.0)


class TinyMistralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import MistralConfig, MistralForCausalLM
        torch.manual_seed(7)
        cls.torch = torch
        cls.config = MistralConfig(
            vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        )
        cls.config._attn_implementation = "sdpa"
        cls.model = MistralForCausalLM(cls.config).eval()

    def test_capture_set_mask_causality_and_cache_isolation(self):
        from transformers import DynamicCache
        prompt_ids = list(range(1, 25))
        trajectory = [25, 26, 27]
        blocks = complete_candidate_blocks(24, 4, 2, 8)
        patch = MistralAttentionPatch(self.model, blocks, 2, 8)
        cache = DynamicCache(config=self.config)
        try:
            prompt = self.torch.tensor([prompt_ids])
            with self.torch.inference_mode(), patch.mode(prefill_query=(16, 20)):
                model_forward(self.model, prompt, cache, 0)
            self.assertEqual(set(patch.prefill_masses), {0, 1})
            self.assertEqual(len(patch.prefill_masses[0]["per_query_remote_mass"]), 4)
            base = clone_dynamic_cache(cache)
            signature = cache_signature(base)
            reference = run_branch(self.model, patch, base, trajectory, 24, 0, 2, capture=True)
            altered = run_branch(self.model, patch, base, trajectory, 24, 0, 2, 0, [0, 1])
            result = effects(reference, altered)
            self.assertEqual(len(result["delta_nll_steps"]), 2)
            self.assertGreaterEqual(result["kl_step0"], 0.0)
            self.assertEqual(cache_signature(base), signature)
            noop = run_branch(self.model, patch, base, trajectory, 24, 0, 2)
            noop_result = effects(reference, noop)
            self.assertEqual(noop_result["delta_nll_mean"], 0.0)
            self.assertEqual(noop_result["kl_mean"], 0.0)
            self.assertEqual(cache_signature(base), signature)
        finally:
            patch.close(self.model)


if __name__ == "__main__":
    unittest.main()
