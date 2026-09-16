import math
import unittest

import numpy as np

from plot_motivation3 import N_LAYERS, layer_utility, water_level


class LayerUtilityTests(unittest.TestCase):
    def test_k50_uses_minimum_empirical_prefix(self):
        utility = layer_utility([4.0, 3.0, 2.0, 1.0])
        self.assertEqual(utility["K50"], 2)
        self.assertLess(utility["empirical"][1], 0.5 * utility["M"])
        self.assertGreaterEqual(utility["empirical"][2], 0.5 * utility["M"])
        self.assertAlmostEqual(utility["lambda"], math.log(2.0) / 2.0)

    def test_empirical_and_surrogate_curves(self):
        utility = layer_utility([4.0, 3.0, 2.0, 1.0])
        np.testing.assert_allclose(utility["empirical"], [0.0, 4.0, 7.0, 9.0, 10.0])
        np.testing.assert_allclose(utility["empirical_marginal"], [4.0, 3.0, 2.0, 1.0])
        self.assertAlmostEqual(utility["surrogate"][0], 0.0)
        self.assertAlmostEqual(utility["surrogate"][utility["K50"]], 0.5 * utility["M"])
        self.assertTrue(np.all(np.diff(utility["surrogate"]) > 0))

    def test_empirical_marginal_sorts_block_mass_descending(self):
        block_mass = [0.2, 1.4, 0.1, 0.8]
        utility = layer_utility(block_mass)
        np.testing.assert_allclose(utility["empirical_marginal"], [1.4, 0.8, 0.2, 0.1])
        np.testing.assert_allclose(utility["empirical_marginal"], np.diff(utility["empirical"]))

    def test_marginals_telescope_to_surrogate(self):
        utility = layer_utility([4.0, 3.0, 2.0, 1.0])
        np.testing.assert_allclose(np.cumsum(utility["marginal"]), utility["surrogate"][1:])
        self.assertTrue(np.all(np.diff(utility["marginal"]) < 0))
        expected_first = utility["M"] * (1.0 - math.exp(-utility["lambda"]))
        self.assertAlmostEqual(utility["marginal"][0], expected_first)


class WaterLevelTests(unittest.TestCase):
    def test_budget_prefix_and_threshold_are_consistent(self):
        utilities = {
            layer: layer_utility([8.0, 4.0, 2.0, 1.0])
            for layer in range(N_LAYERS)
        }
        tau, counts, threshold = water_level(utilities, total_budget=35)
        self.assertEqual(sum(counts), 35)
        self.assertEqual(counts[:3], [2, 2, 2])
        self.assertTrue(all(count == 1 for count in counts[3:]))
        self.assertGreaterEqual(threshold["last_budget_marginal"]["value"], tau)
        self.assertLessEqual(threshold["first_outside_marginal"]["value"], tau)

    def test_tau_uses_last_candidate_when_budget_exhausts_capacity(self):
        utilities = {layer: layer_utility([1.0]) for layer in range(N_LAYERS)}
        tau, counts, threshold = water_level(utilities, total_budget=N_LAYERS)
        self.assertEqual(counts, [1] * N_LAYERS)
        self.assertIsNone(threshold["first_outside_marginal"])
        self.assertAlmostEqual(tau, threshold["last_budget_marginal"]["value"])


if __name__ == "__main__":
    unittest.main()
