import unittest

from plot_motivation3_causal import compute_marginals, ratio_of_sums, validate_recovery_series


class RatioOfSumsTests(unittest.TestCase):
    def test_ratio_of_sums_is_not_mean_of_ratios(self):
        rows = [
            {"kl_mean": 1.0, "a_only_kl_mean": 2.0},
            {"kl_mean": 9.0, "a_only_kl_mean": 10.0},
        ]
        residual, recovery = ratio_of_sums(rows)
        self.assertAlmostEqual(residual, 100.0 * 10.0 / 12.0)
        self.assertAlmostEqual(recovery, 100.0 - 100.0 * 10.0 / 12.0)
        self.assertNotAlmostEqual(residual, 100.0 * ((1.0 / 2.0 + 9.0 / 10.0) / 2.0))


class MarginalTests(unittest.TestCase):
    def test_marginal_intervals_use_interval_width(self):
        recovery = {4: 20.0, 8: 32.0, 12: 40.0, 16: 44.0, 24: 52.0}
        self.assertEqual(
            compute_marginals(recovery),
            {"(0,4]": 5.0, "(4,8]": 3.0, "(8,12]": 2.0,
             "(12,16]": 1.0, "(16,24]": 1.0},
        )


class CompletenessAndMonotonicityTests(unittest.TestCase):
    def test_missing_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Recovery budgets"):
            compute_marginals({4: 1.0, 8: 2.0, 12: 3.0, 16: 4.0})

    def test_complete_monotonic_series_passes(self):
        recovery = {0: 0.0, 4: 1.0, 8: 2.0, 12: 2.0, 16: 5.0, 24: 8.0}
        self.assertEqual(validate_recovery_series(recovery), [0.0, 1.0, 2.0, 2.0, 5.0, 8.0])

    def test_nonmonotonic_series_is_rejected(self):
        recovery = {0: 0.0, 4: 1.0, 8: 2.0, 12: 1.9, 16: 5.0, 24: 8.0}
        with self.assertRaisesRegex(ValueError, "monotonic non-decreasing"):
            validate_recovery_series(recovery)


if __name__ == "__main__":
    unittest.main()
