"""Check continuous source-time integration against independent examples."""

import unittest

import numpy as np

from dsh.validation.convolution import convolve_scored_impulses


class TestContinuousImpulseConvolution(unittest.TestCase):
    def test_fractional_delay_splits_a_top_hat_between_arrival_bins(self):
        result = convolve_scored_impulses(
            [0.5], [2.0], [0.0, 1.0], [1.0], [0.0, 1.0, 2.0]
        )
        np.testing.assert_allclose(result.impulse_fluence, [2.0, 0.0])
        np.testing.assert_allclose(result.expected_flare_fluence, [1.0, 1.0])
        np.testing.assert_allclose(result.conditional_variance, [1.0, 1.0])
        self.assertEqual(result.expected_window_fluence, 2.0)
        self.assertEqual(result.conditional_window_variance, 0.0)

    def test_unequal_cell_durations_use_fluence_not_bin_count(self):
        result = convolve_scored_impulses(
            [0.0], [1.0], [0.0, 1.0, 3.0], [1.0, 3.0], [0.0, 1.0, 2.0, 3.0]
        )
        np.testing.assert_allclose(result.expected_flare_fluence, [0.25, 0.375, 0.375])
        np.testing.assert_allclose(result.expected_flare_fluence.sum(), 1.0)

    def test_conditional_variance_matches_independent_emission_draws(self):
        delays = np.array([0.2, 0.8, 1.1])
        weights = np.array([1.0, 2.0, 0.5])
        result = convolve_scored_impulses(
            delays, weights, [0.0, 1.0], [1.0], [0.0, 1.0, 2.0]
        )
        emission = np.random.default_rng(93).uniform(size=(50_000, len(delays)))
        simulated = np.sum(
            weights[None, :] * ((emission + delays[None, :]) < 1.0), axis=1
        )
        self.assertAlmostEqual(
            simulated.mean(), result.expected_flare_fluence[0], delta=0.02
        )
        self.assertAlmostEqual(
            simulated.var(), result.conditional_variance[0], delta=0.02
        )

    def test_fully_contained_window_has_exactly_zero_sampling_variance(self):
        # For this source CDF, adding the rounded probabilities of the eight
        # separate arrival bins produces 1 - one floating-point ulp. The
        # entire source still lies in the arrival window for every emission.
        source_fluence = np.random.default_rng(25).uniform(size=8)
        result = convolve_scored_impulses(
            [0.0], [1.0], np.arange(9), source_fluence, np.arange(9)
        )
        self.assertAlmostEqual(result.expected_window_fluence, 1.0)
        self.assertEqual(result.conditional_window_variance, 0.0)


if __name__ == "__main__":
    unittest.main()
