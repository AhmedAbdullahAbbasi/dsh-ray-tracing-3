"""Validation tests for physical source-flux sampling."""

import unittest

import jax
import numpy as np
from jax import random

from dsh.sources.models import (
    build_decay_observation_window,
    build_post_peak_exponential_band_source,
    build_powerlaw_band_source,
    build_tabulated_band_source,
    build_variable_powerlaw_source,
    fred_outburst_flux,
    sample_tabulated_band_source,
    sample_variable_powerlaw_source,
)


class TestTabulatedBandSource(unittest.TestCase):
    def test_continuous_powerlaw_energies_and_band_fluence(self):
        source = build_powerlaw_band_source(
            [0.0, 3600.0], [0.038], [2.0, 4.0, 6.0, 10.0], 1.7
        )
        energies = np.asarray(
            sample_tabulated_band_source(random.PRNGKey(23), source, 100_000).energy_kev
        )
        expected_fraction_below_four = (2.0**-0.7 - 4.0**-0.7) / (
            2.0**-0.7 - 10.0**-0.7
        )
        self.assertAlmostEqual(float(source.total_fluence), 136.8, places=3)
        self.assertAlmostEqual(
            float(np.mean(energies < 4.0)), expected_fraction_below_four, delta=0.01
        )
        self.assertGreaterEqual(float(energies.min()), 2.0)
        self.assertLessEqual(float(energies.max()), 10.0)
        self.assertGreater(np.unique(energies).size, 50_000)
        band_flux = np.asarray(source.band_flux[0])
        self.assertAlmostEqual(float(band_flux.sum()), 0.038, places=7)
        self.assertTrue(np.all(band_flux > 0))

    def test_powerlaw_gamma_one_and_invalid_bands(self):
        source = build_powerlaw_band_source([0, 1], [1], [2, 4, 10], 1.0)
        self.assertAlmostEqual(
            float(source.band_flux[0, 0]), np.log(2) / np.log(5), places=6
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            build_powerlaw_band_source([0, 1], [1], [2, 4, 4], 1.7)

    def test_fluence_calculation_and_cdf(self):
        source = build_tabulated_band_source(
            time_edges_s=np.array([0.0, 2.0, 5.0], dtype=np.float32),
            band_flux=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            effective_energy_kev=np.array([3.3, 4.9], dtype=np.float32),
        )

        expected_cells = np.array([[2.0, 4.0], [9.0, 12.0]])
        np.testing.assert_allclose(np.asarray(source.cell_fluence), expected_cells)
        self.assertAlmostEqual(float(source.total_fluence), 27.0)
        self.assertAlmostEqual(float(source.flat_cdf[-1]), 1.0)

    def test_packet_weight_closure_and_reproducibility(self):
        source = build_tabulated_band_source(
            [0.0, 10.0],
            [[1.0, 3.0]],
            [3.3, 4.9],
        )
        sampler = jax.jit(sample_tabulated_band_source, static_argnames=("n_packets",))

        packets_a = sampler(random.PRNGKey(17), source, n_packets=20_000)
        packets_b = sampler(random.PRNGKey(17), source, n_packets=20_000)

        np.testing.assert_array_equal(
            np.asarray(packets_a.spectral_bin_index),
            np.asarray(packets_b.spectral_bin_index),
        )
        np.testing.assert_array_equal(
            np.asarray(packets_a.emission_time_s),
            np.asarray(packets_b.emission_time_s),
        )
        self.assertAlmostEqual(
            float(np.asarray(packets_a.weight_observer_fluence).sum()),
            float(source.total_fluence),
            places=4,
        )

    def test_sampled_band_and_time_distributions(self):
        source = build_tabulated_band_source(
            [0.0, 1.0, 2.0],
            [[1.0, 3.0], [1.0, 3.0]],
            [3.3, 4.9],
        )
        packets = sample_tabulated_band_source(
            random.PRNGKey(9), source, n_packets=100_000
        )

        band_fraction = (
            np.bincount(np.asarray(packets.spectral_bin_index), minlength=2) / 100_000
        )
        np.testing.assert_allclose(band_fraction, [0.25, 0.75], atol=0.01)
        self.assertAlmostEqual(
            float(np.asarray(packets.emission_time_s).mean()), 1.0, places=2
        )
        self.assertGreaterEqual(float(np.asarray(packets.emission_time_s).min()), 0.0)
        self.assertLess(float(np.asarray(packets.emission_time_s).max()), 2.0)

        sampled_energies = np.unique(np.asarray(packets.energy_kev))
        np.testing.assert_allclose(sampled_energies, [3.3, 4.9], rtol=1e-6)

    def test_rejects_ambiguous_or_unphysical_input(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            build_tabulated_band_source([0.0, 2.0, 1.0], [[1.0], [1.0]], [3.3])
        with self.assertRaisesRegex(ValueError, "gap policy"):
            build_tabulated_band_source([0.0, 1.0], [[np.nan]], [3.3])
        with self.assertRaisesRegex(ValueError, "negative"):
            build_tabulated_band_source([0.0, 1.0], [[-1.0]], [3.3])
        with self.assertRaisesRegex(ValueError, "zero total fluence"):
            build_tabulated_band_source([0.0, 1.0], [[0.0]], [3.3])


class TestPostPeakExponentialBandSource(unittest.TestCase):
    def test_exact_fluence_is_independent_of_time_binning(self):
        peak = np.asarray([2.0, 1.0, 0.5])
        baseline = np.asarray([0.2, 0.1, 0.05])
        decay_time = 25.0
        duration = 100.0
        coarse = build_post_peak_exponential_band_source(
            [0.0, 20.0, 50.0, duration],
            [3.3, 4.9, 6.9],
            peak,
            decay_time,
            baseline_band_flux=baseline,
        )
        fine = build_post_peak_exponential_band_source(
            np.linspace(0.0, duration, 1001),
            [3.3, 4.9, 6.9],
            peak,
            decay_time,
            baseline_band_flux=baseline,
        )
        expected_by_band = baseline * duration + (peak - baseline) * decay_time * (
            1.0 - np.exp(-duration / decay_time)
        )
        np.testing.assert_allclose(
            np.asarray(coarse.cell_fluence).sum(axis=0),
            expected_by_band,
            rtol=2.0e-6,
        )
        np.testing.assert_allclose(
            coarse.total_fluence, fine.total_fluence, rtol=2.0e-6
        )
        self.assertTrue(np.all(np.diff(np.asarray(fine.band_flux), axis=0) < 0.0))

    def test_decay_can_begin_after_the_peak(self):
        source = build_post_peak_exponential_band_source(
            [40.0, 50.0, 60.0],
            [3.3],
            [2.0],
            decay_time_s=20.0,
            baseline_band_flux=[0.0],
            peak_time_s=10.0,
        )
        flux_at_interval_start = 2.0 * np.exp(-(40.0 - 10.0) / 20.0)
        self.assertLess(float(source.band_flux[0, 0]), flux_at_interval_start)
        self.assertGreater(float(source.band_flux[0, 0]), 0.0)

    def test_rejects_non_decay_inputs(self):
        with self.assertRaisesRegex(ValueError, "cannot precede"):
            build_post_peak_exponential_band_source([-1.0, 1.0], [3.3], [1.0], 10.0)
        with self.assertRaisesRegex(ValueError, "below baseline"):
            build_post_peak_exponential_band_source(
                [0.0, 1.0],
                [3.3],
                [0.5],
                10.0,
                baseline_band_flux=[1.0],
            )
        with self.assertRaisesRegex(ValueError, "must match"):
            build_post_peak_exponential_band_source([0.0, 1.0], [3.3, 4.9], [1.0], 10.0)


class TestVariablePowerLawSource(unittest.TestCase):
    def test_time_distribution_and_fluence_closure(self):
        source = build_variable_powerlaw_source(
            time_edges_s=[0.0, 1.0, 2.0],
            photon_flux=[1.0, 3.0],
            energy_min_kev=1.0,
            energy_max_kev=10.0,
            photon_index=2.0,
        )
        sampler = jax.jit(
            sample_variable_powerlaw_source, static_argnames=("n_packets",)
        )
        packets = sampler(random.PRNGKey(31), source, n_packets=100_000)

        time_fraction = (
            np.bincount(np.asarray(packets.time_index), minlength=2) / 100_000
        )
        np.testing.assert_allclose(time_fraction, [0.25, 0.75], atol=0.01)
        self.assertAlmostEqual(float(source.total_fluence), 4.0)
        self.assertAlmostEqual(
            float(np.asarray(packets.weight_observer_fluence).sum()), 4.0, places=4
        )
        self.assertTrue(np.all(np.asarray(packets.spectral_bin_index) == -1))

    def test_powerlaw_energy_distribution(self):
        source = build_variable_powerlaw_source(
            time_edges_s=[0.0, 1.0],
            photon_flux=[1.0],
            energy_min_kev=1.0,
            energy_max_kev=10.0,
            photon_index=2.0,
        )
        packets = sample_variable_powerlaw_source(
            random.PRNGKey(12), source, n_packets=100_000
        )
        energies = np.asarray(packets.energy_kev)

        # For Gamma=2 on [1, 10], P(E < 2)=(1 - 1/2)/(1 - 1/10)=5/9.
        self.assertAlmostEqual(float(np.mean(energies < 2.0)), 5.0 / 9.0, places=2)
        self.assertGreaterEqual(float(energies.min()), 1.0)
        self.assertLessEqual(float(energies.max()), 10.0)

    def test_gamma_one_logarithmic_limit(self):
        source = build_variable_powerlaw_source(
            time_edges_s=[0.0, 1.0],
            photon_flux=[1.0],
            energy_min_kev=1.0,
            energy_max_kev=100.0,
            photon_index=1.0,
        )
        packets = sample_variable_powerlaw_source(
            random.PRNGKey(4), source, n_packets=100_000
        )
        # Gamma=1 is uniform in log(E), so half the photons lie below the
        # geometric midpoint sqrt(1*100)=10 keV.
        fraction_below_midpoint = np.mean(np.asarray(packets.energy_kev) < 10.0)
        self.assertAlmostEqual(float(fraction_below_midpoint), 0.5, places=2)

    def test_fred_outburst_is_asymmetric(self):
        edges = np.linspace(0.0, 100.0, 101)
        flux = fred_outburst_flux(
            edges,
            baseline_flux=1.0,
            peak_excess_flux=4.0,
            peak_time_s=20.0,
            rise_time_s=2.0,
            decay_time_s=20.0,
        )
        self.assertEqual(flux.shape, (100,))
        self.assertTrue(np.all(flux >= 1.0))
        # The peak is exactly on a bin boundary. Because the decay is much
        # slower, the first post-peak interval has the largest bin average.
        self.assertEqual(int(np.argmax(flux)), 20)

        # Equal offsets from the peak are brighter on the slower decay than
        # on the fast rise.
        self.assertGreater(float(flux[29]), float(flux[9]))

    def test_fred_exact_fluence_is_stable_to_time_binning(self):
        parameters = {
            "baseline_flux": 0.1,
            "peak_excess_flux": 2.0,
            "peak_time_s": 20.0,
            "rise_time_s": 3.0,
            "decay_time_s": 25.0,
        }
        coarse_edges = np.linspace(0.0, 100.0, 21)
        fine_edges = np.linspace(0.0, 100.0, 2001)
        coarse_flux = fred_outburst_flux(coarse_edges, **parameters)
        fine_flux = fred_outburst_flux(fine_edges, **parameters)
        coarse_fluence = np.sum(coarse_flux * np.diff(coarse_edges))
        fine_fluence = np.sum(fine_flux * np.diff(fine_edges))
        self.assertAlmostEqual(float(coarse_fluence), float(fine_fluence), places=10)

    def test_observation_window_must_be_on_decay(self):
        window = build_decay_observation_window(
            start_s=40.0,
            stop_s=50.0,
            outburst_peak_s=20.0,
        )
        self.assertEqual(float(window.start_s), 40.0)
        self.assertEqual(float(window.stop_s), 50.0)

        with self.assertRaisesRegex(ValueError, "strictly after"):
            build_decay_observation_window(20.0, 30.0, outburst_peak_s=20.0)
        with self.assertRaisesRegex(ValueError, "later than"):
            build_decay_observation_window(50.0, 40.0, outburst_peak_s=20.0)

    def test_rejects_invalid_powerlaw_source(self):
        with self.assertRaisesRegex(ValueError, "one value per time interval"):
            build_variable_powerlaw_source([0.0, 1.0, 2.0], [1.0], 1.0, 10.0, 2.0)
        with self.assertRaisesRegex(ValueError, "energy bounds"):
            build_variable_powerlaw_source([0.0, 1.0], [1.0], 10.0, 1.0, 2.0)
        with self.assertRaisesRegex(ValueError, "gap policy"):
            build_variable_powerlaw_source([0.0, 1.0], [np.nan], 1.0, 10.0, 2.0)


if __name__ == "__main__":
    unittest.main()
