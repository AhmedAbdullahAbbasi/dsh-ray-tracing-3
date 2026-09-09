"""Physics checks for the one-event Monte Carlo transport checkpoint."""

import math
import unittest

import jax
from jax import random
import numpy as np

from utils.clouds import build_angular_distance_cloud
from utils.first_interaction import (
    ABSORBED,
    NO_INTERACTION,
    SCATTERED,
    simulate_first_interactions,
)
from utils.source import (
    build_variable_powerlaw_source,
    fred_outburst_flux,
    sample_variable_powerlaw_source,
)


class TestFirstInteractionMonteCarlo(unittest.TestCase):
    """Compare sampled outcomes with exact finite-slab probabilities."""

    @classmethod
    def setUpClass(cls):
        cls.n_packets = 60_000
        cls.sigma_scattering = 3.0e-22
        cls.sigma_absorption = 2.0e-22
        cls.sigma_total = cls.sigma_scattering + cls.sigma_absorption

        # Four 0.5-kpc radial cells, each carrying 5e20 H cm^-2.
        # Every sightline therefore has NH=2e21 cm^-2 and tau=1 exactly.
        cls.cloud = build_angular_distance_cloud(
            np.full((4, 3, 3), 5.0e20),
            x_centers_arcsec=[-2.0, 0.0, 2.0],
            y_centers_arcsec=[-2.0, 0.0, 2.0],
            z_centers_kpc=[0.25, 0.75, 1.25, 1.75],
            source_distance_kpc=10.0,
        )

        day = 86_400.0
        time_edges = np.linspace(0.0, 30.0 * day, 61)
        photon_flux = fred_outburst_flux(
            time_edges,
            baseline_flux=0.03,
            peak_excess_flux=1.2,
            peak_time_s=2.0 * day,
            rise_time_s=0.4 * day,
            decay_time_s=8.0 * day,
        )
        source = build_variable_powerlaw_source(
            time_edges,
            photon_flux,
            energy_min_kev=2.0,
            energy_max_kev=10.0,
            photon_index=2.2,
        )
        cls.packets = sample_variable_powerlaw_source(
            random.PRNGKey(101), source, cls.n_packets
        )

        cls.origin_pc = np.array([10_000.0, 0.0, 0.0], dtype=np.float32)
        cls.direction = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        compiled = jax.jit(simulate_first_interactions)
        cls.result = compiled(
            random.PRNGKey(202),
            cls.packets,
            cls.cloud,
            cls.origin_pc,
            cls.direction,
            10_000.0,
            cls.sigma_scattering,
            cls.sigma_absorption,
        )
        cls.result.status.block_until_ready()

    def test_outcome_fractions_match_finite_slab_probabilities(self):
        status = np.asarray(self.result.status)
        tau = self.sigma_total * 2.0e21
        interaction_probability = 1.0 - math.exp(-tau)
        expected = {
            NO_INTERACTION: math.exp(-tau),
            SCATTERED: interaction_probability
            * self.sigma_scattering / self.sigma_total,
            ABSORBED: interaction_probability
            * self.sigma_absorption / self.sigma_total,
        }
        for outcome, expected_fraction in expected.items():
            measured_fraction = np.mean(status == outcome)
            self.assertAlmostEqual(measured_fraction, expected_fraction, delta=0.008)

    def test_weighted_outcomes_close_to_input_fluence(self):
        status = np.asarray(self.result.status)
        weights = np.asarray(self.result.weight_observer_fluence)
        total_weight = weights.sum(dtype=np.float64)
        weighted_fractions = np.array(
            [weights[status == value].sum(dtype=np.float64) / total_weight
             for value in (NO_INTERACTION, SCATTERED, ABSORBED)]
        )
        self.assertAlmostEqual(weighted_fractions.sum(), 1.0, places=7)
        self.assertTrue(np.all(weighted_fractions > 0.0))

    def test_sampled_free_path_has_truncated_exponential_mean(self):
        interacted = np.asarray(self.result.status) != NO_INTERACTION
        target_tau = np.asarray(self.result.target_tau)[interacted]
        slab_tau = 1.0
        expected_mean = 1.0 - slab_tau / math.expm1(slab_tau)
        self.assertAlmostEqual(target_tau.mean(), expected_mean, delta=0.006)
        self.assertTrue(np.all((target_tau >= 0.0) & (target_tau < slab_tau)))

    def test_interaction_positions_invert_column_depth(self):
        interacted = np.asarray(self.result.status) != NO_INTERACTION
        target_tau = np.asarray(self.result.target_tau)[interacted]
        distance_pc = np.asarray(self.result.distance_pc)[interacted]

        # The source ray first enters the cloud at 8 kpc from the source.
        # Tau grows linearly from 0 to 1 over the following 2 kpc.
        expected_distance_pc = 8_000.0 + 2_000.0 * target_tau
        np.testing.assert_allclose(distance_pc, expected_distance_pc, atol=0.02)
        self.assertGreaterEqual(distance_pc.min(), 8_000.0)
        self.assertLess(distance_pc.max(), 10_000.0)

    def test_temporary_isotropic_scattering_is_normalized_and_uniform(self):
        scattered = np.asarray(self.result.status) == SCATTERED
        before = np.asarray(self.result.direction_before)[scattered]
        after = np.asarray(self.result.direction_after)[scattered]
        mu = np.sum(before * after, axis=1)

        np.testing.assert_allclose(np.linalg.norm(after, axis=1), 1.0, atol=2.0e-6)
        self.assertAlmostEqual(mu.mean(), 0.0, delta=0.012)
        self.assertAlmostEqual(np.mean(mu**2), 1.0 / 3.0, delta=0.008)

    def test_source_packet_metadata_survives_jitted_interaction(self):
        np.testing.assert_array_equal(self.result.energy_kev, self.packets.energy_kev)
        np.testing.assert_array_equal(
            self.result.emission_time_s, self.packets.emission_time_s
        )
        np.testing.assert_array_equal(
            self.result.weight_observer_fluence,
            self.packets.weight_observer_fluence,
        )
        np.testing.assert_array_equal(self.result.time_index, self.packets.time_index)
        np.testing.assert_array_equal(
            self.result.spectral_bin_index, self.packets.spectral_bin_index
        )

    def test_zero_opacity_reaches_endpoint_without_interaction(self):
        packets = jax.tree.map(lambda value: value[:32], self.packets)
        result = simulate_first_interactions(
            random.PRNGKey(303),
            packets,
            self.cloud,
            self.origin_pc,
            self.direction,
            10_000.0,
            0.0,
            0.0,
        )
        np.testing.assert_array_equal(result.status, NO_INTERACTION)
        np.testing.assert_allclose(result.distance_pc, 10_000.0)
        np.testing.assert_allclose(result.position_pc, np.zeros((32, 3)), atol=1.0e-6)

    def test_rejects_mismatched_ray_shapes(self):
        with self.assertRaisesRegex(ValueError, "origin_pc"):
            simulate_first_interactions(
                random.PRNGKey(404),
                self.packets,
                self.cloud,
                np.zeros((2, 3), dtype=np.float32),
                self.direction,
                10_000.0,
                self.sigma_scattering,
                self.sigma_absorption,
            )


if __name__ == "__main__":
    unittest.main()
