"""Analytic and statistical checkpoints for the DSH validation ladder."""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from dsh.geometry.clouds import build_angular_distance_cloud
from dsh.geometry.coordinates import ARCSEC_TO_RAD, PC_PER_KPC
from dsh.observer.binning import bin_observer_events, build_observer_bin_geometry
from dsh.observer.scoring import score_peeloff_events
from dsh.physics.newdust import (
    build_dust_physics_from_newdust,
    load_newdust_scattering_table,
)
from dsh.sources.launch import LaunchedSourcePackets
from dsh.transport.kernel import (
    DUST_SCATTERING,
    MAX_INTERACTIONS,
    PhotonInteractionRecord,
    PhotonTransportResult,
    transport_photon_batch,
)
from dsh.validation import (
    azimuthal_harmonic_amplitudes,
    causal_discrete_convolution,
    exact_single_scatter_delay_s,
    exact_single_scatter_excess_path_pc,
    finite_screen_ring_bounds_arcsec,
    log_log_power_law_slope,
    phase_containment_angle_rad,
    small_angle_ring_radius_arcsec,
    small_angle_single_scatter_delay_s,
)


def _empty_cloud(source_distance_kpc=10.0):
    return build_angular_distance_cloud(
        np.zeros((2, 2, 2), dtype=np.float64),
        x_centers_arcsec=[-2_000.0, 2_000.0],
        y_centers_arcsec=[-2_000.0, 2_000.0],
        z_centers_kpc=[2.0, 6.0],
        source_distance_kpc=source_distance_kpc,
    )


def _synthetic_single_scatter_ring(fractions, theta_arcsec, azimuth_rad, energy=3.3):
    """Build exact one-scatter histories without using a transport equation."""

    fractions, theta, azimuth = np.broadcast_arrays(
        np.asarray(fractions, dtype=np.float64),
        np.asarray(theta_arcsec, dtype=np.float64) * ARCSEC_TO_RAD,
        np.asarray(azimuth_rad, dtype=np.float64),
    )
    fractions = fractions.reshape(-1)
    theta = theta.reshape(-1)
    azimuth = azimuth.reshape(-1)
    n_packets = fractions.size
    source_distance_pc = 10.0 * PC_PER_KPC
    source = np.array([source_distance_pc, 0.0, 0.0])
    radial_direction = np.column_stack(
        (
            np.cos(theta),
            np.sin(theta) * np.cos(azimuth),
            np.sin(theta) * np.sin(azimuth),
        )
    )
    positions = fractions[:, None] * source_distance_pc * radial_direction
    displacement = positions - source
    path_length = np.linalg.norm(displacement, axis=1)
    incoming_direction = displacement / path_length[:, None]
    incoming = np.column_stack(
        (np.full(n_packets, energy), energy * incoming_direction)
    )
    cumulative_excess = path_length * (1.0 + incoming_direction[:, 0])

    records = PhotonInteractionRecord(
        valid=jnp.ones((n_packets, 1), dtype=bool),
        interaction_type=jnp.full((n_packets, 1), DUST_SCATTERING, dtype=jnp.int32),
        position_pc=jnp.asarray(positions[:, None, :], dtype=jnp.float32),
        incoming_momentum_kev=jnp.asarray(incoming[:, None, :], dtype=jnp.float32),
        outgoing_momentum_kev=jnp.asarray(incoming[:, None, :], dtype=jnp.float32),
        cumulative_path_length_pc=jnp.asarray(path_length[:, None], dtype=jnp.float32),
        cumulative_excess_path_length_pc=jnp.asarray(
            cumulative_excess[:, None], dtype=jnp.float32
        ),
        scattering_order=jnp.ones((n_packets, 1), dtype=jnp.int32),
    )
    transported = PhotonTransportResult(
        position_pc=jnp.asarray(positions, dtype=jnp.float32),
        momentum_kev=jnp.asarray(incoming, dtype=jnp.float32),
        path_length_pc=jnp.asarray(path_length, dtype=jnp.float32),
        excess_path_length_pc=jnp.asarray(cumulative_excess, dtype=jnp.float32),
        deposited_energy_kev=jnp.zeros(n_packets, dtype=jnp.float32),
        n_interactions=jnp.ones(n_packets, dtype=jnp.int32),
        n_scatter=jnp.ones(n_packets, dtype=jnp.int32),
        status=jnp.full(n_packets, MAX_INTERACTIONS, dtype=jnp.int32),
        interactions=records,
    )
    launched = LaunchedSourcePackets(
        position_pc=jnp.broadcast_to(
            jnp.asarray(source, dtype=jnp.float32), (n_packets, 3)
        ),
        momentum_kev=jnp.asarray(incoming, dtype=jnp.float32),
        launch_pdf_per_sr=jnp.ones(n_packets, dtype=jnp.float32),
        isotropic_importance=jnp.full(
            (n_packets,), 1.0 / (4.0 * np.pi), dtype=jnp.float32
        ),
        emission_time_s=jnp.zeros(n_packets, dtype=jnp.float32),
        weight_observer_fluence=jnp.ones(n_packets, dtype=jnp.float32),
        time_index=jnp.zeros(n_packets, dtype=jnp.int32),
        spectral_bin_index=jnp.zeros(n_packets, dtype=jnp.int32),
    )
    return launched, transported


class TestGeometryAndTimeDelay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scattering = load_newdust_scattering_table()
        cls.physics = build_dust_physics_from_newdust(
            scattering, np.zeros_like(scattering.energy_kev)
        )
        cls.cloud = _empty_cloud()

    def test_delta_flare_delay_radius_relation_near_mid_and_far(self):
        fractions = np.array([0.1, 0.5, 0.9])
        theta_arcsec = np.full(3, 120.0)
        launched, transported = _synthetic_single_scatter_ring(
            fractions, theta_arcsec, np.zeros(3)
        )
        events = jax.jit(score_peeloff_events)(
            launched, transported, self.cloud, self.physics
        )

        exact_excess = exact_single_scatter_excess_path_pc(
            10.0, fractions, theta_arcsec * ARCSEC_TO_RAD
        )
        exact_delay = exact_single_scatter_delay_s(
            10.0, fractions, theta_arcsec * ARCSEC_TO_RAD
        )
        np.testing.assert_allclose(
            np.asarray(events.excess_path_length_pc)[:, 0],
            exact_excess,
            rtol=4.0e-4,
            atol=2.0e-8,
        )
        np.testing.assert_allclose(
            np.asarray(events.arrival_time_s)[:, 0], exact_delay, rtol=4.0e-4
        )

        recovered_theta = small_angle_ring_radius_arcsec(
            np.asarray(events.arrival_time_s)[:, 0], 10.0, fractions
        )
        np.testing.assert_allclose(recovered_theta, theta_arcsec, rtol=3.0e-5)
        approximate_delay = small_angle_single_scatter_delay_s(
            10.0, fractions, theta_arcsec * ARCSEC_TO_RAD
        )
        np.testing.assert_allclose(approximate_delay, exact_delay, rtol=5.0e-5)

    def test_screen_thickness_broadens_without_shifting_the_thin_limit(self):
        delay_s = 10.0 * 86_400.0
        center = float(small_angle_ring_radius_arcsec(delay_s, 10.0, 0.5))
        broad = finite_screen_ring_bounds_arcsec(delay_s, 10.0, (0.45, 0.55))
        narrow = finite_screen_ring_bounds_arcsec(delay_s, 10.0, (0.499, 0.501))
        self.assertLess(broad[0], center)
        self.assertGreater(broad[1], center)
        self.assertGreater(broad[1] - broad[0], narrow[1] - narrow[0])
        np.testing.assert_allclose(narrow.mean(), center, rtol=3.0e-6)


class TestScatteringPhysicsScaling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scattering = load_newdust_scattering_table()

    def test_mrn_phase_function_is_forward_peaked(self):
        angles_arcsec = np.array([1.0, 10.0, 100.0, 1_000.0])
        angles_rad = angles_arcsec * ARCSEC_TO_RAD
        rows = []
        for differential in self.scattering.differential_cross_section_cm2_per_sr_per_h:
            rows.append(
                np.interp(
                    np.log(angles_rad),
                    np.log(self.scattering.scattering_angle_rad[1:]),
                    np.log(np.maximum(differential[1:], np.finfo(float).tiny)),
                )
            )
        sampled = np.exp(np.asarray(rows))
        self.assertTrue(np.all(np.diff(sampled, axis=1) < 0.0))
        self.assertTrue(np.all(sampled[:, 0] > 1_000.0 * sampled[:, -1]))

    def test_energy_scaling_of_width_and_total_cross_section(self):
        median_angle = phase_containment_angle_rad(
            self.scattering.scattering_angle_rad,
            self.scattering.scattering_angle_cdf,
            0.5,
        )
        angle_slope = log_log_power_law_slope(self.scattering.energy_kev, median_angle)
        cross_section_slope = log_log_power_law_slope(
            self.scattering.energy_kev,
            self.scattering.scattering_cross_section_cm2_per_h,
        )
        self.assertAlmostEqual(angle_slope, -1.0, delta=2.0e-4)
        self.assertAlmostEqual(cross_section_slope, -2.0, delta=2.0e-4)
        np.testing.assert_allclose(
            self.scattering.energy_kev * median_angle,
            np.mean(self.scattering.energy_kev * median_angle),
            rtol=3.0e-6,
        )


class TestConservationAndSymmetry(unittest.TestCase):
    def test_low_tau_scattered_fluence_matches_finite_slab_probability(self):
        scattering = load_newdust_scattering_table()
        physics = build_dust_physics_from_newdust(
            scattering, np.zeros_like(scattering.energy_kev)
        )
        target_tau = 0.03
        column = target_tau / scattering.scattering_cross_section_cm2_per_h[0]
        cloud = build_angular_distance_cloud(
            np.full((2, 3, 3), 0.5 * column),
            x_centers_arcsec=[-10.0, 0.0, 10.0],
            y_centers_arcsec=[-10.0, 0.0, 10.0],
            z_centers_kpc=[2.0, 4.0],
            source_distance_kpc=10.0,
        )
        n_packets = 50_000
        positions = jnp.broadcast_to(
            jnp.array([10_000.0, 0.0, 0.0], dtype=jnp.float32), (n_packets, 3)
        )
        momenta = jnp.broadcast_to(
            jnp.array([3.3, -3.3, 0.0, 0.0], dtype=jnp.float32), (n_packets, 4)
        )
        result = jax.jit(transport_photon_batch, static_argnames=("max_interactions",))(
            random.PRNGKey(804),
            positions,
            momenta,
            cloud,
            physics,
            max_interactions=1,
        )
        measured = float(np.mean(np.asarray(result.n_scatter) == 1))
        expected = 1.0 - math.exp(-target_tau)
        standard_error = math.sqrt(expected * (1.0 - expected) / n_packets)
        self.assertLess(abs(measured - expected), 5.0 * standard_error)

    def test_uniform_screen_scoring_is_azimuthally_symmetric(self):
        scattering = load_newdust_scattering_table()
        physics = build_dust_physics_from_newdust(
            scattering, np.zeros_like(scattering.energy_kev)
        )
        azimuth = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        launched, transported = _synthetic_single_scatter_ring(
            np.full(azimuth.size, 0.5),
            np.full(azimuth.size, 180.0),
            azimuth,
        )
        events = score_peeloff_events(launched, transported, _empty_cloud(), physics)
        weights = np.asarray(events.weight_observer_fluence)[:, 0]
        delays = np.asarray(events.arrival_time_s)[:, 0]
        np.testing.assert_allclose(weights, weights.mean(), rtol=8.0e-5)
        np.testing.assert_allclose(delays, delays.mean(), rtol=8.0e-5)
        harmonics = azimuthal_harmonic_amplitudes(azimuth, weights, max_order=4)
        np.testing.assert_allclose(harmonics, 0.0, atol=2.0e-5)

    def test_binned_delta_response_convolves_with_source_light_curve(self):
        scattering = load_newdust_scattering_table()
        physics = build_dust_physics_from_newdust(
            scattering, np.zeros_like(scattering.energy_kev)
        )
        launched, transported = _synthetic_single_scatter_ring(
            np.full(3, 0.5), np.array([100.0, 200.0, 300.0]), np.zeros(3)
        )
        source = np.ones(3)  # A three-bin top-hat flare, with fixed trajectories.
        bin_seconds = 4.0 * 86_400.0
        geometry = build_observer_bin_geometry(
            sky_x_edges_arcsec=[-500.0, 500.0],
            sky_y_edges_arcsec=[-500.0, 500.0],
            energy_edges_kev=[3.0, 4.0],
            arrival_time_edges_s=np.arange(7) * bin_seconds,
        )

        def score_and_bin(packets):
            events = score_peeloff_events(packets, transported, _empty_cloud(), physics)
            self.assertEqual(int(np.asarray(events.valid).sum()), 3)
            products = bin_observer_events(events, geometry)
            self.assertEqual(int(np.asarray(products.unbinned_event_count)), 0)
            return np.asarray(products.total_fluence)[:, 0, 0, 0]

        impulse = score_and_bin(launched)
        actual = np.zeros_like(impulse)
        for index, fluence in enumerate(source):
            flare_packets = launched._replace(
                emission_time_s=jnp.full(3, index * bin_seconds, dtype=jnp.float32),
                weight_observer_fluence=jnp.full(3, fluence, dtype=jnp.float32),
            )
            actual += score_and_bin(flare_packets)
        expected = causal_discrete_convolution(source, impulse[:4])
        np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=1.0e-13)


if __name__ == "__main__":
    unittest.main()
