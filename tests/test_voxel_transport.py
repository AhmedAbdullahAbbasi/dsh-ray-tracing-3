"""Physics and API tests for the DSH voxel-transport kernel."""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from dsh.geometry.clouds import build_angular_distance_cloud
from dsh.physics.dust import (
    build_dust_physics_table,
    phase_cdf_from_differential_cross_section,
    remove_small_angle_dsh_geometry_factor,
)
from dsh.transport.kernel import (
    ABSORBED,
    DUST_SCATTERING,
    ESCAPED_OUTER_BOUNDARY,
    INVALID_ENERGY,
    INVALID_STATE,
    MAX_INTERACTIONS,
    NO_INTERACTION,
    PHOTOELECTRIC_ABSORPTION,
    REACHED_OBSERVER_PLANE,
    _direction_from_axis_theta_phi,
    _line_of_sight_excess_factor,
    transport_photon_batch,
    transport_photon_voxels,
)


def isotropic_dust_table(sigma_sca, sigma_abs):
    theta = np.linspace(0.0, np.pi, 2049)
    differential = np.ones((2, theta.size))
    _, cdf = phase_cdf_from_differential_cross_section(theta, differential)
    return build_dust_physics_table(
        energy_kev=[1.0, 10.0],
        scattering_cross_section_cm2_per_h=[sigma_sca, sigma_sca],
        absorption_cross_section_cm2_per_h=[sigma_abs, sigma_abs],
        scattering_angle_rad=theta,
        scattering_angle_cdf=cdf,
    )


class TestVoxelTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Four radial cells carrying a total NH=2e21 cm^-2 on every sightline.
        cls.cloud = build_angular_distance_cloud(
            np.full((4, 3, 3), 5.0e20),
            x_centers_arcsec=[-2.0, 0.0, 2.0],
            y_centers_arcsec=[-2.0, 0.0, 2.0],
            z_centers_kpc=[0.25, 0.75, 1.25, 1.75],
            source_distance_kpc=10.0,
        )
        cls.source_position = jnp.array([10_000.0, 0.0, 0.0])
        cls.momentum = jnp.array([4.0, -4.0, 0.0, 0.0])

    def test_zero_opacity_reaches_observer_plane_without_interaction(self):
        result = transport_photon_voxels(
            random.PRNGKey(0),
            self.source_position,
            self.momentum,
            self.cloud,
            isotropic_dust_table(0.0, 0.0),
            max_interactions=4,
        )
        self.assertEqual(int(result.status), REACHED_OBSERVER_PLANE)
        self.assertEqual(int(result.n_interactions), 0)
        np.testing.assert_allclose(result.position_pc, np.zeros(3), atol=1.0e-3)
        self.assertAlmostEqual(float(result.path_length_pc), 10_000.0, places=3)
        self.assertEqual(float(result.excess_path_length_pc), 0.0)
        np.testing.assert_allclose(result.momentum_kev, self.momentum)
        np.testing.assert_array_equal(result.interactions.valid, False)
        np.testing.assert_array_equal(
            result.interactions.interaction_type, NO_INTERACTION
        )
        np.testing.assert_allclose(result.interactions.position_pc, 0.0)
        self.assertEqual(result.interactions.valid.shape, (4,))
        self.assertEqual(result.interactions.position_pc.shape, (4, 3))
        self.assertEqual(result.interactions.incoming_momentum_kev.shape, (4, 4))

    def test_arcsecond_scattering_angle_survives_float32(self):
        theta = jnp.asarray(1.0e-5, dtype=jnp.float32)
        direction = _direction_from_axis_theta_phi(
            jnp.array([-1.0, 0.0, 0.0], dtype=jnp.float32),
            theta,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        self.assertAlmostEqual(
            float(jnp.linalg.norm(direction[1:])), float(theta), delta=1.0e-11
        )
        self.assertAlmostEqual(
            float(_line_of_sight_excess_factor(direction)),
            0.5 * float(theta) ** 2,
            delta=1.0e-16,
        )

    def test_outward_photon_is_not_misclassified_as_reaching_observer(self):
        result = transport_photon_voxels(
            random.PRNGKey(6),
            self.source_position,
            jnp.array([4.0, 4.0, 0.0, 0.0]),
            self.cloud,
            isotropic_dust_table(0.0, 0.0),
            max_interactions=4,
        )
        self.assertEqual(int(result.status), ESCAPED_OUTER_BOUNDARY)
        self.assertAlmostEqual(float(result.path_length_pc), 0.0, places=6)

    def test_absorption_fraction_matches_finite_slab(self):
        n_packets = 50_000
        sigma_abs = 5.0e-22
        positions = jnp.broadcast_to(self.source_position, (n_packets, 3))
        momenta = jnp.broadcast_to(self.momentum, (n_packets, 4))
        compiled = jax.jit(
            transport_photon_batch, static_argnames=("max_interactions",)
        )
        result = compiled(
            random.PRNGKey(1),
            positions,
            momenta,
            self.cloud,
            isotropic_dust_table(0.0, sigma_abs),
            max_interactions=2,
        )
        measured = np.mean(np.asarray(result.status) == ABSORBED)
        expected = 1.0 - math.exp(-sigma_abs * 2.0e21)
        self.assertAlmostEqual(measured, expected, delta=0.008)
        absorbed = np.asarray(result.status) == ABSORBED
        np.testing.assert_allclose(np.asarray(result.momentum_kev)[absorbed], 0.0)
        np.testing.assert_allclose(
            np.asarray(result.deposited_energy_kev)[absorbed], 4.0
        )
        records = result.interactions
        valid = np.asarray(records.valid)[absorbed]
        np.testing.assert_array_equal(valid.sum(axis=1), 1)
        np.testing.assert_array_equal(
            np.asarray(records.interaction_type)[absorbed, 0],
            PHOTOELECTRIC_ABSORPTION,
        )
        np.testing.assert_allclose(
            np.asarray(records.position_pc)[absorbed, 0],
            np.asarray(result.position_pc)[absorbed],
        )
        np.testing.assert_allclose(
            np.asarray(records.incoming_momentum_kev)[absorbed, 0],
            np.broadcast_to(np.asarray(self.momentum), (absorbed.sum(), 4)),
        )
        np.testing.assert_allclose(
            np.asarray(records.outgoing_momentum_kev)[absorbed, 0], 0.0
        )
        np.testing.assert_allclose(
            np.asarray(records.cumulative_path_length_pc)[absorbed, 0],
            np.asarray(result.path_length_pc)[absorbed],
        )
        np.testing.assert_array_equal(
            np.asarray(records.scattering_order)[absorbed, 0], 0
        )

    def test_scattering_is_elastic_and_preserves_null_four_momentum(self):
        # Use a deliberately broad angular field here.  The real DSH cutout is
        # only a few arcseconds wide, so an isotropically scattered test photon
        # normally leaves it before a second event; that is a geometry effect,
        # not a failure of the multiple-scattering loop.
        wide_cloud = build_angular_distance_cloud(
            np.full((4, 3, 3), 5.0e20),
            x_centers_arcsec=[-120_000.0, 0.0, 120_000.0],
            y_centers_arcsec=[-120_000.0, 0.0, 120_000.0],
            z_centers_kpc=[0.25, 0.75, 1.25, 1.75],
            source_distance_kpc=10.0,
        )
        n_packets = 4096
        positions = jnp.broadcast_to(self.source_position, (n_packets, 3))
        momenta = jnp.broadcast_to(self.momentum, (n_packets, 4))
        result = transport_photon_batch(
            random.PRNGKey(2),
            positions,
            momenta,
            wide_cloud,
            isotropic_dust_table(1.0e-20, 0.0),
            max_interactions=8,
        )
        p4 = np.asarray(result.momentum_kev)
        np.testing.assert_allclose(p4[:, 0], 4.0)
        np.testing.assert_allclose(
            np.linalg.norm(p4[:, 1:], axis=1), p4[:, 0], rtol=2.0e-5
        )
        self.assertGreater(np.max(np.asarray(result.n_scatter)), 1)

        records = result.interactions
        valid = np.asarray(records.valid)
        event_type = np.asarray(records.interaction_type)
        incoming = np.asarray(records.incoming_momentum_kev)
        outgoing = np.asarray(records.outgoing_momentum_kev)
        orders = np.asarray(records.scattering_order)
        np.testing.assert_array_equal(
            valid.sum(axis=1), np.asarray(result.n_interactions)
        )
        np.testing.assert_array_equal(event_type[valid], DUST_SCATTERING)
        np.testing.assert_allclose(incoming[valid, 0], 4.0)
        np.testing.assert_allclose(outgoing[valid, 0], 4.0)
        np.testing.assert_allclose(
            np.linalg.norm(incoming[valid, 1:], axis=1),
            incoming[valid, 0],
            rtol=2.0e-5,
        )
        np.testing.assert_allclose(
            np.linalg.norm(outgoing[valid, 1:], axis=1),
            outgoing[valid, 0],
            rtol=2.0e-5,
        )
        for photon_index in np.flatnonzero(valid.sum(axis=1) > 1)[:32]:
            count = int(valid[photon_index].sum())
            np.testing.assert_array_equal(
                orders[photon_index, :count], np.arange(1, count + 1)
            )
            self.assertTrue(
                np.all(
                    np.diff(
                        np.asarray(records.cumulative_path_length_pc)[
                            photon_index, :count
                        ]
                    )
                    > 0.0
                )
            )

    def test_safety_limit_is_reported_not_misclassified(self):
        # Very large scattering optical depth makes reaching a two-event
        # numerical guard likely.  Guarded photons must remain distinguishable.
        n_packets = 4096
        result = transport_photon_batch(
            random.PRNGKey(3),
            jnp.broadcast_to(self.source_position, (n_packets, 3)),
            jnp.broadcast_to(self.momentum, (n_packets, 4)),
            self.cloud,
            isotropic_dust_table(1.0e-19, 0.0),
            max_interactions=2,
        )
        self.assertGreater(np.mean(np.asarray(result.status) == MAX_INTERACTIONS), 0.0)

    def test_embedded_dsh_geometry_factor_helper(self):
        intrinsic = np.array([2.0, 5.0])
        x = 0.75
        weighted = intrinsic / (1.0 - x) ** 2
        np.testing.assert_allclose(
            remove_small_angle_dsh_geometry_factor(weighted, x), intrinsic
        )

    def test_invalid_energy_and_invalid_four_momentum_are_distinct(self):
        table = isotropic_dust_table(0.0, 0.0)
        invalid_energy = transport_photon_voxels(
            random.PRNGKey(4),
            self.source_position,
            jnp.array([20.0, -20.0, 0.0, 0.0]),
            self.cloud,
            table,
        )
        invalid_state = transport_photon_voxels(
            random.PRNGKey(5),
            self.source_position,
            jnp.array([4.0, -3.0, 0.0, 0.0]),
            self.cloud,
            table,
        )
        self.assertEqual(int(invalid_energy.status), INVALID_ENERGY)
        self.assertEqual(int(invalid_state.status), INVALID_STATE)


if __name__ == "__main__":
    unittest.main()
