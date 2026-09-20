"""Tests for physical source launch geometry and importance weights."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from utils.clouds import build_angular_distance_cloud
from utils.coordinates import sky_position_pc
from utils.dust_physics import build_dust_physics_table
from utils.source import SourcePackets
from utils.source_launch import (
    build_cloud_launch_geometry,
    build_rectangular_launch_geometry,
    sample_source_launches,
)
from utils.voxel_transport import REACHED_OBSERVER_PLANE, transport_photon_batch


def _packets(n_packets):
    return SourcePackets(
        energy_kev=jnp.where(
            jnp.arange(n_packets) % 2 == 0,
            jnp.asarray(3.3, dtype=jnp.float32),
            jnp.asarray(4.9, dtype=jnp.float32),
        ),
        emission_time_s=jnp.arange(n_packets, dtype=jnp.float32) * 2.0,
        weight_observer_fluence=jnp.full(
            (n_packets,), 7.5 / n_packets, dtype=jnp.float32
        ),
        time_index=jnp.arange(n_packets, dtype=jnp.int32) % 7,
        spectral_bin_index=jnp.arange(n_packets, dtype=jnp.int32) % 2,
    )


class TestPhysicalSourceLaunch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cloud = build_angular_distance_cloud(
            np.zeros((3, 3, 3), dtype=np.float64),
            x_centers_arcsec=[-80.0, 20.0, 120.0],
            y_centers_arcsec=[-100.0, 0.0, 100.0],
            z_centers_kpc=[2.0, 4.0, 6.0],
            source_distance_kpc=10.0,
        )
        cls.geometry = build_cloud_launch_geometry(cls.cloud)

    def test_cloud_launch_cone_contains_every_frustum_corner(self):
        source_position = np.asarray(self.geometry.source_position_pc)
        x_bounds = np.asarray(self.geometry.slope_x_bounds)
        y_bounds = np.asarray(self.geometry.slope_y_bounds)

        for distance in np.asarray(self.cloud.z_edges_kpc)[[0, -1]]:
            for sky_x in np.asarray(self.cloud.x_edges_arcsec)[[0, -1]]:
                for sky_y in np.asarray(self.cloud.y_edges_arcsec)[[0, -1]]:
                    point = np.asarray(sky_position_pc(distance, sky_x, sky_y))
                    direction = point - source_position
                    direction /= np.linalg.norm(direction)
                    slope_x = direction[1] / -direction[0]
                    slope_y = direction[2] / -direction[0]
                    self.assertGreaterEqual(slope_x, x_bounds[0])
                    self.assertLessEqual(slope_x, x_bounds[1])
                    self.assertGreaterEqual(slope_y, y_bounds[0])
                    self.assertLessEqual(slope_y, y_bounds[1])

    def test_jitted_launch_has_null_momenta_and_preserves_metadata(self):
        packets = _packets(4096)
        launch_jit = jax.jit(sample_source_launches)
        result = launch_jit(random.PRNGKey(17), packets, self.geometry)

        momentum = np.asarray(result.momentum_kev)
        direction = momentum[:, 1:] / momentum[:, :1]
        slope_x = direction[:, 1] / -direction[:, 0]
        slope_y = direction[:, 2] / -direction[:, 0]
        norm = np.sqrt(1.0 + slope_x**2 + slope_y**2)
        expected_pdf = norm**3 / float(self.geometry.slope_area)

        np.testing.assert_allclose(
            result.position_pc,
            np.broadcast_to(
                np.asarray(self.geometry.source_position_pc), (4096, 3)
            ),
        )
        np.testing.assert_array_equal(momentum[:, 0], packets.energy_kev)
        np.testing.assert_allclose(
            np.linalg.norm(momentum[:, 1:], axis=1),
            momentum[:, 0],
            rtol=2.0e-6,
        )
        self.assertTrue(np.all(momentum[:, 1] < 0.0))
        self.assertTrue(
            np.all(
                (slope_x >= float(self.geometry.slope_x_bounds[0]))
                & (slope_x < float(self.geometry.slope_x_bounds[1]))
            )
        )
        self.assertTrue(
            np.all(
                (slope_y >= float(self.geometry.slope_y_bounds[0]))
                & (slope_y < float(self.geometry.slope_y_bounds[1]))
            )
        )
        np.testing.assert_allclose(result.launch_pdf_per_sr, expected_pdf)
        np.testing.assert_allclose(
            result.isotropic_importance,
            1.0 / (4.0 * np.pi * expected_pdf),
        )
        np.testing.assert_array_equal(result.emission_time_s, packets.emission_time_s)
        np.testing.assert_array_equal(
            result.weight_observer_fluence, packets.weight_observer_fluence
        )
        np.testing.assert_array_equal(result.time_index, packets.time_index)
        np.testing.assert_array_equal(
            result.spectral_bin_index, packets.spectral_bin_index
        )

    def test_importance_measure_is_independent_of_outer_launch_cone(self):
        n_packets = 120_000
        packets = _packets(n_packets)
        inner = build_rectangular_launch_geometry(
            10.0, (-0.10, 0.10), (-0.08, 0.08)
        )
        outer_geometries = (
            build_rectangular_launch_geometry(
                10.0, (-0.20, 0.25), (-0.18, 0.22)
            ),
            build_rectangular_launch_geometry(
                10.0, (-0.50, 0.45), (-0.40, 0.35)
            ),
        )
        expected = float(inner.launch_solid_angle_sr) / (4.0 * np.pi)
        estimates = []
        for index, geometry in enumerate(outer_geometries):
            launched = sample_source_launches(
                random.PRNGKey(100 + index), packets, geometry
            )
            direction = np.asarray(launched.momentum_kev[:, 1:])
            direction = direction / np.linalg.norm(
                direction, axis=1, keepdims=True
            )
            slope_x = direction[:, 1] / -direction[:, 0]
            slope_y = direction[:, 2] / -direction[:, 0]
            inside = (
                (slope_x >= -0.10)
                & (slope_x <= 0.10)
                & (slope_y >= -0.08)
                & (slope_y <= 0.08)
            )
            estimate = np.mean(
                np.asarray(launched.isotropic_importance) * inside,
                dtype=np.float64,
            )
            estimates.append(estimate)
            self.assertAlmostEqual(estimate, expected, delta=1.5e-4)
        self.assertAlmostEqual(estimates[0], estimates[1], delta=2.0e-4)

    def test_launch_output_is_a_voxel_transport_input(self):
        packets = _packets(256)
        launched = sample_source_launches(
            random.PRNGKey(20), packets, self.geometry
        )
        physics = build_dust_physics_table(
            energy_kev=[3.3, 4.9],
            scattering_cross_section_cm2_per_h=[0.0, 0.0],
            absorption_cross_section_cm2_per_h=[0.0, 0.0],
            scattering_angle_rad=[0.0, np.pi],
            scattering_angle_cdf=[[0.0, 1.0], [0.0, 1.0]],
        )
        transport_jit = jax.jit(
            transport_photon_batch, static_argnames=("max_interactions",)
        )
        transported = transport_jit(
            random.PRNGKey(21),
            launched.position_pc,
            launched.momentum_kev,
            self.cloud,
            physics,
            max_interactions=2,
        )

        np.testing.assert_array_equal(
            transported.status, REACHED_OBSERVER_PLANE
        )
        np.testing.assert_array_equal(transported.n_interactions, 0)

    def test_launch_is_reproducible_for_a_fixed_key(self):
        packets = _packets(128)
        first = sample_source_launches(
            random.PRNGKey(31), packets, self.geometry
        )
        repeated = sample_source_launches(
            random.PRNGKey(31), packets, self.geometry
        )
        changed = sample_source_launches(
            random.PRNGKey(32), packets, self.geometry
        )
        np.testing.assert_array_equal(first.momentum_kev, repeated.momentum_kev)
        self.assertFalse(
            np.array_equal(
                np.asarray(first.momentum_kev), np.asarray(changed.momentum_kev)
            )
        )

    def test_rejects_invalid_geometry_and_packet_shapes(self):
        with self.assertRaisesRegex(ValueError, "slope_x_bounds"):
            build_rectangular_launch_geometry(10.0, (0.1, 0.1), (-0.1, 0.1))
        with self.assertRaisesRegex(ValueError, "padding_arcsec"):
            build_cloud_launch_geometry(self.cloud, padding_arcsec=-1.0)

        touching_source = self.cloud._replace(
            z_edges_kpc=self.cloud.z_edges_kpc.at[-1].set(10.0)
        )
        with self.assertRaisesRegex(ValueError, "strictly in front"):
            build_cloud_launch_geometry(touching_source)

        mismatched = _packets(4)._replace(
            weight_observer_fluence=jnp.ones(3, dtype=jnp.float32)
        )
        with self.assertRaisesRegex(ValueError, "same length"):
            sample_source_launches(
                random.PRNGKey(40), mismatched, self.geometry
            )


if __name__ == "__main__":
    unittest.main()
