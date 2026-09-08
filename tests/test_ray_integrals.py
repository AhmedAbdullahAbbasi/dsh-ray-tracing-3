"""Analytic physics tests for exact ray-column integration."""

import unittest

import jax
import numpy as np

from utils.clouds import build_angular_distance_cloud, total_column_map_cm2
from utils.coordinates import angular_offset_direction
from utils.ray_integrals import (
    PC_TO_CM,
    integrate_ray_column_cm2,
    integrate_ray_optical_depth,
)


class TestRayColumnIntegrals(unittest.TestCase):
    def setUp(self):
        self.x_arcsec = np.array([-2.0, 0.0, 2.0])
        self.y_arcsec = np.array([-2.0, 0.0, 2.0])
        self.z_kpc = np.array([0.25, 0.75, 1.25, 1.75])

        spatial = np.array(
            [[1.0, 2.0, 4.0], [3.0, 5.0, 7.0], [6.0, 8.0, 9.0]]
        ) * 1.0e20
        radial = np.array([0.5, 1.0, 2.0, 0.25])
        self.delta_nh = radial[:, None, None] * spatial[None, :, :]
        self.cloud = build_angular_distance_cloud(
            self.delta_nh,
            self.x_arcsec,
            self.y_arcsec,
            self.z_kpc,
            source_distance_kpc=10.0,
        )

    def ray_column(self, x_arcsec, y_arcsec):
        direction = angular_offset_direction(x_arcsec, y_arcsec)
        return integrate_ray_column_cm2(
            self.cloud,
            origin_pc=np.zeros(3, dtype=np.float32),
            direction=direction,
            max_distance_pc=10_000.0,
        )

    def test_radial_sightlines_recover_native_column_pixels(self):
        expected_map = np.asarray(total_column_map_cm2(self.cloud))
        for y_index, y_arcsec in enumerate(self.y_arcsec):
            for x_index, x_arcsec in enumerate(self.x_arcsec):
                measured = float(self.ray_column(x_arcsec, y_arcsec))
                self.assertAlmostEqual(
                    measured / expected_map[y_index, x_index], 1.0, places=5
                )

    def test_multislab_column_is_sum_of_all_radial_increments(self):
        measured = float(self.ray_column(0.0, 0.0))
        expected = float(self.delta_nh[:, 1, 1].sum())
        self.assertAlmostEqual(measured / expected, 1.0, places=5)

    def test_reverse_source_to_observer_ray_has_same_column(self):
        forward = float(self.ray_column(0.0, 0.0))
        reverse = float(
            integrate_ray_column_cm2(
                self.cloud,
                origin_pc=np.array([10_000.0, 0.0, 0.0], dtype=np.float32),
                direction=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                max_distance_pc=10_000.0,
            )
        )
        self.assertAlmostEqual(reverse / forward, 1.0, places=5)

    def test_ray_outside_angular_field_has_zero_column(self):
        measured = float(self.ray_column(20.0, 20.0))
        self.assertEqual(measured, 0.0)

    def test_oblique_ray_crosses_angular_boundary_at_analytic_location(self):
        radial_width_cm = 0.5 * 1000.0 * PC_TO_CM
        n_h_by_x = np.array([1.0, 3.0])
        delta_nh = np.broadcast_to(
            n_h_by_x[None, None, :] * radial_width_cm, (2, 3, 2)
        ).copy()
        cloud = build_angular_distance_cloud(
            delta_nh,
            x_centers_arcsec=[-1500.0, 1500.0],
            y_centers_arcsec=[-1.0, 0.0, 1.0],
            z_centers_kpc=[0.75, 1.25],
            source_distance_kpc=2.0,
        )

        measured = integrate_ray_column_cm2(
            cloud,
            origin_pc=np.array([1000.0, -10.0, 0.0]),
            direction=np.array([0.0, 1.0, 0.0]),
            max_distance_pc=20.0,
        )
        expected = (1.0 * 10.0 + 3.0 * 10.0) * PC_TO_CM
        self.assertAlmostEqual(float(measured / expected), 1.0, places=5)

    def test_optical_depth_is_cross_section_times_integrated_column(self):
        direction = angular_offset_direction(0.0, 0.0)
        origin = np.zeros(3, dtype=np.float32)
        cross_section = 3.0e-22
        column = integrate_ray_column_cm2(
            self.cloud, origin, direction, max_distance_pc=10_000.0
        )
        tau = integrate_ray_optical_depth(
            self.cloud,
            origin,
            direction,
            max_distance_pc=10_000.0,
            cross_section_cm2_per_h=cross_section,
        )
        self.assertAlmostEqual(float(tau / (cross_section * column)), 1.0, places=6)

    def test_integrator_is_jittable(self):
        compiled = jax.jit(integrate_ray_column_cm2)
        measured = compiled(
            self.cloud,
            np.zeros(3, dtype=np.float32),
            angular_offset_direction(2.0, 2.0),
            10_000.0,
        )
        expected = np.asarray(total_column_map_cm2(self.cloud))[2, 2]
        self.assertAlmostEqual(float(measured / expected), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
