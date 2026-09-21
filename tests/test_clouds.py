"""Physics and validation tests for native angular--distance cloud cubes."""

import unittest

import jax
import numpy as np

from dsh.geometry.clouds import (
    KPC_TO_CM,
    build_angular_distance_cloud,
    cloud_from_loaded_fits,
    optical_depth_map,
    recovered_delta_column_cm2,
    total_column_map_cm2,
    transmission_map,
)


class TestAngularDistanceCloud(unittest.TestCase):
    def setUp(self):
        self.x_arcsec = np.array([-1.0, 0.0, 1.0])
        self.y_arcsec = np.array([-0.5, 0.5])
        self.z_kpc = np.array([0.25, 0.75, 1.25, 1.75])
        self.delta_nh = np.full((4, 2, 3), 2.0e20)

    def build(self, **overrides):
        parameters = {
            "delta_nh_cm2": self.delta_nh,
            "x_centers_arcsec": self.x_arcsec,
            "y_centers_arcsec": self.y_arcsec,
            "z_centers_kpc": self.z_kpc,
            "source_distance_kpc": 10.0,
        }
        parameters.update(overrides)
        return build_angular_distance_cloud(**parameters)

    def test_column_and_volume_density_closure(self):
        cloud = self.build()

        expected_width_cm = 0.5 * KPC_TO_CM
        expected_density_cm3 = 2.0e20 / expected_width_cm
        np.testing.assert_allclose(
            np.asarray(cloud.radial_bin_width_cm), expected_width_cm, rtol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(cloud.n_h_cm3), expected_density_cm3, rtol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(recovered_delta_column_cm2(cloud)),
            self.delta_nh,
            rtol=2e-6,
        )
        np.testing.assert_allclose(
            np.asarray(total_column_map_cm2(cloud)),
            np.full((2, 3), 8.0e20),
            rtol=2e-6,
        )

    def test_beer_lambert_optical_depth_and_jit(self):
        cloud = self.build()
        cross_sections = np.array([1.0e-22, 4.0e-22], dtype=np.float32)

        tau = jax.jit(optical_depth_map)(cloud, cross_sections)
        transmission = jax.jit(transmission_map)(cloud, cross_sections)
        expected_tau = cross_sections[:, None, None] * 8.0e20

        self.assertEqual(tau.shape, (2, 2, 3))
        np.testing.assert_allclose(
            np.asarray(tau), np.broadcast_to(expected_tau, (2, 2, 3)), rtol=2e-6
        )
        np.testing.assert_allclose(
            np.asarray(transmission), np.exp(-np.asarray(tau)), rtol=2e-6
        )

    def test_decreasing_axes_are_normalized_without_changing_physics(self):
        reference = self.build()
        reversed_cloud = self.build(
            delta_nh_cm2=self.delta_nh[::-1, ::-1, ::-1],
            x_centers_arcsec=self.x_arcsec[::-1],
            y_centers_arcsec=self.y_arcsec[::-1],
            z_centers_kpc=self.z_kpc[::-1],
        )

        np.testing.assert_array_equal(
            np.asarray(reversed_cloud.delta_nh_cm2),
            np.asarray(reference.delta_nh_cm2),
        )
        np.testing.assert_allclose(
            np.asarray(reversed_cloud.x_edges_arcsec), [-1.5, -0.5, 0.5, 1.5]
        )
        np.testing.assert_allclose(
            np.asarray(reversed_cloud.z_edges_kpc), [0.0, 0.5, 1.0, 1.5, 2.0]
        )

    def test_loaded_fits_adapter_preserves_delta_nh(self):
        loaded = {
            "delta_nh_cm2": self.delta_nh,
            "density": np.zeros_like(self.delta_nh),
            "x_arcsec": self.x_arcsec,
            "y_arcsec": self.y_arcsec,
            "z_kpc": self.z_kpc,
            "unit": "cm-2",
        }
        cloud = cloud_from_loaded_fits(loaded, source_distance_kpc=10.0)
        np.testing.assert_allclose(
            np.asarray(cloud.delta_nh_cm2), self.delta_nh, rtol=2e-6
        )

    def test_rejects_cube_outside_observer_source_interval(self):
        with self.assertRaisesRegex(ValueError, "behind the observer"):
            self.build(z_centers_kpc=[0.0, 0.5, 1.0, 1.5])
        with self.assertRaisesRegex(ValueError, "beyond the source"):
            self.build(source_distance_kpc=1.9)

    def test_rejects_invalid_shape_axes_values_and_units(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.build(delta_nh_cm2=np.zeros((3, 2, 3)))
        with self.assertRaisesRegex(ValueError, "strictly monotonic"):
            self.build(x_centers_arcsec=[-1.0, 0.0, -0.5])
        with self.assertRaisesRegex(ValueError, "negative"):
            bad = self.delta_nh.copy()
            bad[0, 0, 0] = -1.0
            self.build(delta_nh_cm2=bad)
        with self.assertRaisesRegex(ValueError, r"cm\^-2"):
            self.build(unit="arbitrary")


if __name__ == "__main__":
    unittest.main()
