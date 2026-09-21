"""Tests for the physical source-cloud-observer coordinate contract."""

import unittest

import jax
import numpy as np

from dsh.geometry.coordinates import (
    ARCSEC_TO_RAD,
    angular_offset_direction,
    build_sightline_geometry,
    cartesian_to_sky,
    sky_position_pc,
)


class TestPhysicalCoordinates(unittest.TestCase):
    def test_centered_sightline_geometry(self):
        geometry = build_sightline_geometry(10.0)

        np.testing.assert_allclose(geometry.observer_position_pc, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(geometry.source_position_pc, [10_000.0, 0.0, 0.0])
        self.assertAlmostEqual(float(geometry.source_distance_pc), 10_000.0)
        np.testing.assert_allclose(
            geometry.source_to_observer_direction, [-1.0, 0.0, 0.0]
        )

    def test_one_arcsec_transverse_scale(self):
        position = np.asarray(sky_position_pc(1.0, 1.0, 0.0))
        expected_transverse_pc = 1000.0 * np.sin(ARCSEC_TO_RAD)

        self.assertAlmostEqual(position[1], expected_transverse_pc, places=8)
        self.assertAlmostEqual(np.linalg.norm(position), 1000.0, places=5)

    def test_position_round_trip(self):
        distance = np.array([2.5, 7.0], dtype=np.float32)
        sky_x = np.array([-250.0, 120.0], dtype=np.float32)
        sky_y = np.array([40.0, -90.0], dtype=np.float32)

        position = sky_position_pc(distance, sky_x, sky_y)
        recovered_distance, recovered_x, recovered_y = cartesian_to_sky(position)

        np.testing.assert_allclose(recovered_distance, distance, rtol=2e-6)
        np.testing.assert_allclose(recovered_x, sky_x, rtol=2e-6, atol=2e-5)
        np.testing.assert_allclose(recovered_y, sky_y, rtol=2e-6, atol=2e-5)

    def test_array_conversion_is_jittable(self):
        converter = jax.jit(sky_position_pc)
        position = converter(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([0.0, 30.0], dtype=np.float32),
            0.0,
        )

        self.assertEqual(position.shape, (2, 3))
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(position), axis=1), [1000.0, 2000.0], rtol=1e-6
        )

    def test_direction_is_normalized(self):
        direction = np.asarray(angular_offset_direction(500.0, -500.0))
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=6)
        self.assertGreater(direction[0], 0.0)

    def test_rejects_invalid_source_distance(self):
        for invalid in (0.0, -1.0, np.nan, np.inf, [10.0]):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "source_distance_kpc"),
            ):
                build_sightline_geometry(invalid)

    def test_rejects_wrong_cartesian_shape(self):
        with self.assertRaisesRegex(ValueError, "final dimension 3"):
            cartesian_to_sky(np.zeros((2, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
