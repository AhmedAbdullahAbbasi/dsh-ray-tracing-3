"""Independent asymmetric-column fixture checks for Stage 9F."""

import math
import unittest

import numpy as np

from dsh.validation.heterogeneous_observer import (
    PC_TO_CM,
    RADIAL_EDGES_KPC,
    SKY_EDGES,
    host_ray_column,
    scene_columns,
)


class HeterogeneousObserverReferenceTests(unittest.TestCase):
    def test_separated_radial_shells_on_one_observer_ray(self):
        columns = scene_columns(1e22)
        origin = np.array([8000.0, 0.01, 0.02])
        toward_observer = -origin / np.linalg.norm(origin)
        observed = host_ray_column(
            origin, toward_observer, np.linalg.norm(origin), columns
        )
        expected = columns[:, 1, 1].sum()
        self.assertAlmostEqual(observed / expected, 1.0, places=10)

    def test_angled_flight_crosses_a_quadrant_boundary(self):
        columns = scene_columns(1e22)
        angle = math.tan(100.0 * math.pi / (180 * 3600))
        start = np.array([4500.0, -4500.0 * angle, 4500.0 * angle])
        end = np.array([4500.0, 4500.0 * angle, 4500.0 * angle])
        segment = end - start
        observed = host_ray_column(start, segment, np.linalg.norm(segment), columns)
        expected = (columns[2, 1, 0] + columns[2, 1, 1]) * (
            np.linalg.norm(segment) / 2000.0
        )
        self.assertAlmostEqual(observed / expected, 1.0, places=10)

    def test_ray_outside_angular_coverage_has_zero_column(self):
        columns = scene_columns(1e22)
        angle = math.tan(2500.0 * math.pi / (180 * 3600))
        origin = np.array([4500.0, 4500.0 * angle, 0.01])
        direction = np.array([-1.0, -angle, 0.0])
        self.assertEqual(host_ray_column(origin, direction, 300.0, columns), 0.0)

    def test_scene_asymmetry_and_units_are_explicit(self):
        columns = scene_columns(1e22)
        self.assertEqual(columns.shape, (len(RADIAL_EDGES_KPC) - 1,
                                         len(SKY_EDGES) - 1, len(SKY_EDGES) - 1))
        self.assertFalse(np.allclose(columns[2, 0, 0], columns[2, 1, 1]))
        self.assertGreater(PC_TO_CM, 3e18)


if __name__ == "__main__":
    unittest.main()
