"""Tests for the built-in smoke-test scenes and sources."""

import unittest

import numpy as np

from dsh.examples import (
    DAY_S,
    build_synthetic_four_cloud_scene,
    build_v1_decay_source,
    build_v1_test_source,
    post_peak_time_edges_s,
)


class TestBuiltInExamples(unittest.TestCase):
    def test_four_cloud_scene_has_documented_geometry(self):
        cloud = build_synthetic_four_cloud_scene(10.0)

        self.assertEqual(cloud.delta_nh_cm2.shape, (160, 51, 51))
        self.assertTrue(np.all(np.asarray(cloud.delta_nh_cm2) >= 0.0))
        self.assertAlmostEqual(float(cloud.source_distance_kpc), 10.0)

    def test_constant_flare_has_documented_fluence(self):
        source = build_v1_test_source([3.3, 4.9, 6.9])

        self.assertAlmostEqual(float(source.total_fluence), 136.8, places=4)

    def test_post_peak_edges_close_nonintegral_duration(self):
        edges = post_peak_time_edges_s(2.0, 1.0, 0.3) / DAY_S

        np.testing.assert_allclose(edges, [2.0, 2.3, 2.6, 2.9, 3.0])

    def test_decay_source_matches_analytic_fluence(self):
        peak = np.asarray([0.020, 0.012, 0.006])
        source = build_v1_decay_source(
            [3.3, 4.9, 6.9],
            peak_band_flux=peak,
            baseline_band_flux=[0.0, 0.0, 0.0],
            decay_time_days=25.0,
            decay_start_days=0.0,
            decay_duration_days=120.0,
            source_time_bin_days=0.25,
        )
        expected = peak.sum() * 25.0 * DAY_S * (1.0 - np.exp(-120.0 / 25.0))

        np.testing.assert_allclose(source.total_fluence, expected, rtol=2.0e-6)


if __name__ == "__main__":
    unittest.main()
