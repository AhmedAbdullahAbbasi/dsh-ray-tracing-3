"""A fourfold error can survive radial averaging; require an angular check."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from azimuth import analyze_azimuth


class TestAzimuthAnalysis(unittest.TestCase):
    def arrays(self):
        edges = np.arange(-600.0, 604.0, 4.0)
        centers = (edges[:-1] + edges[1:]) / 2
        x, y = np.meshgrid(centers, centers)
        reference = 1e-5 * np.exp(-(x * x + y * y) / (2 * 100**2))
        phi = np.arctan2(y, x)
        return edges, reference, phi

    def test_matches_reference_in_resolved_windows(self):
        edges, reference, _ = self.arrays()
        with tempfile.TemporaryDirectory() as folder:
            d = analyze_azimuth(
                reference, reference, reference**2 / 20, edges, 3000000, Path(folder)
            )
        self.assertTrue(d["sampling_sufficient"])
        self.assertTrue(d["resolved_harmonics_passed"])

    def test_fourfold_pattern_is_rejected(self):
        edges, reference, phi = self.arrays()
        image = reference * (1 + 0.3 * np.cos(4 * phi))
        with tempfile.TemporaryDirectory() as folder:
            d = analyze_azimuth(
                reference, image, image**2 / 20, edges, 3000000, Path(folder)
            )
        self.assertTrue(d["sampling_sufficient"])
        self.assertFalse(d["resolved_harmonics_passed"])

    def test_empty_scores_cannot_pass_sampling(self):
        edges, reference, _ = self.arrays()
        with tempfile.TemporaryDirectory() as folder:
            d = analyze_azimuth(
                reference,
                np.zeros_like(reference),
                np.zeros_like(reference),
                edges,
                3000000,
                Path(folder),
            )
        self.assertFalse(d["sampling_sufficient"])


if __name__ == "__main__":
    unittest.main()
