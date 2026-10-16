"""Covariance and camera-grid checks for independent physical materials."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from scripts.compare_wd01_reflex import combined_moments, independent_ratio
from scripts.plot_wd01_reflex import export_fits, rebin_counts


def toy_run(seed, histories):
    histories = np.asarray(histories, dtype=float)
    # Each history scores two sky pixels in distinct radial/order bins.
    return {
        "seed": seed,
        "packets": len(histories),
        "radial": histories.sum(axis=0),
        "radial_q": histories.T @ histories,
        "order_sum": histories.sum(axis=0),
        "order_q": histories.T @ histories,
        "image": histories.sum(axis=0).reshape(1, 1, 2),
        "image_q": (histories**2).sum(axis=0).reshape(1, 2),
    }


class TestWD01ReflexComparison(unittest.TestCase):
    def test_independent_seed_average_matches_pooled_history_covariance(self):
        first = np.array([[1, 2], [0, 0], [2, 0], [0, 0]])
        second = np.array([[0, 1], [0, 0], [0, 0], [3, 4]])
        combined = combined_moments([toy_run(1, first), toy_run(2, second)])
        # Both stored runs are sums of equally normalized histories. Their
        # average gives pooled per-history weights divided by two.
        pooled = np.concatenate([first, second]) / 2
        expected = np.cov(pooled, rowvar=False, ddof=1) * len(pooled)
        np.testing.assert_allclose(combined["radial"], pooled.sum(axis=0))
        np.testing.assert_allclose(combined["radial_covariance"], expected)
        np.testing.assert_allclose(combined["order_covariance"], expected)
        np.testing.assert_allclose(
            combined["image_variance"].ravel(), expected.diagonal()
        )
        self.assertEqual(combined["histories"], 8)

    def test_rejects_duplicate_seeds_and_unequal_sample_sizes(self):
        run = toy_run(4, [[1, 2], [0, 0]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            combined_moments([run, run])
        with self.assertRaisesRegex(ValueError, "equal-size"):
            combined_moments([run, toy_run(5, [[1, 2]])])

    def test_ratio_error_scales_reference_noise_by_actual_ratio(self):
        ratio, error = independent_ratio(20.0, 10.0, 4.0, 10.0)
        self.assertEqual(ratio, 2)
        self.assertAlmostEqual(error**2, 4 / 100 + 400 * 10 / 10000)
        with self.assertRaises(ValueError):
            independent_ratio(1, 0, 1, 1)

    def test_display_rebin_preserves_counts_and_orientation(self):
        image = np.zeros((6, 6))
        image[0, 0], image[-1, -1] = 3, 7
        rebinned = rebin_counts(image)
        np.testing.assert_array_equal(rebinned, [[3, 0], [0, 7]])

    def test_fits_has_native_grid_and_consistent_fluence_units(self):
        arrays = {
            "reference_counts": np.array([[1, 2], [3, 4]]),
            "expected_counts": np.full((2, 2), 2.0),
            "image_by_order": np.ones((3, 2, 2)),
            "image_mc_variance": np.full((2, 2), 0.25),
        }
        summary = {
            "config": {"pixel_arcsec": 5, "energy_kev": 4.9, "nh_cm2": 1e22},
            "histories": 100,
            "material": {"definition_sha256": "a" * 64},
        }
        with tempfile.TemporaryDirectory() as temporary:
            export_fits(Path(temporary), arrays, summary)
            with fits.open(Path(temporary) / "halo_images.fits", checksum=True) as hdus:
                np.testing.assert_array_equal(
                    hdus["REFLEX"].data, arrays["reference_counts"]
                )
                np.testing.assert_array_equal(hdus["FLUENCE"].data, np.full((2, 2), 3))
                self.assertEqual(hdus["FLUENCE"].header["BUNIT"], "ph cm-2")
                self.assertEqual(hdus["WD01"].header["CDELT2"], 5)
                self.assertEqual(hdus["WD01"].header["CRPIX2"], 1.5)
                self.assertTrue(all(hdu.verify_checksum() for hdu in hdus))


if __name__ == "__main__":
    unittest.main()
