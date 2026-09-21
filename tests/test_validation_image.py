"""Check that scored, binned halo images preserve an impulsive ring."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dsh.observer.scoring import score_peeloff_events
from dsh.physics.newdust import (
    build_dust_physics_from_newdust,
    load_newdust_scattering_table,
)
from dsh.validation import small_angle_ring_radius_arcsec
from dsh.validation.image import bin_and_validate_ring_image

from .test_validation import _empty_cloud, _synthetic_single_scatter_ring


class TestScoredImageGeometry(unittest.TestCase):
    def test_impulse_ring_is_present_in_six_image_time_slices(self):
        scattering = load_newdust_scattering_table()
        physics = build_dust_physics_from_newdust(
            scattering, np.zeros_like(scattering.energy_kev)
        )
        days = np.arange(1.125, 2.625, 0.25)
        theta = small_angle_ring_radius_arcsec(days * 86_400, 10.0, 0.5)
        azimuth = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        launched, transported = _synthetic_single_scatter_ring(
            np.full((6, 32), 0.5),
            np.broadcast_to(theta[:, None], (6, 32)),
            np.broadcast_to(azimuth, (6, 32)),
        )
        events = score_peeloff_events(launched, transported, _empty_cloud(), physics)
        self.assertTrue(np.all(np.asarray(events.valid)))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "impulse_ring.npz"
            x = np.asarray(events.sky_x_arcsec)[:, 0]
            y = np.asarray(events.sky_y_arcsec)[:, 0]
            time = np.asarray(events.arrival_time_s)[:, 0]
            weight = np.asarray(events.weight_observer_fluence)[:, 0]
            result = bin_and_validate_ring_image(
                output,
                x,
                y,
                time,
                weight,
                energy_kev=3.3,
                source_distance_kpc=10.0,
                fractional_distance=0.5,
            )
            self.assertEqual(result.ring_bins_checked, 6)
            self.assertLessEqual(
                result.maximum_ring_bound_violation_arcsec,
                result.half_pixel_diagonal_arcsec,
            )
            with np.load(output) as image:
                self.assertEqual(image["fluence_time_y_x"].shape, (32, 128, 128))
                self.assertEqual(image["event_count_time_y_x"].sum(), 192)
                np.testing.assert_allclose(
                    image["fluence_time_y_x"].sum() + result.unbinned_fluence,
                    np.sum(weight),
                    rtol=1.0e-5,
                )
            if result.fits_path is not None:
                from astropy.io import fits

                with fits.open(result.fits_path) as hdus:
                    self.assertEqual(hdus[0].data.shape, (128, 128))
                    self.assertEqual(hdus["TIME_CUBE"].data.shape, (32, 128, 128))
                    self.assertEqual(hdus["TIME_BINS"].data.shape[0], 32)
                    np.testing.assert_allclose(
                        hdus[0].data.sum(), result.binned_fluence, rtol=1e-5
                    )

            # A one-day time-offset error must be caught by the image check.
            shifted = bin_and_validate_ring_image(
                Path(temporary) / "wrong_time.npz",
                x,
                y,
                time + 86_400.0,
                weight,
                energy_kev=3.3,
                source_distance_kpc=10.0,
                fractional_distance=0.5,
            )
            self.assertGreater(
                shifted.maximum_ring_bound_violation_arcsec,
                shifted.half_pixel_diagonal_arcsec,
            )


if __name__ == "__main__":
    unittest.main()
