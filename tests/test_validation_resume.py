"""Check that added screen photons do not require repeating passed physics gates."""

import unittest
from dataclasses import asdict
from types import SimpleNamespace

from dsh.validation.image import ImageValidation, RingSliceMeasurement
from scripts.rerun_screen_images import _saved_primary_result, _update_screen_checks


def _image(fraction, n_slices):
    slices = tuple(
        RingSliceMeasurement(
            time_bin_index=i,
            time_start_s=i * 21_600.0,
            time_end_s=(i + 1) * 21_600.0,
            event_count=40,
            measured_median_radius_arcsec=(1.0 - fraction) / fraction * (80 + i),
            analytic_midpoint_radius_arcsec=(1.0 - fraction) / fraction * (80 + i),
        )
        for i in range(4, 4 + n_slices)
    )
    return ImageValidation(
        path=f"screen_{fraction}.npz",
        binned_fluence=0.009,
        unbinned_fluence=0.001,
        ring_bins_checked=n_slices,
        maximum_ring_bound_violation_arcsec=0.0,
        half_pixel_diagonal_arcsec=1.0,
        ring_slices=slices,
    )


def _result(fraction, n_slices):
    return SimpleNamespace(
        image=_image(fraction, n_slices),
        packet_count=1_000_000,
        scored_event_count=5_000,
        scored_observer_fluence=0.01,
        maximum_relative_delay_error=1.0e-7,
    )


class TestResumeScreenImages(unittest.TestCase):
    def test_rechecks_screens_and_keeps_the_primary_energy_run(self):
        primary = _result(0.5, 12)
        primary.packet_count = 4_000_000
        report = {
            "simulation": {
                "uniform_screen": [
                    {
                        "image": asdict(primary.image),
                        "packet_count": primary.packet_count,
                        "scored_event_count": primary.scored_event_count,
                        "scored_observer_fluence": primary.scored_observer_fluence,
                        "maximum_relative_delay_error": (
                            primary.maximum_relative_delay_error
                        ),
                    }
                ],
                "image_screen_sweep_3p3_kev": [],
            },
            "checks": {
                "all_passed": False,
                "energy_width_scaling": True,
                "image_screen_geometry": False,
                "image_screen_fraction_order": False,
                "image_screen_fluence_closure": True,
            },
        }
        restored = _saved_primary_result(report)
        self.assertEqual(restored.packet_count, 4_000_000)
        incomplete = [(0.1, _result(0.1, 5)), (0.5, restored), (0.9, _result(0.9, 6))]
        self.assertFalse(_update_screen_checks(report, incomplete)["all_passed"])

        complete = [(0.1, _result(0.1, 6)), (0.5, restored), (0.9, _result(0.9, 6))]
        checks = _update_screen_checks(report, complete)
        self.assertTrue(checks["all_passed"])
        self.assertTrue(checks["energy_width_scaling"])
        self.assertEqual(
            report["simulation"]["uniform_screen"][0]["packet_count"], 4_000_000
        )
        self.assertEqual(
            len(
                report["simulation"]["image_screen_sweep_3p3_kev"][0]["image"][
                    "ring_slices"
                ]
            ),
            6,
        )


if __name__ == "__main__":
    unittest.main()
