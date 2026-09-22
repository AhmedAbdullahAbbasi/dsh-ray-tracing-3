"""Prevent sparse images and cross-format mismatches from passing a flare gate."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dsh.validation.flare_production import inspect_flare_arrays, snapshot_slice


class TestFlareProductionScreen(unittest.TestCase):
    def test_sparse_one_day_images_fail_even_when_events_are_positive(self):
        edges = np.asarray([0, 3, 4, 6, 7, 9, 10, 60]) * 86_400
        counts = np.zeros((7, 3, 4, 4), dtype=np.int32)
        for t in (1, 3, 5):
            counts[t, 0, :2, :2] = 5
        first = counts * 0.75
        multi = counts * 0.25

        def screen(values, status):
            return inspect_flare_arrays(
                counts=values,
                total=first + multi,
                first=first,
                multiple=multi,
                arrival_edges_s=edges,
                status_counts=status,
                expected_packets=100,
                coarse_factor=2,
                min_snapshot_events=10,
                min_supported_cell_events=10,
                min_supported_event_fraction=0.8,
            )

        good = screen(counts, [0, 80, 0, 20, 0, 0, 0])
        self.assertTrue(good["all_passed"])
        self.assertEqual([r["event_count"] for r in good["snapshot_rows"]], [20] * 3)

        sparse = counts.copy()
        sparse[3, 0] = 1
        sparse_result = screen(sparse, [0, 80, 0, 20, 0, 0, 0])
        self.assertEqual(sparse_result["snapshot_rows"][1]["event_count"], 16)
        self.assertFalse(sparse_result["snapshot_rows"][1]["spatial_support_check"])
        self.assertFalse(sparse_result["all_passed"])
        energy_error = screen(counts, [0, 78, 0, 20, 0, 2, 0])
        self.assertFalse(energy_error["checks"]["clean_transport"])
        with self.assertRaisesRegex(ValueError, "not bounded"):
            snapshot_slice(edges, 11, 1)


@unittest.skipUnless(importlib.util.find_spec("astropy"), "requires Astropy")
class TestFlareProductionFitsAgreement(unittest.TestCase):
    def test_extracts_three_fits_and_catches_a_displaced_npz_event(self):
        from astropy.io import fits

        from scripts.audit_four_cloud_flare import audit_run
        from scripts.extract_flare_snapshots import extract_snapshots
        from tests.test_flare_snapshots import TestFlareSnapshots

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_fits = root / "full.fits"
            TestFlareSnapshots()._write_input(full_fits)
            snapshot_dir = root / "snapshots"
            extract_snapshots(full_fits, snapshot_dir)
            with fits.open(full_fits) as hdul:
                records = {
                    "total_fluence": hdul["TOTAL4D"].data.copy(),
                    "first_scatter_fluence": hdul["FIRST4D"].data.copy(),
                    "multiple_scatter_fluence": hdul["MULTI4D"].data.copy(),
                    "event_count": hdul["EVENT4D"].data.copy(),
                }
            base = {
                **records,
                "requested_packet_count": 2_500_000,
                "source_packet_count": 2_500_000,
                "source_spectrum": "hard-state-powerlaw",
                "source_model": "constant-flare",
                "energy_edges_kev": [2, 4, 6, 10],
                "source_photon_index": 1.7,
                "material_tables": "2-10",
                "scattering_table_sha256": "a" * 64,
                "absorption_table_sha256": "b" * 64,
                "transport_status_count": [0, 2_499_999, 0, 1, 0, 0, 0],
                "arrival_time_edges_s": np.arange(13) * 86_400,
                "binned_event_count": 3,
                "valid_event_count": 3,
                "unbinned_event_count": 0,
                "scored_observer_fluence": 6.0,
                "binned_weight_observer_fluence": 6.0,
                "unbinned_weight_observer_fluence": 0.0,
                "outside_sky_event_count": 0,
                "outside_energy_event_count": 0,
                "outside_arrival_time_weight_observer_fluence": 0.0,
                "cloud_description": "test cloud",
                "source_total_fluence": 10.0,
                "scored_observer_event_count": 3,
                "outside_arrival_time_event_count": 0,
            }
            npz = root / "fake.npz"
            np.savez(npz, **base)
            options = {
                "full_fits": full_fits,
                "snapshot_dir": snapshot_dir,
                "coarse_factor": 1,
                "min_snapshot_events": 1,
                "min_supported_cell_events": 1,
            }
            report = audit_run(npz, **options)
            self.assertTrue(all(report["fits_checks"].values()))
            self.assertEqual(
                [x["event_count"] for x in report["snapshot_rows"]], [1] * 3
            )
            displaced = base["event_count"].copy()
            displaced[3, 0, 0, 1] = 0
            displaced[3, 0, 0, 0] = 1
            np.savez(npz, **{**base, "event_count": displaced})
            altered = audit_run(npz, **options)
            self.assertFalse(altered["fits_checks"]["event4d"])
            self.assertFalse(altered["fits_checks"]["snapshot_3_pixels"])


if __name__ == "__main__":
    unittest.main()
