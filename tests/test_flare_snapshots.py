"""Verify local flare snapshots select the intended, disjoint arrival intervals."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class TestArrivalEdges(unittest.TestCase):
    def test_sparse_edges_cover_snapshots_and_background(self):
        from dsh.cli import _arrival_edges

        seconds = _arrival_edges(60, 1, [0, 3, 4, 6, 7, 9, 10, 60])
        np.testing.assert_array_equal(
            seconds, np.asarray([0, 3, 4, 6, 7, 9, 10, 60]) * 86_400
        )
        for invalid in ([3, 4], [0, 4, 3], [0, 3, 3], [0, np.nan]):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "strictly increasing"),
            ):
                _arrival_edges(60, 1, invalid)


class TestHardStateCli(unittest.TestCase):
    def test_runner_receives_continuous_source_and_three_output_bands(self):
        from dsh.cli import main

        class StopBeforeSimulation(Exception):
            pass

        def check_simulation_inputs(
            key, source, launch, cloud, physics, bins, **kwargs
        ):
            np.testing.assert_allclose(
                np.asarray(source.energy_edges_kev), [2.0, 4.0, 6.0, 10.0]
            )
            np.testing.assert_allclose(
                np.asarray(bins.energy_edges_kev), [2.0, 4.0, 6.0, 10.0]
            )
            self.assertAlmostEqual(float(source.photon_index), 1.7)
            self.assertAlmostEqual(float(source.total_fluence), 136.8, places=3)
            self.assertEqual(kwargs["total_packets"], 2_500_000)
            raise StopBeforeSimulation

        with tempfile.TemporaryDirectory() as directory:
            argv = [
                "run_dsh_v1",
                "--materials",
                "2-10",
                "--source-spectrum",
                "hard-state-powerlaw",
                "--packets",
                "2500000",
                "--output",
                str(Path(directory) / "mocked.npz"),
            ]
            with (
                patch("sys.argv", argv),
                patch(
                    "dsh.cli.run_tabulated_source_to_observer_chunked",
                    side_effect=check_simulation_inputs,
                ),
                self.assertRaises(StopBeforeSimulation),
            ):
                main()
            self.assertFalse((Path(directory) / "mocked.npz").exists())


@unittest.skipUnless(importlib.util.find_spec("astropy"), "requires Astropy")
class TestFlareSnapshots(unittest.TestCase):
    def _write_input(self, path, *, limit_count=0):
        from astropy.io import fits

        cube = np.zeros((12, 3, 2, 2), dtype=np.float32)
        first = cube.copy()
        multiple = cube.copy()
        first[3, 0, 0, 1] = 2.0
        multiple[6, 1, 1, 0] = 1.0
        first[9, 2, 1, 1] = 3.0
        cube[:] = first + multiple
        counts = (cube > 0).astype(np.int32)
        primary = fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32))
        primary.header["NPACKETS"] = 2_500_000
        primary.header["MATMODEL"] = "2-10"
        primary.header["SRCSPEC"] = "hard-state-powerlaw"
        primary.header["PHINDEX"] = 1.7
        primary.header["EMINKEV"] = 2.0
        primary.header["EMAXKEV"] = 10.0
        primary.header["SCATSHA"] = "a" * 64
        primary.header["ABSSHA"] = "b" * 64
        primary.header["CTYPE1"] = "XOFFSET"
        primary.header["CRVAL1"] = 5.0
        seconds = np.arange(13, dtype=np.float64) * 86_400.0
        time_bins = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="LOW", format="D", array=seconds[:-1]),
                fits.Column(name="HIGH", format="D", array=seconds[1:]),
            ],
            name="TIME_BINS",
        )
        energy_bins = fits.BinTableHDU.from_columns(
            [fits.Column(name="LOW", format="D", array=[2.5, 4.1, 5.9])],
            name="ENERGY_BINS",
        )
        statuses = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="STATUS_CODE", format="J", array=np.arange(7)),
                fits.Column(
                    name="COUNT",
                    format="K",
                    array=np.asarray(
                        [0, 2_499_999 - limit_count, 0, 1, limit_count, 0, 0],
                        dtype=np.int64,
                    ),
                ),
            ],
            name="STATUS",
        )
        diagnostics = fits.BinTableHDU.from_columns(
            [fits.Column(name="BINNED_EVENT_COUNT", format="K", array=[3])],
            name="DIAGNOSTICS",
        )
        fits.HDUList(
            [
                primary,
                fits.ImageHDU(cube, name="TOTAL4D"),
                fits.ImageHDU(first, name="FIRST4D"),
                fits.ImageHDU(multiple, name="MULTI4D"),
                fits.ImageHDU(counts, name="EVENT4D"),
                time_bins,
                energy_bins,
                statuses,
                diagnostics,
            ]
        ).writeto(path, checksum=True)

    def test_images_use_three_separate_one_day_arrival_windows(self):
        from astropy.io import fits

        from scripts.extract_flare_snapshots import extract_snapshots

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_fits = root / "complete.fits"
            self._write_input(input_fits)
            report = extract_snapshots(input_fits, root / "snapshots")
            self.assertEqual(
                [entry["arrival_start_day"] for entry in report["products"]],
                [3, 6, 9],
            )
            self.assertEqual(
                [entry["observer_fluence_ph_cm2"] for entry in report["products"]],
                [2.0, 1.0, 3.0],
            )
            self.assertEqual(report["diagnostics"]["binned_event_count"], 3)
            self.assertEqual(report["source_spectrum"], "hard-state-powerlaw")
            self.assertEqual(report["energy_range_kev"], [2.0, 10.0])
            for start, expected in ((3, 2.0), (6, 1.0), (9, 3.0)):
                path = (
                    root
                    / "snapshots"
                    / f"flare_day_{start:03d}_to_{start + 1:03d}.fits"
                )
                with fits.open(path, checksum=True) as hdul:
                    hdul.verify("exception")
                    self.assertEqual(hdul[0].data.shape, (2, 2))
                    self.assertEqual(float(hdul[0].data.sum()), expected)
                    self.assertEqual(hdul[0].header["TSTART"], start * 86_400)
                    self.assertEqual(hdul[0].header["CRVAL1"], 5.0)
                    self.assertEqual(int(hdul["EVENTIMG"].data.sum()), 1)
            self.assertTrue(
                (root / "snapshots" / "flare_snapshots_manifest.json").is_file()
            )

    def test_numerical_limit_prevents_success_outputs(self):
        from scripts.extract_flare_snapshots import extract_snapshots

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_fits = root / "complete.fits"
            self._write_input(input_fits, limit_count=1)
            with self.assertRaisesRegex(ValueError, "numerical/invalid"):
                extract_snapshots(input_fits, root / "snapshots")
            self.assertFalse((root / "snapshots").exists())

    def test_rejects_old_three_energy_flare(self):
        from astropy.io import fits

        from scripts.extract_flare_snapshots import extract_snapshots

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_fits = root / "complete.fits"
            self._write_input(input_fits)
            with fits.open(input_fits, mode="update") as hdul:
                hdul[0].header["SRCSPEC"] = "representative"
            with self.assertRaisesRegex(ValueError, "continuously sampled"):
                extract_snapshots(input_fits, root / "snapshots")
            self.assertFalse((root / "snapshots").exists())


if __name__ == "__main__":
    unittest.main()
