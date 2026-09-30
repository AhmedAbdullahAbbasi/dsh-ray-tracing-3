"""A generated line-only run produces auditable dated schema-7 images."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from dsh.build import build_run_plan
from dsh.config import build_run, load_run_config, run_configured_simulation
from scripts.extract_configured_line_snapshots import extract_snapshots
from scripts.prepare_realistic_line_run import prepare_run


class RealisticLineWorkflowTests(unittest.TestCase):
    def test_generated_pilot_and_snapshots_close_to_saved_history_moments(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cloud = fits.PrimaryHDU(np.ones((6, 2, 2), dtype=np.float64) * 1e21)
            cloud.header["BUNIT"] = "cm-2"
            for axis, kind, unit, origin, step in (
                (1, "XOFFSET", "arcsec", -900.0, 1800.0),
                (2, "YOFFSET", "arcsec", -900.0, 1800.0),
                (3, "DISTANCE", "kpc", 2.5, 1.0),
            ):
                cloud.header[f"CTYPE{axis}"] = kind
                cloud.header[f"CUNIT{axis}"] = unit
                cloud.header[f"CRPIX{axis}"] = 1.0
                cloud.header[f"CRVAL{axis}"] = origin
                cloud.header[f"CDELT{axis}"] = step
            cube_path = root / "cloud.fits"
            cloud.writeto(cube_path, checksum=True)
            pilot_path, full_path = prepare_run(
                cube_path, root / "generated", pilot_packets=256, packets=2_500_000
            )
            self.assertEqual(load_run_config(full_path).packets, 2_500_000)
            config = load_run_config(pilot_path)
            _, cells, _, scene, _, bins = build_run(config)
            self.assertEqual(config.components, "lines")
            self.assertAlmostEqual(float(cells.total_fluence), 136.8, places=4)
            np.testing.assert_array_equal(np.asarray(cells.kind), [0])
            self.assertAlmostEqual(float(scene.source_distance_kpc), 10.5)
            self.assertEqual(len(bins.arrival_time_edges_s), 8)
            plan = build_run_plan(config)
            self.assertEqual(plan.packets, config.packets)
            np.testing.assert_array_equal(
                plan.source.total_fluence, cells.total_fluence
            )
            np.testing.assert_array_equal(plan.cloud.delta_nh_cm2, scene.delta_nh_cm2)
            self.assertTrue(run_configured_simulation(config)["numerical_passed"])
            report = extract_snapshots(
                pilot_path, days=(3, 6, 9), output_dir=root / "snapshots"
            )
            self.assertTrue(report["product_integrity_passed"])
            self.assertFalse(report["validated_science_product"])
            self.assertEqual(len(report["snapshots"]), 3)
            manifest = json.loads(
                (root / "snapshots" / "line_snapshots_manifest.json").read_text()
            )
            self.assertAlmostEqual(manifest["source_energy_kev"], 5.35, places=5)
            with np.load(config.output_npz, allow_pickle=False) as archive:
                for day, time_index in ((3, 1), (6, 3), (9, 5)):
                    path = (
                        root / "snapshots" / f"line_day_{day:03d}_to_{day + 1:03d}.fits"
                    )
                    with fits.open(path, checksum=True) as hdus:
                        self.assertTrue(all(hdu.verify_checksum() == 1 for hdu in hdus))
                        self.assertEqual(hdus[0].header["SNAPSCHE"], 1)
                        np.testing.assert_allclose(
                            hdus[0].data,
                            archive["total_fluence"][time_index, 0],
                            rtol=2e-6,
                            atol=1e-12,
                        )
                        n = config.packets
                        image = np.asarray(archive["total_fluence"][time_index, 0])
                        q = np.asarray(archive["total_fluence_squared"][time_index, 0])
                        expected_error = np.sqrt(
                            n / (n - 1) * np.maximum(q - image * image / n, 0)
                        )
                        np.testing.assert_allclose(
                            hdus["STDIMG"].data, expected_error, rtol=2e-6, atol=1e-12
                        )
            command_output = root / "command_snapshots"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dsh.command",
                    "snapshot",
                    str(pilot_path),
                    "--output-dir",
                    str(command_output),
                    "--days",
                    "3",
                    "6",
                    "9",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn("line_day_003_to_004.fits", completed.stdout)
            command_manifest = json.loads(
                (command_output / "line_snapshots_manifest.json").read_text()
            )
            self.assertEqual(command_manifest, manifest)
            # Invalid windows exit before producing a misleading snapshot.
            rejected = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dsh.command",
                    "snapshot",
                    str(pilot_path),
                    "--output-dir",
                    str(root / "invalid_snapshots"),
                    "--days",
                    "2",
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("must match one arrival bin", rejected.stderr)
            self.assertFalse((root / "invalid_snapshots").exists())
            with self.assertRaisesRegex(FileExistsError, "already exist"):
                prepare_run(cube_path, root / "generated")


if __name__ == "__main__":
    unittest.main()
