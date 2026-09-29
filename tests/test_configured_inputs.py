"""A mixed external source and independent materials reach schema-7 products."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from dsh.config import build_run, load_run_config, run_configured_simulation
from dsh.materials import packaged_material_paths
from dsh.sources.source_fits import SourceFluxFile, write_source_fits


class ConfiguredInputTests(unittest.TestCase):
    def test_mixed_file_run_records_source_and_material_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source_path = write_source_fits(
                base / "source.fits",
                SourceFluxFile(
                    time_edges_s=np.array([0.0, 3600.0]),
                    continuum_energy_edges_kev=np.array([2.0, 4.0]),
                    continuum_flux=np.array([[0.01]]),
                    continuum_shape=("FLAT",),
                    continuum_photon_index=np.array([[0.0]]),
                    line_energy_kev=np.array([6.4]),
                    line_flux=np.array([[0.001]]),
                    line_labels=("FeKa",),
                ),
            )
            scattering, absorption, grid = packaged_material_paths("2-10")
            config_path = base / "run.toml"
            config_path.write_text(
                f"""format_version = 1
[run]
name = "input_test"
packets = 32
chunk_size = 16
max_interactions = 16
seed = 91
[scene]
kind = "example_four_cloud"
source_distance_kpc = 10.0
[source]
file = "source.fits"
components = "both"
[materials]
scattering = "{scattering}"
absorption = "{absorption}"
grid = "{grid}"
[observer]
time_edges_days = [0.0, 60.0]
energy_edges_kev = [2.0, 4.0, 6.0, 10.0]
[output]
npz = "run.npz"
fits = "run.fits"
"""
            )
            config = load_run_config(config_path)
            _, cells, _, _, _, _ = build_run(config)
            self.assertAlmostEqual(float(cells.total_fluence), 39.6, places=4)
            report = run_configured_simulation(config)
            self.assertTrue(report["numerical_passed"])
            self.assertFalse(report["validated_science_product"])
            self.assertEqual(
                report["source_sha256"],
                hashlib.sha256(source_path.read_bytes()).hexdigest(),
            )
            with np.load(config.output_npz, allow_pickle=False) as archive:
                self.assertEqual(int(archive["output_schema_version"]), 7)
                self.assertEqual(int(archive["history_count"]), 32)
                np.testing.assert_array_equal(archive["source_cell_kind"], [1, 0])
                self.assertEqual(
                    str(archive["source_fits_sha256"]), report["source_sha256"]
                )
                self.assertEqual(
                    hashlib.sha256(
                        str(archive["resolved_config"]).encode()
                    ).hexdigest(),
                    str(archive["resolved_config_sha256"]),
                )
            with fits.open(config.output_fits, checksum=True) as hdus:
                self.assertTrue(all(hdu.verify_checksum() == 1 for hdu in hdus))
                self.assertEqual(list(hdus["SOURCE"].data["KIND"]), [1, 0])
                self.assertEqual(hdus[0].header["SRCFSHA"], report["source_sha256"])
                self.assertEqual(hdus[0].header["OUTSCHEM"], 7)

    def test_rejects_source_outside_material_energy_range(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            write_source_fits(
                base / "source.fits",
                SourceFluxFile(
                    np.array([0.0, 1.0]),
                    np.array([]),
                    np.empty((1, 0)),
                    (),
                    np.empty((1, 0)),
                    np.array([10.1]),
                    np.array([[1.0]]),
                    ("too_high",),
                ),
            )
            scattering, absorption, grid = packaged_material_paths("2-10")
            path = base / "run.toml"
            path.write_text(
                f"""format_version = 1
[run]
name = "out_of_range"
packets = 2
chunk_size = 2
max_interactions = 16
seed = 1
[scene]
kind = "example_four_cloud"
source_distance_kpc = 10.0
[source]
file = "source.fits"
[materials]
scattering = "{scattering}"
absorption = "{absorption}"
grid = "{grid}"
[observer]
time_edges_days = [0.0, 60.0]
energy_edges_kev = [2.0, 10.0]
[output]
npz = "run.npz"
"""
            )
            with self.assertRaisesRegex(ValueError, "material support"):
                build_run(load_run_config(path))

    def test_config_rejects_unknown_keys_and_input_overwrite(self):
        example = Path(__file__).resolve().parents[1] / "configs/file_input_smoke.toml"
        original = example.read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.toml"
            path.write_text(
                original.replace("packets = 4096", "packets = 4096\npakets = 42")
            )
            with self.assertRaisesRegex(ValueError, "unknown keys"):
                load_run_config(path)
            source = (
                example.parent
                / "../dsh/data/examples/one_hour_hard_state_with_line.fits"
            ).resolve()
            path.write_text(
                original.replace(
                    "../dsh/data/examples/one_hour_hard_state_with_line.fits",
                    str(source),
                ).replace(
                    'fits = "../outputs/file_input_smoke.fits"', f'fits = "{source}"'
                )
            )
            with self.assertRaisesRegex(ValueError, "overwrite input"):
                load_run_config(path)


if __name__ == "__main__":
    unittest.main()
