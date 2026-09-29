"""A mixed external source and independent materials reach schema-7 products."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from dsh.config import (
    audit_configured_run,
    build_run,
    load_run_config,
    run_configured_simulation,
)
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
scattering = "{scattering.as_posix()}"
absorption = "{absorption.as_posix()}"
grid = "{grid.as_posix()}"
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
scattering = "{scattering.as_posix()}"
absorption = "{absorption.as_posix()}"
grid = "{grid.as_posix()}"
[observer]
time_edges_days = [0.0, 60.0]
energy_edges_kev = [2.0, 10.0]
[output]
npz = "run.npz"
"""
            )
            with self.assertRaisesRegex(ValueError, "material support"):
                build_run(load_run_config(path))

    def test_monochromatic_flare_through_asymmetric_fits_cube(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            columns = (
                np.array(
                    [
                        [[2, 1, 1], [1, 4, 2], [1, 2, 3]],
                        [[1, 3, 1], [2, 1, 4], [1, 2, 1]],
                        [[3, 1, 2], [1, 2, 1], [4, 1, 2]],
                        [[1, 2, 4], [3, 1, 2], [1, 4, 1]],
                    ],
                    dtype=np.float32,
                )
                * 1.0e21
            )
            cloud_path = base / "asymmetric_cloud.fits"
            cloud_hdu = fits.PrimaryHDU(columns)
            cloud_hdu.header["BUNIT"] = "cm-2"
            for axis, kind, unit, start, step in (
                (1, "XOFFSET", "arcsec", -40.0, 40.0),
                (2, "YOFFSET", "arcsec", -40.0, 40.0),
                (3, "DISTANCE", "kpc", 2.0, 2.0),
            ):
                cloud_hdu.header[f"CTYPE{axis}"] = kind
                cloud_hdu.header[f"CUNIT{axis}"] = unit
                cloud_hdu.header[f"CRPIX{axis}"] = 1.0
                cloud_hdu.header[f"CRVAL{axis}"] = start
                cloud_hdu.header[f"CDELT{axis}"] = step
            cloud_hdu.writeto(cloud_path, checksum=True)
            write_source_fits(
                base / "line.fits",
                SourceFluxFile(
                    np.array([0.0, 3600.0]),
                    np.array([]),
                    np.empty((1, 0)),
                    (),
                    np.empty((1, 0)),
                    np.array([5.35]),
                    np.array([[0.038]]),
                    ("test_line",),
                ),
            )
            scattering, absorption, grid = packaged_material_paths("2-10")
            config_path = base / "line.toml"
            config_path.write_text(
                f"""format_version = 1
[run]
name = "asymmetric_line"
packets = 512
chunk_size = 128
max_interactions = 16
seed = 314
[scene]
kind = "fits"
path = "asymmetric_cloud.fits"
source_distance_kpc = 10.0
[source]
file = "line.fits"
components = "lines"
[materials]
scattering = "{scattering.as_posix()}"
absorption = "{absorption.as_posix()}"
grid = "{grid.as_posix()}"
[observer]
time_edges_days = [0.0, 1.0, 3.0, 10.0, 60.0]
energy_edges_kev = [2.0, 5.0, 6.0, 10.0]
[output]
npz = "line.npz"
fits = "line_output.fits"
"""
            )
            config = load_run_config(config_path)
            _, cells, _, cloud, _, _ = build_run(config)
            self.assertAlmostEqual(float(cells.total_fluence), 136.8, places=4)
            np.testing.assert_allclose(np.asarray(cloud.delta_nh_cm2), columns)
            self.assertFalse(np.allclose(columns[:, 0, 0], columns[:, 1, 1]))
            report = run_configured_simulation(config)
            cloud_sha = hashlib.sha256(cloud_path.read_bytes()).hexdigest()
            self.assertEqual(report["cloud_sha256"], cloud_sha)
            self.assertTrue(report["numerical_passed"])
            with np.load(config.output_npz, allow_pickle=False) as archive:
                self.assertEqual(int(archive["output_schema_version"]), 7)
                self.assertEqual(str(archive["cloud_fits_sha256"]), cloud_sha)
                np.testing.assert_array_equal(archive["source_cell_kind"], [0])
                self.assertEqual(int(archive["history_count"]), 512)
                self.assertEqual(int(archive["transport_status_count"].sum()), 512)
                np.testing.assert_allclose(archive["cloud_delta_nh_cm2"], columns)
                self.assertEqual(archive["total_fluence"].shape, (4, 3, 3, 3))
            with fits.open(config.output_fits, checksum=True) as hdus:
                self.assertTrue(all(hdu.verify_checksum() == 1 for hdu in hdus))
                self.assertEqual(hdus[0].header["CLDFSHA"], cloud_sha)
                self.assertEqual(hdus["SOURCE"].data["KIND"].tolist(), [0])
                self.assertAlmostEqual(
                    float(hdus["SOURCE"].data["ENERGY_LOW"][0]), 5.35, places=5
                )
                self.assertEqual(
                    hdus["SOURCE"].data["ENERGY_LOW"].tolist(),
                    hdus["SOURCE"].data["ENERGY_HIGH"].tolist(),
                )
            audit = audit_configured_run(config)
            self.assertTrue(audit["all_passed"], audit["checks"])
            report_path = config.output_npz.with_suffix(".run_report.json")
            changed_report = json.loads(report_path.read_text())
            changed_report["cloud_sha256"] = "0" * 64
            report_path.write_text(json.dumps(changed_report))
            self.assertFalse(audit_configured_run(config)["checks"]["run_report"])

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
                    source.as_posix(),
                ).replace(
                    'fits = "../outputs/file_input_smoke.fits"',
                    f'fits = "{source.as_posix()}"',
                )
            )
            with self.assertRaisesRegex(ValueError, "overwrite input"):
                load_run_config(path)


if __name__ == "__main__":
    unittest.main()
