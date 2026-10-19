"""Independent fluence contracts, file failures, and read-only CLI checks."""

import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from dsh.build import build_run_plan
from dsh.command import bootstrap, main
from dsh.config import load_run_config
from dsh.run import publish_run
from tests.config_fixtures import ROOT, config_text, write_config


class ConfiguredInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="configured inputs ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        bootstrap("cpu")

    def example(self, name):
        text = (ROOT / "configs" / (name + ".toml")).read_text()
        text = text.replace('"../dsh/data/', '"' + (ROOT / "dsh/data").as_posix() + "/")
        text = text.replace('"data/', '"' + (ROOT / "configs/data").as_posix() + "/")
        return text.replace('"../outputs/' + name + '"', '"products"')

    def assert_cli_error(self, path, field):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(main(["check", str(path)]), 2)
        self.assertIn(field, stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertFalse((self.root / "products").exists())
        self.assertFalse((self.root / "results with spaces").exists())

    def test_all_source_modes_preserve_independently_integrated_fluence(self):
        expected = {
            "smoke_legacy": 0.038 * 3600,
            "smoke_wd01": 0.038 * 3600,
            "example_cloud_fits": 0.038 * 3600,
            "example_band_powerlaw": (0.038 + 0.019) * 43200,
            "example_history": (0.038 + 0.019) * 43200,
            "example_flux_table": (0.038 + 0.019 + 0.0095) * 43200,
            "example_exponential": 0.003 * 86400 + 0.035 * 43200 * (1 - math.exp(-2)),
        }
        for name, fluence in expected.items():
            with self.subTest(name=name):
                plan = build_run_plan(
                    load_run_config(write_config(self.root, self.example(name)))
                )
                self.assertAlmostEqual(
                    float(plan.source.total_fluence), fluence, delta=fluence * 2e-6
                )
                self.assertEqual(plan.summary["observer_shape_teyx"][:2], [6, 3])
        self.assertFalse((self.root / "products").exists())

    def test_check_never_calls_transport_or_publication_for_any_example(self):
        with (
            patch(
                "dsh.run.publish_run", side_effect=AssertionError("publication called")
            ),
            patch(
                "dsh.run.execute_plan", side_effect=AssertionError("transport called")
            ),
        ):
            for path in sorted((ROOT / "configs").glob("*.toml")):
                config = write_config(self.root, self.example(path.stem))
                with (
                    self.subTest(name=path.stem),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(main(["check", str(config)]), 0)
        self.assertFalse((self.root / "products").exists())

    def test_native_fits_example_preserves_column_and_physical_axes(self):
        plan = build_run_plan(
            load_run_config(write_config(self.root, self.example("example_cloud_fits")))
        )
        self.assertEqual(plan.cloud.delta_nh_cm2.shape, (4, 9, 9))
        np.testing.assert_allclose(
            np.asarray(plan.cloud.delta_nh_cm2).sum(axis=0), 4e20, rtol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(plan.cloud.n_h_cm3)
            * np.asarray(plan.cloud.radial_bin_width_cm)[:, None, None],
            plan.cloud.delta_nh_cm2,
            rtol=1e-6,
        )
        np.testing.assert_array_equal(plan.cloud.z_edges_kpc, [1.5, 2.5, 3.5, 4.5, 5.5])
        np.testing.assert_array_equal(plan.cloud.x_edges_arcsec, np.arange(-90, 91, 20))

    def test_flux_file_errors_are_field_specific_and_do_not_publish(self):
        table = self.root / "flux.txt"
        text = self.example("example_flux_table").replace(
            (ROOT / "configs/data/example_flux.txt").as_posix(), table.as_posix()
        )
        path = write_config(self.root, text)
        cases = (
            ("# empty\n", "source.file"),
            ("60000 0 .01 .01\n", "source.flux_columns"),
            ("60000 0 .01 .01 .01\n60000 0 .01 .01 .01\n", "MJD times"),
            ("59999 0 .01 .01 .01\n", "MJD times"),
            ("60000 0 .01 .01 .01\n60001 0 .01 .01 .01\n", "max_interval_days"),
            ("60000 0 nan .01 .01\n", "source.flux_columns"),
            ("60000 0 -.01 .01 .01\n", "source.flux_columns"),
            ("not numbers\n", "source.file"),
        )
        for contents, field in cases:
            table.write_text(contents)
            with self.subTest(contents=contents):
                self.assert_cli_error(path, field)

    def test_bad_history_contract_and_json_are_field_specific(self):
        history = self.root / "history.json"
        text = self.example("example_history").replace(
            (ROOT / "configs/data/example_history.json").as_posix(), history.as_posix()
        )
        path = write_config(self.root, text)
        for contents in ("{}", "{invalid", "[]"):
            history.write_text(contents)
            with self.subTest(contents=contents):
                self.assert_cli_error(path, "source.file")

    def test_missing_fits_header_and_invalid_geometry_are_field_specific(self):
        path = write_config(self.root, fits_scene=True)
        original = fits.getheader(self.root / "cloud.fits")
        data = fits.getdata(self.root / "cloud.fits", memmap=False)
        for key, value in (
            ("CRVAL1", None),
            ("CTYPE1", "RA---TAN"),
            ("BUNIT", "cm-3"),
            ("CRVAL3", 12),
        ):
            header = original.copy()
            if value is None:
                del header[key]
            else:
                header[key] = value
            fits.writeto(self.root / "cloud.fits", data, header, overwrite=True)
            with self.subTest(key=key):
                self.assert_cli_error(path, "scene.cloud_file")

    def test_malformed_material_metadata_identifies_materials(self):
        scattering = self.root / "scattering.npz"
        scattering.write_bytes(
            (ROOT / "dsh/data/newdust/mrn_rg_drude_v1.npz").read_bytes()
        )
        scattering.with_suffix(".json").write_text("{}")
        text = config_text().replace(
            (ROOT / "dsh/data/newdust/mrn_rg_drude_v1.npz").as_posix(),
            scattering.as_posix(),
        )
        self.assert_cli_error(write_config(self.root, text), "materials")

    def test_single_format_runs_publish_only_the_selected_physical_product(self):
        for selected, excluded in (("npz", "fits"), ("fits", "npz")):
            text = config_text().replace('["npz", "fits"]', f'["{selected}"]')
            text = text.replace("results with spaces", selected)
            path = write_config(self.root, text, fits_scene=True)
            config = load_run_config(path)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(publish_run(build_run_plan(config)), 0)
            self.assertTrue((config.output.directory / f"test.{selected}").is_file())
            self.assertFalse((config.output.directory / f"test.{excluded}").exists())
            self.assertTrue((config.output.directory / "complete.json").is_file())
