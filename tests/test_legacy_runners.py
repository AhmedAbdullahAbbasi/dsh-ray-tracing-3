"""Legacy translation, physical selections and shared publication contracts."""

import contextlib
import errno
import io
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dsh.build import build_run_plan
from dsh.cli import main as cli_main
from dsh.cli import parse_arguments
from dsh.command import bootstrap
from dsh.compat import _toml, cli_translation, root_translation, run_root
from dsh.config import ConfigError, load_run_config
from dsh.plots import plot_arrays
from dsh.run import _publish
from tests.config_fixtures import ROOT, config_text, write_config


class LegacyRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="legacy runners ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        bootstrap("cpu")

    def settings(self, **changes):
        values = {
            key: value
            for key, value in runpy.run_path(str(ROOT / "sim_config.py")).items()
            if key.isupper()
        }
        values.update(
            DEVICE="cpu",
            PACKETS=37,
            CHUNK_SIZE=16,
            MAX_INTERACTIONS=32,
            CLOUD_FITS_PATH=str(ROOT / "configs/data/example_cloud.fits"),
            SOURCE_MODEL="custom",
            ENERGY_EDGES_KEV=None,
            CUSTOM_TIME_EDGES_DAYS=(0.0, 0.01),
            CUSTOM_BAND_FLUXES=((0.02, 0.012, 0.006),),
            ARRIVAL_DAYS=2.5,
            TIME_BIN_DAYS=1.0,
            IMAGE_EPOCH_DAYS=None,
            IMAGE_ENERGY_KEV=None,
            OUTPUT_PNG=str(self.root / "images/intensity.png"),
            OUTPUT_COUNT_PNG=str(self.root / "counts/events.png"),
            OUTPUT_COUNT_NPY=str(self.root / "maps/events.npy"),
            PROGRESS_LOG=str(self.root / "logs/progress.log"),
            IMAGE_DPI=60,
        )
        values.update(changes)
        path = self.root / "settings.py"
        path.write_text(
            "\n".join(key + " = " + repr(value) for key, value in values.items()),
            encoding="utf-8",
        )
        return SimpleNamespace(**runpy.run_path(str(path)))

    def plan(self, data, mapping=None, extra=()):
        path = self.root / "translated.toml"
        path.write_text(_toml(data))
        return build_run_plan(
            load_run_config(path), publication_paths=mapping, extra_inputs=extra
        )

    def test_entry_point_imports_do_not_load_settings_or_numerical_libraries(self):
        script = "import sys; import dsh.cli; import run_simulation; assert not {'jax','numpy','astropy','matplotlib','sim_config'} & set(sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_keeps_equal_width_arrival_bins_and_separate_output_paths(self):
        args = parse_arguments(
            [
                "--arrival-days",
                "2.5",
                "--time-bin-days",
                "1",
                "--output",
                str(self.root / "custom name.npz"),
                "--fits-output",
                str(self.root / "other folder/cube.fits"),
            ]
        )
        data, mapping, _ = cli_translation(args, "cpu")
        plan = self.plan(data, mapping)
        np.testing.assert_array_equal(
            plan.bins.arrival_time_edges_s,
            np.linspace(0, 2.5 * 86400, 4).astype(np.float32),
        )
        self.assertEqual(
            mapping[plan.config.name + ".npz"], self.root / "custom name.npz"
        )
        self.assertEqual(
            mapping[plan.config.name + ".fits"], self.root / "other folder/cube.fits"
        )
        self.assertFalse(plan.config.output.directory.exists())

    def test_disc_keeps_selected_density_radius_and_radial_widths(self):
        settings = self.settings(
            CLOUD_FITS_PATH=None,
            GRID_X_ARCSEC=(-20.0, 30.0, 10.0),
            GRID_Y_ARCSEC=(-20.0, 30.0, 10.0),
            GRID_Z_KPC=(2.0, 5.0, 1.0),
            DISC_CENTER_Z_KPC=3.0,
            DISC_THICKNESS_KPC=1.0,
            DISC_RADIUS_ARCSEC=15.0,
            DISC_N_H_CM3=2.0,
        )
        data, mapping, _, extra, _ = root_translation(settings)
        plan = self.plan(data, mapping, extra)
        expected = np.zeros((3, 5, 5), dtype=np.float32)
        expected[1, 1:4, 1:4] = 2
        np.testing.assert_allclose(plan.cloud.n_h_cm3, expected, rtol=1e-6)
        np.testing.assert_allclose(
            plan.cloud.delta_nh_cm2,
            expected * np.float32(3.0856775814913673e21),
            rtol=1e-6,
        )
        self.assertIn(str(Path(settings.__file__)), plan.input_hashes)

    def test_flux_translation_records_epoch_cadence_columns_and_unknown_scale(self):
        flux = self.root / "flux.txt"
        np.savetxt(
            flux,
            [
                [60000, 999, 0.02, 0.01],
                [60000.5, 998, 0.01, 0.005],
                [60001.5, 997, 0.005, 0.0025],
            ],
        )
        settings = self.settings(
            SOURCE_MODEL="flux-file",
            FLUX_FILE=str(flux),
            ENERGY_EDGES_KEV=(2.25, 3.15, 6.0),
            FLUX_COLUMNS=(2, 3),
            ARRIVAL_DAYS=5.0,
        )
        data, mapping, metadata, extra, _ = root_translation(settings)
        plan = self.plan(data, mapping, extra)
        np.testing.assert_array_equal(
            plan.source.time_edges_s, np.array([0, 0.5, 1.5, 2.5]) * 86400
        )
        np.testing.assert_array_equal(
            plan.source.band_flux,
            np.array([[0.02, 0.01], [0.01, 0.005], [0.005, 0.0025]], dtype=np.float32),
        )
        self.assertEqual(data["source"]["flux_columns"], [3, 4])
        self.assertEqual(data["source"]["time_scale"], "unspecified")
        self.assertEqual(plan.summary["source_time_scale"], "unspecified")
        self.assertIn("last measured cadence", metadata["legacy_flux_policy"])

    def test_representative_flux_file_keeps_three_discrete_energies(self):
        flux = self.root / "flux.txt"
        np.savetxt(flux, [[60000, 0.02, 0.012, 0.006], [60001, 0.01, 0.006, 0.003]])
        settings = self.settings(
            SOURCE_MODEL="flux-file", FLUX_FILE=str(flux), ARRIVAL_DAYS=3.0
        )
        data, _, _, _, _ = root_translation(settings)
        plan = self.plan(data)
        self.assertIsNone(plan.source.energy_edges_kev)
        np.testing.assert_array_equal(
            plan.source.effective_energy_kev,
            np.array([3.3, 4.9, 6.9], dtype=np.float32),
        )
        self.assertEqual(data["source"]["flux_columns"], [2, 3, 4])

    def test_root_check_never_executes_transport_or_creates_output_directories(self):
        settings = self.settings()
        with (
            patch(
                "dsh.run.publish_run", side_effect=AssertionError("publication called")
            ),
            patch(
                "dsh.run.execute_plan", side_effect=AssertionError("transport called")
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_root(settings, check=True), 0)
        for directory in ("images", "counts", "maps", "logs"):
            self.assertFalse((self.root / directory).exists())

    def test_cli_real_success_has_mapped_products_and_completion_manifest(self):
        output, fits_output = (
            self.root / "out with spaces/result.npz",
            self.root / "other directory/result.fits",
        )
        args = [
            "--cloud-fits",
            str(ROOT / "configs/data/example_cloud.fits"),
            "--packets",
            "37",
            "--chunk-size",
            "16",
            "--max-interactions",
            "32",
            "--output",
            str(output),
            "--fits-output",
            str(fits_output),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli_main(args), 0)
        self.assertTrue(output.is_file())
        self.assertTrue(fits_output.is_file())
        complete = json.loads((output.parent / "result.run/complete.json").read_text())
        self.assertEqual(Path(complete["output_paths"]["result.fits"]), fits_output)
        with np.load(output, allow_pickle=False) as data:
            self.assertTrue(bool(data["numerical_complete"]))
            self.assertEqual(int(data["history_count"]), 37)

    def test_cli_capped_run_returns_three_and_withholds_both_physical_products(self):
        output = self.root / "capped.npz"
        with contextlib.redirect_stdout(io.StringIO()):
            code = cli_main(
                [
                    "--packets",
                    "128",
                    "--chunk-size",
                    "32",
                    "--max-interactions",
                    "1",
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(code, 3)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".fits").exists())
        self.assertFalse((self.root / "capped.run/complete.json").exists())
        report = json.loads((self.root / "capped.run/report.json").read_text())
        self.assertGreater(report["numerical_failure_histories"], 0)

    def test_mapped_output_cannot_overwrite_any_input(self):
        path = write_config(self.root)
        config = load_run_config(path)
        with self.assertRaisesRegex(ConfigError, "overwrite input"):
            build_run_plan(
                config, publication_paths={"test.npz": config.materials.scattering_file}
            )
        self.assertFalse(config.output.directory.exists())

    def test_mapped_products_cannot_alias_or_redirect_the_completion_marker(self):
        config = load_run_config(write_config(self.root))
        with self.assertRaisesRegex(ConfigError, "collide"):
            build_run_plan(
                config,
                publication_paths={
                    "test.npz": self.root / "same",
                    "test.fits": self.root / "same",
                },
            )
        with self.assertRaisesRegex(ConfigError, "provenance"):
            build_run_plan(
                config, publication_paths={"complete.json": self.root / "redirect"}
            )

    def test_cross_device_publication_stages_a_complete_file_at_destination(self):
        source, target = self.root / "source", self.root / "other/file"
        source.write_bytes(b"complete bytes")
        import os

        original = os.replace
        calls = []

        def replace(src, dst):
            calls.append((Path(src), Path(dst)))
            if Path(src) == source:
                raise OSError(errno.EXDEV, "cross-device link")
            return original(src, dst)

        with patch("dsh.run.os.replace", side_effect=replace):
            _publish(source, target, True)
        self.assertEqual(target.read_bytes(), b"complete bytes")
        self.assertFalse(source.exists())
        self.assertEqual(len(calls), 2)
        self.assertEqual(list(target.parent.iterdir()), [target])

    def test_selected_epoch_band_and_nonuniform_pixel_area_have_correct_units(self):
        config = load_run_config(write_config(self.root))
        from dataclasses import replace

        plots = replace(
            config.plots,
            epoch_days=(1.0, 2.0),
            energy_selection="nearest",
            energy_kev=5.0,
            density_per_kev=True,
        )
        config = replace(config, plots=plots)
        bins = SimpleNamespace(
            arrival_time_edges_s=np.array([0, 1, 2, 3]) * 86400,
            energy_edges_kev=np.array([2, 4, 6]),
            sky_x_edges_arcsec=np.array([0, 1, 3]),
            sky_y_edges_arcsec=np.array([0, 1, 3]),
        )
        areas = np.array([[1, 2], [2, 4]])
        image = np.zeros((3, 2, 2, 2))
        image[1, 1] = areas * 86400 * 2
        events = np.zeros_like(image, dtype=int)
        events[1, 1] = 7
        result = SimpleNamespace(
            products=SimpleNamespace(total_fluence=image, event_count=events)
        )
        selected, counts = plot_arrays(
            SimpleNamespace(config=config, bins=bins), result
        )
        np.testing.assert_array_equal(selected, np.ones((2, 2)))
        np.testing.assert_array_equal(counts, np.full((2, 2), 7))

    def test_new_mode_conflicts_fail_with_field_specific_errors(self):
        for text, field in (
            (
                config_text() + '\n[plots]\nenergy_selection="nearest"\n',
                "plots.energy_kev",
            ),
            (config_text() + '\n[plots]\ncrop="centered"\n', "plots.fov_arcsec"),
            (
                config_text().replace("[scene]", "[scene]\ndisc_radius_arcsec=10"),
                "scene",
            ),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ConfigError, field):
                load_run_config(write_config(self.root, text))
