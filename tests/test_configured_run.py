"""Configured adapters, exact seeded parity, quality and product publication."""

import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from jax import random

from dsh.build import build_run_plan
from dsh.command import bootstrap, main
from dsh.config import load_run_config
from dsh.config.schema import MaterialConfig
from dsh.inputs.material import load_material
from dsh.inputs.source import load_source
from dsh.physics.materials import load_2_10_material_tables
from dsh.physics.wd01_ism import load_wd01_ism
from dsh.pipeline import run_tabulated_source_to_observer_chunked
from dsh.plots import plot_arrays
from dsh.quality import evaluate_quality
from dsh.run import execute_plan, publish_run
from dsh.sources.histories import build_spectral_history_source
from dsh.sources.models import build_tabulated_band_source
from tests.config_fixtures import ROOT, write_config


class ConfiguredRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="configured run ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = write_config(self.root, fits_scene=True)
        self.config = load_run_config(self.path)
        bootstrap("cpu")

    def plan(self):
        return build_run_plan(self.config)

    def test_seeded_execution_matches_explicit_existing_builder_with_short_tail(self):
        plan = self.plan()
        reference_source = build_tabulated_band_source(
            [0, 864], [[0.02, 0.012, 0.006]], [3.3, 4.9, 6.9]
        )
        for actual, reference in zip(plan.source, reference_source, strict=True):
            if actual is not None:
                np.testing.assert_array_equal(actual, reference)
        reference = run_tabulated_source_to_observer_chunked(
            random.PRNGKey(2026),
            reference_source,
            plan.launch,
            plan.cloud,
            plan.physics,
            plan.bins,
            total_packets=37,
            chunk_size=16,
            max_interactions=8,
        )
        actual = execute_plan(plan)
        for actual_group, reference_group in zip(actual, reference, strict=True):
            for actual_array, reference_array in zip(
                actual_group, reference_group, strict=True
            ):
                np.testing.assert_array_equal(actual_array, reference_array)
        self.assertEqual(int(actual.products.history_count), 37)

    def test_flux_columns_are_full_table_indices_and_band_flux_is_not_resplit(self):
        file = self.root / "flux.txt"
        np.savetxt(file, [[57625, 900, 2, 1000, 3], [57626, 901, 4, 1001, 5]])
        source_config = dataclasses.replace(
            self.config.source,
            kind="flux_table",
            file=file,
            time_column=1,
            flux_columns=(3, 5),
            time_scale="TT",
            epoch_mjd=57625,
            last_interval_days=0.5,
            max_interval_days=1,
            energy_edges_kev=(2, 4, 10),
        )
        source, metadata = load_source(source_config)
        np.testing.assert_array_equal(source.time_edges_s, [0, 86400, 129600])
        np.testing.assert_array_equal(source.band_flux, [[2, 3], [4, 5]])
        self.assertAlmostEqual(
            float(source.total_fluence), 5 * 86400 + 9 * 43200, delta=1
        )
        self.assertEqual(metadata["source_time_scale"], "TT")
        np.savetxt(file, [[57625, 0, 2, 0, 3]])
        one, _ = load_source(source_config)
        np.testing.assert_array_equal(one.time_edges_s, [0, 43200])
        np.savetxt(file, [[57625, 0, 2, 0, 3], [57627, 0, 4, 0, 5]])
        with self.assertRaisesRegex(ValueError, "max_interval_days"):
            load_source(source_config)

    def test_pair_and_declared_adapters_preserve_existing_physics_arrays(self):
        data = ROOT / "dsh/data"
        choices = [
            (
                MaterialConfig(
                    "table_pair",
                    scattering_file=data / "scattering/rg_drude_2_10_v2.npz",
                    absorption_file=data / "absorption/tbabs_wilm_vern_2_10_v2.npz",
                    grid_file=data / "materials/material_grid_2_10_v2.json",
                ),
                load_2_10_material_tables()[-1],
            ),
            (
                MaterialConfig(
                    "declared_bundle",
                    directory=data / "materials/wd01_wilm_c50_si90_v1",
                ),
                load_wd01_ism(data / "materials/wd01_wilm_c50_si90_v1")[-1],
            ),
        ]
        for choice, reference in choices:
            actual = load_material(choice)[2]
            for a, b in zip(actual, reference, strict=True):
                if a is not None:
                    np.testing.assert_array_equal(a, b)

    def test_history_adapter_preserves_epoch_slopes_and_band_flux(self):
        path = self.root / "history.json"
        payload = {
            "schema_version": 1,
            "flux_convention": "unabsorbed observer-equivalent",
            "units": {"time": "s", "energy": "keV", "band_flux": "ph cm^-2 s^-1"},
            "epoch": "MJD 57625",
            "time_scale": "TT",
            "time_edges_s": [0, 10, 20],
            "energy_edges_kev": [2, 4, 10],
            "band_flux": [[1, 2], [3, 4]],
            "photon_index": [[1.7, 2.1], [2.0, 1.8]],
        }
        path.write_text(json.dumps(payload))
        actual, metadata = load_source(
            dataclasses.replace(self.config.source, kind="spectral_history", file=path)
        )
        expected = build_spectral_history_source(
            payload["time_edges_s"],
            payload["energy_edges_kev"],
            payload["band_flux"],
            payload["photon_index"],
        )
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(metadata["source_time_scale"], "TT")
        self.assertEqual(metadata["source_epoch"], "MJD 57625")

    def test_runtime_precision_errors_do_not_clip_to_valid_endpoints(self):
        source_config = dataclasses.replace(
            self.config.source,
            kind="powerlaw",
            energy_edges_kev=(2, 10),
            photon_flux=(0.038,),
            photon_index=-100,
        )
        with self.assertRaisesRegex(ValueError, "sampling precision"):
            load_source(source_config)
        with self.assertRaisesRegex(ValueError, "restart"):
            bootstrap("gpu")

    def test_check_does_not_run_transport_or_create_output(self):
        with patch(
            "dsh.run.execute_plan", side_effect=AssertionError("transport called")
        ):
            self.assertEqual(main(["check", str(self.path)]), 0)
        self.assertFalse(self.config.output.directory.exists())

    def test_product_roundtrip_provenance_and_completion_hashes(self):
        plan = self.plan()
        self.assertEqual(publish_run(plan), 0)
        output = self.config.output.directory
        manifest = json.loads((output / "complete.json").read_text())
        for name, digest in manifest["output_sha256"].items():
            self.assertEqual(
                hashlib.sha256((output / name).read_bytes()).hexdigest(), digest
            )
        self.assertEqual(
            (output / "original.toml").read_bytes(), self.config.config_bytes
        )
        with (
            np.load(output / "test.npz") as data,
            fits.open(output / "test.fits") as hdul,
        ):
            self.assertEqual(str(data["config_sha256"]), self.config.config_sha256)
            self.assertTrue(bool(data["numerical_complete"]))
            self.assertEqual(hdul[0].header["CFGSHA"], self.config.config_sha256)
            self.assertTrue(hdul[0].header["NUMCOMP"])
            np.testing.assert_allclose(
                hdul[0].data,
                data["total_fluence"].sum(axis=(0, 1)),
                atol=1e-20,
                rtol=1e-6,
            )
        with self.assertRaisesRegex(ValueError, "destination exists"):
            publish_run(plan)

    def test_incomplete_transport_is_nonzero_and_never_has_success_marker(self):
        plan = self.plan()
        result = execute_plan(plan)
        counts = (
            jnp.zeros_like(result.diagnostics.transport_status_count)
            .at[1]
            .set(36)
            .at[4]
            .set(1)
        )
        result = result._replace(
            diagnostics=result.diagnostics._replace(transport_status_count=counts)
        )
        self.assertFalse(evaluate_quality(result, 37)["numerical_complete"])
        with patch("dsh.run.execute_plan", return_value=result):
            self.assertEqual(publish_run(plan), 3)
        output = self.config.output.directory
        self.assertFalse((output / "complete.json").exists())
        self.assertFalse((output / "test.npz").exists())
        self.assertEqual(
            json.loads((output / "report.json").read_text())["status"], "incomplete"
        )

    def test_diagnostic_incomplete_products_are_explicitly_labeled(self):
        config = dataclasses.replace(
            self.config,
            quality=dataclasses.replace(self.config.quality, allow_incomplete=True),
        )
        plan = build_run_plan(config)
        result = execute_plan(plan)
        counts = jnp.zeros_like(result.diagnostics.transport_status_count).at[4].set(37)
        result = result._replace(
            diagnostics=result.diagnostics._replace(transport_status_count=counts)
        )
        with patch("dsh.run.execute_plan", return_value=result):
            self.assertEqual(publish_run(plan), 3)
        with np.load(config.output.directory / "diagnostic_test.npz") as data:
            self.assertFalse(bool(data["numerical_complete"]))
            self.assertTrue(bool(data["diagnostic_only"]))
        self.assertFalse((config.output.directory / "complete.json").exists())

    def test_export_failure_does_not_publish_partial_group(self):
        plan = self.plan()
        with (
            patch(
                "dsh.run.write_ideal_observer_fits", side_effect=OSError("interrupted")
            ),
            self.assertRaisesRegex(OSError, "interrupted"),
        ):
            publish_run(plan)
        self.assertFalse((self.config.output.directory / "test.npz").exists())
        self.assertFalse((self.config.output.directory / "complete.json").exists())

    def test_changed_input_is_rejected_before_transport(self):
        plan = self.plan()
        self.path.write_text(self.path.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            execute_plan(plan)

    def test_nonuniform_pixel_intensity_is_constant(self):
        bins = SimpleNamespace(
            sky_x_edges_arcsec=np.array([0, 1, 3]),
            sky_y_edges_arcsec=np.array([0, 1, 3]),
            energy_edges_kev=np.array([2, 4]),
            arrival_time_edges_s=np.array([0, 3]),
        )
        area = np.array([[1, 2], [2, 4]])
        result = SimpleNamespace(
            products=SimpleNamespace(
                total_fluence=(area * 6)[None, None],
                event_count=np.ones((1, 1, 2, 2), dtype=int),
            )
        )
        plan = SimpleNamespace(bins=bins, config=self.config)
        intensity, events = plot_arrays(plan, result)
        np.testing.assert_array_equal(intensity, np.ones((2, 2)))
        np.testing.assert_array_equal(events, np.ones((2, 2)))
