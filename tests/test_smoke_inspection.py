"""A real product fixture and counterexamples for the independent smoke gate."""

import contextlib
import io
import json
import platform
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dsh.build import build_run_plan
from dsh.command import bootstrap
from dsh.config import load_run_config
from dsh.run import publish_run
from scripts.validate_configured_smoke import (
    ValidationError,
    digest,
    inspect_complete,
    prepare_case,
    run_smoke,
)
from tests.config_fixtures import ROOT, write_config


class SmokeInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory(prefix="smoke fixture ")
        cls.addClassCleanup(cls.fixture.cleanup)
        path = write_config(cls.fixture.name, fits_scene=True)
        bootstrap("cpu")
        cls.config = load_run_config(path)
        with contextlib.redirect_stdout(io.StringIO()):
            if publish_run(build_run_plan(cls.config)) != 0:
                raise AssertionError("real product fixture did not complete")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="smoke inspection ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.products = self.root / "products"
        shutil.copytree(self.config.output.directory, self.products)

    def test_real_complete_products_pass(self):
        result = inspect_complete(self.products)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["packets"], 37)
        self.assertEqual(result["shape_time_energy_y_x"], [3, 3, 3, 3])

    def test_changed_output_fails_hash_gate(self):
        with (self.products / "run.log").open("a") as stream:
            stream.write("corruption")
        with self.assertRaisesRegex(ValidationError, "output hash mismatch"):
            inspect_complete(self.products)

    def test_rehashed_incomplete_flag_still_fails_physical_gate(self):
        path = self.products / "test.npz"
        with np.load(path, allow_pickle=False) as original:
            arrays = {key: original[key] for key in original.files}
        arrays["numerical_complete"] = np.asarray(False)
        np.savez_compressed(path, **arrays)
        marker = self.products / "complete.json"
        complete = json.loads(marker.read_text())
        complete["output_sha256"][path.name] = digest(path)
        marker.write_text(json.dumps(complete))
        with self.assertRaisesRegex(ValidationError, "completion flags differ"):
            inspect_complete(self.products)

    def test_generated_config_is_explicit_and_resolves_in_a_path_with_spaces(self):
        for name in ("smoke_legacy", "smoke_wd01", "capped_legacy"):
            path, products = prepare_case(
                ROOT, self.root, name, capped=name == "capped_legacy"
            )
            config = load_run_config(path)
            self.assertEqual(config.output.directory, products)
            self.assertTrue(config.plots.enabled)
            self.assertTrue(config.plots.save_scored_event_map)
            self.assertFalse(products.exists())
            if name == "capped_legacy":
                self.assertEqual(config.execution.max_interactions, 1)
                self.assertEqual(config.execution.packets, 128)

    def test_failed_check_still_produces_an_honest_report(self):
        with (
            patch("scripts.validate_configured_smoke.invoke", return_value=2),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(run_smoke(ROOT, self.root), 1)
        summary = json.loads(next(self.root.glob("*/smoke_summary.json")).read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["windows_execution"], platform.system() == "Windows")
        self.assertFalse(summary["scientific_convergence_claimed"])
        self.assertTrue(next(self.root.glob("*/smoke_report.zip")).is_file())
