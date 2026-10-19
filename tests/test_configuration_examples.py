"""Portable example preparation preserves settings and reports failed checks."""

import contextlib
import io
import json
import platform
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dsh.config import load_run_config
from scripts.validate_configuration_examples import (
    EXAMPLES,
    prepare_example,
    run_validation,
)
from tests.config_fixtures import ROOT


class ConfigurationExampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="example validation ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_prepared_examples_preserve_physics_and_use_fresh_destinations(self):
        for name in EXAMPLES:
            path, products = prepare_example(ROOT, self.root, name)
            config = load_run_config(path)
            self.assertEqual(config.output.directory, products)
            self.assertEqual(config.execution.packets, 64)
            self.assertTrue(config.plots.enabled)
            self.assertTrue(config.plots.save_scored_event_map)
            self.assertFalse(products.exists())
            original_text = (ROOT / "configs" / (name + ".toml")).read_text()
            self.assertEqual(config.name, name)
            self.assertIn(f'kind = "{config.source.kind}"', original_text)
            if config.source.file:
                self.assertEqual(config.source.file.parent, ROOT / "configs/data")
            if config.scene.cloud_file:
                self.assertEqual(config.scene.cloud_file.parent, ROOT / "configs/data")

    def assert_report(self):
        summary = json.loads(next(self.root.glob("*/stage2_summary.json")).read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["windows_execution"], platform.system() == "Windows")
        self.assertFalse(summary["scientific_convergence_claimed"])
        with zipfile.ZipFile(next(self.root.glob("*/stage2_report.zip"))) as report:
            self.assertIsNone(report.testzip())
            self.assertIn("stage2_summary.json", report.namelist())
            self.assertIn("tests.log", report.namelist())
        return summary

    def test_failed_tests_stop_before_running_examples_and_pack_report(self):
        with (
            patch(
                "scripts.validate_configuration_examples.subprocess.run",
                return_value=SimpleNamespace(returncode=1),
            ),
            patch(
                "scripts.validate_configuration_examples.invoke",
                side_effect=AssertionError("example ran after failed tests"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(run_validation(ROOT, self.root), 1)
        summary = self.assert_report()
        self.assertEqual(summary["tests_exit"], 1)
        self.assertEqual(summary["cases"], {})

    def test_failed_example_check_does_not_run_transport_and_packs_report(self):
        with (
            patch(
                "scripts.validate_configuration_examples.subprocess.run",
                return_value=SimpleNamespace(returncode=0),
            ),
            patch(
                "scripts.validate_configuration_examples.invoke", return_value=2
            ) as invoke,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(run_validation(ROOT, self.root), 1)
        summary = self.assert_report()
        self.assertEqual(summary["cases"]["smoke_legacy"]["check_exit"], 2)
        self.assertEqual(invoke.call_count, 1)
        self.assertEqual(invoke.call_args.args[1], "check")
