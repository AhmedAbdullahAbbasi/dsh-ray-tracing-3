"""Profiling measures synchronized seeded results without production exports."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dsh.profiling import profile_config, result_fingerprint, synchronize
from tests.config_fixtures import ROOT, config_text, write_config


class ProfilingTests(unittest.TestCase):
    def test_import_is_host_only_and_synchronization_waits_for_arrays(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                'import sys; import dsh.profiling; assert not {"jax","numpy","astropy","matplotlib"} & set(sys.modules)',
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        class Pending:
            waited = False

            def block_until_ready(self):
                self.waited = True

        pending = Pending()
        synchronize((pending, [None]))
        self.assertTrue(pending.waited)

    def test_fingerprint_tracks_shape_dtype_and_values(self):
        original = result_fingerprint((np.array([1.0, 2.0]), 3))
        self.assertNotEqual(original, result_fingerprint((np.array([1.0, 3.0]), 3)))
        self.assertNotEqual(original, result_fingerprint((np.array([[1.0, 2.0]]), 3)))
        self.assertNotEqual(
            original, result_fingerprint((np.array([1.0, 2.0], dtype=np.float32), 3))
        )

    def test_real_profile_repeats_exactly_and_creates_no_production_products(self):
        with tempfile.TemporaryDirectory(prefix="profile paths ") as temp:
            root = Path(temp)
            path = write_config(root)
            report = profile_config(path, root / "profile", repeats=1)
            self.assertEqual(report["status"], "passed", report.get("error"))
            self.assertTrue(report["seeded_repeats_exact"])
            self.assertGreater(report["first_execution_seconds"], 0)
            self.assertEqual(len(report["repeat_execution_seconds"]), 1)
            self.assertFalse((root / "results with spaces").exists())
            self.assertTrue((root / "profile/calls.pstats").is_file())
            self.assertFalse(
                json.loads((root / "profile/profile.json").read_text())[
                    "production_products_published"
                ]
            )

    def test_incomplete_profile_is_rejected_without_production_marker(self):
        with tempfile.TemporaryDirectory(prefix="profile capped ") as temp:
            root = Path(temp)
            text = (
                config_text()
                .replace("packets = 37", "packets = 128")
                .replace("max_interactions = 8", "max_interactions = 1")
            )
            report = profile_config(
                write_config(root, text), root / "profile", repeats=1
            )
            self.assertEqual(report["status"], "failed")
            self.assertIn("incomplete", report["error"])
            self.assertFalse(report.get("seeded_repeats_exact", False))

    def test_profile_cannot_use_the_production_output_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = write_config(root)
            with self.assertRaisesRegex(ValueError, "outside the production"):
                profile_config(
                    path, root / "results with spaces" / "profile", repeats=1
                )
            self.assertFalse((root / "results with spaces").exists())

    def test_existing_profile_directory_is_not_overwritten(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            self.assertRaisesRegex(ValueError, "must be new"),
        ):
            profile_config("not-a-config.toml", Path(temp), repeats=1)
