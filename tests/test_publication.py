"""Shared publication contracts, including interrupted and stale-input runs."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from dsh.config import load_run_config
from dsh.io.provenance import hash_files
from dsh.io.publication import (
    publish_file,
    publish_outputs,
    staged_run,
    write_run_records,
)
from tests.config_fixtures import config_text, write_config


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="publication contract ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_run_config(write_config(self.root))
        self.hashes = hash_files((self.config.config_path,))

    def records(self, stage):
        paths = []
        for suffix in ("npz", "fits"):
            path = stage / ("test." + suffix)
            path.write_bytes(suffix.encode())
            paths.append(path)
        log = stage / "run.log"
        log.write_text("log\n")
        report = {"status": "complete", "numerical_complete": True}
        return write_run_records(
            stage, self.config, self.hashes, {}, report, paths, log_path=log
        )

    def test_complete_marker_contains_exact_hashes_and_stays_last(self):
        observed = []

        def publish(source, target, overwrite):
            observed.append(source.name)
            publish_file(source, target, overwrite)

        with staged_run(self.config, self.hashes) as stage:
            paths = self.records(stage)
            publish_outputs(
                stage, self.config, self.hashes, paths, complete=True, publisher=publish
            )
        self.assertEqual(observed[-1], "complete.json")
        directory = self.config.output.directory
        marker = json.loads((directory / "complete.json").read_text())
        for name, digest in marker["output_sha256"].items():
            self.assertEqual(
                hashlib.sha256((directory / name).read_bytes()).hexdigest(), digest
            )
        self.assertNotIn("output_paths", marker)

    def test_interrupted_group_never_publishes_completion_marker(self):
        calls = []

        def publish(source, target, overwrite):
            calls.append(source.name)
            if len(calls) == 2:
                raise OSError("interrupted publication")
            publish_file(source, target, overwrite)

        with (
            self.assertRaisesRegex(OSError, "interrupted"),
            staged_run(self.config, self.hashes) as stage,
        ):
            publish_outputs(
                stage,
                self.config,
                self.hashes,
                self.records(stage),
                complete=True,
                publisher=publish,
            )
        self.assertTrue((self.config.output.directory / "test.npz").exists())
        self.assertFalse((self.config.output.directory / "complete.json").exists())

    def test_changed_input_prevents_any_publication(self):
        with (
            self.assertRaisesRegex(ValueError, "inputs changed"),
            staged_run(self.config, self.hashes) as stage,
        ):
            paths = self.records(stage)
            self.config.config_path.write_bytes(
                self.config.config_bytes + b"\n# modified\n"
            )
            publish_outputs(stage, self.config, self.hashes, paths, complete=True)
        self.assertEqual(list(self.config.output.directory.iterdir()), [])

    def test_failed_overwrite_invalidates_old_marker_and_cleans_staging(self):
        self.config = load_run_config(
            write_config(
                self.root,
                config_text() + "overwrite = true\n",
            )
        )
        self.hashes = hash_files((self.config.config_path,))
        directory = self.config.output.directory
        directory.mkdir(parents=True)
        (directory / "complete.json").write_text("{}")
        with (
            self.assertRaisesRegex(OSError, "failed"),
            staged_run(self.config, self.hashes),
        ):
            self.assertFalse((directory / "complete.json").exists())
            raise OSError("failed export")
        self.assertEqual(list(directory.iterdir()), [])

    def test_no_overwrite_race_preserves_the_existing_file(self):
        def publish(source, target, overwrite):
            target.write_bytes(b"existing run")
            publish_file(source, target, overwrite)

        with (
            self.assertRaises(FileExistsError),
            staged_run(self.config, self.hashes) as stage,
        ):
            publish_outputs(
                stage,
                self.config,
                self.hashes,
                self.records(stage),
                complete=True,
                publisher=publish,
            )
        self.assertEqual(
            (self.config.output.directory / "test.npz").read_bytes(), b"existing run"
        )
        self.assertFalse((self.config.output.directory / "complete.json").exists())
