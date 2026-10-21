"""Configured response parity, grouping, archive compatibility and quality gates."""

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dsh.command import bootstrap, main
from dsh.config import ConfigError, load_run_config
from dsh.observer.response import (
    SpectralResponse,
    load_spectral_response,
    write_spectral_response,
)
from dsh.observer.response_transport import collect_spectral_response
from dsh.response_run import build_response_plan, publish_response_run
from tests.config_fixtures import ROOT


class ResponseWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="response workflow ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        bootstrap("cpu")

    def collection(self, **changes):
        text = (ROOT / "configs/response/collect.toml").read_text()
        text = text.replace("../../dsh/data/", (ROOT / "dsh/data").as_posix() + "/")
        text = text.replace(
            "../../outputs/response_collect", (self.root / "collect").as_posix()
        )
        text = text.replace("packets = 64", "packets = 33").replace(
            "chunk_size = 32", "chunk_size = 16"
        )
        for old, new in changes.items():
            text = text.replace(old, new)
        path = self.root / "collect.toml"
        path.write_text(text)
        return load_run_config(path)

    def archive(self):
        response = SpectralResponse(
            energy_kev=np.array([3.0, 3.0]),
            delay_s=np.zeros(2),
            sky_x_arcsec=np.full(2, 0.5),
            sky_y_arcsec=np.zeros(2),
            scattering_order=np.array([1, 2]),
            history_index=np.array([0, 0]),
            weight_kev=np.full(2, 0.5),
            n_histories=2,
            energy_support_kev=(2.0, 4.0),
            provenance=dict.fromkeys(
                ("material_sha256", "cloud_sha256", "launch_sha256", "runtime_sha256"),
                "a" * 64,
            ),
        )
        path = self.root / "old archive.npz"
        write_spectral_response(path, response)
        return path

    def folding(self, archive=None, extra=""):
        archive = archive or self.archive()
        text = f"""config_version=1
[run]
name="fold"
workflow="response_fold"
[source]
kind="powerlaw"
time_edges_days=[0.0,0.000011574074074074073]
energy_edges_kev=[2.0,4.0]
photon_flux=[2.0]
photon_index=0.0
[response]
file={json.dumps(archive.as_posix())}
{extra}
[observer]
arrival_time_edges_days=[0.0,0.000011574074074074073]
energy_edges_kev=[2.0,4.0]
radial_edges_arcsec=[0.0,1.0]
[output]
directory={json.dumps((self.root / "fold").as_posix())}
"""
        path = self.root / "fold.toml"
        path.write_text(text)
        return load_run_config(path)

    def assert_hashes(self, directory):
        marker = json.loads((directory / "complete.json").read_text())
        for name, digest in marker["output_sha256"].items():
            self.assertEqual(
                hashlib.sha256((directory / name).read_bytes()).hexdigest(), digest
            )

    def test_collection_matches_original_collector_including_one_packet_remainder(self):
        plan = build_response_plan(self.collection())
        expected = collect_spectral_response(
            plan.source,
            plan.cloud,
            plan.physics,
            plan.launch,
            n_packets=33,
            chunk_size=16,
            max_interactions=32,
            seed=2026,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(publish_response_run(plan), 0)
        actual = load_spectral_response(
            plan.config.output.directory / "response_collect.npz"
        )
        for name in (
            "energy_kev",
            "delay_s",
            "sky_x_arcsec",
            "sky_y_arcsec",
            "history_index",
            "scattering_order",
            "weight_kev",
        ):
            np.testing.assert_array_equal(
                getattr(actual, name), getattr(expected, name)
            )
        self.assertEqual(actual.n_histories, 33)
        self.assertEqual(sum(actual.provenance["transport_status_counts"]), 33)
        self.assert_hashes(plan.config.output.directory)

    def test_legacy_schema_one_archive_folds_with_photon_grouped_covariance(self):
        config = self.folding()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["run", str(config.config_path)]), 0)
        with np.load(config.output.directory / "fold.npz", allow_pickle=False) as data:
            np.testing.assert_allclose(data["fluence"], [[[1.0]]], rtol=1e-6)
            np.testing.assert_allclose(data["standard_error"], [[[1.0]]], rtol=1e-6)
            np.testing.assert_allclose(data["covariance"], [[1.0]], rtol=1e-6)
        self.assert_hashes(config.output.directory)

    def test_order_selection_and_no_covariance(self):
        config = self.folding(extra="scattering_orders=[1]\ncovariance=false")
        plan = build_response_plan(config)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(publish_response_run(plan), 0)
        with np.load(config.output.directory / "fold.npz") as data:
            self.assertNotIn("covariance", data.files)
            np.testing.assert_allclose(data["fluence"], [[[0.5]]], rtol=1e-6)
            np.testing.assert_allclose(data["standard_error"], [[[0.5]]], rtol=1e-6)

    def test_check_has_no_transport_or_products(self):
        config = self.collection()
        with (
            patch(
                "dsh.response_run.collect_spectral_response",
                side_effect=AssertionError("transport"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(["check", str(config.config_path)]), 0)
        self.assertFalse(config.output.directory.exists())

    def test_cap_quality_failure_withholds_production_response(self):
        config = self.collection(
            **{
                "max_interactions = 32": "max_interactions = 1",
                "packets = 33": "packets = 128",
            }
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["run", str(config.config_path)]), 3)
        self.assertFalse((config.output.directory / "response_collect.npz").exists())
        self.assertFalse((config.output.directory / "complete.json").exists())
        self.assertFalse(
            json.loads((config.output.directory / "report.json").read_text())[
                "numerical_complete"
            ]
        )

    def test_bad_proposal_rejected_before_collection(self):
        config = self.collection(
            **{
                "photon_flux = [0.038]": "photon_flux = [0.038, 0.02]",
                "time_edges_days = [0.0, 0.041666666666666664]": "time_edges_days = [0.0, 0.02, 0.04]",
            }
        )
        with self.assertRaisesRegex(ValueError, "one interval"):
            build_response_plan(config)

    def test_archive_corruption_and_support_mismatch_rejected(self):
        config = self.folding()
        text = config.config_path.read_text().replace(
            "energy_edges_kev=[2.0,4.0]", "energy_edges_kev=[1.0,4.0]"
        )
        config.config_path.write_text(text)
        with self.assertRaisesRegex(ValueError, "support"):
            build_response_plan(load_run_config(config.config_path))
        config = load_run_config(config.config_path)
        config.response.file.write_bytes(b"not an archive")
        with self.assertRaisesRegex(ConfigError, "response.file"):
            build_response_plan(config)

    def test_input_edit_and_output_collision_are_rejected(self):
        config = self.folding()
        plan = build_response_plan(config)
        config.response.file.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            publish_response_run(plan)
        config = self.folding()
        path = self.root / "fold" / "fold.npz"
        path.parent.mkdir()
        path.write_bytes(config.response.file.read_bytes())
        import os

        path.unlink()
        os.link(config.response.file, path)
        with self.assertRaisesRegex(ConfigError, "overwrite input"):
            load_run_config(config.config_path)

    def test_nonfinite_folding_withholds_production_product(self):
        config = self.folding()
        from dataclasses import replace

        response = load_spectral_response(config.response.file)
        response = replace(response, weight_kev=np.full(2, 1e308))
        write_spectral_response(config.response.file, response)
        with (
            np.errstate(over="ignore", invalid="ignore"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(["run", str(config.config_path)]), 3)
        self.assertFalse((config.output.directory / "fold.npz").exists())
        self.assertFalse((config.output.directory / "complete.json").exists())

    def test_response_schema_rejects_irrelevant_and_unsafe_settings(self):
        config = self.collection()
        original = config.config_path.read_text()
        for old, new in (
            ("chunk_size = 16", "chunk_size = 1"),
            ("packets = 33", "packets = 1"),
            ('kind = "powerlaw"', 'kind = "representative"'),
        ):
            config.config_path.write_text(original.replace(old, new))
            with self.subTest(new=new), self.assertRaises(ConfigError):
                load_run_config(config.config_path)
        config = self.folding()
        original = config.config_path.read_text()
        for text in (
            original + "\n[execution]\npackets=10\n",
            original + '\n[execution]\ndevice="gpu"\n',
            original.replace("[response]", "[response]\nscattering_orders=[1,1]"),
            original.replace(
                "radial_edges_arcsec=[0.0,1.0]", "radial_edges_arcsec=[0.0,0.0]"
            ),
        ):
            config.config_path.write_text(text)
            with self.assertRaises(ConfigError):
                load_run_config(config.config_path)

    def test_host_parser_imports_no_numerical_libraries(self):
        config = self.collection()
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                'import sys; from dsh.config import load_run_config; load_run_config(sys.argv[1]); assert not {"jax","numpy","astropy","matplotlib"} & set(sys.modules)',
                str(config.config_path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
