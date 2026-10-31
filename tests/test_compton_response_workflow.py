"""Compton response collection/folding through the public configured interface."""

import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

import jax
import numpy as np

from dsh.command import main
from dsh.config import load_run_config
from dsh.observer.inelastic_response import InelasticSpectralResponse
from dsh.observer.response import load_spectral_response, write_spectral_response
from dsh.observer.response_transport import collect_spectral_response
from dsh.response_run import (
    build_response_plan,
    execute_response_plan,
    publish_response_run,
)
from tests.test_configured_compton import smoke_text
from tests.test_inelastic_response import response


def collection_text(directory):
    text = smoke_text(directory).replace(
        'name = "cold_compton_smoke"', 'name = "collect"\nworkflow = "response_collect"'
    )
    text = re.sub(
        r"\[source\].*?\[materials\]",
        '[source]\nkind="powerlaw"\ntime_edges_days=[0.0,0.01]\nenergy_edges_kev=[3.3,8.0]\nphoton_flux=[0.038]\nphoton_index=1.7\n[materials]',
        text,
        flags=re.S,
    )
    text = re.sub(r"\[observer\].*?\[output\]", "[output]", text, flags=re.S)
    return text.replace('formats = ["npz", "fits"]\n', "")


class ComptonResponseWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="compton response ")
        cls.root = Path(cls.temp.name)
        cls.path = cls.root / "collect.toml"
        cls.path.write_text(collection_text(cls.root / "collect"))
        cls.config = load_run_config(cls.path)
        cls.plan = build_response_plan(cls.config)
        cls.archive = execute_response_plan(cls.plan)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        jax.clear_caches()

    def test_collector_keeps_initial_observed_incident_and_prefix_information(self):
        r = self.archive
        self.assertIsInstance(r, InelasticSpectralResponse)
        self.assertEqual(r.n_histories, 257)
        self.assertEqual(sum(r.provenance["transport_status_counts"]), 257)
        self.assertEqual(len(r.provenance["transport_status_counts"]), 9)
        self.assertGreater(int(np.sum(r.compton_order > 0)), 0)
        self.assertTrue(np.all(r.energy_kev <= r.source_energy_kev * (1 + 2e-6)))
        self.assertIn("compton", r.provenance["configured_run"])
        np.testing.assert_array_equal(
            r.scattering_order, r.dust_order + r.compton_order
        )

    def test_configured_collection_matches_explicit_api_with_one_packet_tail(self):
        expected = collect_spectral_response(
            self.plan.source,
            self.plan.cloud,
            self.plan.physics,
            self.plan.launch,
            n_packets=257,
            chunk_size=128,
            max_interactions=24,
            seed=2049,
            compton=self.plan.compton,
            energy_cutoff_kev=2.0,
        )
        for name in self.archive.__dataclass_fields__:
            if isinstance(getattr(self.archive, name), np.ndarray):
                np.testing.assert_array_equal(
                    getattr(self.archive, name), getattr(expected, name)
                )

    def test_version_two_collection_is_published_and_reloads(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(publish_response_run(self.plan), 0)
        output = self.config.output.directory
        actual = load_spectral_response(output / "collect.npz")
        self.assertIsInstance(actual, InelasticSpectralResponse)
        np.testing.assert_array_equal(
            actual.source_energy_kev, self.archive.source_energy_kev
        )
        self.assertTrue((output / "complete.json").is_file())

    def test_cutoff_cannot_publish_a_complete_response(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cutoff.toml"
            text = (
                collection_text(Path(directory) / "out")
                .replace("energy_edges_kev=[3.3,8.0]", "energy_edges_kev=[3.3,3.31]")
                .replace("energy_cutoff_kev = 2.0", "energy_cutoff_kev = 3.2999")
            )
            path.write_text(text)
            config = load_run_config(path)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["run", str(path)]), 3)
            self.assertFalse((config.output.directory / "complete.json").exists())
            self.assertFalse((config.output.directory / "collect.npz").exists())

    def test_folding_writes_components_at_observed_energy_with_initial_energy_weights(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "response.npz"
            write_spectral_response(archive, response())
            path = root / "fold.toml"
            path.write_text(f"""config_version=1
[run]
name="fold"
workflow="response_fold"
[source]
kind="powerlaw"
time_edges_days=[0.0,0.000011574074074074073]
energy_edges_kev=[400.0,600.0]
photon_flux=[200.0]
photon_index=2.0
[response]
file={json.dumps(archive.as_posix())}
[observer]
arrival_time_edges_days=[0.0,0.000011574074074074073]
energy_edges_kev=[200.0,300.0,600.0]
radial_edges_arcsec=[0.0,1.0]
[output]
directory={json.dumps((root / "folded").as_posix())}
""")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["run", str(path)]), 0)
            with np.load(root / "folded/fold.npz") as data:
                np.testing.assert_allclose(
                    data["fluence"], [[[1.92], [0.96]]], rtol=2e-6
                )
                np.testing.assert_allclose(
                    sum(
                        data[name + "__fluence"]
                        for name in ("dust_only", "compton_only", "mixed")
                    ),
                    data["fluence"],
                    rtol=1e-14,
                )
                self.assertEqual(
                    json.loads(str(data["metadata_json"]))["schema_version"], 2
                )
                self.assertIn("covariance", data.files)
