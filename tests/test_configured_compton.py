"""Configured Compton checks, quality gates, publication, and material identity."""

import contextlib
import dataclasses
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np
from astropy.io import fits

from dsh.build import build_run_plan
from dsh.command import main
from dsh.config import ConfigError, load_run_config
from dsh.config.load import input_paths
from dsh.config.schema import ComptonConfig
from dsh.physics.compton import (
    build_free_electron_compton_table,
    write_compton_material_table,
)
from dsh.quality import evaluate_quality
from dsh.run import execute_plan, publish_run
from tests.config_fixtures import ROOT


def smoke_text(directory):
    text = (ROOT / "configs/compton/cold_smoke.toml").read_text()
    text = text.replace("../../dsh/data/", (ROOT / "dsh/data").as_posix() + "/")
    text = text.replace("../../outputs/cold_compton_smoke", Path(directory).as_posix())
    return text.replace("packets = 4096", "packets = 257").replace(
        "chunk_size = 256", "chunk_size = 128"
    )


class ConfiguredComptonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="configured compton ")
        cls.root = Path(cls.temp.name)
        cls.path = cls.root / "run.toml"
        cls.path.write_text(smoke_text(cls.root / "outputs"))
        cls.config = load_run_config(cls.path)
        cls.plan = build_run_plan(cls.config)
        cls.result = execute_plan(cls.plan)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        jax.clear_caches()

    def test_check_performs_no_transport_or_publication(self):
        with (
            patch("dsh.run.execute_plan", side_effect=AssertionError("transport")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(["check", str(self.path)]), 0)
        self.assertFalse((self.root / "outputs").exists())

    def test_chunk_remainder_components_and_energy_accounting(self):
        r = self.result
        q = evaluate_quality(r, 257)
        self.assertTrue(q["numerical_complete"], q)
        self.assertGreater(int(r.inelastic_diagnostics.analog_compton_count), 0)
        self.assertEqual(int(r.products.history_count), 257)
        np.testing.assert_allclose(
            r.products.total_fluence,
            sum(group.total_fluence for group in r.components[1:]),
            rtol=5e-6,
            atol=1e-20,
        )
        self.assertLess(
            float(r.inelastic_diagnostics.max_energy_closure_error_kev), 2e-5
        )

    def test_npz_fits_schema_and_completion_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            config = dataclasses.replace(
                self.config,
                output=dataclasses.replace(
                    self.config.output, directory=Path(directory) / "published"
                ),
            )
            plan = dataclasses.replace(self.plan, config=config)
            with (
                patch("dsh.run.execute_plan", return_value=self.result),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(publish_run(plan), 0)
            output = config.output.directory
            with (
                np.load(output / "cold_compton_smoke.npz") as data,
                fits.open(output / "cold_compton_smoke.fits", checksum=True) as hdus,
            ):
                self.assertEqual(int(data["output_schema_version"]), 9)
                self.assertEqual(len(data["transport_status_labels"]), 9)
                self.assertTrue(hdus[0].header["COMPTON"])
                np.testing.assert_allclose(
                    data["compton_only__total_fluence"],
                    hdus["COMPT4D"].data,
                    rtol=1e-14,
                )
                np.testing.assert_allclose(
                    data["mixed__total_fluence_squared"],
                    hdus["MIXEDQ4D"].data,
                    rtol=1e-14,
                )
                np.testing.assert_allclose(
                    data["compton_energy_kev"],
                    hdus["COMPT_ENERGY_KEV"].data,
                    rtol=1e-14,
                )
                self.assertEqual(hdus["STATUS"].data.shape, (9,))
            marker = json.loads((output / "complete.json").read_text())
            for name, digest in marker["output_sha256"].items():
                self.assertEqual(
                    hashlib.sha256((output / name).read_bytes()).hexdigest(), digest
                )
            resolved = json.loads((output / "resolved.json").read_text())
            self.assertIn("compton", resolved["configuration"])

    def test_cutoff_support_and_missing_phase_block_complete_publication(self):
        for field, value in (
            ("unsupported_observer_escape_count", 1),
            ("unsupported_observer_phase_count", 1),
            ("max_energy_closure_error_kev", 0.1),
        ):
            r = self.result._replace(
                inelastic_diagnostics=self.result.inelastic_diagnostics._replace(
                    **{field: value}
                )
            )
            self.assertFalse(evaluate_quality(r, 257)["numerical_complete"])
        for status in (4, 7, 8):
            counts = self.result.diagnostics.transport_status_count.copy()
            origin = np.flatnonzero(counts[1:4])[0] + 1
            counts[origin] -= 1
            counts[status] += 1
            r = self.result._replace(
                diagnostics=self.result.diagnostics._replace(
                    transport_status_count=counts
                )
            )
            self.assertFalse(evaluate_quality(r, 257)["numerical_complete"])
        with tempfile.TemporaryDirectory() as directory:
            config = dataclasses.replace(
                self.config,
                output=dataclasses.replace(
                    self.config.output, directory=Path(directory) / "failed"
                ),
            )
            with (
                patch("dsh.run.execute_plan", return_value=r),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    publish_run(dataclasses.replace(self.plan, config=config)), 3
                )
            self.assertFalse((config.output.directory / "complete.json").exists())
            self.assertFalse(
                (config.output.directory / "cold_compton_smoke.npz").exists()
            )

    def test_host_parser_remains_free_of_numerical_imports(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                'import sys; from dsh.config import load_run_config; load_run_config(sys.argv[1]); assert not {"jax","numpy","astropy","matplotlib"} & set(sys.modules)',
                str(self.path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cold_source_file_is_in_input_provenance(self):
        self.assertTrue(
            any(
                path.name == "scattering_functions.json"
                for path in input_paths(self.config)
            )
        )

    def test_incomplete_diagnostic_exports_are_labeled(self):
        extra = self.result.inelastic_diagnostics._replace(
            unsupported_observer_escape_count=1
        )
        result = self.result._replace(inelastic_diagnostics=extra)
        with tempfile.TemporaryDirectory() as directory:
            config = dataclasses.replace(
                self.config,
                output=dataclasses.replace(
                    self.config.output, directory=Path(directory) / "diagnostic"
                ),
                quality=dataclasses.replace(self.config.quality, allow_incomplete=True),
            )
            with (
                patch("dsh.run.execute_plan", return_value=result),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    publish_run(dataclasses.replace(self.plan, config=config)), 3
                )
            with np.load(
                config.output.directory / "diagnostic_cold_compton_smoke.npz"
            ) as data:
                self.assertTrue(bool(data["diagnostic_only"]))
                self.assertFalse(bool(data["numerical_complete"]))
                self.assertEqual(
                    int(data["inelastic_unsupported_observer_escape_count"]), 1
                )
            self.assertFalse((config.output.directory / "complete.json").exists())

    def test_strict_modes_require_explicit_grain_policy_and_cutoff(self):
        original = self.path.read_text()
        for text in (
            original.replace('grain_electrons = "atomic_incoherent"\n', ""),
            original.replace("energy_cutoff_kev = 2.0\n", ""),
            original.replace("energy_cutoff_kev = 2.0", "energy_cutoff_kev = true"),
            original.replace('kind = "cold_material"', 'kind = "disabled"'),
            original.replace(
                'grain_electrons = "atomic_incoherent"', 'grain_electrons = "free"'
            ),
        ):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.toml"
                path.write_text(text)
                with self.assertRaises(ConfigError):
                    load_run_config(path)

    def test_external_material_hash_and_support_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compton.npz"
            digest = self.plan.metadata["material_definition_sha256"]
            for material, energies, cutoff in (
                ("b" * 64, [2.0, 10.0], 2.0),
                (digest, [4.0, 10.0], 4.0),
                (digest, [2.0, 10.0], 1.0),
                (digest, [2.0, 10.0], 4.0),
            ):
                write_compton_material_table(
                    path,
                    build_free_electron_compton_table(
                        energies,
                        electrons_per_h=1.0,
                        material_definition_sha256=material,
                        cosine_nodes=65,
                    ),
                )
                config = dataclasses.replace(
                    self.config,
                    compton=ComptonConfig(
                        "external_table", file=path, energy_cutoff_kev=cutoff
                    ),
                )
                with (
                    self.subTest(material=material, energies=energies, cutoff=cutoff),
                    self.assertRaises(ValueError),
                ):
                    build_run_plan(config)
            write_compton_material_table(
                path,
                build_free_electron_compton_table(
                    [2.0, 10.0],
                    electrons_per_h=1.0,
                    material_definition_sha256=digest,
                    cosine_nodes=65,
                ),
            )
            config = dataclasses.replace(
                self.config,
                compton=ComptonConfig(
                    "external_table", file=path, energy_cutoff_kev=2.0
                ),
            )
            self.assertIsNotNone(build_run_plan(config).compton)
