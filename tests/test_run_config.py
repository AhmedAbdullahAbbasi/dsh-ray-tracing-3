"""Configuration contracts, path safety and runtime import isolation."""

import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dsh.config import ConfigError, load_run_config
from tests.config_fixtures import ROOT, config_text, write_config


class RunConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dsh config ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_defaults_are_immutable_explicit_and_json_serializable(self):
        config = load_run_config(write_config(self.root))
        self.assertEqual(config.observer.arrival.edges_days, (0, 1, 2, 2.5))
        self.assertEqual(config.output.directory, self.root / "results with spaces")
        self.assertFalse(config.output.directory.exists())
        self.assertFalse(config.quality.allow_incomplete)
        self.assertNotIn("photon_index", config.to_dict()["source"])
        self.assertEqual(config.to_dict()["execution"]["launch_sampler"], "radial")
        json.dumps(config.to_dict(), allow_nan=False)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.name = "changed"

    def test_documented_examples_resolve(self):
        for name in ("smoke_legacy.toml", "smoke_wd01.toml"):
            config = load_run_config(ROOT / "configs" / name)
            self.assertEqual(config.execution.device, "cpu")

    def test_rejects_bad_fields_types_modes_and_versions(self):
        cases = [
            ("config_version = 1", "config_version = true", "config_version"),
            ("config_version = 1", "config_version = 2", "config_version"),
            ('name = "test"', 'name = "../test"', "run.name"),
            ("packets = 37", "packets = true", "execution.packets"),
            ("packets = 37", "packets = 0", "execution.packets"),
            ("packets = 37", "packets = 3.7", "execution.packets"),
            ("seed = 2026", "seed = 4294967296", "execution.seed"),
            (
                "source_distance_kpc = 10.0",
                "source_distance_kpc = nan",
                "scene.source_distance_kpc",
            ),
            ('device = "cpu"', 'device = "auto"', "execution.device"),
            ("[run]", '[run]\nworkflow = "fold"', "run.workflow"),
            ("[scene]", "[scene]\nsource_pixel = [1, 1]", "scene"),
            ("[source]", '[source]\nfile = "unknown"', "source"),
            (
                'kind = "table_pair"',
                'kind = "table_pair"\ndirectory = "x"',
                "materials",
            ),
            (
                'kind = "uniform"',
                'kind = "uniform"\nedges_days = [0, 1]',
                "observer.arrival",
            ),
            ("step_days = 1.0", "step_days = 0", "observer.arrival.step_days"),
            ("[output]", "[output]\noverwrite = 1", "output.overwrite"),
            ('formats = ["npz", "fits"]', "formats = []", "output.formats"),
            ("[output]", '[scheduler]\npartition = "gpu"\n[output]', "config"),
            ("[output]", "[plots]\nscales = []\n[output]", "plots.scales"),
            ("[output]", '[logging]\nfile = "../log"\n[output]', "logging.file"),
        ]
        for old, new, field in cases:
            with (
                self.subTest(field=field, new=new),
                self.assertRaisesRegex(ConfigError, field),
            ):
                load_run_config(
                    write_config(self.root, config_text().replace(old, new))
                )

    def test_duplicate_keys_and_missing_files_have_clear_errors(self):
        with self.assertRaisesRegex(ConfigError, "run.toml"):
            load_run_config(
                write_config(self.root, config_text() + '\nformats = ["npz"]')
            )
        with self.assertRaisesRegex(ConfigError, "materials.scattering_file"):
            load_run_config(
                write_config(
                    self.root,
                    config_text().replace("mrn_rg_drude_v1.npz", "missing.npz"),
                )
            )

    def test_resolution_is_independent_of_working_directory(self):
        path = write_config(self.root)
        reference = load_run_config(path).to_dict()
        original = Path.cwd()
        try:
            os.chdir(self.root.parent)
            self.assertEqual(load_run_config(path).to_dict(), reference)
        finally:
            os.chdir(original)

    def test_paths_do_not_expand_environment_variables(self):
        with self.assertRaisesRegex(ConfigError, "output.directory"):
            load_run_config(
                write_config(
                    self.root,
                    config_text().replace("results with spaces", "$HOME/output"),
                )
            )

    def test_prevents_overwriting_config_symlinks_and_hardlinked_inputs(self):
        text = config_text().replace(
            'directory = "results with spaces"', 'directory = "."\noverwrite = true'
        )
        with self.assertRaisesRegex(ConfigError, "overwrite input"):
            load_run_config(write_config(self.root, text, filename="original.toml"))
        target = self.root / "results with spaces"
        target.mkdir()
        source = ROOT / "dsh/data/newdust/mrn_rg_drude_v1.npz"
        collision = target / "test.npz"
        try:
            collision.symlink_to(source)
        except OSError:
            self.skipTest("symlinks not available on this host")
        with self.assertRaisesRegex(ConfigError, "overwrite input"):
            load_run_config(
                write_config(
                    self.root,
                    config_text().replace("[output]", "[output]\noverwrite = true"),
                )
            )
        collision.unlink()
        try:
            os.link(source, collision)
        except OSError:
            self.skipTest("hardlinks not available across fixture filesystem")
        with self.assertRaisesRegex(ConfigError, "overwrite input"):
            load_run_config(
                write_config(
                    self.root,
                    config_text().replace("[output]", "[output]\noverwrite = true"),
                )
            )

    def test_parser_imports_no_numerical_runtime_and_creates_no_outputs(self):
        path = write_config(self.root)
        script = """import sys
from dsh.config import load_run_config
from dsh.command import bootstrap
config = load_run_config(sys.argv[1])
assert not {'jax', 'numpy', 'astropy', 'matplotlib'} & set(sys.modules)
assert not config.output.directory.exists()
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_tomli_fallback_route_without_tomllib(self):
        path = write_config(self.root)
        script = """import builtins, sys
original = builtins.__import__
def imports(name, *args, **kwargs):
    if name == 'tomllib':
        raise ModuleNotFoundError('Python 3.10 does not provide tomllib')
    return original(name, *args, **kwargs)
builtins.__import__ = imports
from dsh.config import load_run_config
assert load_run_config(sys.argv[1]).name == 'test'
assert 'tomli' in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if "No module named 'tomli'" in result.stderr:
            self.skipTest("install tomli to exercise the Python 3.10 import route")
        self.assertEqual(result.returncode, 0, result.stderr)
