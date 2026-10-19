"""Host-only conflict checks and actionable configuration diagnostics."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dsh.config import ConfigError, load_run_config
from tests.config_fixtures import ROOT, config_text, write_config


class ConfigurationUsabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="configuration usability ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_all_seven_examples_parse_without_a_numerical_runtime(self):
        script = """import sys
from pathlib import Path
from dsh.config import load_run_config
repo, temp = Path(sys.argv[1]), Path(sys.argv[2])
paths = sorted((repo / 'configs').glob('*.toml'))
assert len(paths) == 7
configs = []
for path in paths:
    text = path.read_text().replace('"../dsh/data/', '"' + (repo / 'dsh/data').as_posix() + '/')
    text = text.replace('"data/', '"' + (repo / 'configs/data').as_posix() + '/')
    text = text.replace('"../outputs/', '"' + temp.as_posix() + '/products/')
    copied = temp / path.name
    copied.write_text(text)
    configs.append(load_run_config(copied))
assert {c.source.kind for c in configs} == {
    'representative', 'powerlaw', 'band_powerlaw', 'exponential_decay',
    'spectral_history', 'flux_table'}
assert {c.scene.kind for c in configs} == {'fits', 'example_four_cloud'}
assert not {'jax', 'numpy', 'astropy', 'matplotlib'} & set(sys.modules)
assert all(not c.output.directory.exists() for c in configs)
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(ROOT), str(self.root)],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_typo_hint_and_selected_mode_are_named(self):
        with self.assertRaisesRegex(ConfigError, "did you mean 'packets'"):
            load_run_config(
                write_config(self.root, config_text().replace("packets =", "pakets ="))
            )
        with self.assertRaisesRegex(
            ConfigError, "source \\(kind=representative\\).*photon_flux"
        ):
            load_run_config(
                write_config(
                    self.root,
                    config_text().replace("[source]", "[source]\nphoton_flux = [1]"),
                )
            )

    def test_zero_flux_and_observer_conflicts_fail_during_host_parsing(self):
        for old, new, field in (
            ("[[0.02, 0.012, 0.006]]", "[[0, 0, 0]]", "source.band_flux"),
            ("[2.5, 4.1, 5.9, 7.9]", "[4.1, 5.9, 7.9]", "observer.energy_edges_kev"),
            ("stop_days = 2.5", "stop_days = 0.01", "observer.arrival"),
            ("step_days = 1.0", "step_days = 1e-320", "observer.arrival"),
            ("stop_days = 2.5", "stop_days = 1e308", "observer.arrival"),
            (
                "source_distance_kpc = 10.0",
                "source_distance_kpc = " + "9" * 400,
                "scene.source_distance_kpc",
            ),
        ):
            with self.subTest(new=new), self.assertRaisesRegex(ConfigError, field):
                load_run_config(
                    write_config(self.root, config_text().replace(old, new))
                )

    def test_zero_broadband_flux_is_rejected(self):
        text = (ROOT / "configs/smoke_wd01.toml").read_text()
        text = text.replace('"../dsh/data/', '"' + (ROOT / "dsh/data").as_posix() + "/")
        with self.assertRaisesRegex(ConfigError, "source.photon_flux"):
            load_run_config(write_config(self.root, text.replace("[0.038]", "[0.0]")))

    def test_decay_peak_cannot_lie_below_its_baseline(self):
        text = (
            config_text()
            .replace(
                'kind = "representative"',
                'kind = "exponential_decay"\ndecay_time_days = 1\nbaseline_band_flux = [0.03, 0, 0]',
            )
            .replace(
                "band_flux = [[0.02, 0.012, 0.006]]",
                "peak_band_flux = [0.02, 0.012, 0.006]",
            )
        )
        with self.assertRaisesRegex(ConfigError, "source.peak_band_flux.*baseline"):
            load_run_config(write_config(self.root, text))

    def test_flux_final_interval_cannot_exceed_gap_policy(self):
        text = (ROOT / "configs/example_flux_table.toml").read_text()
        text = text.replace('"../dsh/data/', '"' + (ROOT / "dsh/data").as_posix() + "/")
        text = text.replace('"data/', '"' + (ROOT / "configs/data").as_posix() + "/")
        with self.assertRaisesRegex(ConfigError, "source.last_interval_days"):
            load_run_config(
                write_config(
                    self.root,
                    text.replace(
                        "last_interval_days = 0.5", "last_interval_days = 1.0"
                    ),
                )
            )

    def test_plot_epoch_must_select_whole_unambiguous_bins(self):
        for epoch in ("[0, 1.5]", "[0, 3]", "[0, 0.0000000001]"):
            text = config_text() + f"\n[plots]\nepoch_days = {epoch}\n"
            with (
                self.subTest(epoch=epoch),
                self.assertRaisesRegex(ConfigError, "plots.epoch_days"),
            ):
                load_run_config(write_config(self.root, text))

    def test_output_parent_file_is_rejected_without_creating_directories(self):
        parent = self.root / "parent file"
        parent.write_text("input")
        with self.assertRaisesRegex(ConfigError, "output.directory.*parent"):
            load_run_config(
                write_config(
                    self.root,
                    config_text().replace("results with spaces", "parent file/results"),
                )
            )
        self.assertEqual(parent.read_text(), "input")

    def test_existing_output_message_explains_how_to_repeat(self):
        output = self.root / "results with spaces"
        output.mkdir()
        (output / "test.npz").write_bytes(b"existing product")
        with self.assertRaisesRegex(ConfigError, "output.overwrite = true"):
            load_run_config(write_config(self.root))
        self.assertEqual((output / "test.npz").read_bytes(), b"existing product")
