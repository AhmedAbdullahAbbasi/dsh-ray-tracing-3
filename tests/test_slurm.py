"""Scheduler contracts without requiring Slurm or a GPU on the test host."""

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dsh.cluster import execute, job_plan, parse_slurm, submit_job
from dsh.command import main
from dsh.config import ConfigError, load_run_config
from tests.config_fixtures import ROOT, config_text, write_config


class SlurmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="slurm paths ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def config(self, extra="\n[slurm]\n"):
        return load_run_config(write_config(self.root, config_text() + extra))

    def test_cpu_plan_is_pure_and_preserves_absolute_paths(self):
        config = self.config('\n[slurm]\npartition="compute"\naccount="research"\n')
        plan = job_plan(config)
        self.assertIn("--partition=compute", plan["submission_arguments"])
        self.assertFalse(any("gres" in s for s in plan["submission_arguments"]))
        self.assertIn(str(config.config_path), plan["script"])
        self.assertFalse(config.output.directory.exists())
        self.assertFalse(Path(plan["log_directory"]).exists())
        self.assertIn(config.config_sha256, plan["script"])
        self.assertIn("--ntasks=1", plan["submission_arguments"])

    def test_gpu_plan_can_be_built_without_jax_or_cuda(self):
        path = write_config(
            self.root,
            config_text().replace('device = "cpu"', 'device = "gpu"')
            + '\n[slurm]\ngpus=1\nexclude="cn[05-06]"\n',
        )
        # Fixture may omit the default CPU device.
        text = path.read_text()
        if 'device = "gpu"' not in text:
            path.write_text(text.replace("[execution]", '[execution]\ndevice="gpu"'))
        plan = job_plan(load_run_config(path))
        self.assertIn("--gres=gpu:1", plan["submission_arguments"])
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                'import sys; from dsh.command import main; code=main(["slurm-plan",sys.argv[1]]); assert not {"jax","numpy","astropy","matplotlib"} & set(sys.modules); raise SystemExit(code)',
                str(path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_cluster_examples_select_expected_resources(self):
        for device in ("cpu", "gpu"):
            config = load_run_config(ROOT / "configs/slurm" / (device + ".toml"))
            self.assertEqual(config.execution.device, device)
            self.assertEqual(config.slurm.gpus, int(device == "gpu"))
            job_plan(config)

    def test_bad_settings_and_cpu_gpu_conflicts(self):
        cases = [
            ({"gpus": 1}, "cpu"),
            ({}, "gpu"),
            ({"gpus": True}, "gpu"),
            ({"cpus_per_task": 0}, "cpu"),
            ({"memory": "16G\n--x"}, "cpu"),
            ({"partition": "x;touch BAD"}, "cpu"),
            ({"time": "00:00:00"}, "cpu"),
            ({"time": "01:99:00"}, "cpu"),
            ({"nodes": 2}, "cpu"),
            ({"python": "python\nBAD"}, "cpu"),
        ]
        for data, device in cases:
            with self.subTest(data=data), self.assertRaises(ConfigError):
                parse_slurm(data, self.root, device)

    def test_submission_uses_argv_and_stdin_not_shell(self):
        config = self.config()
        plan = job_plan(config)
        with (
            patch("dsh.cluster.sys.platform", "linux"),
            patch.dict(os.environ, {}, clear=True),
            patch(
                "dsh.cluster.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, "12345;cluster\n", ""),
            ) as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(submit_job(config, plan), 0)
        run.assert_called_once_with(
            plan["submission_arguments"],
            input=plan["script"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertTrue(Path(plan["log_directory"]).is_dir())

    def test_submit_errors_never_retry_or_resubmit(self):
        config = self.config()
        plan = job_plan(config)
        with (
            patch("dsh.cluster.sys.platform", "linux"),
            patch.dict(os.environ, {}, clear=True),
        ):
            for result in (
                subprocess.CompletedProcess([], 1, "", "denied"),
                subprocess.CompletedProcess([], 0, "unexpected", ""),
            ):
                with (
                    patch("dsh.cluster.subprocess.run", return_value=result) as run,
                    self.assertRaises(ConfigError),
                ):
                    submit_job(config, plan)
                self.assertEqual(run.call_count, 1)
            with (
                patch.dict(os.environ, {"SLURM_JOB_ID": "55"}),
                patch("dsh.cluster.subprocess.run") as run,
                self.assertRaises(ConfigError),
            ):
                submit_job(config, plan)
            run.assert_not_called()

    def test_queue_edits_fail_before_numerical_import(self):
        path = write_config(self.root)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.write_bytes(path.read_bytes() + b"\n# edited\n")
        with (
            patch("dsh.command.main") as run,
            self.assertRaisesRegex(ConfigError, "changed while queued"),
        ):
            execute(path, digest)
        run.assert_not_called()

    def test_worker_delegates_and_preserves_failure_exit(self):
        config = self.config()
        with patch("dsh.command.main", return_value=3) as run:
            self.assertEqual(execute(config.config_path, config.config_sha256), 3)
        run.assert_called_once_with(["run", str(config.config_path)])

    def test_plan_command_does_not_submit_or_create_files(self):
        config = self.config()
        output = io.StringIO()
        with (
            patch("dsh.cluster.subprocess.run") as run,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(main(["slurm-plan", str(config.config_path)]), 0)
        run.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["device"], "cpu")
        self.assertFalse(config.output.directory.exists())

    def test_missing_slurm_and_bad_working_directory(self):
        with self.assertRaisesRegex(ConfigError, "slurm"):
            job_plan(load_run_config(write_config(self.root)))
        with self.assertRaisesRegex(ConfigError, "working_directory"):
            job_plan(self.config('\n[slurm]\nworking_directory="missing"\n'))
