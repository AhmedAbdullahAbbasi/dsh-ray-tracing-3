"""Standard-library Slurm planning; numerical imports occur only in the worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

from .config import ConfigError, load_run_config
from .config.schema import SlurmConfig


def parse_slurm(data, base, device):
    if not isinstance(data, dict):
        raise ConfigError("slurm: expected a TOML table")
    unknown = set(data) - {f.name for f in fields(SlurmConfig)}
    if unknown:
        raise ConfigError(f"slurm: unknown settings {sorted(unknown)}")
    values = {}
    for key, value in data.items():
        label = "slurm." + key
        if key in {"cpus_per_task", "gpus"}:
            minimum = 1 if key == "cpus_per_task" else 0
            if type(value) is not int or not minimum <= value <= 1024:
                raise ConfigError(f"{label}: expected integer {minimum}–1024")
        elif key in {"working_directory", "log_directory"}:
            if (
                not isinstance(value, str)
                or not value
                or any(c in value for c in "\n\r\0")
            ):
                raise ConfigError(f"{label}: expected a nonempty path")
            value = (base / value).resolve()
        elif key == "python":
            if (
                not isinstance(value, str)
                or not value
                or any(c in value for c in "\n\r\0")
            ):
                raise ConfigError(
                    f"{label}: expected one executable path, without control characters"
                )
            # An executable name uses PATH; a relative path is config-relative.
            if "/" in value and not value.startswith("/"):
                value = str((base / value).resolve())
        else:
            patterns = {
                "time": r"(?:[0-9]+-)?[0-9]+:[0-5][0-9]:[0-5][0-9]",
                "memory": r"[1-9][0-9]*(?:[KMGT])?",
                "exclude": r"[A-Za-z0-9_,.\[\]-]+",
            }
            pattern = patterns.get(key, r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
            if not isinstance(value, str) or not re.fullmatch(pattern, value):
                raise ConfigError(f"{label}: invalid value")
            if key == "time" and not any(int(x) for x in re.split("[-:]", value)):
                raise ConfigError("slurm.time: must be positive")
        values[key] = value
    result = SlurmConfig(**values)
    if (device == "gpu") != (result.gpus > 0):
        raise ConfigError("slurm.gpus: request GPUs exactly when execution.device=gpu")
    return result


def job_plan(config):
    settings = config.slurm
    if settings is None:
        raise ConfigError("slurm: add a [slurm] table before planning submission")
    work = settings.working_directory or config.config_path.parent
    logs = settings.log_directory or config.output.directory.parent / "slurm-logs"
    if not work.is_dir():
        raise ConfigError(f"slurm.working_directory: not a directory: {work}")
    for parent in (logs, *logs.parents):
        if parent.exists():
            if not parent.is_dir():
                raise ConfigError(
                    f"slurm.log_directory: parent is not a directory: {parent}"
                )
            break
    arguments = [
        "sbatch",
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        f"--job-name={config.name}",
        f"--cpus-per-task={settings.cpus_per_task}",
        f"--mem={settings.memory}",
        f"--time={settings.time}",
        f"--chdir={work}",
        f"--output={logs / '%x_%j.out'}",
        f"--error={logs / '%x_%j.err'}",
        "--export=ALL",
    ]
    for key in ("partition", "account", "qos", "exclude"):
        value = getattr(settings, key)
        if value:
            arguments.append(f"--{key}={value}")
    if settings.gpus:
        arguments.append(f"--gres=gpu:{settings.gpus}")
    command = [
        settings.python,
        "-m",
        "dsh.cluster",
        "execute",
        str(config.config_path),
        config.config_sha256,
    ]
    # sbatch reads the script from stdin. Resource options are argv values,
    # never interpolated shell directives or evaluated shell commands.
    script = (
        "#!/bin/bash\nset -euo pipefail\n"
        f"export OMP_NUM_THREADS={settings.cpus_per_task}\n"
        "export PYTHONUNBUFFERED=1\n" + "exec " + shlex.join(command) + "\n"
    )
    return {
        "submission_arguments": arguments,
        "script": script,
        "config_sha256": config.config_sha256,
        "log_directory": str(logs),
        "device": config.execution.device,
        "validation": "schema and paths only; numerical preflight runs on allocated node",
    }


def submit_job(config, plan):
    if sys.platform == "win32":
        raise ConfigError(
            "Slurm submission requires the cluster's Linux login environment"
        )
    if os.environ.get("SLURM_JOB_ID"):
        raise ConfigError("submission inside an existing Slurm job is disabled")
    if (
        hashlib.sha256(config.config_path.read_bytes()).hexdigest()
        != config.config_sha256
    ):
        raise ConfigError("configuration changed after submission planning")
    Path(plan["log_directory"]).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        plan["submission_arguments"],
        input=plan["script"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ConfigError("sbatch failed: " + result.stderr.strip())
    identifier = result.stdout.strip()
    if not re.fullmatch(r"[0-9]+(?:;[A-Za-z0-9_.-]+)?", identifier):
        raise ConfigError(
            "sbatch returned an unexpected job identifier; check squeue before retrying"
        )
    print(
        json.dumps(
            {
                "job_id": identifier,
                "config_sha256": config.config_sha256,
                "log_directory": plan["log_directory"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def execute(config_path, expected_sha256):
    # Refuse edits while queued before importing JAX. Input hashes and numerical
    # configuration checks are performed by the shared builder on this node.
    path = Path(config_path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ConfigError(
            "configuration changed while queued; resubmit the intended file"
        )
    from .command import main as configured_main

    return configured_main(["run", str(path)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "submit", "execute"))
    parser.add_argument("config")
    parser.add_argument("expected_sha256", nargs="?")
    args = parser.parse_args(argv)
    try:
        if args.command == "execute":
            if args.expected_sha256 is None:
                raise ConfigError("execute requires the submission-time SHA256")
            return execute(args.config, args.expected_sha256)
        config = load_run_config(args.config)
        plan = job_plan(config)
        if args.command == "plan":
            print(json.dumps(plan, indent=2), flush=True)
            return 0
        return submit_job(config, plan)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"dsh cluster: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
