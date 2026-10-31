"""Accept configured Compton simulation, response collection/folding and regressions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE = "546e6cdfaf70e60d4bd815811ab0ff060f1cedc9"
CORE_FILES = (
    "dsh/transport/kernel.py",
    "dsh/transport/inelastic.py",
    "dsh/observer/scoring.py",
    "dsh/observer/inelastic.py",
    "dsh/physics/dust.py",
    "dsh/physics/compton.py",
    "dsh/physics/cold_compton.py",
    "dsh/inputs/material.py",
    "dsh/pipeline.py",
)
FOCUSED = (
    "tests.test_inelastic_transport",
    "tests.test_inelastic_observer",
    "tests.test_inelastic_response",
    "tests.test_configured_compton",
    "tests.test_compton_response_workflow",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=Path("outputs/compton-configured")
    )
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    destination = args.output_directory.resolve()
    session = destination / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    session.mkdir(parents=True)
    report = {
        "stage": "Compton configured workflow integration",
        "status": "failed",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "configured_compton_enabled": True,
        "detector_response_added": False,
        "reflex_transport_comparison_performed": False,
        "checks": [],
    }
    try:
        report["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        report["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("jax", "numpy", "astropy")
        }
        report["unchanged_core_sha256"] = {}
        for relative in CORE_FILES:
            expected = subprocess.check_output(
                ["git", "show", BASE + ":" + relative], cwd=repo
            ).replace(b"\r\n", b"\n")
            actual = (repo / relative).read_bytes().replace(b"\r\n", b"\n")
            if actual != expected:
                raise ValueError("accepted numerical core changed: " + relative)
            report["unchanged_core_sha256"][relative] = hashlib.sha256(
                actual
            ).hexdigest()
        from scripts.validate_cleanup_stage6 import frozen_evidence

        report["frozen_evidence"] = frozen_evidence(repo)
        commands = [
            ("compton-float32", "0", ["-m", "unittest", *FOCUSED, "-v"]),
            ("compton-float64", "1", ["-m", "unittest", *FOCUSED, "-v"]),
            (
                "physics-float64",
                "1",
                [
                    "-m",
                    "unittest",
                    "tests.test_compton",
                    "tests.test_cold_compton",
                    "-v",
                ],
            ),
        ]
        remaining = [
            "tests." + path.stem
            for path in sorted((repo / "tests").glob("test_*.py"))
            if "tests." + path.stem not in FOCUSED
        ]
        # Isolate groups to bound compiler caches and reset test environments.
        commands += [
            (
                f"regression-{index // 6 + 1}",
                "0",
                ["-m", "unittest", *remaining[index : index + 6], "-v"],
            )
            for index in range(0, len(remaining), 6)
        ]
        commands += [
            (
                "configured-" + precision,
                flag,
                [
                    "-m",
                    "scripts.compton_configured_checks",
                    "--output-directory",
                    str(session / ("configured-" + precision)),
                ],
            )
            for precision, flag in (("float32", "0"), ("float64", "1"))
        ]
        for name, flag, arguments in commands:
            print("Running " + name + " ...", flush=True)
            log = session / (name + ".log")
            started = time.perf_counter()
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(
                    [sys.executable, *arguments],
                    cwd=repo,
                    env=dict(
                        os.environ,
                        JAX_PLATFORMS="cpu",
                        JAX_ENABLE_X64=flag,
                        PYTHONIOENCODING="utf-8",
                    ),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            text = log.read_text(encoding="utf-8")
            tests = re.search(r"Ran (\d+) tests?", text)
            skipped = re.search(r"OK \(skipped=(\d+)\)", text)
            report["checks"].append(
                {
                    "name": name,
                    "exit_code": result.returncode,
                    "tests": int(tests.group(1)) if tests else 0,
                    "skipped": int(skipped.group(1)) if skipped else 0,
                    "seconds": time.perf_counter() - started,
                    "log": log.name,
                }
            )
            if result.returncode:
                raise ValueError(name + " failed; see " + str(log))
        report["configured_workflows"] = {
            precision: json.loads(
                (session / ("configured-" + precision) / "checks.json").read_text(
                    encoding="utf-8"
                )
            )
            for precision in ("float32", "float64")
        }
        report["test_cases"] = sum(value["tests"] for value in report["checks"])
        report["skipped_tests"] = sum(value["skipped"] for value in report["checks"])
        report["passed_tests"] = report["test_cases"] - report["skipped_tests"]
        report["source_sha256"] = {
            path.relative_to(repo).as_posix(): hashlib.sha256(
                path.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for directory in ("dsh", "tests", "configs/compton")
            for path in sorted((repo / directory).rglob("*"))
            if path.suffix in {".py", ".toml"}
        }
        report["status"] = "passed"
        report["scope"] = (
            "Configured Compton simulation and version-2 physical halo responses accepted. No detector response, external Compton transport equivalence, or Monte Carlo convergence claim."
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        (session / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        archive = destination / "compton_configured_report.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(session.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(destination).as_posix())
        print("Status: " + report["status"], flush=True)
        print("Report: " + str(archive), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
