"""Accept coupled Compton transport/scoring and all earlier frozen checkpoints."""

from __future__ import annotations

import argparse
import hashlib
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=Path("outputs/compton-transport")
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
        "stage": "Compton coupled transport and observer core",
        "status": "failed",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "inelastic_core_enabled": True,
        "configured_pipeline_compton_enabled": False,
        "reflex_equivalence_claimed": False,
        "checks": [],
    }
    commands = [
        (
            "inelastic-float32",
            "0",
            [
                "-m",
                "unittest",
                "tests.test_inelastic_transport",
                "tests.test_inelastic_observer",
                "-v",
            ],
        ),
        (
            "inelastic-float64",
            "1",
            [
                "-m",
                "unittest",
                "tests.test_inelastic_transport",
                "tests.test_inelastic_observer",
                "-v",
            ],
        ),
        (
            "controls-float32",
            "0",
            [
                "-m",
                "scripts.compton_transport_checks",
                "--output-directory",
                str(session / "controls-float32"),
            ],
        ),
        (
            "controls-float64",
            "1",
            [
                "-m",
                "scripts.compton_transport_checks",
                "--output-directory",
                str(session / "controls-float64"),
            ],
        ),
        (
            "earlier-checkpoints",
            "0",
            [
                "-m",
                "scripts.validate_cold_compton",
                "--output-directory",
                str(session / "earlier-checkpoints"),
            ],
        ),
    ]
    try:
        report["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        report["git_branch"] = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repo, text=True
        ).strip()
        report["source_hashes"] = {
            relative: hashlib.sha256(
                (repo / relative).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for relative in (
                "dsh/transport/inelastic.py",
                "dsh/observer/inelastic.py",
                "tests/test_inelastic_transport.py",
                "tests/test_inelastic_observer.py",
                "scripts/compton_transport_checks.py",
                "scripts/validate_compton_transport.py",
                "docs/COMPTON_TRANSPORT_STAGE2.md",
            )
        }
        for name, x64, arguments in commands:
            print("Running " + name + " ...", flush=True)
            started = time.perf_counter()
            log = session / (name + ".log")
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    [sys.executable, *arguments],
                    cwd=repo,
                    env=dict(
                        os.environ,
                        JAX_PLATFORMS="cpu",
                        JAX_ENABLE_X64=x64,
                        PYTHONIOENCODING="utf-8",
                    ),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            count = re.search(r"Ran (\d+) tests?", log.read_text(encoding="utf-8"))
            report["checks"].append(
                {
                    "name": name,
                    "exit_code": completed.returncode,
                    "tests": int(count.group(1)) if count else None,
                    "seconds": time.perf_counter() - started,
                    "log": log.name,
                }
            )
            if completed.returncode:
                raise RuntimeError(name + " failed; see " + str(log))
        report["controls"] = {
            precision: json.loads(
                (session / ("controls-" + precision) / "checks.json").read_text(
                    encoding="utf-8"
                )
            )
            for precision in ("float32", "float64")
        }
        # Count the earlier reports once each; parent log files intentionally
        # contain progress messages rather than copied unittest output.
        earlier_reports = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((session / "earlier-checkpoints").rglob("report.json"))
        ]
        if len(earlier_reports) != 2 or any(
            value["status"] != "passed" for value in earlier_reports
        ):
            raise ValueError("missing or failing earlier acceptance report")
        report["earlier_test_executions"] = sum(
            check.get("tests") or 0
            for value in earlier_reports
            for check in value["checks"]
        )
        report["new_test_executions"] = sum(
            check["tests"] or 0 for check in report["checks"]
        )
        report["total_test_executions"] = (
            report["new_test_executions"] + report["earlier_test_executions"]
        )
        if (
            report["new_test_executions"] != 64
            or report["earlier_test_executions"] != 140
        ):
            raise ValueError("unexpected test count; inspect incomplete acceptance")
        report["frozen_evidence"] = next(
            value["frozen_evidence"]
            for value in earlier_reports
            if "frozen_evidence" in value
        )
        report["status"] = "passed"
        report["scope"] = (
            "Core transport/scoring accepted, including current-energy opacity, observer-specific recoil, prefix components and diagnosed truncation. TOML/response/archive integration and external transport benchmarks remain subsequent work."
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        (session / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        archive = destination / "compton_transport_report.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(session.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(destination).as_posix())
        print("Status: " + report["status"], flush=True)
        print("Report: " + str(archive), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
