"""Portable Slurm contract checks; never submit a scheduler job."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    session = (
        repo
        / "outputs/slurm-stage4"
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
    )
    session.mkdir(parents=True)
    summary = {
        "stage": 4,
        "platform": platform.system(),
        "python": platform.python_version(),
        "status": "failed",
        "actual_slurm_submission_tested": False,
        "gpu_execution_tested": False,
    }
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    try:
        identity = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        summary["git_head"] = (
            identity.stdout.strip() if identity.returncode == 0 else "unavailable"
        )
        with (session / "tests.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "tests.test_slurm",
                    "tests.test_run_config",
                    "tests.test_configuration_usability",
                    "-v",
                ],
                cwd=repo,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        summary["tests_exit"] = result.returncode
        if result.returncode:
            raise ValueError("Portable tests failed; inspect tests.log")
        # Keep the example config in configs so its existing relative inputs resolve.
        config = repo / "configs" / ("stage4-check-" + uuid.uuid4().hex + ".toml")
        try:
            text = (repo / "configs/smoke_legacy.toml").read_text(encoding="utf-8")
            text += '\n[slurm]\npartition="compute"\ncpus_per_task=2\ngpus=0\n'
            config.write_text(text, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "dsh.command", "slurm-plan", str(config)],
                cwd=repo,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            (session / "plan.log").write_text(result.stderr, encoding="utf-8")
            if result.returncode:
                raise ValueError("Plan command failed")
            plan = json.loads(result.stdout)
            (session / "job_plan.json").write_text(
                json.dumps(plan, indent=2) + "\n", encoding="utf-8"
            )
            (session / "job.sh").write_text(plan["script"], encoding="utf-8")
            if (
                not plan["submission_arguments"][0] == "sbatch"
                or "--ntasks=1" not in plan["submission_arguments"]
            ):
                raise ValueError("Unexpected scheduler plan")
            summary["plan_exit"] = result.returncode
        finally:
            config.unlink(missing_ok=True)
        summary["status"] = "passed"
    except (OSError, ValueError) as error:
        summary["error"] = str(error)
    finally:
        (session / "stage4_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        archive = session / "stage4_report.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for file in sorted(session.iterdir()):
                if file != archive and file.is_file():
                    z.write(file, file.name)
        print("Stage 4 " + summary["status"] + "; report: " + str(archive), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
