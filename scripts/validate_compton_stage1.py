"""Validate the Compton foundation and unchanged production paths; save a report."""

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

BASE = "39e1790e991cc83c392962587e806210f044a267"
PROTECTED_SOURCE = (
    "dsh/physics/dust.py",
    "dsh/physics/materials.py",
    "dsh/inputs/material.py",
    "dsh/transport/kernel.py",
    "dsh/observer/scoring.py",
    "dsh/observer/response.py",
    "dsh/observer/response_transport.py",
    "dsh/config/schema.py",
    "dsh/run.py",
    "dsh/build.py",
    "dsh/pipeline.py",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=Path("outputs/compton-stage1")
    )
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    destination = args.output_directory.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    session = destination / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    session.mkdir()
    summary = {
        "stage": "Compton 1: physics and material foundation",
        "status": "failed",
        "base": BASE,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "production_compton_enabled": False,
        "bound_electron_model_validated": False,
        "reflex_transport_comparison_performed": False,
        "checks": [],
    }

    def git(*arguments):
        return subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    try:
        summary["git_head"] = git("rev-parse", "HEAD").stdout.decode().strip()
        summary["git_branch"] = git("branch", "--show-current").stdout.decode().strip()
        summary["production_source_unchanged"] = {}
        for relative in PROTECTED_SOURCE:
            expected = git("show", BASE + ":" + relative).stdout.replace(b"\r\n", b"\n")
            actual = (repo / relative).read_bytes().replace(b"\r\n", b"\n")
            if actual != expected:
                raise ValueError("Unexpected production source edit: " + relative)
            summary["production_source_unchanged"][relative] = hashlib.sha256(
                actual
            ).hexdigest()
        from scripts.validate_cleanup_stage6 import frozen_evidence

        summary["frozen_evidence"] = frozen_evidence(repo)
        checks = [
            ("compton-float32", "0", ["tests.test_compton"]),
            ("compton-float64", "1", ["tests.test_compton"]),
            (
                "dust-only-regression",
                "0",
                [
                    "tests.test_voxel_transport",
                    "tests.test_observer",
                    "tests.test_material_v2",
                    "tests.test_configured_run",
                    "tests.test_spectral_histories",
                    "tests.test_response_workflows",
                ],
            ),
        ]
        for name, x64, modules in checks:
            print("Running " + name + " ...", flush=True)
            env = dict(
                os.environ,
                JAX_PLATFORMS="cpu",
                JAX_ENABLE_X64=x64,
                PYTHONIOENCODING="utf-8",
            )
            start = time.perf_counter()
            log = session / (name + ".log")
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    [sys.executable, "-m", "unittest", *modules, "-v"],
                    cwd=repo,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            text = log.read_text(encoding="utf-8")
            count = re.search(r"Ran (\d+) tests?", text)
            summary["checks"].append(
                {
                    "name": name,
                    "exit_code": completed.returncode,
                    "tests": int(count.group(1)) if count else None,
                    "seconds": time.perf_counter() - start,
                    "log": log.name,
                }
            )
            if completed.returncode:
                raise RuntimeError(name + " failed; see " + str(log))

        import jax
        import numpy as np

        from dsh.physics.compton import (
            build_free_electron_compton_table,
            write_compton_material_table,
        )

        summary["versions"] = {"jax": jax.__version__, "numpy": np.__version__}
        recipe = {
            "kind": "synthetic fully ionized H-He example",
            "H_atoms_per_h": 1.0,
            "He_atoms_per_h": 0.1,
            "electrons_per_h": 1.2,
            "scope": "Free-electron reference only; not the neutral RefleX material.",
        }
        recipe_text = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
        recipe_sha = hashlib.sha256(recipe_text.encode()).hexdigest()
        (session / "free_electron_demo_recipe.json").write_text(
            recipe_text, encoding="utf-8"
        )
        table = build_free_electron_compton_table(
            [0.1, 0.2, 0.5, 1.0, 2.0, 3.3, 4.9, 6.9, 10.0, 30.0, 100.0, 511.0, 1000.0],
            electrons_per_h=1.2,
            material_definition_sha256=recipe_sha,
        )
        write_compton_material_table(session / "free_electron_demo.npz", table)
        summary["demo_material"] = dict(table.metadata)
        summary["demo_material"]["energy_kev"] = table.energy_kev.tolist()
        summary["demo_material"]["cross_section_cm2_per_h"] = (
            table.cross_section_cm2_per_h.tolist()
        )
        summary["source_hashes"] = {
            name: hashlib.sha256(
                (repo / name).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for name in (
                "dsh/physics/compton.py",
                "tests/test_compton.py",
                "scripts/validate_compton_stage1.py",
            )
        }
        summary["status"] = "passed"
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
    finally:
        (session / "report.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        archive = destination / "compton_stage1_report.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for file in sorted(session.iterdir()):
                if file.is_file():
                    bundle.write(file, session.name + "/" + file.name)
        print("Status: " + summary["status"], flush=True)
        print("Report: " + str(archive), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
