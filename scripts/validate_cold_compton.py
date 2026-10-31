"""Validate cold material physics, frozen reference diagnostics, and Stage 1."""

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
        "--output-directory", type=Path, default=Path("outputs/compton-cold-material")
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
        "stage": "Compton cold-material prescription",
        "status": "failed",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "production_compton_enabled": False,
        "reflex_angular_equivalence": False,
        "checks": [],
    }
    try:
        report["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        for name, x64, command in [
            ("cold-float32", "0", ["-m", "unittest", "tests.test_cold_compton", "-v"]),
            ("cold-float64", "1", ["-m", "unittest", "tests.test_cold_compton", "-v"]),
            (
                "stage1-and-dust-regression",
                "0",
                [
                    "-m",
                    "scripts.validate_compton_stage1",
                    "--output-directory",
                    str(session / "stage1"),
                ],
            ),
        ]:
            print("Running " + name + " ...", flush=True)
            log = session / (name + ".log")
            start = time.perf_counter()
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(
                    [sys.executable, *command],
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
                    "exit_code": result.returncode,
                    "tests": int(count.group(1)) if count else None,
                    "seconds": time.perf_counter() - start,
                    "log": log.name,
                }
            )
            if result.returncode:
                raise RuntimeError(name + " failed; see " + str(log))
        import numpy as np

        from dsh.physics.cold_compton import SOURCE_PATH, build_cold_compton_table
        from dsh.physics.compton import write_compton_material_table
        from dsh.physics.material_definitions import load_material_definition
        from scripts.cold_compton_reference import assess_reference

        material = load_material_definition(
            SOURCE_PATH.parent / "reflex_reference_definition.json"
        )
        energy = np.unique(np.r_[np.geomspace(2, 10, 129), 3.3, 4.9, 6.9])
        report["materials"] = {}
        for policy in ("omit", "atomic_incoherent"):
            built = build_cold_compton_table(material, energy, grain_electrons=policy)
            write_compton_material_table(
                session / ("cold_" + policy + ".npz"), built.table
            )
            report["materials"][policy] = {
                "metadata": built.table.metadata,
                "anchor_energy_kev": [3.3, 4.9, 6.9],
                "anchor_cross_section_cm2_per_h": built.table.cross_section_cm2_per_h[
                    np.searchsorted(energy, [3.3, 4.9, 6.9])
                ].tolist(),
                "maximum_angular_quadrature_relative_difference": built.table.metadata[
                    "maximum_angular_quadrature_relative_difference"
                ],
            }
        report["material_definition"] = material.as_dict()
        report["abundance_audit"] = material.audit()
        report["reference_diagnostics"] = assess_reference()
        if not report["reference_diagnostics"][
            "native_interpolation_opacity_within_0p5_percent"
        ]:
            raise ValueError(
                "cold opacity reference diagnostic exceeded declared 0.5% tolerance"
            )
        if not report["reference_diagnostics"]["angular_discrepancy_demonstrated"]:
            raise ValueError(
                "frozen angular diagnostic no longer reproduces the recorded result"
            )
        report["source_hashes"] = {
            name: hashlib.sha256(
                (repo / name).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for name in [
                "dsh/physics/cold_compton.py",
                "tests/test_cold_compton.py",
                "scripts/cold_compton_reference.py",
                "scripts/validate_cold_compton.py",
            ]
        }
        report["status"] = "passed"
        report["scientific_scope"] = (
            "Physical-provider acceptance passed. RefleX effective scaling, interpolation and angular differences remain explicitly distinct; production transport is unchanged."
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        (session / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        archive = destination / "compton_cold_material_report.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(session.rglob("*")):
                if path.is_file():
                    bundle.write(path, str(path.relative_to(destination)))
        print("Status: " + report["status"], flush=True)
        print("Report: " + str(archive), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
