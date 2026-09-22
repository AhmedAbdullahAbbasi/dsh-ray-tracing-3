"""Run the deterministic Stage 9B virtual observer attenuation gate locally.

Uses frozen scattering records and an independent NumPy radial-shell column
reference. This does not launch or rerun a large photon transport simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import jax
import numpy as np

from dsh.physics.absorption import (
    DEFAULT_TBABS_TABLE,
    load_photoelectric_absorption_table,
)
from dsh.physics.materials import (
    DEFAULT_2_10_ABSORPTION,
    DEFAULT_2_10_SCATTERING,
    load_2_10_material_tables,
)
from dsh.physics.newdust import DEFAULT_NEWDUST_TABLE, load_newdust_scattering_table
from dsh.validation.escape import run_escape_attenuation_case


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materials", choices=("v1", "2-10", "both"), default="both")
    args = parser.parse_args()
    models = []
    if args.materials in {"v1", "both"}:
        models.append(
            (
                "v1",
                load_newdust_scattering_table(),
                load_photoelectric_absorption_table(),
                DEFAULT_NEWDUST_TABLE,
                DEFAULT_TBABS_TABLE,
                (3.3, 4.9, 6.9),
            )
        )
    if args.materials in {"2-10", "both"}:
        scattering, absorption, _ = load_2_10_material_tables()
        models.append(
            (
                "2-10",
                scattering,
                absorption,
                DEFAULT_2_10_SCATTERING,
                DEFAULT_2_10_ABSORPTION,
                (2.0, 3.3, 4.9, 5.35, 6.9, 10.0),
            )
        )
    report = {
        "stage": "9B_observer_escape_attenuation",
        "scope": (
            "fixed scatter histories; NumPy radial-shell foreground reference; "
            "no new analog transport or multiple-scatter statistics"
        ),
        "git_head": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "backend": jax.default_backend(),
        "materials": [],
    }
    for (
        label,
        scattering,
        absorption,
        scattering_path,
        absorption_path,
        energies,
    ) in models:
        model = {
            "name": label,
            "table_sha256": {
                "scattering": hashlib.sha256(scattering_path.read_bytes()).hexdigest(),
                "absorption": hashlib.sha256(absorption_path.read_bytes()).hexdigest(),
            },
            "energy_kev": list(energies),
            "cases": [],
        }
        for energy in energies:
            print(f"{label}: {energy:g} keV", flush=True)
            model["cases"].append(
                run_escape_attenuation_case(scattering, absorption, energy)
            )
        model["all_passed"] = all(case["all_passed"] for case in model["cases"])
        report["materials"].append(model)
    report["all_passed"] = all(model["all_passed"] for model in report["materials"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}; all_passed={report['all_passed']}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
