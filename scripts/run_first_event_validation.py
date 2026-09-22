"""Controlled Stage 9A first-event probability and location validation.

Run from the repository root, e.g.::

    python -m scripts.run_first_event_validation --output validation_outputs/first_events.json

Results at small packet counts are smoke checks only; the report retains
individual statistical residuals and numerical checks for each ray.
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
from jax import random

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
from dsh.validation.first_events import run_first_event_case


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packets-per-ray", type=int, default=20_000)
    parser.add_argument("--chunk-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=981)
    parser.add_argument("--sigma-limit", type=float, default=5.0)
    parser.add_argument("--materials", choices=("v1", "2-10"), default="v1")
    parser.add_argument("--energy-indices", type=int, nargs="+")
    parser.add_argument(
        "--modes", nargs="+", default=["zero", "absorption", "scattering", "both"]
    )
    parser.add_argument(
        "--target-tau-sca-3p3", type=float, nargs="+", default=[0.0, 0.1, 0.3, 1.0]
    )
    args = parser.parse_args()
    if args.materials == "2-10":
        scattering, absorption, _ = load_2_10_material_tables()
        scattering_path, absorption_path = (
            DEFAULT_2_10_SCATTERING,
            DEFAULT_2_10_ABSORPTION,
        )
    else:
        scattering = load_newdust_scattering_table()
        absorption = load_photoelectric_absorption_table()
        scattering_path, absorption_path = DEFAULT_NEWDUST_TABLE, DEFAULT_TBABS_TABLE
    energy_indices = (
        args.energy_indices
        if args.energy_indices is not None
        else [
            int(np.flatnonzero(scattering.energy_kev == energy)[0])
            for energy in (3.3, 4.9, 6.9)
        ]
    )
    if any(
        index < 0 or index >= scattering.energy_kev.size for index in energy_indices
    ):
        parser.error("--energy-indices must select nodes in the material table")
    report = {
        "stage": "9A_first_events",
        "model": "constant-density 4-5 kpc radial shell; central and 0.2 rad source rays",
        "references": "independent NumPy sphere chords and truncated exponential first-event law",
        "interpretation": "MAX_INTERACTIONS is expected after a first scatter at cap=1; no terminal multiple-scattering claims",
        "git_head": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "backend": jax.default_backend(),
        "table_sha256": {
            "scattering": _sha256(scattering_path),
            "absorption": _sha256(absorption_path),
        },
        "config": {
            "materials": args.materials,
            "seed": args.seed,
            "packets_per_ray": args.packets_per_ray,
            "chunk_size": args.chunk_size,
            "sigma_limit": args.sigma_limit,
            "energy_indices": energy_indices,
            "modes": args.modes,
            "target_tau_sca_3p3": args.target_tau_sca_3p3,
        },
        "cases": [],
    }
    case_index = 0
    for tau in args.target_tau_sca_3p3:
        for mode in args.modes:
            # Include one zero-column and one nonzero-column zero-opacity
            # control. Other zero-column modes duplicate these outcomes.
            if tau == 0.0 and mode != "zero":
                continue
            if mode == "zero" and tau not in {0.0, 1.0}:
                continue
            for energy_index in energy_indices:
                print(f"{mode}, tau3p3={tau}, energy_index={energy_index}", flush=True)
                case = run_first_event_case(
                    random.fold_in(random.PRNGKey(args.seed), case_index),
                    scattering,
                    absorption,
                    energy_index=energy_index,
                    mode=mode,
                    target_tau_sca_3p3=tau,
                    packets_per_ray=args.packets_per_ray,
                    chunk_size=args.chunk_size,
                    sigma_limit=args.sigma_limit,
                )
                report["cases"].append(case)
                case_index += 1
    if not report["cases"]:
        parser.error("selected modes/optical depths produce no cases")
    report["all_passed"] = all(case["all_passed"] for case in report["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}; all_passed={report['all_passed']}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
