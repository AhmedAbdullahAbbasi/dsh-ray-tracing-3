"""Stage 9C: validate pure-scattering repeated flights using real dust tables."""

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

from dsh.physics.materials import (
    DEFAULT_2_10_SCATTERING,
    load_2_10_material_tables,
)
from dsh.physics.newdust import DEFAULT_NEWDUST_TABLE, load_newdust_scattering_table
from dsh.validation.multiple_scattering import run_multiple_scattering_case


def _git(*args):
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--packets", type=int, default=20_000)
    parser.add_argument("--chunk-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=927)
    parser.add_argument("--sigma-limit", type=float, default=5.0)
    parser.add_argument("--min-expected-events", type=int, default=20)
    parser.add_argument("--materials", choices=("v1", "2-10", "both"), default="both")
    args = parser.parse_args()
    models = []
    if args.materials in {"v1", "both"}:
        models.append(
            ("v1", load_newdust_scattering_table(), DEFAULT_NEWDUST_TABLE, (3.3, 6.9))
        )
    if args.materials in {"2-10", "both"}:
        models.append(
            (
                "2-10",
                load_2_10_material_tables()[0],
                DEFAULT_2_10_SCATTERING,
                (2.0, 5.35, 10.0),
            )
        )
    if args.packets <= 0 or args.chunk_size <= 0:
        parser.error("--packets and --chunk-size must be positive")
    report = {
        "stage": "9C_repeated_pure_scattering",
        "scope": "conditional full-flight hazard, position, phase, and scattering-order tests; analog absorption is zero",
        "reference": "independent NumPy sphere chords and conditional exponential hazards",
        "git_head": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "backend": jax.default_backend(),
        "config": {
            "packets_per_case": args.packets,
            "chunk_size": args.chunk_size,
            "seed": args.seed,
            "sigma_limit": args.sigma_limit,
            "min_expected_events_per_flight": args.min_expected_events,
            "max_interactions": 4,
            "target_tau_scattering": [1.5, 2.4],
        },
        "materials": [],
    }
    for name, scattering, path, energies in models:
        model = {
            "name": name,
            "scattering_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "cases": [],
        }
        for energy in energies:
            for tau in (1.5, 2.4):
                print(f"{name}: {energy:g} keV, tau={tau:g}", flush=True)
                case_id = len(model["cases"]) + 10 * (name == "2-10")
                model["cases"].append(
                    run_multiple_scattering_case(
                        random.fold_in(random.PRNGKey(args.seed), case_id),
                        scattering,
                        energy_kev=energy,
                        target_tau=tau,
                        packets=args.packets,
                        chunk_size=args.chunk_size,
                        sigma_limit=args.sigma_limit,
                        min_expected_events=args.min_expected_events,
                    )
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
