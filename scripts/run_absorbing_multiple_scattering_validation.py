"""Stage 9D: absorption-enabled repeated-scattering validation on real tables."""

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
from dsh.validation.absorbing_multiple_scattering import (
    run_absorbing_multiple_scattering_case,
)


def _git(*args):
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packets", type=int, default=20_000)
    parser.add_argument("--chunk-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=1047)
    parser.add_argument("--sigma-limit", type=float, default=5.0)
    parser.add_argument("--min-expected-category", type=float, default=20.0)
    parser.add_argument("--materials", choices=("v1", "2-10", "both"), default="both")
    args = parser.parse_args()
    if args.packets <= 0 or args.chunk_size <= 0:
        parser.error("packet count and chunk size must be positive")
    models = []
    if args.materials in {"v1", "both"}:
        models.append(
            (
                "v1",
                load_newdust_scattering_table(),
                load_photoelectric_absorption_table(),
                DEFAULT_NEWDUST_TABLE,
                DEFAULT_TBABS_TABLE,
                (3.3, 6.9),
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
                (2.0, 5.35, 10.0),
            )
        )
    report = {
        "stage": "9D_absorption_enabled_repeated_scattering",
        "scope": "conditional analog scatter/absorb/escape and event positions through four flights; no production flare",
        "reference": "independent NumPy sphere chords, finite-path competing exponential risks, and real scattering/absorption tables",
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
            "min_expected_category_per_flight": args.min_expected_category,
            "target_tau_scattering": [1.5, 2.4],
            "max_interactions": 4,
        },
        "materials": [],
    }
    for name, scattering, absorption, sca_path, abs_path, energies in models:
        model = {
            "name": name,
            "table_sha256": {
                "scattering": hashlib.sha256(sca_path.read_bytes()).hexdigest(),
                "absorption": hashlib.sha256(abs_path.read_bytes()).hexdigest(),
            },
            "cases": [],
        }
        for energy in energies:
            for tau in (1.5, 2.4):
                print(f"{name}: {energy:g} keV, scattering tau={tau:g}", flush=True)
                case_id = len(model["cases"]) + 10 * (name == "2-10")
                model["cases"].append(
                    run_absorbing_multiple_scattering_case(
                        random.fold_in(random.PRNGKey(args.seed), case_id),
                        scattering,
                        absorption,
                        energy_kev=energy,
                        target_tau_scattering=tau,
                        packets=args.packets,
                        chunk_size=args.chunk_size,
                        sigma_limit=args.sigma_limit,
                        min_expected_category=args.min_expected_category,
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
