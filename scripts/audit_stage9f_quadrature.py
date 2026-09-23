"""Re-evaluate Stage 9F first-order references without rerunning photon transport.

The original Cartesian slope quadrature can cancel large changes between
adjacent narrow time bins. This audit uses a radial integral with exact
azimuthal support for the symmetric 9F shell launch cone; it reads the saved
analog estimates and uncertainty from the original JSON reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from dsh.physics.materials import (
    DEFAULT_2_10_ABSORPTION,
    DEFAULT_2_10_SCATTERING,
    load_2_10_material_tables,
)
from dsh.validation.absorbed_observer import first_order_radial_quadrature

BOUNDS = ((-0.003, 0.003), (-0.003, 0.003))


def audit_case(report, physics, digests):
    if report.get("stage") != "9F_absorbed_observer":
        raise ValueError("expected a Stage 9F absorbed-observer report")
    case = report["case"]
    config = report["config"]
    energy = float(case["energy_kev"])
    column = float(case["column_cm2"])
    time_edges = np.asarray(case["time_edges_s"], dtype=np.float64)
    if (
        not np.isfinite(column)
        or column <= 0
        or time_edges.shape != (5,)
        or not np.all(np.isfinite(time_edges))
        or not np.all(np.diff(time_edges) > 0)
    ):
        raise ValueError("unexpected Stage 9F shell column or time-bin edges")
    if config["packets_per_seed"] < 2 or len(config["seeds"]) < 3:
        raise ValueError("incomplete Stage 9F photon ensemble")

    coarse = first_order_radial_quadrature(
        physics, energy, column, BOUNDS, time_edges, n_radius=24, n_depth=24
    )
    fine = first_order_radial_quadrature(
        physics, energy, column, BOUNDS, time_edges, n_radius=64, n_depth=56
    )
    relative_bins = np.abs(fine - coarse) / np.maximum(np.abs(fine), 1e-100)
    original = case["quadrature"]
    old_coarse = np.asarray(original["coarse"], dtype=np.float64)
    old_fine = np.asarray(original["fine"], dtype=np.float64)
    if old_coarse.shape != fine.shape or old_fine.shape != fine.shape:
        raise ValueError("original quadrature has unexpected time bins")

    def compare(name, sel):
        old = case[name]
        old_quadrature_error = float(abs(old_coarse[sel].sum() - old_fine[sel].sum()))
        combined_se = float(old["standard_error"])
        photon_se = math.sqrt(max(combined_se**2 - old_quadrature_error**2, 0.0))
        radial_quadrature_error = float(abs(coarse[sel].sum() - fine[sel].sum()))
        total_se = math.hypot(photon_se, radial_quadrature_error)
        analog = float(old["analog"])
        reference = float(fine[sel].sum())
        z = (analog - reference) / total_se if total_se > 0 else float("inf")
        return {
            "analog": analog,
            "radial_reference": reference,
            "photon_standard_error_from_saved_report": photon_se,
            "radial_quadrature_difference": radial_quadrature_error,
            "z": z,
            "passed": bool(abs(z) <= config["sigma_limit"]),
        }

    integrated = compare("first_order_reference", slice(None))
    early = compare("first_order_early_reference", slice(0, 2))
    tolerance = float(config["quadrature_rtol"])
    cartesian_difference = float(abs(old_fine.sum() - fine.sum()) / fine.sum())
    checks = {
        "original_report_passed": bool(report["all_passed"] and case["all_passed"]),
        "material_digests_match": all(
            report.get(name) == digest for name, digest in digests.items()
        ),
        "each_radial_time_bin_converged": bool(np.all(relative_bins <= tolerance)),
        "integrated_cartesian_crosscheck": bool(cartesian_difference <= tolerance),
        "integrated_first_order_agrees": integrated["passed"],
        "early_first_order_agrees": early["passed"],
    }
    return {
        "energy_kev": energy,
        "source_git_head": report["git_head"],
        "source_packet_count": config["packets_per_seed"] * len(config["seeds"]),
        "method": "axisymmetric-shell radial slope quadrature; exact rectangular azimuth",
        "radial_coarse": coarse.tolist(),
        "radial_fine": fine.tolist(),
        "relative_each_time_bin_change": relative_bins.tolist(),
        "per_bin_relative_tolerance": tolerance,
        "cartesian_fine_integrated_relative_difference": cartesian_difference,
        "integrated_first_order": integrated,
        "early_first_order": early,
        "checks": checks,
        "all_passed": bool(all(checks.values())),
        "limitation": "Saved Stage 9F reports contain no per-time-bin analog moments; this audits only the numerical reference in individual bins.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    physics = load_2_10_material_tables()[2]
    digests = {
        "scattering_table_sha256": hashlib.sha256(
            DEFAULT_2_10_SCATTERING.read_bytes()
        ).hexdigest(),
        "absorption_table_sha256": hashlib.sha256(
            DEFAULT_2_10_ABSORPTION.read_bytes()
        ).hexdigest(),
    }
    cases = []
    for path in args.reports:
        source_bytes = path.read_bytes()
        original = json.loads(source_bytes)
        case = audit_case(original, physics, digests)
        case["source_report"] = str(path)
        case["source_report_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        cases.append(case)
    audit = {"stage": "9F_radial_quadrature_audit", "cases": cases}
    audit["all_passed"] = all(c["all_passed"] for c in cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved {args.output}; all_passed={audit['all_passed']}")
    return 0 if audit["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
