"""Scan TBabs and write an adaptive common energy grid for offline tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dsh.physics.material_grid import initial_material_grid, refine_material_grid
from scripts.generate_tbabs_table import (
    BIN_HALF_WIDTH_KEV,
    REFERENCE_NH22,
    _extract_one_energy,
    resolve_xspec,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xspec", default="xspec")
    parser.add_argument("--initial-step-kev", type=float, default=0.1)
    parser.add_argument("--relative-tolerance", type=float, default=5e-3)
    parser.add_argument("--minimum-interval-kev", type=float, default=1e-4)
    parser.add_argument("--maximum-nodes", type=int, default=2048)
    args = parser.parse_args()
    xspec = resolve_xspec(args.xspec)
    versions: set[str] = set()

    def evaluate(energy_kev: float) -> float:
        sigma, _, version = _extract_one_energy(
            xspec, energy_kev, REFERENCE_NH22, BIN_HALF_WIDTH_KEV
        )
        if version is not None:
            versions.add(version)
        return sigma

    energy, edge_intervals, max_error = refine_material_grid(
        evaluate,
        initial_energy_kev=initial_material_grid(args.initial_step_kev),
        relative_tolerance=args.relative_tolerance,
        minimum_interval_kev=args.minimum_interval_kev,
        maximum_nodes=args.maximum_nodes,
    )
    if len(versions) > 1:
        raise RuntimeError("XSPEC version changed during energy-grid generation")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "energy_kev": energy.tolist(),
        "range_kev": [2.0, 10.0],
        "initial_step_kev": args.initial_step_kev,
        "relative_tolerance": args.relative_tolerance,
        "minimum_interval_kev": args.minimum_interval_kev,
        "max_smooth_midpoint_relative_error": max_error,
        "edge_intervals_kev_and_relative_error": edge_intervals,
        "model": "XSPEC TBabs version 2, abund wilm",
        "producer_version": next(iter(versions), "unknown"),
        "note": (
            "Log-energy/log-cross-section midpoint scan; edge intervals remain "
            "approximate and are localized to the stated minimum interval. "
            "Only the sampled energy range is supported."
        ),
    }
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output}: {energy.size} energies, {len(edge_intervals)} edge intervals"
    )


if __name__ == "__main__":
    main()
