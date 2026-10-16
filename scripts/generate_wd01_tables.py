"""Offline WD01/D03 dust scattering and grain-absorption producer.

Set MIEPYTHON_USE_JIT=1 for faster generation. These tables do not include gas
or H2 absorption and do not change the default simulation material.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from dsh.physics.grain_populations import WD01Population
from dsh.physics.material_definitions import ElementalAbundances, load_wilm_abundances
from dsh.physics.optical_constants import default_wd01_energy_grid
from dsh.physics.wd01_tables import (
    calculate_wd01_tables,
    default_wd01_angle_grid,
    write_wd01_tables,
)


def _one_energy(energy, population, angle, method, order):
    return calculate_wd01_tables(
        population, [energy], angle, method=method, quadrature_order=order
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--normalization",
        choices=("wd01_original", "d03_0p93"),
        default="wd01_original",
    )
    parser.add_argument("--carbon-scale", type=float, default=1)
    parser.add_argument("--silicate-scale", type=float, default=1)
    parser.add_argument("--method", choices=("mie", "mie_adt"), default="mie")
    parser.add_argument("--energies", type=float, nargs="+")
    parser.add_argument("--quadrature-order", type=int, default=16)
    parser.add_argument("--forward-nodes", type=int, default=2048)
    parser.add_argument("--tail-nodes", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--abundances",
        help="wilm, or a JSON file containing abundance_id, atoms_per_h, reference",
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    population = WD01Population(
        args.normalization, args.carbon_scale, args.silicate_scale
    )
    definition = None
    if args.abundances:
        if args.abundances == "wilm":
            abundances = load_wilm_abundances()
        else:
            data = json.loads(Path(args.abundances).read_text(encoding="utf-8"))
            abundances = ElementalAbundances(
                **{
                    key: data[key]
                    for key in ("abundance_id", "atoms_per_h", "reference")
                }
            )
        # Fail before any expensive scattering work if the chemical budget fails.
        definition = population.material_definition(abundances, method=args.method)
    if args.audit_only:
        print(
            json.dumps(
                {
                    "population": population.as_dict(),
                    "inventory_wd01": population.inventory(),
                    "inventory_d03": population.inventory(density_model="d03"),
                    "chemical_audit": definition.audit()
                    if definition is not None
                    else None,
                },
                indent=2,
            )
        )
        return
    if args.output is None:
        parser.error("--output is required unless --audit-only is selected")
    energy = np.asarray(
        default_wd01_energy_grid() if args.energies is None else args.energies,
        dtype=np.float64,
    )
    if (
        energy.ndim != 1
        or energy.size < 2
        or np.any(~np.isfinite(energy))
        or np.any(np.diff(energy) <= 0)
        or energy[0] < 2
        or energy[-1] > 10
    ):
        parser.error(
            "energies must contain at least two increasing nodes within 2--10 keV"
        )
    angle = default_wd01_angle_grid(args.forward_nodes, args.tail_nodes)
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.workers == 1:
        arrays = calculate_wd01_tables(
            population,
            energy,
            angle,
            method=args.method,
            quadrature_order=args.quadrature_order,
            progress=lambda current, total, value: print(
                f"{current}/{total}: {value:.9g} keV", flush=True
            ),
        )
    else:
        rows = []
        work = partial(
            _one_energy,
            population=population,
            angle=angle,
            method=args.method,
            order=args.quadrature_order,
        )
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, row in enumerate(pool.map(work, energy)):
                rows.append(row)
                print(f"{i + 1}/{len(energy)}: {float(energy[i]):.9g} keV", flush=True)
        if any(
            row["producer_configuration"] != rows[0]["producer_configuration"]
            for row in rows[1:]
        ):
            raise ValueError("worker results have different producer configurations")
        arrays = {}
        for key in rows[0]:
            if key in ("producer_configuration", "scattering_angle_rad"):
                arrays[key] = rows[0][key]
            else:
                arrays[key] = np.concatenate(
                    [row[key] for row in rows],
                    axis=1
                    if key.startswith("component_")
                    or key
                    in (
                        "raw_angular_closure_relative_error",
                        "omitted_phase_probability",
                        "adt_quadrature_nodes",
                    )
                    else 0,
                )
    write_wd01_tables(
        args.output,
        arrays,
        population,
        method=args.method,
        quadrature_order=args.quadrature_order,
        material_definition=definition,
    )
    print(f"Wrote independent WD01 dust tables to {args.output}")


if __name__ == "__main__":
    main()
