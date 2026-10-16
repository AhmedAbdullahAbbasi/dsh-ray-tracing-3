"""Build an explicitly renormalized Wilms-budget WD01-shape ISM material.

This experimental recipe preserves shapes, not the published WD01 abundance
normalization or its extinction fit. Target C/Si gas fractions are required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dsh.physics.material_definitions import load_wilm_abundances
from dsh.physics.wd01_ism import build_wd01_ism, wilm_budget_population, write_wd01_ism


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-dust", type=Path, default=Path("dsh/data/materials/wd01_d03_v1")
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--carbon-gas-fraction", type=float, required=True)
    p.add_argument("--silicon-gas-fraction", type=float, required=True)
    p.add_argument("--h2-fraction", type=float, required=True)
    a = p.parse_args()
    population = wilm_budget_population(
        carbon_gas_fraction=a.carbon_gas_fraction,
        silicon_gas_fraction=a.silicon_gas_fraction,
    )
    material = build_wd01_ism(
        a.source_dust,
        population,
        load_wilm_abundances(),
        molecular_hydrogen_fraction=a.h2_fraction,
    )
    recipe = {
        "recipe": "experimental WD01 shapes normalized to explicit C/Si depletion targets under Wilms abundances",
        "carbon_gas_fraction": a.carbon_gas_fraction,
        "silicon_gas_fraction": a.silicon_gas_fraction,
        "molecular_hydrogen_nuclei_fraction": a.h2_fraction,
        "population": population.as_dict(),
        "other_depletions": "derived from graphite/MgFeSiO4 inventory; not the default TBabs dust mixture",
        "extinction_curve_validation": "not performed; this recipe is not nominal published WD01",
    }
    write_wd01_ism(a.output, material, recipe=recipe)
    print(f"Wrote full material {material[0].sha256} at {a.output}")


if __name__ == "__main__":
    main()
