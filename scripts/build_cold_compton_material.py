"""Build an offline cold Compton archive from a complete material definition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dsh.physics.cold_compton import GRAIN_POLICIES, build_cold_compton_table
from dsh.physics.compton import write_compton_material_table
from dsh.physics.material_definitions import load_material_definition


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-definition", type=Path, required=True)
    parser.add_argument("--grain-electrons", choices=GRAIN_POLICIES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cosine-nodes", type=int, default=4097)
    args = parser.parse_args(argv)
    material = load_material_definition(args.material_definition)
    energy = np.unique(np.r_[np.geomspace(2, 10, 129), 3.3, 4.9, 6.9])
    result = build_cold_compton_table(
        material,
        energy,
        grain_electrons=args.grain_electrons,
        cosine_nodes=args.cosine_nodes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_compton_material_table(args.output, result.table)
    summary = {
        "production_compton_enabled": False,
        "material_definition": material.as_dict(),
        "abundance_audit": result.abundance_audit,
        "metadata": result.table.metadata,
        "energy_kev": energy.tolist(),
        "cross_section_cm2_per_h": result.table.cross_section_cm2_per_h.tolist(),
        "component_cross_sections_cm2_per_h": {
            name: values.tolist()
            for name, values in result.component_cross_sections_cm2_per_h.items()
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("Built: " + str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
