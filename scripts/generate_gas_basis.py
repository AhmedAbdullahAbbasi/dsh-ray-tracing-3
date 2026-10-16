"""Generate the portable, abundance-independent 2--10 keV gas/H2 basis."""

from __future__ import annotations

import argparse
from pathlib import Path

from dsh.physics.gas_absorption import (
    TBnewGasBackend,
    generate_atomic_basis,
    verner_edge_catalog,
    write_atomic_basis,
)
from dsh.physics.optical_constants import default_wd01_energy_grid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", type=Path, required=True)
    p.add_argument("--phfit-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    b = generate_atomic_basis(
        TBnewGasBackend(a.backend),
        default_wd01_energy_grid(),
        verner_edge_catalog(a.phfit_source),
    )
    b = write_atomic_basis(a.output, b)
    print(f"Wrote {len(b.energy_kev)} energies and 30 neutral atoms plus H2")


if __name__ == "__main__":
    main()
