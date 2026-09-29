"""Validate material files once, then provide the existing JAX physics table.

The NPZ payload and adjacent JSON file are one versioned component. The
component loaders check units, intrinsic quantities, numerical closure and
SHA-256 provenance. Scattering and absorption are independently selectable;
the current transport requires their energy nodes to match exactly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..physics.absorption import (
    DEFAULT_TBABS_TABLE,
    PhotoelectricAbsorptionTable,
    load_photoelectric_absorption_table,
)
from ..physics.dust import DustPhysicsTable
from ..physics.material_grid import load_material_grid
from ..physics.materials import (
    DEFAULT_2_10_ABSORPTION,
    DEFAULT_2_10_GRID,
    DEFAULT_2_10_SCATTERING,
)
from ..physics.newdust import (
    DEFAULT_NEWDUST_TABLE,
    NewDustScatteringTable,
    build_dust_physics_from_tables,
    load_newdust_scattering_table,
)


@dataclass(frozen=True)
class MaterialInputs:
    """Host-side provenance and the numerical table consumed by transport."""

    scattering: NewDustScatteringTable
    absorption: PhotoelectricAbsorptionTable
    physics: DustPhysicsTable
    scattering_path: Path
    absorption_path: Path
    grid_path: Path | None

    @property
    def scattering_sha256(self) -> str:
        return str(self.scattering.metadata["table_sha256"])

    @property
    def absorption_sha256(self) -> str:
        return str(self.absorption.metadata["table_sha256"])


def packaged_material_paths(name: str) -> tuple[Path, Path, Path | None]:
    """Locate byte-identical legacy inputs without exposing model names to core."""

    if name == "v1":
        return DEFAULT_NEWDUST_TABLE, DEFAULT_TBABS_TABLE, None
    if name == "2-10":
        return DEFAULT_2_10_SCATTERING, DEFAULT_2_10_ABSORPTION, DEFAULT_2_10_GRID
    raise ValueError(f"unknown packaged material pair: {name}")


def load_material_inputs(
    scattering_path: str | Path,
    absorption_path: str | Path,
    *,
    grid_path: str | Path | None = None,
) -> MaterialInputs:
    """Load independent components and reject mismatched numerical grids.

    If a generated pair declares a shared-grid digest, supplying the grid
    file additionally checks its actual bytes (permitting only LF/CRLF text
    conversion). Replacing TBabs with another model requires a new absorption
    NPZ and JSON on the chosen grid, without editing the transport kernel.
    """

    scattering_file = Path(scattering_path)
    absorption_file = Path(absorption_path)
    scattering = load_newdust_scattering_table(scattering_file)
    absorption = load_photoelectric_absorption_table(absorption_file)
    physics = build_dust_physics_from_tables(scattering, absorption)

    grid_file = None if grid_path is None else Path(grid_path)
    if grid_file is not None:
        grid = load_material_grid(grid_file)
        if not np.array_equal(grid, scattering.energy_kev):
            raise ValueError("material files do not match the specified energy grid")
        declared = scattering.metadata.get("shared_energy_grid_sha256")
        if declared is not None:
            normalized = grid_file.read_bytes().replace(b"\r\n", b"\n")
            if hashlib.sha256(normalized).hexdigest() != declared:
                raise ValueError("material energy-grid provenance checksum mismatch")

    return MaterialInputs(
        scattering=scattering,
        absorption=absorption,
        physics=physics,
        scattering_path=scattering_file,
        absorption_path=absorption_file,
        grid_path=grid_file,
    )
