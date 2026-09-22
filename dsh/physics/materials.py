"""Opt-in matching 2–10 keV scattering and absorption materials."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .absorption import load_photoelectric_absorption_table
from .material_grid import load_material_grid
from .newdust import (
    build_dust_physics_from_tables,
    load_newdust_scattering_table,
)

_DATA = Path(__file__).resolve().parent.parent / "data"
DEFAULT_2_10_GRID = _DATA / "materials" / "material_grid_2_10_v2.json"
DEFAULT_2_10_SCATTERING = _DATA / "scattering" / "rg_drude_2_10_v2.npz"
DEFAULT_2_10_ABSORPTION = _DATA / "absorption" / "tbabs_wilm_vern_2_10_v2.npz"


def load_2_10_material_tables():
    """Load checked, exactly matched XSPEC TBabs and Gaussian RG/Drude tables.

    The frozen V1 tables remain the default in their existing loaders. This
    helper checks that both V2 tables refer to the bundled grid file before
    returning the host tables. It also constructs the JAX physics table once
    to validate their joint normalization and shapes.
    """

    grid = load_material_grid(DEFAULT_2_10_GRID)
    # Git may check text files out with CRLF on Windows while the XSPEC grid
    # was generated with LF. Only this newline conversion is permitted: the
    # checksum still fails for any change to the actual grid contents.
    grid_bytes = DEFAULT_2_10_GRID.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(grid_bytes).hexdigest()
    scattering = load_newdust_scattering_table(DEFAULT_2_10_SCATTERING)
    absorption = load_photoelectric_absorption_table(DEFAULT_2_10_ABSORPTION)
    if not np.array_equal(grid, scattering.energy_kev) or not np.array_equal(
        grid, absorption.energy_kev
    ):
        raise ValueError("2–10 keV tables do not match the bundled material grid")
    if (
        scattering.metadata.get("shared_energy_grid_sha256") != digest
        or absorption.metadata.get("shared_energy_grid_sha256") != digest
    ):
        raise ValueError("2–10 keV table/grid provenance checksum mismatch")
    physics = build_dust_physics_from_tables(scattering, absorption)
    return scattering, absorption, physics
