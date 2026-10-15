"""Optional material provider for development/physics-improvements at 7deb50c.

Place this module and its materials/ directory in the repository root, or put
this directory on sys.path. The original MRN/TBabs defaults are unchanged.
"""
from pathlib import Path

import numpy as np


def load_reflex_materials(directory=None, *, absorption_enabled=True):
    from dsh.physics.absorption import load_photoelectric_absorption_table
    from dsh.physics.newdust import (
        build_dust_physics_from_newdust,
        build_dust_physics_from_tables,
        load_newdust_scattering_table,
    )

    root = Path(directory) if directory is not None else Path(__file__).parent / "materials"
    scattering = load_newdust_scattering_table(root / "reflex_scattering_v1.npz")
    absorption = load_photoelectric_absorption_table(root / "reflex_absorption_v1.npz")
    if absorption_enabled:
        physics = build_dust_physics_from_tables(scattering, absorption)
    else:
        physics = build_dust_physics_from_newdust(scattering, np.zeros_like(scattering.energy_kev))
    return scattering, absorption, physics
