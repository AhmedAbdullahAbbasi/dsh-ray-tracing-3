"""Select explicit material files without coupling materials to source mode."""

import hashlib

import numpy as np

from ..physics.absorption import load_photoelectric_absorption_table
from ..physics.material_grid import load_material_grid
from ..physics.newdust import (
    build_dust_physics_from_tables,
    load_newdust_scattering_table,
)
from ..physics.wd01_ism import load_wd01_ism


def load_material(config):
    if config.kind == "declared_bundle":
        definition, scattering, _, absorption, physics = load_wd01_ism(config.directory)
        return (
            scattering,
            absorption,
            physics,
            {
                "kind": config.kind,
                "definition_sha256": definition.sha256,
                "definition": definition.as_dict(),
                "description": "Declared budgeted material; recipe and components define its physics. WD01-shaped bundles are not identical RefleX models.",
            },
        )
    scattering = load_newdust_scattering_table(config.scattering_file)
    absorption = load_photoelectric_absorption_table(config.absorption_file)
    if config.grid_file is not None:
        grid = load_material_grid(config.grid_file)
        digest = hashlib.sha256(
            config.grid_file.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for table in (scattering, absorption):
            if (
                not np.array_equal(grid, table.energy_kev)
                or table.metadata.get("shared_energy_grid_sha256") != digest
            ):
                raise ValueError(
                    "materials.grid_file: table/grid values or normalized-text checksum differ"
                )
    elif any(
        table.metadata.get("shared_energy_grid_sha256")
        for table in (scattering, absorption)
    ):
        raise ValueError(
            "materials.grid_file: required by the selected tables' shared-grid declaration"
        )
    physics = build_dust_physics_from_tables(scattering, absorption)
    return (
        scattering,
        absorption,
        physics,
        {
            "kind": config.kind,
            "scattering": dict(scattering.metadata),
            "absorption": dict(absorption.metadata),
        },
    )
