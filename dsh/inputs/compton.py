"""Composition-checked optional Compton material and common energy support."""

from __future__ import annotations

import numpy as np

from ..physics.cold_compton import ENERGY_LIMITS_KEV, build_cold_compton_table
from ..physics.compton import load_compton_material_table, to_compton_physics
from ..physics.material_definitions import load_material_definition


def load_compton(config, material_config, scattering, absorption, physics, identity):
    if not config.enabled:
        return None, None
    digest = identity.get("definition_sha256")
    if digest is None:
        digest = scattering.metadata.get("material_definition_sha256")
    if not digest or any(
        table.metadata.get("material_definition_sha256") != digest
        for table in (scattering, absorption)
    ):
        raise ValueError(
            "compton: dust and absorption require one matching declared material_definition_sha256"
        )
    if config.kind == "cold_material":
        definition = load_material_definition(
            material_config.directory / "material_definition.json"
        )
        if definition.sha256 != digest:
            raise ValueError(
                "compton: material definition does not match dust/absorption"
            )
        grid = np.asarray(physics.energy_kev, dtype=float)
        grid = np.unique(
            np.r_[
                grid[(grid >= ENERGY_LIMITS_KEV[0]) & (grid <= ENERGY_LIMITS_KEV[1])],
                ENERGY_LIMITS_KEV,
            ]
        )
        table = build_cold_compton_table(
            definition, grid, grain_electrons=config.grain_electrons
        ).table
    else:
        table = load_compton_material_table(
            config.file, expected_material_definition_sha256=digest
        )
    low = max(float(physics.energy_kev[0]), float(table.energy_kev[0]))
    high = min(float(physics.energy_kev[-1]), float(table.energy_kev[-1]))
    if not low <= config.energy_cutoff_kev < high:
        raise ValueError(
            "compton.energy_cutoff_kev: must lie inside common material support, below its upper endpoint"
        )
    metadata = {
        "kind": config.kind,
        "energy_cutoff_kev": config.energy_cutoff_kev,
        "common_energy_support_kev": [low, high],
        "material_definition_sha256": digest,
        "table": table.metadata,
    }
    return to_compton_physics(table), metadata


def check_compton_source_support(source, compton, settings):
    if compton is None:
        return
    energies = np.asarray(
        source.energy_edges_kev
        if source.energy_edges_kev is not None
        else source.effective_energy_kev
    )
    if energies.min() < float(compton.energy_kev[0]) or energies.max() > float(
        compton.energy_kev[-1]
    ):
        raise ValueError(
            "source: energy support extends beyond Compton material tables"
        )
    if energies.min() < settings.energy_cutoff_kev:
        raise ValueError("source: initial energies lie below compton.energy_cutoff_kev")
