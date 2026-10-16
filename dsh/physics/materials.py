"""Opt-in matching 2–10 keV scattering and absorption materials."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .absorption import (
    PhotoelectricAbsorptionTable,
    load_photoelectric_absorption_table,
)
from .material_definitions import MaterialDefinition
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


def build_declared_material_tables(definition, scattering, absorption_components):
    """Assemble a chemically declared material without overlapping absorption.

    This opt-in builder leaves the accepted legacy/effective providers intact.
    Producers must attach the same ``material_definition_sha256`` to every
    table. Each absorption table declares ``absorption_coverage`` using one or
    more of ``atomic_gas``, ``molecular_hydrogen``, and ``grains``. A full ISM
    absorption table covers all three, so it cannot be added to grain absorption.

    The declaration establishes input provenance and an elemental budget. It
    cannot independently prove that a producer computed the declared physics.
    Cross-section/phase closure is checked by the existing numerical builder.
    """
    if not isinstance(definition, MaterialDefinition):
        raise ValueError("definition must be a MaterialDefinition")
    definition.audit()
    digest = definition.sha256
    if scattering.metadata.get("material_definition_sha256") != digest:
        raise ValueError("scattering material definition does not match")
    if scattering.metadata.get("scattering_method") != definition.scattering_method:
        raise ValueError("scattering method does not match material definition")

    components = tuple(absorption_components)
    if not components:
        raise ValueError("explicit absorption components are required")
    required = {"atomic_gas", "molecular_hydrogen", "grains"}
    covered = set()
    arrays, sources = [], []
    grid_sha = scattering.metadata.get("shared_energy_grid_sha256")
    for component in components:
        if not isinstance(component, PhotoelectricAbsorptionTable):
            raise ValueError(
                "absorption components must be PhotoelectricAbsorptionTable values"
            )
        metadata = component.metadata
        if metadata.get("material_definition_sha256") != digest:
            raise ValueError("absorption material definition does not match")
        if not np.array_equal(scattering.energy_kev, component.energy_kev):
            raise ValueError("absorption components must have identical energy grids")
        if metadata.get("shared_energy_grid_sha256") != grid_sha:
            raise ValueError("absorption component shared-grid checksum differs")
        if metadata.get("edge_energies_kev", []) != scattering.metadata.get(
            "edge_energies_kev", []
        ):
            raise ValueError("absorption component edge declaration differs")
        if (
            metadata.get("quantity") != "photoelectric_absorption_cross_section"
            or metadata.get("cross_section_unit") != "cm2 H-1"
        ):
            raise ValueError(
                "absorption components must be intrinsic photoelectric cm2 H-1"
            )
        declaration = metadata.get("absorption_coverage")
        if (
            not isinstance(declaration, (list, tuple))
            or not declaration
            or any(not isinstance(part, str) for part in declaration)
        ):
            raise ValueError("absorption_coverage must explicitly list covered phases")
        coverage = set(declaration)
        if len(coverage) != len(declaration) or not coverage.issubset(required):
            raise ValueError("invalid or duplicate absorption_coverage phase")
        overlap = covered.intersection(coverage)
        if overlap:
            raise ValueError(f"overlapping absorption contributions: {sorted(overlap)}")
        covered.update(coverage)
        values = np.asarray(
            component.absorption_cross_section_cm2_per_h, dtype=np.float64
        )
        if (
            values.shape != scattering.energy_kev.shape
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
        ):
            raise ValueError(
                "absorption components must be finite nonnegative energy arrays"
            )
        arrays.append(values)
        sources.append(
            {
                "producer": metadata.get("producer"),
                "table_sha256": metadata.get("table_sha256"),
                "absorption_coverage": sorted(coverage),
            }
        )
    if covered != required:
        raise ValueError(f"missing absorption phases: {sorted(required - covered)}")
    total = np.sum(arrays, axis=0, dtype=np.float64)
    if not np.all(np.isfinite(total)):
        raise ValueError("combined absorption is nonfinite")
    metadata = {
        "schema_version": 1,
        "quantity": "photoelectric_absorption_cross_section",
        "cross_section_unit": "cm2 H-1",
        "material_definition_sha256": digest,
        "producer": "DSH disjoint absorption component sum",
        "absorption_coverage": sorted(covered),
        "components": sources,
    }
    if grid_sha is not None:
        metadata["shared_energy_grid_sha256"] = grid_sha
    if "edge_energies_kev" in scattering.metadata:
        metadata["edge_energies_kev"] = scattering.metadata["edge_energies_kev"]
    absorption = PhotoelectricAbsorptionTable(
        energy_kev=np.array(scattering.energy_kev, copy=True),
        absorption_cross_section_cm2_per_h=total,
        metadata=metadata,
    )
    physics = build_dust_physics_from_tables(scattering, absorption)
    return scattering, absorption, physics
