"""Budget-declared WD01-shape gas/H2/grain materials, with explicit energy edges."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .absorption import (
    PhotoelectricAbsorptionTable,
    load_photoelectric_absorption_table,
)
from .dust import phase_cdf_from_differential_cross_section
from .energy_edges import bracketed_edge_grid
from .gas_absorption import load_atomic_basis, residual_gas_components
from .grain_populations import WD01Population
from .material_definitions import load_material_definition, load_wilm_abundances
from .materials import build_declared_material_tables
from .newdust import NewDustScatteringTable, load_newdust_scattering_table
from .wd01_tables import load_wd01_dust_tables


def wilm_budget_population(*, carbon_gas_fraction, silicon_gas_fraction):
    """Explicitly renormalize WD01 shapes to C/Si depletion targets.

    This is a new, experimental population, not nominal published WD01 and
    not the complete default TBabs dust mixture. Mg/Fe/O residuals follow
    MgFeSiO4 stoichiometry; other elements remain in the gas for this model.
    """
    fractions = (carbon_gas_fraction, silicon_gas_fraction)
    if any(not np.isfinite(f) or not 0 <= f < 1 for f in fractions):
        raise ValueError("target gas fractions must be finite in [0,1)")
    base = WD01Population().inventory(density_model="d03")
    abundances = load_wilm_abundances()
    population = WD01Population(
        carbon_scale=abundances.atoms_per_h["C"]
        * (1 - carbon_gas_fraction)
        / base["carbonaceous"]["atoms_per_h"]["C"],
        silicate_scale=abundances.atoms_per_h["Si"]
        * (1 - silicon_gas_fraction)
        / base["silicate"]["atoms_per_h"]["Si"],
    )
    population.material_definition(abundances)
    return population


def build_wd01_ism(
    source_directory, population, abundances, *, molecular_hydrogen_fraction, basis=None
):
    """Compose a full explicitly budgeted sphere material without new Mie solves.

    Existing independently checked component laws are scaled linearly in
    number/H, then regridded with the runtime's normalized-phase mixture.
    Chemistry and optical constants are otherwise unchanged. A full source
    population/material audit precedes construction.
    """
    source_directory = Path(source_directory)
    scattering, grains = load_wd01_dust_tables(source_directory)
    info = scattering.metadata["population"]
    source_population = WD01Population(
        **{
            key: info[key]
            for key in (
                "normalization",
                "carbon_scale",
                "silicate_scale",
                "maximum_radius_um",
            )
        }
    )
    if (
        source_population.sha256 != scattering.metadata["population_sha256"]
        or population.maximum_radius_um != source_population.maximum_radius_um
    ):
        raise ValueError("source and requested population shapes differ")
    method = scattering.metadata["scattering_method"]
    definition = population.material_definition(
        abundances,
        method=method,
        molecular_hydrogen_fraction=molecular_hydrogen_fraction,
    )
    basis = load_atomic_basis() if basis is None else basis
    diagnostic_path = source_directory / "wd01_producer_diagnostics.npz"
    with np.load(diagnostic_path, allow_pickle=False) as data:
        component = np.array(data["component_differential"])
        sca = np.array(data["component_mie_scattering"])
        absorbed = np.array(data["component_mie_absorption"])
        diagnostic_energy = np.array(data["energy_kev"])
    if (
        not np.array_equal(diagnostic_energy, scattering.energy_kev)
        or component.shape
        != (2, *scattering.differential_cross_section_cm2_per_sr_per_h.shape)
        or sca.shape != (2, len(diagnostic_energy))
        or absorbed.shape != sca.shape
        or any(
            np.any(~np.isfinite(a)) or np.any(a < 0) for a in (component, sca, absorbed)
        )
        or not np.allclose(
            component.sum(axis=0),
            scattering.differential_cross_section_cm2_per_sr_per_h,
            rtol=1e-12,
            atol=0,
        )
        or not np.allclose(
            sca.sum(axis=0),
            scattering.scattering_cross_section_cm2_per_h,
            rtol=1e-12,
            atol=0,
        )
        or not np.allclose(
            absorbed.sum(axis=0),
            grains.absorption_cross_section_cm2_per_h,
            rtol=1e-12,
            atol=0,
        )
    ):
        raise ValueError(
            "source component diagnostics do not match checked dust tables"
        )
    scales = np.array(
        [
            population.base_scale
            * population.carbon_scale
            / (source_population.base_scale * source_population.carbon_scale),
            population.base_scale
            * population.silicate_scale
            / (source_population.base_scale * source_population.silicate_scale),
        ]
    )
    source_sca = scales @ sca
    source_abs = scales @ absorbed
    source_diff = np.einsum("c,cea->ea", scales, component)
    source_phase = source_diff / source_sca[:, None]
    edges = basis.metadata["edge_energies_kev"]
    energy = bracketed_edge_grid(
        np.union1d(basis.energy_kev, scattering.energy_kev), edges
    )
    lower = np.clip(
        np.searchsorted(scattering.energy_kev, energy, side="right") - 1,
        0,
        len(scattering.energy_kev) - 2,
    )
    upper = lower + 1
    fraction = np.log(energy / scattering.energy_kev[lower]) / np.log(
        scattering.energy_kev[upper] / scattering.energy_kev[lower]
    )
    sigma = np.exp(
        np.interp(np.log(energy), np.log(scattering.energy_kev), np.log(source_sca))
    )
    differential = sigma[:, None] * (
        (1 - fraction[:, None]) * source_phase[lower]
        + fraction[:, None] * source_phase[upper]
    )
    sigma, cdf = phase_cdf_from_differential_cross_section(
        scattering.scattering_angle_rad, differential
    )
    grid_sha = hashlib.sha256(energy.astype("<f8").tobytes()).hexdigest()
    common = {
        "schema_version": 1,
        "material_definition_sha256": definition.sha256,
        "shared_energy_grid_sha256": grid_sha,
        "edge_energies_kev": edges,
        "energy_kev": energy.tolist(),
        "population": population.as_dict(),
        "population_sha256": population.sha256,
        "scattering_method": method,
        "normalization": "per H nucleus",
        "source_scattering_sha256": scattering.metadata["table_sha256"],
        "optical_source_sha256": scattering.metadata["optical_source_sha256"],
        "source_scattering_metadata_sha256": hashlib.sha256(
            (source_directory / "wd01_scattering_v1.json")
            .read_bytes()
            .replace(b"\r\n", b"\n")
        ).hexdigest(),
        "source_grain_absorption_sha256": grains.metadata["table_sha256"],
        "source_component_diagnostics_sha256": hashlib.sha256(
            diagnostic_path.read_bytes()
        ).hexdigest(),
        "component_number_scale_relative_to_source": scales.tolist(),
        "dust_energy_interpolation": "log total cross sections; convex mixture of normalized source phase laws; no extrapolation",
        "edge_interpolation": "one-sided endpoint values within narrow brackets; high side at threshold",
        "model_scope": "WD01-shaped graphite/MgFeSiO4 spheres with explicitly selected abundances; no PAH molecular optics, ionized gas or additional solid species",
    }
    new_scattering = NewDustScatteringTable(
        energy,
        scattering.scattering_angle_rad.copy(),
        differential,
        sigma,
        cdf,
        dict(
            common,
            producer="DSH checked WD01 components, explicit number scaling and energy regrid",
            quantity="intrinsic_differential_cross_section",
            differential_cross_section_unit="cm2 sr-1 H-1",
            integrated_cross_section_unit="cm2 H-1",
        ),
    )
    gas = residual_gas_components(definition, basis, energy, edge_energies_kev=edges)
    gas = tuple(
        replace(table, metadata=dict(table.metadata, **common)) for table in gas
    )
    grain_sigma = np.exp(
        np.interp(np.log(energy), np.log(scattering.energy_kev), np.log(source_abs))
    )
    new_grains = PhotoelectricAbsorptionTable(
        energy.copy(),
        grain_sigma,
        dict(
            common,
            producer="DSH Mie grain absorption, explicit number scaling and energy regrid",
            quantity="photoelectric_absorption_cross_section",
            cross_section_unit="cm2 H-1",
            absorption_coverage=["grains"],
        ),
    )
    components = (*gas, new_grains)
    _, total, physics = build_declared_material_tables(
        definition, new_scattering, components
    )
    return definition, new_scattering, components, total, physics


def write_wd01_ism(directory, material, *, recipe):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    definition, scattering, components, total, _ = material
    (directory / "material_definition.json").write_text(
        json.dumps(definition.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (directory / "abundance_audit.json").write_text(
        json.dumps(definition.audit(), indent=2) + "\n", encoding="utf-8"
    )
    (directory / "recipe.json").write_text(
        json.dumps(recipe, indent=2) + "\n", encoding="utf-8"
    )
    path = directory / "scattering.npz"
    np.savez_compressed(
        path,
        energy_kev=scattering.energy_kev,
        scattering_angle_rad=scattering.scattering_angle_rad,
        differential_cross_section_cm2_per_sr_per_h=scattering.differential_cross_section_cm2_per_sr_per_h,
        scattering_cross_section_cm2_per_h=scattering.scattering_cross_section_cm2_per_h,
        scattering_angle_cdf=scattering.scattering_angle_cdf,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            dict(
                scattering.metadata,
                table_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for name, table in zip(
        ("atomic_gas", "molecular_hydrogen", "grains", "total_absorption"),
        (*components, total),
        strict=True,
    ):
        path = directory / f"{name}.npz"
        np.savez_compressed(
            path,
            energy_kev=table.energy_kev,
            absorption_cross_section_cm2_per_h=table.absorption_cross_section_cm2_per_h,
        )
        metadata = dict(
            table.metadata,
            energy_kev=table.energy_kev.tolist(),
            table_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        if name == "total_absorption":
            metadata["components"] = [
                {
                    "producer": component.metadata.get("producer"),
                    "table_sha256": hashlib.sha256(
                        (directory / f"{phase}.npz").read_bytes()
                    ).hexdigest(),
                    "absorption_coverage": [phase],
                }
                for phase, component in zip(
                    ("atomic_gas", "molecular_hydrogen", "grains"),
                    components,
                    strict=True,
                )
            ]
        path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    return load_wd01_ism(directory)


def load_wd01_ism(directory):
    directory = Path(directory)
    definition = load_material_definition(directory / "material_definition.json")
    scattering = load_newdust_scattering_table(directory / "scattering.npz")
    components = tuple(
        load_photoelectric_absorption_table(directory / f"{name}.npz")
        for name in ("atomic_gas", "molecular_hydrogen", "grains")
    )
    scattering, total, physics = build_declared_material_tables(
        definition, scattering, components
    )
    stored = load_photoelectric_absorption_table(directory / "total_absorption.npz")
    if (
        stored.metadata.get("material_definition_sha256") != definition.sha256
        or stored.metadata.get("shared_energy_grid_sha256")
        != total.metadata.get("shared_energy_grid_sha256")
        or stored.metadata.get("edge_energies_kev")
        != total.metadata.get("edge_energies_kev")
        or stored.metadata.get("absorption_coverage")
        != total.metadata.get("absorption_coverage")
        or not np.array_equal(stored.energy_kev, total.energy_kev)
        or stored.metadata.get("components") != total.metadata["components"]
        or not np.array_equal(
            stored.absorption_cross_section_cm2_per_h,
            total.absorption_cross_section_cm2_per_h,
        )
    ):
        raise ValueError("stored total absorption does not match declared components")
    return definition, scattering, components, stored, physics
