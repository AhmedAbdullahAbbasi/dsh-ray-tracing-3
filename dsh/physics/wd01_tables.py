"""Generate independent WD01-size-law/D03-optics dust material tables.

These are dust scattering plus grain absorption, not total ISM absorption.
Gas and H2 terms must be supplied separately for a complete material. Existing
production defaults and the RefleX compatibility provider are not replaced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .absorption import load_photoelectric_absorption_table
from .dust import phase_cdf_from_differential_cross_section
from .grain_populations import WD01Population
from .newdust import load_newdust_scattering_table
from .optical_constants import default_wd01_energy_grid, load_draine_optical_constants
from .sphere_scattering import (
    ADT_MAXIMUM_ANGLE_RAD,
    ADT_MAXIMUM_INDEX_CONTRAST,
    ADT_MAXIMUM_PHASE_SHIFT,
    ADT_MINIMUM_SIZE_PARAMETER,
    hybrid_sphere,
    mie_sphere,
)


def _producer_source_hashes():
    directory = Path(__file__).parent
    return {
        name: hashlib.sha256(
            (directory / name).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for name in (
            "grain_populations.py",
            "optical_constants.py",
            "sphere_scattering.py",
            "wd01_tables.py",
        )
    }


def default_wd01_angle_grid(forward_nodes=2048, tail_nodes=512):
    """Logarithmic forward and outer grids with an explicit ADT seam node."""
    if forward_nodes < 64 or tail_nodes < 64:
        raise ValueError("angle grids need at least 64 forward and outer nodes")
    return np.r_[
        0.0,
        np.geomspace(1e-9, ADT_MAXIMUM_ANGLE_RAD, forward_nodes),
        np.geomspace(ADT_MAXIMUM_ANGLE_RAD, np.pi, tail_nodes + 1)[1:],
    ]


def calculate_wd01_tables(
    population=None,
    energy_kev=None,
    angle_rad=None,
    *,
    method="mie",
    quadrature_order=16,
    maximum_raw_closure_error=3e-3,
    progress=None,
):
    """Integrate cross sections over grain sizes and graphite orientations.

    Each component's angular integral is checked against its independently
    computed Mie total before a small closure correction is applied. Totals,
    differential law and CDF subsequently share exactly one normalization.
    Tiny quadrature-node phase contributions may be skipped only when their
    summed Mie scattering probability is below 1e-10 per component/energy.
    The integrated scattering and absorption totals still include those nodes.
    """
    population = WD01Population() if population is None else population
    if not isinstance(population, WD01Population):
        raise ValueError("population must be WD01Population")
    energy = np.asarray(
        default_wd01_energy_grid() if energy_kev is None else energy_kev,
        dtype=np.float64,
    )
    angle = np.asarray(
        default_wd01_angle_grid() if angle_rad is None else angle_rad, dtype=np.float64
    )
    if (
        energy.ndim != 1
        or energy.size < 1
        or np.any(~np.isfinite(energy))
        or np.any(np.diff(energy) <= 0)
        or energy[0] < 2
        or energy[-1] > 10
    ):
        raise ValueError("energies must increase within 2--10 keV")
    if (
        angle.ndim != 1
        or angle.size < 2
        or angle[0] != 0
        or angle[-1] != np.pi
        or np.any(~np.isfinite(angle))
        or np.any(np.diff(angle) <= 0)
    ):
        raise ValueError("angles must increase from zero to pi")
    if method not in ("mie", "mie_adt"):
        raise ValueError("method must be mie or mie_adt")
    if (
        not np.isfinite(maximum_raw_closure_error)
        or not 0 < maximum_raw_closure_error <= 0.01
    ):
        raise ValueError("raw angular closure tolerance must be positive and <=1%")
    sphere = mie_sphere if method == "mie" else hybrid_sphere
    optical = {
        name: load_draine_optical_constants(name)
        for name in ("silicate", "graphite_parallel", "graphite_perpendicular")
    }
    shape = (2, energy.size, angle.size)
    component_differential = np.zeros(shape)
    component_sca = np.zeros(shape[:2])
    component_abs = np.zeros(shape[:2])
    raw_closure = np.zeros(shape[:2])
    skipped_fraction = np.zeros(shape[:2])
    adt_nodes = np.zeros(shape[:2], dtype=int)
    for i, e in enumerate(energy):
        for c, component in enumerate(("carbonaceous", "silicate")):
            radius, count = population.quadrature(component, quadrature_order)
            orientations = (
                (("graphite_parallel", 1 / 3), ("graphite_perpendicular", 2 / 3))
                if c == 0
                else (("silicate", 1.0),)
            )
            nodes = []
            for optical_name, orientation_weight in orientations:
                index = optical[optical_name].index_at_energy(e)
                for a, number in zip(radius, count, strict=True):
                    # Exact integrated Mie efficiencies first: they provide a
                    # quantitative bound on any omitted angular contribution.
                    result = mie_sphere(a, e, index, [0.0])
                    weight = number * orientation_weight
                    nodes.append((a, index, weight, result.scattering_cm2))
                    component_sca[c, i] += weight * result.scattering_cm2
                    component_abs[c, i] += weight * result.absorption_cm2
            total = component_sca[c, i]
            if not total > 0:
                raise ValueError("WD01 component has no scattering opacity")
            for a, index, weight, cross_section in nodes:
                contribution = weight * cross_section
                if contribution <= total * 1e-13:
                    skipped_fraction[c, i] += contribution / total
                    continue
                result = sphere(a, e, index, angle)
                component_differential[c, i] += weight * result.differential_cm2_per_sr
                adt_nodes[c, i] += result.phase_method != "mie"
            if skipped_fraction[c, i] > 1e-10:
                raise ValueError("omitted phase probability exceeds its explicit bound")
            raw = np.trapezoid(
                2 * np.pi * np.sin(angle) * component_differential[c, i], angle
            )
            raw_closure[c, i] = raw / total - 1
            if (
                not np.isfinite(raw)
                or raw <= 0
                or abs(raw_closure[c, i]) > maximum_raw_closure_error
            ):
                raise ValueError(
                    f"angular closure failed for {component} at {e:g} keV: {raw_closure[c, i]:g}; refine the angular grid or use full Mie"
                )
            component_differential[c, i] *= total / raw
        if progress is not None:
            progress(i + 1, energy.size, float(e))
    differential = component_differential.sum(axis=0)
    sigma_sca, cdf = phase_cdf_from_differential_cross_section(angle, differential)
    return {
        "producer_configuration": {
            "population_sha256": population.sha256,
            "method": method,
            "quadrature_order": quadrature_order,
            "angle_sha256": hashlib.sha256(angle.tobytes()).hexdigest(),
            "source_sha256": _producer_source_hashes(),
        },
        "energy_kev": energy,
        "scattering_angle_rad": angle,
        "differential_cross_section_cm2_per_sr_per_h": differential,
        "scattering_cross_section_cm2_per_h": sigma_sca,
        "scattering_angle_cdf": cdf,
        "grain_absorption_cross_section_cm2_per_h": component_abs.sum(axis=0),
        "component_differential": component_differential,
        "component_mie_scattering": component_sca,
        "component_mie_absorption": component_abs,
        "raw_angular_closure_relative_error": raw_closure,
        "omitted_phase_probability": skipped_fraction,
        "adt_quadrature_nodes": adt_nodes,
    }


def write_wd01_tables(
    directory, arrays, population, *, method, quadrature_order, material_definition=None
):
    """Write the existing portable runtime schema plus producer diagnostics."""
    import miepython

    directory = Path(directory)
    expected_configuration = {
        "population_sha256": population.sha256,
        "method": method,
        "quadrature_order": quadrature_order,
        "angle_sha256": hashlib.sha256(
            arrays["scattering_angle_rad"].tobytes()
        ).hexdigest(),
        "source_sha256": _producer_source_hashes(),
    }
    if arrays.get("producer_configuration") != expected_configuration:
        raise ValueError("writer inputs differ from calculated producer configuration")
    if arrays["energy_kev"].size < 2:
        raise ValueError("runtime tables require at least two energy nodes")
    if material_definition is not None:
        material_definition.audit()
        expected = population.material_definition(
            material_definition.abundances,
            method=method,
            molecular_hydrogen_fraction=material_definition.molecular_hydrogen_fraction,
        )
        if expected.sha256 != material_definition.sha256:
            raise ValueError(
                "material definition differs from generated WD01 population"
            )
    directory.mkdir(parents=True, exist_ok=True)
    optical = {
        name: load_draine_optical_constants(name)
        for name in ("silicate", "graphite_parallel", "graphite_perpendicular")
    }
    metadata = {
        "schema_version": 1,
        "energy_kev": arrays["energy_kev"].tolist(),
        "producer": "DSH independent WD01/D03 homogeneous-sphere producer",
        "miepython_version": miepython.__version__,
        "numpy_version": np.__version__,
        "miepython_jit_enabled": bool(miepython.USE_JIT),
        "producer_source_sha256": arrays["producer_configuration"]["source_sha256"],
        "scattering_method": method,
        "population": population.as_dict(),
        "population_sha256": population.sha256,
        "optical_source_sha256": {
            name: table.source_sha256 for name, table in optical.items()
        },
        "quadrature_order_per_log_radius_interval": quadrature_order,
        "number_of_angles": arrays["scattering_angle_rad"].size,
        "maximum_raw_angular_closure_relative_error": float(
            np.max(np.abs(arrays["raw_angular_closure_relative_error"]))
        ),
        "maximum_omitted_phase_probability": float(
            np.max(arrays["omitted_phase_probability"])
        ),
        "grain_inventory_wd01_densities": population.inventory(),
        "grain_inventory_d03_optical_densities": population.inventory(
            density_model="d03"
        ),
        "absorption_scope": "grains only; gas and H2 absorption are not included",
        "carbon_optics": "bulk graphite sphere approximation for all carbonaceous sizes; not a PAH molecular model",
        "energy_interpolation_scope": "native D03 samples retained; high-resolution interpolation across edges has not yet been validated",
        "adt_forward_domain": {
            "minimum_x": ADT_MINIMUM_SIZE_PARAMETER,
            "maximum_abs_m_minus_1": ADT_MAXIMUM_INDEX_CONTRAST,
            "maximum_abs_rho": ADT_MAXIMUM_PHASE_SHIFT,
            "maximum_angle_rad": ADT_MAXIMUM_ANGLE_RAD,
            "outside_domain": "exact Mie",
            "integrated_totals": "exact Mie",
        },
        "chemical_status": "grain inventory only; no total abundance set selected",
    }
    if material_definition is not None:
        metadata["material_definition_sha256"] = material_definition.sha256
        metadata["chemical_status"] = "passed explicit abundance audit"
        (directory / "material_definition.json").write_text(
            json.dumps(material_definition.as_dict(), indent=2) + "\n", encoding="utf-8"
        )
    scattering_keys = (
        "energy_kev",
        "scattering_angle_rad",
        "differential_cross_section_cm2_per_sr_per_h",
        "scattering_cross_section_cm2_per_h",
        "scattering_angle_cdf",
    )
    scattering_path = directory / "wd01_scattering_v1.npz"
    absorption_path = directory / "wd01_grain_absorption_v1.npz"
    np.savez_compressed(
        scattering_path, **{key: arrays[key] for key in scattering_keys}
    )
    np.savez_compressed(
        absorption_path,
        energy_kev=arrays["energy_kev"],
        absorption_cross_section_cm2_per_h=arrays[
            "grain_absorption_cross_section_cm2_per_h"
        ],
    )
    scatter_metadata = dict(
        metadata,
        quantity="intrinsic_differential_cross_section",
        differential_cross_section_unit="cm2 sr-1 H-1",
        integrated_cross_section_unit="cm2 H-1",
        table_sha256=hashlib.sha256(scattering_path.read_bytes()).hexdigest(),
    )
    absorb_metadata = dict(
        metadata,
        quantity="photoelectric_absorption_cross_section",
        cross_section_unit="cm2 H-1",
        absorption_coverage=["grains"],
        table_sha256=hashlib.sha256(absorption_path.read_bytes()).hexdigest(),
    )
    for path, contents in (
        (scattering_path, scatter_metadata),
        (absorption_path, absorb_metadata),
    ):
        path.with_suffix(".json").write_text(
            json.dumps(contents, indent=2) + "\n", encoding="utf-8"
        )
    np.savez_compressed(
        directory / "wd01_producer_diagnostics.npz",
        **{
            key: arrays[key]
            for key in (
                "energy_kev",
                "component_differential",
                "component_mie_scattering",
                "component_mie_absorption",
                "raw_angular_closure_relative_error",
                "omitted_phase_probability",
                "adt_quadrature_nodes",
            )
        },
    )
    # Exercise the real existing loaders before declaring generation complete.
    return load_wd01_dust_tables(directory)


def load_wd01_dust_tables(directory):
    """Return scattering and explicitly grain-only absorption host tables."""
    directory = Path(directory)
    scattering = load_newdust_scattering_table(directory / "wd01_scattering_v1.npz")
    absorption = load_photoelectric_absorption_table(
        directory / "wd01_grain_absorption_v1.npz"
    )
    for table in (scattering, absorption):
        if table.metadata.get("population_sha256") != scattering.metadata.get(
            "population_sha256"
        ):
            raise ValueError("WD01 population provenance mismatch")
        if table.metadata.get("optical_source_sha256") != scattering.metadata.get(
            "optical_source_sha256"
        ):
            raise ValueError("WD01 optical provenance mismatch")
    if not np.array_equal(
        scattering.energy_kev, absorption.energy_kev
    ) or absorption.metadata.get("absorption_coverage") != ["grains"]:
        raise ValueError(
            "WD01 dust tables must share an energy grid and contain grain absorption only"
        )
    return scattering, absorption
