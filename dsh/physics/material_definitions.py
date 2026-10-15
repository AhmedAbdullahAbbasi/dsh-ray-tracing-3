"""Host-side chemical definitions for future independently generated materials.

All abundances and grain masses refer to one hydrogen nucleus, not one H2
molecule or one gram of gas. Size-distribution and optical-constant identifiers
describe the inputs to a producer; this module does not calculate cross sections.
Existing effective material tables remain usable without these declarations.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from types import MappingProxyType

# CIAAW 2024 abridged standard atomic weights, H through Zn. These are a
# declared mass-conversion convention, not an astrophysical isotope model.
# https://ciaaw.org/abridged-atomic-weights.htm
ATOMIC_WEIGHTS_U = MappingProxyType(
    dict(
        zip(
            [
                "H",
                "He",
                "Li",
                "Be",
                "B",
                "C",
                "N",
                "O",
                "F",
                "Ne",
                "Na",
                "Mg",
                "Al",
                "Si",
                "P",
                "S",
                "Cl",
                "Ar",
                "K",
                "Ca",
                "Sc",
                "Ti",
                "V",
                "Cr",
                "Mn",
                "Fe",
                "Co",
                "Ni",
                "Cu",
                "Zn",
            ],
            (
                1.0080,
                4.0026,
                6.94,
                9.0122,
                10.81,
                12.011,
                14.007,
                15.999,
                18.998,
                20.180,
                22.990,
                24.305,
                26.982,
                28.085,
                30.974,
                32.06,
                35.45,
                39.95,
                39.098,
                40.078,
                44.956,
                47.867,
                50.942,
                51.996,
                54.938,
                55.845,
                58.933,
                58.693,
                63.546,
                65.38,
            ),
            strict=True,
        )
    )
)
# 2022 CODATA atomic mass constant, converted from kg to g.
ATOMIC_MASS_UNIT_G = 1.66053906892e-24
MASS_CONVENTION = "CIAAW-2024-abridged/CODATA-2022-u"
DEFAULT_ABUNDANCES = (
    Path(__file__).resolve().parent.parent / "data/materials/wilm_abundances_v1.json"
)


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(
            f"{label} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return value


def _identifier(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _finite_sum(values, label):
    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise ValueError(f"{label} exceeds finite precision") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must remain finite")
    return result


def _element_map(values, label, *, positive=False):
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{label} must be a nonempty element mapping")
    result = {}
    for element, value in values.items():
        if element not in ATOMIC_WEIGHTS_U:
            raise ValueError(f"unsupported element in {label}: {element}")
        result[element] = _number(value, f"{label}.{element}", positive=positive)
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True)
class ElementalAbundances:
    """Total gas-plus-solid atoms per hydrogen nucleus; H is exactly one."""

    abundance_id: str
    atoms_per_h: Mapping[str, float]
    reference: str

    def __post_init__(self):
        _identifier(self.abundance_id, "abundance_id")
        _identifier(self.reference, "abundance reference")
        values = _element_map(self.atoms_per_h, "atoms_per_h")
        if values.get("H") != 1.0:
            raise ValueError("atoms_per_h must contain H=1 hydrogen nucleus")
        object.__setattr__(self, "atoms_per_h", values)

    def as_dict(self):
        return {
            "abundance_id": self.abundance_id,
            "atoms_per_h": dict(self.atoms_per_h),
            "reference": self.reference,
        }


def load_wilm_abundances() -> ElementalAbundances:
    """Load the explicit XSPEC Wilms vector, including its zero abundances."""
    data = json.loads(DEFAULT_ABUNDANCES.read_text(encoding="utf-8"))
    if (
        data.get("schema_version") != 1
        or data.get("normalization") != "atoms per H nucleus"
    ):
        raise ValueError("unsupported abundance schema or normalization")
    return ElementalAbundances(
        **{k: data[k] for k in ("abundance_id", "atoms_per_h", "reference")}
    )


@dataclass(frozen=True)
class GrainComponent:
    """A solid's composition and independently specified population mass.

    ``mass_per_h_g`` is the integral of the population's grain masses per H.
    A future size-distribution producer must verify that integral; an identifier
    alone does not establish that it implemented WD01, ZDA04, or another model.
    Stoichiometric coefficients may be fractional for a declared homogeneous
    mixture. Porosity/effective-medium prescriptions belong to the producer's
    optical-constant definition and are not inferred here.
    """

    component_id: str
    stoichiometry: Mapping[str, float]
    density_g_cm3: float
    mass_per_h_g: float
    size_distribution_id: str
    optical_constants_id: str
    reference: str
    size_parameters: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for name in (
            "component_id",
            "size_distribution_id",
            "optical_constants_id",
            "reference",
        ):
            _identifier(getattr(self, name), name)
        object.__setattr__(
            self,
            "stoichiometry",
            _element_map(self.stoichiometry, "stoichiometry", positive=True),
        )
        for name in ("density_g_cm3", "mass_per_h_g"):
            object.__setattr__(
                self,
                name,
                _number(getattr(self, name), name, positive=name == "density_g_cm3"),
            )
        if not isinstance(self.size_parameters, Mapping):
            raise ValueError("size_parameters must be a mapping")
        parameters = {}
        for name, value in self.size_parameters.items():
            _identifier(name, "size parameter name")
            # Shape exponents may be negative; all parameters must be finite.
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"size parameter {name} must be finite and numeric")
            parameters[name] = float(value)
        object.__setattr__(
            self, "size_parameters", MappingProxyType(dict(sorted(parameters.items())))
        )
        _ = self.formula_weight_u

    @property
    def formula_weight_u(self):
        return _finite_sum(
            (ATOMIC_WEIGHTS_U[e] * n for e, n in self.stoichiometry.items()),
            "grain formula weight",
        )

    def atoms_per_h(self):
        units_per_h = self.mass_per_h_g / (self.formula_weight_u * ATOMIC_MASS_UNIT_G)
        return {
            e: _number(n * units_per_h, f"grain {e} atoms/H")
            for e, n in self.stoichiometry.items()
        }

    def as_dict(self):
        return {
            "component_id": self.component_id,
            "stoichiometry": dict(self.stoichiometry),
            "density_g_cm3": self.density_g_cm3,
            "mass_per_h_g": self.mass_per_h_g,
            "size_distribution_id": self.size_distribution_id,
            "optical_constants_id": self.optical_constants_id,
            "reference": self.reference,
            "size_parameters": dict(self.size_parameters),
        }


@dataclass(frozen=True)
class MaterialDefinition:
    """Reproducible chemical inputs, independent of geometry and source flux.

    The H2 fraction counts hydrogen nuclei: f_H2 = 2*n(H2)/n_H. Hydrogen
    assigned to solids is unavailable to H2. Non-hydrogen gas molecules and
    ionization are outside this first contract. Gas fractions are derived from
    the solid inventory, never silently adjusted to fit an abundance budget.
    """

    material_id: str
    abundances: ElementalAbundances
    grains: tuple[GrainComponent, ...]
    molecular_hydrogen_fraction: float
    scattering_method: str
    reference: str

    def __post_init__(self):
        for name in ("material_id", "scattering_method", "reference"):
            _identifier(getattr(self, name), name)
        if not isinstance(self.abundances, ElementalAbundances):
            raise ValueError("abundances must be ElementalAbundances")
        grains = tuple(self.grains)
        if any(not isinstance(g, GrainComponent) for g in grains):
            raise ValueError("grains must contain GrainComponent values")
        if len({g.component_id for g in grains}) != len(grains):
            raise ValueError("grain component_id values must be unique")
        object.__setattr__(self, "grains", grains)
        fraction = _number(
            self.molecular_hydrogen_fraction, "molecular_hydrogen_fraction"
        )
        if fraction > 1:
            raise ValueError("molecular_hydrogen_fraction must lie in [0, 1]")
        object.__setattr__(self, "molecular_hydrogen_fraction", fraction)

    def as_dict(self):
        return {
            "schema_version": 1,
            "normalization": "per H nucleus",
            "mass_convention": MASS_CONVENTION,
            "material_id": self.material_id,
            "abundances": self.abundances.as_dict(),
            "grains": [
                g.as_dict() for g in sorted(self.grains, key=lambda g: g.component_id)
            ],
            "molecular_hydrogen_fraction": self.molecular_hydrogen_fraction,
            "scattering_method": self.scattering_method,
            "reference": self.reference,
        }

    @property
    def sha256(self):
        canonical = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def audit(self):
        """Return the closed elemental inventory or reject an overdrawn element."""
        solid_maps = [g.atoms_per_h() for g in self.grains]
        elements = set(self.abundances.atoms_per_h).union(
            *(m.keys() for m in solid_maps)
        )
        rows = {}
        for element in sorted(elements):
            total = self.abundances.atoms_per_h.get(element, 0.0)
            solid = _finite_sum(
                (m.get(element, 0.0) for m in solid_maps), "solid atoms/H"
            )
            tolerance = 32 * math.ulp(max(total, solid))
            if solid > total + tolerance:
                raise ValueError(
                    f"grain inventory exceeds {element} abundance: {solid:.12g} > {total:.12g} atoms/H"
                )
            gas = max(0.0, total - solid)
            rows[element] = {
                "total_atoms_per_h": total,
                "grain_atoms_per_h": solid,
                "gas_atoms_per_h": gas,
                "gas_fraction": gas / total if total > 0 else None,
            }
        gas_h = rows["H"]["gas_atoms_per_h"]
        if self.molecular_hydrogen_fraction > gas_h + 32 * math.ulp(gas_h):
            raise ValueError(
                "molecular hydrogen plus grain hydrogen exceeds H abundance"
            )
        dust_mass = _finite_sum((g.mass_per_h_g for g in self.grains), "dust mass/H")
        total_mass = ATOMIC_MASS_UNIT_G * _finite_sum(
            (ATOMIC_WEIGHTS_U[e] * row["total_atoms_per_h"] for e, row in rows.items()),
            "total mass/u/H",
        )
        gas_mass = ATOMIC_MASS_UNIT_G * _finite_sum(
            (ATOMIC_WEIGHTS_U[e] * row["gas_atoms_per_h"] for e, row in rows.items()),
            "gas mass/u/H",
        )
        if gas_mass <= 0:
            raise ValueError("material must contain nonzero gas mass")
        if not math.isclose(dust_mass + gas_mass, total_mass, rel_tol=1e-13, abs_tol=0):
            raise ValueError("gas-plus-grain mass inventory does not close")
        return {
            "material_id": self.material_id,
            "material_definition_sha256": self.sha256,
            "normalization": "per H nucleus",
            "mass_convention": MASS_CONVENTION,
            "elements": rows,
            "dust_mass_per_h_g": dust_mass,
            "gas_mass_per_h_g": gas_mass,
            "total_mass_per_h_g": total_mass,
            "dust_to_gas_mass_ratio": dust_mass / gas_mass,
            "atomic_hydrogen_atoms_per_h": max(
                0.0, gas_h - self.molecular_hydrogen_fraction
            ),
            "molecular_hydrogen_molecules_per_h": self.molecular_hydrogen_fraction / 2,
            "grain_hydrogen_atoms_per_h": rows["H"]["grain_atoms_per_h"],
        }


def load_material_definition(path: str | Path) -> MaterialDefinition:
    """Read a complete explicit definition; reject unsupported conventions."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("material definition must be a JSON object")
    for key, expected in (
        ("schema_version", 1),
        ("normalization", "per H nucleus"),
        ("mass_convention", MASS_CONVENTION),
    ):
        if data.pop(key, None) != expected:
            raise ValueError(f"unsupported material {key}")
    try:
        data["abundances"] = ElementalAbundances(**data["abundances"])
        data["grains"] = tuple(GrainComponent(**g) for g in data["grains"])
        definition = MaterialDefinition(**data)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid material definition: {exc}") from exc
    definition.audit()
    return definition
