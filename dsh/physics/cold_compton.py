"""Cold neutral incoherent scattering; independent of production transport.

The model is unpolarized KN times an incoherent scattering function, with
stationary-electron recoil. Atomic, molecular and solid inventories are
explicit. This is not a shell-resolved or Doppler-broadened Compton kernel.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .compton import (
    ELECTRON_REST_ENERGY_KEV,
    THOMSON_CROSS_SECTION_CM2,
    ComptonMaterialTable,
    build_compton_material_table,
)
from .material_definitions import ATOMIC_WEIGHTS_U, MaterialDefinition

HC_KEV_ANGSTROM = 12.398419843320026  # exact SI h, c, elementary charge
BOHR_RADIUS_ANGSTROM = 0.529177210544  # CODATA 2022
SOURCE_PATH = (
    Path(__file__).resolve().parent.parent
    / "data/materials/cold_compton_v1/scattering_functions.json"
)
ENERGY_LIMITS_KEV = (2.0, 10.0)
ATOMIC_NUMBERS = {symbol: z for z, symbol in enumerate(ATOMIC_WEIGHTS_U, 1)}
GRAIN_POLICIES = ("omit", "atomic_incoherent")


def _canonical_sha(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _monotone_cubic(x, y, query):
    """Shape-preserving Hermite interpolation (PCHIP), using only NumPy."""
    h = np.diff(x)
    delta = np.diff(y) / h
    slopes = np.zeros_like(y)
    same_sign = delta[:-1] * delta[1:] > 0
    w1, w2 = 2 * h[1:] + h[:-1], h[1:] + 2 * h[:-1]
    # Divide only where both secants are nonzero and have the same sign.
    a, b = delta[:-1][same_sign], delta[1:][same_sign]
    slopes[1:-1][same_sign] = (w1[same_sign] + w2[same_sign]) / (
        w1[same_sign] / a + w2[same_sign] / b
    )
    for endpoint, h0, h1, d0, d1 in (
        (0, h[0], h[1], delta[0], delta[1]),
        (-1, h[-1], h[-2], delta[-1], delta[-2]),
    ):
        slope = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(slope) != np.sign(d0):
            slope = 0.0
        elif np.sign(d0) != np.sign(d1) and abs(slope) > 3 * abs(d0):
            slope = 3 * d0
        slopes[endpoint] = slope
    left = np.clip(np.searchsorted(x, query, side="right") - 1, 0, len(x) - 2)
    t = (query - x[left]) / h[left]
    return (
        (2 * t**3 - 3 * t**2 + 1) * y[left]
        + (t**3 - 2 * t**2 + t) * h[left] * slopes[left]
        + (-2 * t**3 + 3 * t**2) * y[left + 1]
        + (t**3 - t**2) * h[left] * slopes[left + 1]
    )


@dataclass(frozen=True)
class IncoherentFunctions:
    """Offline source data, with explicit interpolation and zero-angle limit."""

    x_inverse_angstrom: np.ndarray
    atomic_s: np.ndarray
    h2_x_inverse_angstrom: np.ndarray
    h2_s_per_h_nucleus: np.ndarray
    sha256: str
    provenance: dict

    def evaluate(self, species, x_inverse_angstrom):
        x = np.asarray(x_inverse_angstrom, dtype=float)
        if not np.isfinite(x).all() or np.any(x < 0):
            raise ValueError(
                "momentum-transfer parameter must be finite and nonnegative"
            )
        if species == "H":
            # Exact isolated-H Pirenne law. expm1 preserves the x -> 0 limit.
            return -np.expm1(-4 * np.log1p((2 * np.pi * BOHR_RADIUS_ANGSTROM * x) ** 2))
        if species == "H2":
            grid, values = self.h2_x_inverse_angstrom, self.h2_s_per_h_nucleus
        elif species in ATOMIC_NUMBERS:
            grid = self.x_inverse_angstrom
            values = self.atomic_s[ATOMIC_NUMBERS[species] - 1]
        else:
            raise ValueError("unsupported neutral scattering species: " + str(species))
        if np.any(x > grid[-1]):
            raise ValueError("scattering-function request exceeds source support")
        # S ~ x^2 at the origin. Positive interior nodes use log-log linear
        # interpolation. No spline overshoot and no constant forward floor.
        positive = np.maximum(x, grid[0])
        interpolation = _monotone_cubic if species == "H2" else np.interp
        if species == "H2":
            result = np.exp(
                interpolation(np.log(grid), np.log(values), np.log(positive))
            )
        else:
            result = np.exp(
                interpolation(np.log(positive), np.log(grid), np.log(values))
            )
        return np.where(x < grid[0], values[0] * (x / grid[0]) ** 2, result)


def load_incoherent_functions(path=SOURCE_PATH):
    """Load checksummed numeric data without xraylib, SciPy or network access."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = data.pop("content_sha256", None)
    actual = _canonical_sha(data)
    if expected != actual:
        raise ValueError("incoherent source checksum mismatch")
    if data.get("schema_version") != 1 or data.get("x_unit") != "angstrom^-1":
        raise ValueError("unsupported incoherent source schema/units")
    if data.get("h2_normalization") != "per H nucleus in H2":
        raise ValueError("H2 scattering-function normalization must be explicit")
    x = np.array(data["x_inverse_angstrom"], dtype=float)
    atomic = np.array(data["atomic_s"], dtype=float)
    h2_x = np.array(data["h2_x_inverse_angstrom"], dtype=float)
    h2 = np.array(data["h2_s_per_h_nucleus"], dtype=float)
    for grid in (x, h2_x):
        if (
            grid.ndim != 1
            or len(grid) < 3
            or not np.isfinite(grid).all()
            or np.any(grid <= 0)
            or np.any(np.diff(grid) <= 0)
        ):
            raise ValueError("incoherent source grids must be positive and increasing")
    if atomic.shape != (30, x.size) or h2.shape != h2_x.shape:
        raise ValueError("incoherent source array shape mismatch")
    for values, upper in ((atomic, np.arange(1, 31)[:, None]), (h2, 1.0)):
        if (
            not np.isfinite(values).all()
            or np.any(values <= 0)
            or np.any(values > upper)
            or np.any(np.diff(values, axis=-1) < 0)
        ):
            raise ValueError(
                "incoherent functions must increase within electron limits"
            )
    for array in (x, atomic, h2_x, h2):
        array.flags.writeable = False
    return IncoherentFunctions(x, atomic, h2_x, h2, actual, data["provenance"])


def cold_scattering_inventory(material, *, grain_electrons):
    """Return target numbers per H; molecular H2 entries count MOLECULES.

    ``omit`` explicitly excludes solid electrons, for the RefleX control.
    ``atomic_incoherent`` treats each solid atom as a neutral isolated atom
    for INCOHERENT scattering; its coherent Mie/ADT scattering is separate.
    Neither option implies a validated solid-state incoherent calculation.
    """
    if not isinstance(material, MaterialDefinition):
        raise ValueError("a complete MaterialDefinition is required")
    if grain_electrons not in GRAIN_POLICIES:
        raise ValueError("grain_electrons must explicitly be omit or atomic_incoherent")
    audit = material.audit()
    targets = {
        "gas:H": audit["atomic_hydrogen_atoms_per_h"],
        "gas:H2": audit["molecular_hydrogen_molecules_per_h"],
    }
    for element, row in audit["elements"].items():
        if element != "H" and row["gas_atoms_per_h"] > 0:
            targets["gas:" + element] = row["gas_atoms_per_h"]
        if row["grain_atoms_per_h"] > 0:
            targets["solid:" + element] = (
                row["grain_atoms_per_h"]
                if grain_electrons == "atomic_incoherent"
                else 0.0
            )
    return {key: value for key, value in targets.items() if value > 0}, audit


def _checked_coordinates(energy_kev, cosine):
    energy, mu = np.broadcast_arrays(
        np.asarray(energy_kev, dtype=float), np.asarray(cosine, dtype=float)
    )
    if (
        not np.isfinite(energy).all()
        or np.any(energy < ENERGY_LIMITS_KEV[0])
        or np.any(energy > ENERGY_LIMITS_KEV[1])
    ):
        raise ValueError("cold provider supports incident energies in 2-10 keV only")
    if not np.isfinite(mu).all() or np.any(abs(mu) > 1):
        raise ValueError("physical scattering cosine must lie in [-1, 1]")
    return energy, mu


def compton_transfer_inverse_angstrom(energy_kev, cosine):
    """Half the photon momentum transfer / h [angstrom^-1].

    Hubbell 1975 Eq. (2), converted to the tabulated x argument. Recoil is
    retained: x approaches sin(theta/2)/lambda at small energy or angle.
    """
    energy, mu = _checked_coordinates(energy_kev, cosine)
    epsilon = energy / ELECTRON_REST_ENERGY_KEV
    u = (1 - mu) / 2
    return (
        energy
        / HC_KEV_ANGSTROM
        * np.sqrt(u * (1 + (epsilon**2 + 2 * epsilon) * u))
        / (1 + 2 * epsilon * u)
    )


def cold_compton_components(
    material, energy_kev, cosine, *, grain_electrons, functions=None
):
    """Host d sigma/d Omega by target [cm2 sr-1 H-1]; no geometry factors."""
    functions = load_incoherent_functions() if functions is None else functions
    targets, _ = cold_scattering_inventory(material, grain_electrons=grain_electrons)
    energy, mu = _checked_coordinates(energy_kev, cosine)
    x = compton_transfer_inverse_angstrom(energy, mu)
    k = 1 / (1 + energy / ELECTRON_REST_ENERGY_KEV * (1 - mu))
    kn = 3 * THOMSON_CROSS_SECTION_CM2 / (16 * np.pi) * (k**3 + k - k**2 * (1 - mu**2))
    return {
        label: kn
        * number
        * (2.0 if label.endswith(":H2") else 1.0)
        * functions.evaluate(label.split(":")[1], x)
        for label, number in targets.items()
    }


@dataclass(frozen=True)
class ColdComptonBuild:
    table: ComptonMaterialTable
    component_cross_sections_cm2_per_h: dict
    abundance_audit: dict


def build_cold_compton_table(
    material, energy_kev, *, grain_electrons, cosine_nodes=4097, functions=None
):
    """Build a material with one consistent differential law, total and CDF.

    No RefleX total is fitted or attached to an unrelated phase function.
    Incident-energy support is explicit; downscatter termination is a later
    transport concern. Existing production material selection is unchanged.
    """
    if (
        isinstance(cosine_nodes, bool)
        or not isinstance(cosine_nodes, int)
        or cosine_nodes < 33
    ):
        raise ValueError("cosine_nodes must be an integer >= 33")
    energy = np.asarray(energy_kev, dtype=float)
    if energy.ndim != 1 or energy.size < 2 or np.any(np.diff(energy) <= 0):
        raise ValueError("supply at least two increasing incident energies")
    functions = load_incoherent_functions() if functions is None else functions
    mu = np.linspace(-1, 1, cosine_nodes)
    components = cold_compton_components(
        material,
        energy[:, None],
        mu[None, :],
        grain_electrons=grain_electrons,
        functions=functions,
    )
    if not components:
        raise ValueError("selected inventory has zero incoherent opacity")
    differential = sum(components.values())
    _, audit = cold_scattering_inventory(material, grain_electrons=grain_electrons)
    metadata = {
        "model": "external_incoherent",
        "cold_model": "neutral_scattering_functions_v1",
        "electron_state": "bound",
        "normalization": "per H nucleus",
        "polarization": "unpolarized",
        "energy_redistribution": "stationary_electron_compton",
        "material_definition_sha256": material.sha256,
        "producer": "DSH cold Compton provider v1",
        "reference": "Hubbell et al. 1975, doi:10.1063/1.555523; source ledger included",
        "source_data_sha256": functions.sha256,
        "source_provenance": functions.provenance,
        "grain_electrons": grain_electrons,
        "h2_normalization": "targets count molecules; S is per nucleus; multiply by 2 once",
        "source_interpolation": "atomic log-log linear; H2 log-log PCHIP; quadratic continuation to x=0",
        "momentum_transfer": "Hubbell 1975 Eq.2 recoil-corrected q, converted to tabulated x",
        "hydrogen_law": "analytic isolated-H Pirenne, CODATA-2022 Bohr radius",
        "incident_energy_limits_kev": list(ENERGY_LIMITS_KEV),
        "limitations": [
            "no shell selection, binding-energy thresholds, Doppler broadening or polarization",
            "solid incoherent scattering omitted or isolated-atom approximation as declared",
            "Rayleigh is a separate channel, not included in this Compton table",
            "no claim of RefleX angular equivalence",
        ],
    }
    table = build_compton_material_table(energy, mu, differential, metadata=metadata)
    totals = {
        label: np.pi
        * np.sum(np.diff(mu)[None, :] * (values[:, 1:] + values[:, :-1]), axis=1)
        for label, values in components.items()
    }
    # Independent nodes and quadrature weights test discretization, not data accuracy.
    qmu, weights = np.polynomial.legendre.leggauss(512)
    reference = sum(
        cold_compton_components(
            material,
            energy[:, None],
            qmu[None, :],
            grain_electrons=grain_electrons,
            functions=functions,
        ).values()
    ) @ (2 * np.pi * weights)
    error = float(np.max(abs(table.cross_section_cm2_per_h / reference - 1)))
    if not np.isfinite(error) or error > 5e-4:
        raise ValueError("cold angular grid exceeds 5e-4 quadrature tolerance")
    table.metadata["maximum_angular_quadrature_relative_difference"] = error
    return ColdComptonBuild(table, totals, audit)
