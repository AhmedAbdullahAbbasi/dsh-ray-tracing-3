"""Explicit residual-gas/H2 absorption from a portable atomic cross-section basis."""

from __future__ import annotations

import ctypes
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .absorption import PhotoelectricAbsorptionTable
from .energy_edges import bracketed_edge_grid, edge_intervals
from .material_definitions import ATOMIC_WEIGHTS_U, MaterialDefinition

ELEMENTS = tuple(ATOMIC_WEIGHTS_U)
TBNEW_ELEMENTS = (1, 2, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 24, 26, 27, 28)
DEFAULT_GAS_BASIS = (
    Path(__file__).resolve().parent.parent
    / "data/absorption/tbnew_verner_basis_2_10_v1.npz"
)


class TBnewGasBackend:
    """Offline, single-threaded access to pinned author C/Fortran routines."""

    def __init__(self, library):
        path = Path(library).resolve()
        self.metadata = json.loads(
            path.with_suffix(".json").read_text(encoding="utf-8")
        )
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != self.metadata["library_sha256"]
        ):
            raise ValueError("TBnew backend library checksum mismatch")
        self.library = ctypes.CDLL(str(path))
        self.library.dsh_atomic_sigma.argtypes = [ctypes.c_int, ctypes.c_double]
        self.library.dsh_atomic_sigma.restype = ctypes.c_double
        self.library.dsh_h2_sigma.argtypes = [ctypes.c_double]
        self.library.dsh_h2_sigma.restype = ctypes.c_double
        self.library.dsh_tbnew_gas_transmission.argtypes = [ctypes.c_double] * 4 + [
            ctypes.POINTER(ctypes.c_double)
        ]
        self.library.dsh_tbnew_gas_transmission.restype = ctypes.c_double

    def evaluate(self, energy_kev):
        e = np.atleast_1d(np.asarray(energy_kev, dtype=float))
        if e.ndim != 1 or np.any(~np.isfinite(e)) or np.any((e < 2) | (e > 10)):
            raise ValueError("gas backend supports finite 2--10 keV energies")
        atomic = np.array(
            [
                [self.library.dsh_atomic_sigma(z, value) for value in e]
                for z in range(1, 31)
            ]
        )
        h2 = np.array([self.library.dsh_h2_sigma(value) for value in e])
        if (
            np.any(~np.isfinite(atomic))
            or np.any(atomic <= 0)
            or np.any(~np.isfinite(h2))
            or np.any(h2 <= 0)
        ):
            raise ValueError("gas backend returned invalid cross sections")
        return atomic, h2

    def core_transmission(
        self, energy_kev, nh_cm2, residual, fraction_h2, *, half_width_kev=0
    ):
        """Independent upstream tbnew summation, restricted to its 18 atoms.

        rho=0 and explicit negative element columns avoid both grain shielding
        and external abundance lookups. Atomic cross sections are averaged at
        the two bin endpoints by the upstream routine.
        """
        residual = np.ascontiguousarray(residual, dtype=np.float64)
        if (
            residual.shape != (30,)
            or np.any(~np.isfinite(residual))
            or np.any(residual < 0)
            or residual[0] <= 0
            or not 0 <= fraction_h2 <= residual[0]
            or not np.isfinite(nh_cm2)
            or nh_cm2 <= 0
            or not np.isfinite(half_width_kev)
            or half_width_kev < 0
            or not np.isfinite(energy_kev)
            or energy_kev - half_width_kev < 2
            or energy_kev + half_width_kev > 10
        ):
            raise ValueError("invalid core-transmission gas inventory")
        return float(
            self.library.dsh_tbnew_gas_transmission(
                float(energy_kev - half_width_kev),
                float(energy_kev + half_width_kev),
                float(nh_cm2),
                float(fraction_h2),
                residual.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        )


def verner_edge_catalog(source_file):
    """Neutral-atom shell thresholds from the pinned PHFIT2 data declarations."""
    source = Path(source_file).read_bytes()
    if (
        hashlib.sha256(source).hexdigest()
        != "8df982535a4a2b952e33ca2ffb6d81600e156d7373fbf547a0c0170c1cecc296"
    ):
        raise ValueError("PHFIT2 source checksum mismatch")
    records = re.findall(
        r"DATA\s*\(PH1\(I,\s*(\d+),\s*(\d+),\s*(\d+)\),I=1,6\)\s*/([^/]+)/",
        source.decode(),
        re.I,
    )
    result = []
    for z, electrons, shell, values in records:
        if z == electrons:
            edge = float(np.float32(float(values.split(",")[0]))) / 1000
            if 2 < edge < 10:
                result.append(
                    {
                        "element": ELEMENTS[int(z) - 1],
                        "shell": int(shell),
                        "energy_kev": edge,
                    }
                )
    if len(result) != 16:
        raise ValueError("unexpected neutral PHFIT2 2--10 keV edge catalog")
    return sorted(result, key=lambda x: x["energy_kev"])


@dataclass(frozen=True)
class AtomicAbsorptionBasis:
    energy_kev: np.ndarray
    atomic_cm2: np.ndarray
    h2_cm2: np.ndarray
    metadata: dict

    def evaluate(self, energy_kev):
        e = np.asarray(energy_kev, dtype=float)
        if np.any(~np.isfinite(e)) or np.any(
            (e < self.energy_kev[0]) | (e > self.energy_kev[-1])
        ):
            raise ValueError("energy outside atomic absorption basis")
        high = np.clip(
            np.searchsorted(self.energy_kev, e, side="right"),
            1,
            len(self.energy_kev) - 1,
        )
        low = high - 1
        f = np.log(e / self.energy_kev[low]) / np.log(
            self.energy_kev[high] / self.energy_kev[low]
        )
        edge = edge_intervals(self.energy_kev, self.metadata["edge_energies_kev"])[low]
        f = np.where(edge > 0, (e >= edge).astype(float), f)

        def interp(values):
            return np.exp(
                np.log(values[..., low])
                + f * np.log(values[..., high] / values[..., low])
            )

        return interp(self.atomic_cm2), interp(self.h2_cm2)


def generate_atomic_basis(
    backend, initial_energy_kev, edge_catalog, *, relative_tolerance=3e-4
):
    edges = [record["energy_kev"] for record in edge_catalog]
    energy = bracketed_edge_grid(initial_energy_kev, edges)
    for _ in range(20):
        atomic, h2 = backend.evaluate(energy)
        middle = np.sqrt(energy[:-1] * energy[1:])
        ma, mh = backend.evaluate(middle)
        values = np.vstack((atomic, h2))
        midvalues = np.vstack((ma, mh))
        errors = np.max(
            abs(np.sqrt(values[:, :-1] * values[:, 1:]) / midvalues - 1), axis=0
        )
        smooth = edge_intervals(energy, edges) == 0
        split = middle[smooth & (errors > relative_tolerance)]
        if split.size == 0:
            return AtomicAbsorptionBasis(
                energy,
                atomic,
                h2,
                {
                    "schema_version": 1,
                    "quantity": "neutral_atomic_and_molecular_absorption_basis",
                    "atomic_cross_section_unit": "cm2 atom-1",
                    "molecular_cross_section_unit": "cm2 molecule-1",
                    "elements": ELEMENTS,
                    "energy_kev": energy.tolist(),
                    "edge_energies_kev": edges,
                    "edge_catalog": edge_catalog,
                    "relative_smooth_midpoint_tolerance": relative_tolerance,
                    "maximum_smooth_midpoint_relative_error": float(
                        np.max(errors[smooth])
                    ),
                    "backend": backend.metadata,
                    "additional_elements_outside_tbnew": [
                        ELEMENTS[i - 1] for i in range(1, 31) if i not in TBNEW_ELEMENTS
                    ],
                    "scope": "2--10 keV neutral gas plus H2; no solid absorption or scattering; standalone author routines, not an XSPEC-process validation",
                },
            )
        energy = np.sort(np.r_[energy, split])
        if energy.size > 8192:
            break
    raise ValueError("atomic basis did not converge")


def write_atomic_basis(path, basis):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        energy_kev=basis.energy_kev,
        atomic_cm2=basis.atomic_cm2,
        h2_cm2=basis.h2_cm2,
    )
    metadata = dict(
        basis.metadata, table_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return load_atomic_basis(path)


def load_atomic_basis(path=DEFAULT_GAS_BASIS):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != 1
        or metadata.get("quantity") != "neutral_atomic_and_molecular_absorption_basis"
        or tuple(metadata.get("elements", ())) != ELEMENTS
        or metadata.get("atomic_cross_section_unit") != "cm2 atom-1"
        or metadata.get("molecular_cross_section_unit") != "cm2 molecule-1"
    ):
        raise ValueError("unsupported atomic absorption basis")
    if metadata.get("table_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("atomic absorption basis checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"energy_kev", "atomic_cm2", "h2_cm2"}:
            raise ValueError("invalid atomic absorption basis members")
        energy, atomic, h2 = (
            np.array(data[key], dtype=float)
            for key in ("energy_kev", "atomic_cm2", "h2_cm2")
        )
    if (
        energy.ndim != 1
        or energy.size < 2
        or atomic.shape != (30, energy.size)
        or h2.shape != energy.shape
        or energy[0] != 2
        or energy[-1] != 10
        or np.any(~np.isfinite(atomic))
        or np.any(atomic <= 0)
        or np.any(~np.isfinite(h2))
        or np.any(h2 <= 0)
        or not np.array_equal(energy, metadata["energy_kev"])
    ):
        raise ValueError("invalid atomic absorption basis arrays")
    edge_intervals(energy, metadata["edge_energies_kev"])
    for a in (energy, atomic, h2):
        a.flags.writeable = False
    return AtomicAbsorptionBasis(energy, atomic, h2, metadata)


def residual_gas_components(
    definition, basis, energy_kev=None, *, edge_energies_kev=None
):
    if not isinstance(definition, MaterialDefinition):
        raise ValueError("explicit material definition required")
    audit = definition.audit()
    energy = np.asarray(
        basis.energy_kev if energy_kev is None else energy_kev, dtype=float
    )
    atomic, molecule = basis.evaluate(energy)
    counts = np.array(
        [audit["elements"].get(e, {}).get("gas_atoms_per_h", 0) for e in ELEMENTS]
    )
    counts[0] = audit["atomic_hydrogen_atoms_per_h"]
    sigma_atom = counts @ atomic
    sigma_h2 = audit["molecular_hydrogen_molecules_per_h"] * molecule
    metadata = {
        "schema_version": 1,
        "quantity": "photoelectric_absorption_cross_section",
        "cross_section_unit": "cm2 H-1",
        "producer": "TBnew/Verner residual gas, explicit audited element counts",
        "material_definition_sha256": definition.sha256,
        "atomic_basis_sha256": basis.metadata.get("table_sha256"),
        "edge_energies_kev": list(
            basis.metadata["edge_energies_kev"]
            if edge_energies_kev is None
            else edge_energies_kev
        ),
    }
    return tuple(
        PhotoelectricAbsorptionTable(
            energy.copy(), sigma, dict(metadata, absorption_coverage=[phase])
        )
        for sigma, phase in (
            (sigma_atom, "atomic_gas"),
            (sigma_h2, "molecular_hydrogen"),
        )
    )
