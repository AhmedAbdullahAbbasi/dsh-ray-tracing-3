"""Load and validate intrinsic scattering tables generated with NewDust.

NewDust's :class:`ScreenGalHalo` output is an observer-space halo kernel.  It
contains both the physical phase function and the thin-screen geometry.  The
voxel transport kernel instead needs local material properties, independent
of source and observer positions.  The table stored with this project is
therefore generated directly from ``GrainPop.int_diff`` and normalized per H
atom; it is not a table of ``ScreenGalHalo.norm_int`` values.

This module is deliberately NumPy-only preprocessing.  It validates the
tabulation on the host and then constructs the fixed-shape JAX table used in
the transport hot path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .absorption import PhotoelectricAbsorptionTable
from .dust import (
    DustPhysicsTable,
    build_dust_physics_table,
    phase_cdf_from_differential_cross_section,
)

DEFAULT_NEWDUST_TABLE = (
    Path(__file__).resolve().parent.parent / "data" / "newdust" / "mrn_rg_drude_v1.npz"
)


@dataclass(frozen=True)
class NewDustScatteringTable:
    r"""Validated host-side intrinsic scattering data.

    ``differential_cross_section_cm2_per_sr_per_h`` is intrinsic
    :math:`d\sigma/d\Omega`, not an observer-space halo surface brightness.
    ``scattering_cross_section_cm2_per_h`` is the numerical solid-angle
    integral of that same table, so the interaction opacity and sampled phase
    function have exactly the same normalization.
    """

    energy_kev: np.ndarray
    scattering_angle_rad: np.ndarray
    differential_cross_section_cm2_per_sr_per_h: np.ndarray
    scattering_cross_section_cm2_per_h: np.ndarray
    scattering_angle_cdf: np.ndarray
    metadata: Mapping[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_path(table_path: Path) -> Path:
    return table_path.with_suffix(".json")


def load_newdust_scattering_table(
    path: str | Path = DEFAULT_NEWDUST_TABLE,
) -> NewDustScatteringTable:
    """Load a generated NewDust table and fail closed on bad provenance.

    The adjacent JSON file is part of the table contract.  Its checksum makes
    it difficult to accidentally pair physics metadata with different array
    bytes.
    """

    table_path = Path(path)
    metadata_path = _metadata_path(table_path)
    if not table_path.is_file():
        raise FileNotFoundError(f"NewDust table not found: {table_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"NewDust metadata not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported NewDust table schema_version")
    if metadata.get("table_sha256") != _sha256(table_path):
        raise ValueError("NewDust table checksum does not match its metadata")
    if metadata.get("quantity") != "intrinsic_differential_cross_section":
        raise ValueError("NewDust table is not marked as an intrinsic cross-section")
    if metadata.get("differential_cross_section_unit") != "cm2 sr-1 H-1":
        raise ValueError("unsupported NewDust differential cross-section unit")

    required = {
        "energy_kev",
        "scattering_angle_rad",
        "differential_cross_section_cm2_per_sr_per_h",
        "scattering_cross_section_cm2_per_h",
        "scattering_angle_cdf",
    }
    with np.load(table_path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        extra = set(archive.files).difference(required)
        if missing or extra:
            raise ValueError(
                f"invalid NewDust array members; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        energy = np.asarray(archive["energy_kev"], dtype=np.float64)
        angle = np.asarray(archive["scattering_angle_rad"], dtype=np.float64)
        differential = np.asarray(
            archive["differential_cross_section_cm2_per_sr_per_h"],
            dtype=np.float64,
        )
        sigma = np.asarray(
            archive["scattering_cross_section_cm2_per_h"], dtype=np.float64
        )
        cdf = np.asarray(archive["scattering_angle_cdf"], dtype=np.float64)

    if energy.ndim != 1 or energy.size < 2:
        raise ValueError("NewDust energy_kev must contain at least two values")
    if angle.ndim != 1 or angle.size < 2:
        raise ValueError("NewDust scattering_angle_rad must be one-dimensional")
    if not np.all(np.isfinite(energy)) or not np.all(np.diff(energy) > 0.0):
        raise ValueError("NewDust energies must be finite and strictly increasing")
    if (
        not np.all(np.isfinite(angle))
        or angle[0] != 0.0
        or angle[-1] != np.pi
        or not np.all(np.diff(angle) > 0.0)
    ):
        raise ValueError("NewDust angles must increase exactly from zero to pi")
    expected_shape = (energy.size, angle.size)
    if differential.shape != expected_shape or cdf.shape != expected_shape:
        raise ValueError("NewDust differential cross-section/CDF shape mismatch")
    if sigma.shape != energy.shape:
        raise ValueError("NewDust integrated cross-section shape mismatch")
    if not np.all(np.isfinite(differential)) or np.any(differential < 0.0):
        raise ValueError("NewDust differential cross-sections must be nonnegative")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("NewDust integrated cross-sections must be positive")

    integrated, rebuilt_cdf = phase_cdf_from_differential_cross_section(
        angle, differential
    )
    if not np.allclose(sigma, integrated, rtol=5.0e-12, atol=0.0):
        raise ValueError(
            "NewDust integrated cross-section does not match dSigma/dOmega"
        )
    if not np.allclose(cdf, rebuilt_cdf, rtol=5.0e-12, atol=5.0e-14):
        raise ValueError("NewDust CDF does not match dSigma/dOmega")

    metadata_energies = np.asarray(metadata.get("energy_kev", []), dtype=float)
    if metadata_energies.shape != energy.shape or not np.array_equal(
        metadata_energies, energy
    ):
        raise ValueError("NewDust energy metadata does not match the arrays")

    return NewDustScatteringTable(
        energy_kev=energy,
        scattering_angle_rad=angle,
        differential_cross_section_cm2_per_sr_per_h=differential,
        scattering_cross_section_cm2_per_h=sigma,
        scattering_angle_cdf=cdf,
        metadata=metadata,
    )


def build_dust_physics_from_newdust(
    scattering: NewDustScatteringTable,
    absorption_cross_section_cm2_per_h,
) -> DustPhysicsTable:
    """Combine NewDust scattering with an explicit absorption prescription.

    NewDust's v1 MRN/RG-Drude table is a dust-scattering calculation.  The
    caller must supply photoelectric absorption cross-sections on the same
    energy grid; there is intentionally no silent zero-absorption default.
    """

    absorption = np.asarray(absorption_cross_section_cm2_per_h, dtype=np.float64)
    if absorption.shape != scattering.energy_kev.shape:
        raise ValueError(
            "absorption_cross_section_cm2_per_h must have shape (n_energy,)"
        )
    return build_dust_physics_table(
        energy_kev=scattering.energy_kev,
        scattering_cross_section_cm2_per_h=(
            scattering.scattering_cross_section_cm2_per_h
        ),
        absorption_cross_section_cm2_per_h=absorption,
        scattering_angle_rad=scattering.scattering_angle_rad,
        scattering_angle_cdf=scattering.scattering_angle_cdf,
        differential_cross_section_cm2_per_sr_per_h=(
            scattering.differential_cross_section_cm2_per_sr_per_h
        ),
    )


def build_dust_physics_from_tables(
    scattering: NewDustScatteringTable,
    absorption: PhotoelectricAbsorptionTable,
) -> DustPhysicsTable:
    """Combine independently versioned scattering and absorption tables.

    Version 1 deliberately requires identical energy axes.  Interpolating one
    three-point table onto the other would hide a physically weak
    approximation.  A later dense common energy grid can use the same API.
    """

    if not np.array_equal(scattering.energy_kev, absorption.energy_kev):
        raise ValueError(
            "scattering and absorption tables must have identical energy grids"
        )
    return build_dust_physics_from_newdust(
        scattering,
        absorption.absorption_cross_section_cm2_per_h,
    )
