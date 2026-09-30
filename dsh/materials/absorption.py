"""Load source-independent photoelectric absorption cross-sections.

The voxel transport needs an intrinsic material coefficient,
``sigma_abs(E)`` in ``cm^2/H``.  A broad-band transmission curve is not such
a coefficient: it also depends on the source spectrum, band boundaries, and
column density through spectral hardening.

The checked-in Version-1 table is extracted from the XSPEC ``tbabs`` model at
the same three representative energies as the NewDust scattering table.  The
schema is deliberately an arbitrary one-dimensional energy axis so a later
dense tabulation can replace it without changing the transport interface.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_TBABS_TABLE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "absorption"
    / "tbabs_wilm_vern_v1.npz"
)


@dataclass(frozen=True)
class PhotoelectricAbsorptionTable:
    """Validated host-side absorption cross-section per hydrogen atom."""

    energy_kev: np.ndarray
    absorption_cross_section_cm2_per_h: np.ndarray
    metadata: Mapping[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_photoelectric_absorption_table(
    path: str | Path = DEFAULT_TBABS_TABLE,
) -> PhotoelectricAbsorptionTable:
    """Load a generated absorption table and validate arrays and provenance."""

    table_path = Path(path)
    metadata_path = table_path.with_suffix(".json")
    if not table_path.is_file():
        raise FileNotFoundError(f"absorption table not found: {table_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"absorption-table metadata not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported absorption table schema_version")
    if metadata.get("quantity") != "photoelectric_absorption_cross_section":
        raise ValueError("table is not marked as a photoelectric cross-section")
    if metadata.get("cross_section_unit") != "cm2 H-1":
        raise ValueError("unsupported absorption cross-section unit")
    if metadata.get("table_sha256") != _sha256(table_path):
        raise ValueError("absorption table checksum does not match its metadata")

    required = {"energy_kev", "absorption_cross_section_cm2_per_h"}
    with np.load(table_path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        extra = set(archive.files).difference(required)
        if missing or extra:
            raise ValueError(
                f"invalid absorption array members; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        energy = np.asarray(archive["energy_kev"], dtype=np.float64)
        sigma = np.asarray(
            archive["absorption_cross_section_cm2_per_h"], dtype=np.float64
        )

    if energy.ndim != 1 or energy.size == 0:
        raise ValueError("absorption energy_kev must be a nonempty 1D array")
    if not np.all(np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("absorption energies must be finite and positive")
    if np.any(np.diff(energy) <= 0.0):
        raise ValueError("absorption energies must be strictly increasing")
    if sigma.shape != energy.shape:
        raise ValueError("absorption cross-section shape must match energy_kev")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("absorption cross-sections must be finite and positive")

    metadata_energy = np.asarray(metadata.get("energy_kev", []), dtype=float)
    if metadata_energy.shape != energy.shape or not np.array_equal(
        metadata_energy, energy
    ):
        raise ValueError("absorption energy metadata does not match the arrays")

    return PhotoelectricAbsorptionTable(
        energy_kev=energy,
        absorption_cross_section_cm2_per_h=sigma,
        metadata=metadata,
    )


def monochromatic_transmission(
    absorption: PhotoelectricAbsorptionTable,
    hydrogen_column_cm2,
):
    """Return ``exp(-NH*sigma_abs(E))`` for one or more hydrogen columns.

    A scalar column returns shape ``(n_energy,)``.  An input with shape
    ``column_shape`` returns ``column_shape + (n_energy,)``.
    """

    column = np.asarray(hydrogen_column_cm2, dtype=np.float64)
    if not np.all(np.isfinite(column)) or np.any(column < 0.0):
        raise ValueError("hydrogen_column_cm2 must be finite and nonnegative")
    return np.exp(-column[..., None] * absorption.absorption_cross_section_cm2_per_h)
