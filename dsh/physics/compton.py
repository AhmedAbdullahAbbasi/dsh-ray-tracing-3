"""Compton physics foundation; not yet connected to production transport.

Unpolarized photons, stationary-electron recoil, intrinsic cross-sections.
The analytic reference is per FREE electron. External bound-electron angular
tables must be supplied and declared explicitly; no neutral-ISM recipe is
inferred from an electron/H number. See docs/COMPTON_STAGE1.md.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import random

from ..io.atomic import atomic_savez_compressed

# NIST 2022 CODATA; m^2 -> cm^2 and MeV -> keV.
THOMSON_CROSS_SECTION_CM2 = 6.6524587051e-25
ELECTRON_REST_ENERGY_KEV = 510.99895069
_SCHEMA = 1
_ARRAYS = ("energy_kev", "cosine", "differential_cm2_per_sr_per_h")


def _series_coefficient(n):
    # Integrate the x^n coefficient of (3/8)[k^3+k-(1-mu^2)k^2],
    # k=1/[1+x(1-mu)], over mu=-1..1. Exact host arithmetic.
    a = (Fraction((n + 1) * (n + 2), 2) + 1) * Fraction(2 ** (n + 1), n + 1)
    b = (n + 1) * (Fraction(2 ** (n + 3), n + 2) - Fraction(2 ** (n + 3), n + 3))
    return float(Fraction(3, 8) * (-1) ** n * (a - b))


_TOTAL_SERIES = tuple(_series_coefficient(n) for n in range(33))


def _float_array(value):
    array = jnp.asarray(value)
    return array.astype(jnp.result_type(array, jnp.float32))


def compton_recoil_kinematics(energy_kev, cosine):
    """Return (outgoing photon energy, electron kinetic energy), in keV.

    Both inputs broadcast. Invalid/nonfinite inputs return NaNs. E=0 is
    admitted only as the continuous zero-energy limit. The recoil expression
    avoids cancellation in E-E' for small energy transfer.
    """
    energy, mu = jnp.broadcast_arrays(_float_array(energy_kev), _float_array(cosine))
    valid = jnp.isfinite(energy) & (energy >= 0) & jnp.isfinite(mu) & (abs(mu) <= 1)
    transfer = energy / ELECTRON_REST_ENERGY_KEV * (1 - mu)
    outgoing = energy / (1 + transfer)
    recoil = energy * (transfer / (1 + transfer))
    return jnp.where(valid, outgoing, jnp.nan), jnp.where(valid, recoil, jnp.nan)


def klein_nishina_differential_cm2_per_sr(energy_kev, cosine):
    """Unpolarized d sigma/d Omega per stationary FREE electron [cm2/sr]."""
    energy, mu = jnp.broadcast_arrays(_float_array(energy_kev), _float_array(cosine))
    valid = jnp.isfinite(energy) & (energy >= 0) & jnp.isfinite(mu) & (abs(mu) <= 1)
    k = 1 / (1 + energy / ELECTRON_REST_ENERGY_KEV * (1 - mu))
    value = (
        3 * THOMSON_CROSS_SECTION_CM2 / (16 * jnp.pi) * (k**3 + k - k**2 * (1 - mu**2))
    )
    return jnp.where(valid, value, jnp.nan)


def klein_nishina_total_cm2(energy_kev):
    """Analytic total per free electron, stable in the Thomson limit.

    A convergent integrated series is used for E/mc2 <= 1/8. The closed
    expression is used above that boundary. Neither branch integrates the
    tabulated angular grid. Invalid/nonfinite energies return NaNs.
    """
    energy = _float_array(energy_kev)
    x = energy / ELECTRON_REST_ENERGY_KEV
    series = jnp.polyval(
        jnp.asarray(_TOTAL_SERIES[::-1], dtype=x.dtype), jnp.clip(x, 0.0, 0.125)
    )
    safe_x = jnp.maximum(x, 0.125)
    logterm = jnp.log1p(2 * safe_x)
    closed = 0.75 * (
        (1 + safe_x)
        / safe_x**3
        * (2 * safe_x * (1 + safe_x) / (1 + 2 * safe_x) - logterm)
        + logterm / (2 * safe_x)
        - (1 + 3 * safe_x) / (1 + 2 * safe_x) ** 2
    )
    value = THOMSON_CROSS_SECTION_CM2 * jnp.where(x <= 0.125, series, closed)
    return jnp.where(jnp.isfinite(energy) & (energy >= 0), value, jnp.nan)


@dataclass(frozen=True)
class ComptonMaterialTable:
    """Host-side table. The runtime law is piecewise linear in cosine.

    Cross-sections are per H nucleus. CDF runs from backward (mu=-1) to
    forward (mu=+1), covering the FULL sphere with uniform azimuth.
    Energy interpolation mixes normalized angular laws linearly in log(E).
    """

    energy_kev: np.ndarray
    cosine: np.ndarray
    differential_cm2_per_sr_per_h: np.ndarray
    cross_section_cm2_per_h: np.ndarray
    cosine_cdf: np.ndarray
    metadata: dict


class ComptonPhysicsTable(NamedTuple):
    """Numeric-only JAX pytree, separate from the existing DustPhysicsTable."""

    energy_kev: jnp.ndarray
    cosine: jnp.ndarray
    cross_section_cm2_per_h: jnp.ndarray
    phase_pdf_per_sr: jnp.ndarray
    cosine_cdf: jnp.ndarray


class ComptonEvaluation(NamedTuple):
    supported: jnp.ndarray
    cross_section_cm2_per_h: jnp.ndarray
    phase_pdf_per_sr: jnp.ndarray


class ComptonSample(NamedTuple):
    supported: jnp.ndarray
    cosine: jnp.ndarray
    azimuth_rad: jnp.ndarray
    outgoing_energy_kev: jnp.ndarray
    recoil_energy_kev: jnp.ndarray


def _checked_metadata(metadata):
    value = json.loads(json.dumps(metadata, allow_nan=False))
    if value.get("model") not in ("free_electron_kn", "external_incoherent"):
        raise ValueError("unsupported Compton model")
    if value.get("normalization") != "per H nucleus":
        raise ValueError("Compton normalization must be per H nucleus")
    if value.get("energy_redistribution") != "stationary_electron_compton":
        raise ValueError("unsupported Compton energy redistribution")
    if value.get("polarization") != "unpolarized":
        raise ValueError("only unpolarized Compton tables are supported")
    if value.get("electron_state") not in ("free", "bound"):
        raise ValueError("explicit free/bound electron_state is required")
    for field in ("producer", "reference"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ValueError(f"Compton {field} is required")
    digest = value.get("material_definition_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("material_definition_sha256 must be a lowercase SHA256")
    if value["model"] == "free_electron_kn":
        count = value.get("electrons_per_h")
        if (
            value["electron_state"] != "free"
            or isinstance(count, bool)
            or not isinstance(count, (int, float))
            or not np.isfinite(count)
            or count <= 0
        ):
            raise ValueError(
                "free_electron_kn requires explicit positive electrons_per_h"
            )
    return value


def build_compton_material_table(
    energy_kev,
    cosine,
    differential_cm2_per_sr_per_h,
    *,
    metadata,
    expected_cross_section_cm2_per_h=None,
):
    """Validate a full angular law and derive its consistent total and CDF.

    Supplying only a total opacity is insufficient. No rescaling/repair of
    external data is done. Optional supplied totals must match the angular
    integral to rtol=5e-4; the runtime total is always that integral.
    """
    energy, mu, differential = [
        np.array(a, dtype=np.float64, copy=True)
        for a in (energy_kev, cosine, differential_cm2_per_sr_per_h)
    ]
    if (
        energy.ndim != 1
        or energy.size < 2
        or not np.isfinite(energy).all()
        or np.any(energy <= 0)
        or np.any(np.diff(energy) <= 0)
    ):
        raise ValueError("Compton energies must be finite, positive and increasing")
    if (
        mu.ndim != 1
        or mu.size < 3
        or not np.isfinite(mu).all()
        or np.any(np.diff(mu) <= 0)
        or mu[0] != -1
        or mu[-1] != 1
    ):
        raise ValueError("Compton cosine grid must increase from -1 to +1")
    if (
        differential.shape != (energy.size, mu.size)
        or not np.isfinite(differential).all()
        or np.any(differential < 0)
    ):
        raise ValueError(
            "Compton differential must be finite nonnegative (energy,cosine)"
        )
    increments = (
        np.pi * np.diff(mu)[None, :] * (differential[:, 1:] + differential[:, :-1])
    )
    cumulative = np.concatenate(
        [np.zeros((energy.size, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    total = cumulative[:, -1].copy()
    if not np.isfinite(total).all() or np.any(total <= 0):
        raise ValueError("Compton angular integrals must be positive and finite")
    if expected_cross_section_cm2_per_h is not None:
        supplied = np.asarray(expected_cross_section_cm2_per_h, dtype=float)
        if (
            supplied.shape != total.shape
            or not np.isfinite(supplied).all()
            or not np.allclose(supplied, total, rtol=5e-4, atol=0)
        ):
            raise ValueError("Compton angular integral disagrees with supplied total")
    cdf = cumulative / total[:, None]
    cdf[:, -1] = 1.0
    for array in (energy, mu, differential, total, cdf):
        array.flags.writeable = False
    return ComptonMaterialTable(
        energy, mu, differential, total, cdf, _checked_metadata(metadata)
    )


def build_free_electron_compton_table(
    energy_kev,
    *,
    electrons_per_h,
    material_definition_sha256,
    cosine_nodes=4097,
):
    """Produce an explicit free-electron reference, not a cold-ISM adapter.

    The full-sphere table approximates the continuous KN law. Its total is
    derived from its angular grid; producer accuracy is checked independently
    against the analytic total. No default electron abundance is assumed.
    """
    if (
        isinstance(electrons_per_h, bool)
        or not np.isscalar(electrons_per_h)
        or not np.isfinite(electrons_per_h)
        or electrons_per_h <= 0
    ):
        raise ValueError("electrons_per_h must be explicitly positive and finite")
    if (
        isinstance(cosine_nodes, bool)
        or not isinstance(cosine_nodes, int)
        or cosine_nodes < 3
    ):
        raise ValueError("cosine_nodes must be an integer >=3")
    energy = np.asarray(energy_kev, dtype=float)
    if energy.ndim != 1:
        raise ValueError("Compton energy grid must be one-dimensional")
    mu = np.linspace(-1, 1, cosine_nodes)
    k = 1 / (1 + energy[:, None] / ELECTRON_REST_ENERGY_KEV * (1 - mu))
    diff = (
        electrons_per_h
        * 3
        * THOMSON_CROSS_SECTION_CM2
        / (16 * np.pi)
        * (k**3 + k - k**2 * (1 - mu**2))
    )
    metadata = {
        "model": "free_electron_kn",
        "electron_state": "free",
        "electrons_per_h": float(electrons_per_h),
        "normalization": "per H nucleus",
        "polarization": "unpolarized",
        "energy_redistribution": "stationary_electron_compton",
        "material_definition_sha256": material_definition_sha256,
        "producer": "DSH Compton foundation v1",
        "reference": "Klein & Nishina (1929); NIST 2022 CODATA",
    }
    table = build_compton_material_table(energy, mu, diff, metadata=metadata)
    x = energy / ELECTRON_REST_ENERGY_KEV
    series = np.polynomial.polynomial.polyval(np.minimum(x, 0.125), _TOTAL_SERIES)
    y = np.maximum(x, 0.125)
    exact = 0.75 * (
        (1 + y) / y**3 * (2 * y * (1 + y) / (1 + 2 * y) - np.log1p(2 * y))
        + np.log1p(2 * y) / (2 * y)
        - (1 + 3 * y) / (1 + 2 * y) ** 2
    )
    exact = (
        electrons_per_h
        * THOMSON_CROSS_SECTION_CM2
        * np.where(x <= 0.125, series, exact)
    )
    error = float(np.max(abs(table.cross_section_cm2_per_h / exact - 1)))
    if not np.isfinite(error) or error > 5e-4:
        raise ValueError(
            "free-electron angular grid is too coarse; increase cosine_nodes"
        )
    table.metadata["maximum_angular_integral_relative_error"] = error
    return table


def to_compton_physics(table):
    """Convert a validated host table to JAX arrays; no production hookup."""
    return ComptonPhysicsTable(
        *map(
            jnp.asarray,
            (
                table.energy_kev,
                table.cosine,
                table.cross_section_cm2_per_h,
                table.differential_cm2_per_sr_per_h
                / table.cross_section_cm2_per_h[:, None],
                table.cosine_cdf,
            ),
        )
    )


def _energy_bracket(table, energy_kev):
    energy = _float_array(energy_kev)
    supported = (
        jnp.isfinite(energy)
        & (energy >= table.energy_kev[0])
        & (energy <= table.energy_kev[-1])
    )
    safe = jnp.clip(
        jnp.where(jnp.isfinite(energy), energy, table.energy_kev[0]),
        table.energy_kev[0],
        table.energy_kev[-1],
    )
    hi = jnp.clip(
        jnp.searchsorted(table.energy_kev, safe, side="right"),
        1,
        table.energy_kev.size - 1,
    )
    lo = hi - 1
    fraction = jnp.log(safe / table.energy_kev[lo]) / jnp.log(
        table.energy_kev[hi] / table.energy_kev[lo]
    )
    return supported, lo, hi, fraction


def evaluate_compton(table, energy_kev, cosine):
    """Return support, total [cm2/H], normalized phase [sr^-1]; broadcasts.

    Unsupported evaluations return supported=False and zero values. Internal
    safe indexing does not turn an unsupported energy into a valid event.
    """
    energy, mu = jnp.broadcast_arrays(_float_array(energy_kev), _float_array(cosine))
    supported, lo, hi, f = _energy_bracket(table, energy)
    supported &= jnp.isfinite(mu) & (mu >= -1) & (mu <= 1)
    safe_mu = jnp.clip(jnp.where(jnp.isfinite(mu), mu, 0.0), -1, 1)
    right = jnp.clip(
        jnp.searchsorted(table.cosine, safe_mu, side="right"), 1, table.cosine.size - 1
    )
    left = right - 1
    t = (safe_mu - table.cosine[left]) / (table.cosine[right] - table.cosine[left])
    lower = (1 - t) * table.phase_pdf_per_sr[lo, left] + t * table.phase_pdf_per_sr[
        lo, right
    ]
    upper = (1 - t) * table.phase_pdf_per_sr[hi, left] + t * table.phase_pdf_per_sr[
        hi, right
    ]
    phase = (1 - f) * lower + f * upper
    sigma = jnp.exp(
        (1 - f) * jnp.log(table.cross_section_cm2_per_h[lo])
        + f * jnp.log(table.cross_section_cm2_per_h[hi])
    )
    return ComptonEvaluation(
        supported, jnp.where(supported, sigma, 0.0), jnp.where(supported, phase, 0.0)
    )


def sample_compton_scattering(key, table, energy_kev):
    """One scalar-energy sample; use vmap for packets. JIT-compatible.

    Invert the integral of the SAME piecewise-linear phase used by evaluate.
    Linear interpolation of CDF abscissae would instead sample a different,
    piecewise-constant phase. Zero-density intervals are skipped.
    """
    energy = _float_array(energy_kev)
    if energy.ndim != 0:
        raise ValueError("sample_compton_scattering requires scalar energy; use vmap")
    supported, lo, hi, f = _energy_bracket(table, energy)
    cdf = (1 - f) * table.cosine_cdf[lo] + f * table.cosine_cdf[hi]
    phase = (1 - f) * table.phase_pdf_per_sr[lo] + f * table.phase_pdf_per_sr[hi]
    key_u, key_phi = random.split(key)
    u = random.uniform(key_u, dtype=energy.dtype)
    right = jnp.clip(jnp.searchsorted(cdf, u, side="right"), 1, cdf.size - 1)
    left = right - 1
    width = cdf[right] - cdf[left]
    area = jnp.clip((u - cdf[left]) / jnp.maximum(width, 1e-30), 0.0, 1.0)
    p0, p1 = phase[left], phase[right]
    root = jnp.sqrt(jnp.maximum(p0 * p0 + (p1 - p0) * area * (p0 + p1), 0.0))
    t = area * (p0 + p1) / jnp.maximum(p0 + root, 1e-30)
    mu = table.cosine[left] + jnp.clip(t, 0.0, 1.0) * (
        table.cosine[right] - table.cosine[left]
    )
    outgoing, recoil = compton_recoil_kinematics(energy, mu)
    supported &= width > 0
    azimuth = 2 * jnp.pi * random.uniform(key_phi, dtype=energy.dtype)
    return ComptonSample(
        supported,
        jnp.where(supported, mu, 0.0),
        jnp.where(supported, azimuth, 0.0),
        jnp.where(supported, outgoing, 0.0),
        jnp.where(supported, recoil, 0.0),
    )


def _fingerprint(arrays):
    digest = hashlib.sha256()
    for name in _ARRAYS:
        a = np.ascontiguousarray(arrays[name], dtype="<f8")
        digest.update(name.encode())
        digest.update(json.dumps(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def write_compton_material_table(path, table):
    """Atomically save physical arrays and checksummed model provenance."""
    checked = build_compton_material_table(
        table.energy_kev,
        table.cosine,
        table.differential_cm2_per_sr_per_h,
        metadata=table.metadata,
        expected_cross_section_cm2_per_h=table.cross_section_cm2_per_h,
    )
    arrays = {name: getattr(checked, name) for name in _ARRAYS}
    metadata = {
        "schema_version": _SCHEMA,
        "units": "cm2 sr-1 H-1",
        "material": checked.metadata,
        "arrays_sha256": _fingerprint(arrays),
    }
    text = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Compton material path must end in .npz")
    return atomic_savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(text),
        metadata_sha256=np.asarray(hashlib.sha256(text.encode()).hexdigest()),
    )


def load_compton_material_table(path, *, expected_material_definition_sha256):
    """Load without pickle; verify checksums and the requested composition."""
    with np.load(path, allow_pickle=False) as archive:
        text = str(archive["metadata_json"].item())
        if hashlib.sha256(text.encode()).hexdigest() != str(
            archive["metadata_sha256"].item()
        ):
            raise ValueError("Compton metadata checksum mismatch")
        metadata = json.loads(text)
        if (
            metadata.get("schema_version") != _SCHEMA
            or metadata.get("units") != "cm2 sr-1 H-1"
        ):
            raise ValueError("unsupported Compton material schema/units")
        arrays = {name: np.array(archive[name], copy=True) for name in _ARRAYS}
    if _fingerprint(arrays) != metadata.get("arrays_sha256"):
        raise ValueError("Compton physical-array checksum mismatch")
    table = build_compton_material_table(**arrays, metadata=metadata["material"])
    if (
        table.metadata["material_definition_sha256"]
        != expected_material_definition_sha256
    ):
        raise ValueError(
            "Compton material definition does not match requested composition"
        )
    return table
