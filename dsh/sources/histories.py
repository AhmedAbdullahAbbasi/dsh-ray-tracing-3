"""Intrinsic, piecewise power-law spectral histories with explicit units.

Flux is constant in time inside each interval. Each energy band has its own
integrated photon flux and photon index, so breaks must be explicit band edges.
No absorption, gap filling, or temporal/energy extrapolation is applied here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .models import TabulatedBandSource, build_tabulated_band_source

FLUX_CONVENTION = "unabsorbed observer-equivalent"
HISTORY_UNITS = {"time": "s", "energy": "keV", "band_flux": "ph cm^-2 s^-1"}


def powerlaw_integral(low, high, photon_index):
    """Integral of E**(-Gamma), including the continuous Gamma=1 limit."""
    low, high, gamma = np.broadcast_arrays(
        np.asarray(low, dtype=np.float64),
        np.asarray(high, dtype=np.float64),
        np.asarray(photon_index, dtype=np.float64),
    )
    exponent = 1.0 - gamma
    log_ratio = np.log(high / low)
    x = exponent * log_ratio
    ratio = np.divide(np.expm1(x), x, out=np.ones_like(x), where=x != 0)
    return np.exp(exponent * np.log(low)) * log_ratio * ratio


def _edges(values, name, *, positive=False):
    edges = np.asarray(values, dtype=np.float64)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or not np.all(np.diff(edges) > 0)
        or (positive and edges[0] <= 0)
    ):
        raise ValueError(f"{name} must be finite, strictly increasing edges")
    return edges


def build_spectral_history_source(
    time_edges_s, energy_edges_kev, band_flux, photon_index
) -> TabulatedBandSource:
    """Build continuous spectra with one slope and integrated flux per cell.

    ``band_flux`` is (n_time, n_band), in ph cm^-2 s^-1. ``photon_index`` is
    scalar or has that same shape. Zero-flux cells are allowed; missing data
    must be resolved explicitly before building a history. Source epoch and
    time scale belong to the file adapter, not JIT data.
    """
    times = _edges(time_edges_s, "time_edges_s")
    energies = _edges(energy_edges_kev, "energy_edges_kev", positive=True)
    flux = np.asarray(band_flux, dtype=np.float64)
    shape = (times.size - 1, energies.size - 1)
    if flux.shape != shape:
        raise ValueError("band_flux must have shape (n_time, n_band)")
    gamma = np.asarray(photon_index, dtype=np.float64)
    if gamma.ndim == 0:
        gamma = np.full(shape, gamma)
    if gamma.shape != shape or not np.all(np.isfinite(gamma)):
        raise ValueError("photon_index must be finite and scalar or (n_time, n_band)")
    photons = powerlaw_integral(energies[:-1], energies[1:], gamma)
    if not np.all(np.isfinite(photons)) or np.any(photons <= 0):
        raise ValueError("power-law cell integrals must be finite and positive")
    # Representative energies are only plotting labels for this model. Sampling
    # and response folding always use the full band edges and cell slopes.
    source = build_tabulated_band_source(
        times, flux, np.sqrt(energies[:-1] * energies[1:])
    )
    dtype = source.band_flux.dtype
    # The inverse CDF evaluates expm1((1-Gamma)*log(E_hi/E_lo)) at
    # runtime precision. Reject excessive range instead of clipping overflowed
    # samples onto a band endpoint and silently changing the spectrum.
    log_range = np.abs((1.0 - gamma) * np.log(energies[1:] / energies[:-1]))
    if np.any(log_range > np.log(np.finfo(np.dtype(dtype)).max) - 4):
        raise ValueError("power-law cell dynamic range exceeds JAX sampling precision")
    source = source._replace(
        energy_edges_kev=jnp.asarray(energies, dtype=dtype),
        photon_index=jnp.asarray(gamma, dtype=dtype),
    )
    for name in ("time_edges_s", "energy_edges_kev"):
        if not np.all(np.diff(np.asarray(getattr(source, name))) > 0):
            raise ValueError(f"{name} collapse at the configured JAX precision")
    if (
        not np.all(np.isfinite(np.asarray(source.cell_fluence)))
        or not np.isfinite(float(source.total_fluence))
        or not np.all(np.isfinite(np.asarray(source.photon_index)))
    ):
        raise ValueError("history is not finite at the configured JAX precision")
    if np.any((flux > 0) & (np.asarray(source.cell_fluence) <= 0)):
        raise ValueError(
            "positive history cells underflow at the configured JAX precision"
        )
    increments = np.diff(np.r_[0.0, np.asarray(source.flat_cdf, dtype=float)])
    if np.any((flux.ravel() > 0) & (increments <= 0)):
        raise ValueError("positive history cells disappear from the JAX sampling CDF")
    return source


def build_powerlaw_history_source(
    time_edges_s, photon_flux, energy_edges_kev, photon_index
) -> TabulatedBandSource:
    """One continuous power law per time interval, normalized over all bands."""
    times = _edges(time_edges_s, "time_edges_s")
    energies = _edges(energy_edges_kev, "energy_edges_kev", positive=True)
    flux = np.asarray(photon_flux, dtype=np.float64)
    gamma = np.asarray(photon_index, dtype=np.float64)
    n_time = times.size - 1
    if flux.shape != (n_time,):
        raise ValueError("photon_flux must have one value per source time interval")
    if gamma.ndim == 0:
        gamma = np.full(n_time, gamma)
    if gamma.shape != (n_time,) or not np.all(np.isfinite(gamma)):
        raise ValueError(
            "photon_index must be finite and scalar or one per time interval"
        )
    cells = np.broadcast_to(gamma[:, None], (n_time, energies.size - 1))
    integrals = powerlaw_integral(energies[:-1], energies[1:], cells)
    bands = flux[:, None] * integrals / integrals.sum(axis=1, keepdims=True)
    return build_spectral_history_source(times, energies, bands, cells)


def spectral_density(source: TabulatedBandSource, energy_kev):
    """Return F_E(t_i, E_j) [ph cm^-2 s^-1 keV^-1], zero outside support."""
    if source.energy_edges_kev is None or source.photon_index is None:
        raise ValueError(
            "continuous spectral histories require energy edges and slopes"
        )
    energy = np.asarray(energy_kev, dtype=np.float64)
    if energy.ndim != 1 or not np.all(np.isfinite(energy)) or np.any(energy <= 0):
        raise ValueError("energy_kev must be a finite positive vector")
    edges = np.asarray(source.energy_edges_kev, dtype=np.float64)
    flux = np.asarray(source.band_flux, dtype=np.float64)
    gamma = np.broadcast_to(np.asarray(source.photon_index), flux.shape).astype(float)
    band = np.clip(np.searchsorted(edges, energy, side="right") - 1, 0, len(edges) - 2)
    normalization = powerlaw_integral(edges[:-1], edges[1:], gamma)
    density = flux[:, band] * np.exp(-gamma[:, band] * np.log(energy))
    density /= normalization[:, band]
    density[:, (energy < edges[0]) | (energy > edges[-1])] = 0
    return density


def integrated_spectral_density(source, energy_kev, start_s, stop_s):
    """Exact time integral at each energy over its own [start, stop] interval."""
    energy = np.asarray(energy_kev, dtype=float)
    start, stop = np.broadcast_arrays(
        np.asarray(start_s, dtype=float), np.asarray(stop_s, dtype=float)
    )
    if start.ndim == 0:
        start, stop = np.full(energy.shape, start), np.full(energy.shape, stop)
    if (
        start.shape != energy.shape
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(stop))
        or np.any(stop < start)
    ):
        raise ValueError("time bounds must be finite, ordered, and match energies")
    times = np.asarray(source.time_edges_s, dtype=float)
    overlap = np.maximum(
        np.minimum(times[1:, None], stop) - np.maximum(times[:-1, None], start), 0
    )
    return np.sum(spectral_density(source, energy) * overlap, axis=0)


def load_spectral_history(path):
    """Load a versioned intrinsic history; return the source and file metadata."""
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    required = {
        "schema_version",
        "flux_convention",
        "units",
        "epoch",
        "time_scale",
        "time_edges_s",
        "energy_edges_kev",
        "band_flux",
        "photon_index",
    }
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError("history requires exactly the documented schema fields")
    if data["schema_version"] != 1:
        raise ValueError("unsupported history schema_version")
    if data["flux_convention"] != FLUX_CONVENTION or data["units"] != HISTORY_UNITS:
        raise ValueError(
            "history must declare intrinsic flux convention and exact units"
        )
    if not all(
        isinstance(data[key], str) and data[key].strip()
        for key in ("epoch", "time_scale")
    ):
        raise ValueError("history requires explicit epoch and time_scale")
    source = build_spectral_history_source(
        data["time_edges_s"],
        data["energy_edges_kev"],
        data["band_flux"],
        data["photon_index"],
    )
    return source, {
        "source_history_sha256": hashlib.sha256(raw).hexdigest(),
        "source_epoch": data["epoch"],
        "source_time_scale": data["time_scale"],
    }
