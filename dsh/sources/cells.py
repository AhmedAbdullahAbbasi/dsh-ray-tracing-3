"""Compile canonical source fluxes into one JAX-ready photon proposal."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import random

from .models import SourcePackets, _sample_powerlaw_energy
from .source_fits import SourceFluxFile

LINE = 0
FLAT = 1
POWERLAW = 2


class SourceCells(NamedTuple):
    """Positive-fluence cells; all fields are numerical JAX pytrees."""

    time_edges_s: jnp.ndarray
    start_s: jnp.ndarray
    stop_s: jnp.ndarray
    energy_low_kev: jnp.ndarray
    energy_high_kev: jnp.ndarray
    photon_index: jnp.ndarray
    kind: jnp.ndarray
    time_index: jnp.ndarray
    spectral_bin_index: jnp.ndarray
    cell_fluence: jnp.ndarray
    flat_cdf: jnp.ndarray
    total_fluence: jnp.ndarray


def build_source_cells(
    source: SourceFluxFile, *, components: str = "both"
) -> SourceCells:
    """Select continuum/lines and preserve the selected absolute fluence."""

    if components not in {"both", "continuum", "lines"}:
        raise ValueError("components must be both, continuum or lines")
    n_cont = source.continuum_flux.shape[1]
    n_line = source.line_energy_kev.size
    if components == "continuum" and not n_cont:
        raise ValueError("source file has no continuum")
    if components == "lines" and not n_line:
        raise ValueError("source file has no lines")
    durations = np.diff(source.time_edges_s)
    records = []
    for t, duration in enumerate(durations):
        if components != "lines":
            for b in range(n_cont):
                fluence = float(source.continuum_flux[t, b] * duration)
                if fluence > 0.0:
                    kind = FLAT if source.continuum_shape[b] == "FLAT" else POWERLAW
                    records.append(
                        (
                            t,
                            b,
                            kind,
                            source.continuum_energy_edges_kev[b],
                            source.continuum_energy_edges_kev[b + 1],
                            source.continuum_photon_index[t, b],
                            fluence,
                        )
                    )
        if components != "continuum":
            for b in range(n_line):
                fluence = float(source.line_flux[t, b] * duration)
                if fluence > 0.0:
                    energy = source.line_energy_kev[b]
                    records.append((t, n_cont + b, LINE, energy, energy, 0.0, fluence))
    if not records:
        raise ValueError("selected source components have zero total fluence")
    values = np.asarray(records, dtype=np.float64)
    fluence = values[:, 6]
    total = fluence.sum(dtype=np.float64)
    cdf = np.cumsum(fluence, dtype=np.float64) / total
    cdf[-1] = 1.0
    return SourceCells(
        time_edges_s=jnp.asarray(source.time_edges_s),
        start_s=jnp.asarray(source.time_edges_s[values[:, 0].astype(np.int32)]),
        stop_s=jnp.asarray(source.time_edges_s[values[:, 0].astype(np.int32) + 1]),
        energy_low_kev=jnp.asarray(values[:, 3]),
        energy_high_kev=jnp.asarray(values[:, 4]),
        photon_index=jnp.asarray(values[:, 5]),
        kind=jnp.asarray(values[:, 2], dtype=jnp.int32),
        time_index=jnp.asarray(values[:, 0], dtype=jnp.int32),
        spectral_bin_index=jnp.asarray(values[:, 1], dtype=jnp.int32),
        cell_fluence=jnp.asarray(fluence),
        flat_cdf=jnp.asarray(cdf),
        total_fluence=jnp.asarray(total),
    )


def sample_source_cells(key, source: SourceCells, n_packets: int) -> SourcePackets:
    """Draw line, uniform-continuum and per-cell power-law photons."""

    if n_packets <= 0:
        raise ValueError("n_packets must be positive")
    key_cell, key_time, key_energy = random.split(key, 3)
    u_cell = random.uniform(key_cell, shape=(n_packets,))
    u_time = random.uniform(key_time, shape=(n_packets,))
    u_energy = random.uniform(key_energy, shape=(n_packets,))
    cell = jnp.searchsorted(source.flat_cdf, u_cell, side="right")
    cell = jnp.minimum(cell, source.flat_cdf.size - 1)
    start = source.start_s[cell]
    stop = source.stop_s[cell]
    low = source.energy_low_kev[cell]
    high = source.energy_high_kev[cell]
    kind = source.kind[cell]
    flat_energy = jnp.clip(low + u_energy * (high - low), low, high)
    # Reuse the validated stable inverse CDF. Its independent random stream
    # does not alter the cell and time proposals; a future sampler can accept
    # pre-drawn uniforms without changing the physical distribution.
    power_energy = _sample_powerlaw_energy(
        random.fold_in(key_energy, 1),
        low,
        jnp.where(kind == LINE, low * 2.0, high),
        source.photon_index[cell],
        n_packets,
    )
    energy = jnp.where(
        kind == LINE, low, jnp.where(kind == FLAT, flat_energy, power_energy)
    )
    return SourcePackets(
        energy_kev=energy,
        emission_time_s=start + (stop - start) * u_time,
        weight_observer_fluence=jnp.full(
            (n_packets,),
            source.total_fluence / n_packets,
            dtype=source.total_fluence.dtype,
        ),
        time_index=source.time_index[cell],
        spectral_bin_index=source.spectral_bin_index[cell],
    )
