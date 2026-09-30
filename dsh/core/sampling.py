"""Numerical photon and angle proposals; no source-file handling."""

from __future__ import annotations

import jax.numpy as jnp
from jax import random

from dsh.contracts import (
    FLAT,
    LINE,
    SourceCells,
    SourcePackets,
    TabulatedBandSource,
    VariablePowerLawSource,
)
from dsh.contracts import POWERLAW as POWERLAW


def sample_tabulated_band_source(
    key,
    source: TabulatedBandSource,
    n_packets: int,
) -> SourcePackets:
    """Draw a packet batch from a tabulated band light curve.

    A time-band cell is selected in proportion to its integrated fluence, and
    emission time is uniform inside that cell because its flux is defined to
    be piecewise constant. The equal packet weights sum to the table's total
    observer fluence.

    ``n_packets`` determines output shapes and must be static under
    :func:`jax.jit`.
    """

    if n_packets <= 0:
        raise ValueError("n_packets must be positive")

    if source.energy_edges_kev is None:
        key_cell, key_time = random.split(key)
    else:
        key_cell, key_time, key_energy = random.split(key, 3)
    u_cell = random.uniform(key_cell, shape=(n_packets,))
    u_time = random.uniform(key_time, shape=(n_packets,))

    flat_index = jnp.searchsorted(source.flat_cdf, u_cell, side="right")
    flat_index = jnp.minimum(flat_index, source.flat_cdf.size - 1)

    n_band = source.effective_energy_kev.size
    time_index = flat_index // n_band
    band_index = flat_index % n_band

    t0 = source.time_edges_s[time_index]
    t1 = source.time_edges_s[time_index + 1]
    emission_time_s = t0 + (t1 - t0) * u_time
    if source.energy_edges_kev is None:
        energy_kev = source.effective_energy_kev[band_index]
    else:
        energy_kev = _sample_powerlaw_energy(
            key_energy,
            source.energy_edges_kev[band_index],
            source.energy_edges_kev[band_index + 1],
            source.photon_index,
            n_packets,
        )
    weight = jnp.full(
        (n_packets,),
        source.total_fluence / n_packets,
        dtype=source.total_fluence.dtype,
    )

    return SourcePackets(
        energy_kev=energy_kev,
        emission_time_s=emission_time_s,
        weight_observer_fluence=weight,
        time_index=time_index.astype(jnp.int32),
        spectral_bin_index=band_index.astype(jnp.int32),
    )


def _sample_powerlaw_energy(
    key, energy_min_kev, energy_max_kev, photon_index, n_packets
):
    """Sample exactly from ``p(E) proportional to E**(-photon_index)``."""

    u = random.uniform(key, shape=(n_packets,))
    alpha = 1.0 - photon_index
    use_log_limit = jnp.abs(alpha) < 1.0e-6

    log_ratio = jnp.log(energy_max_kev / energy_min_kev)
    alpha_safe = jnp.where(use_log_limit, 1.0, alpha)
    log_scaled = jnp.log1p(u * jnp.expm1(alpha_safe * log_ratio)) / alpha_safe
    energy = energy_min_kev * jnp.exp(
        jnp.where(use_log_limit, u * log_ratio, log_scaled)
    )
    # Float32 inverse-CDF evaluation can round one ULP below the lower bound
    # (observed at 2 keV in a 2.5M-packet run). Preserve the requested support.
    return jnp.clip(energy, energy_min_kev, energy_max_kev)


def sample_variable_powerlaw_source(
    key,
    source: VariablePowerLawSource,
    n_packets: int,
) -> SourcePackets:
    """Draw energies and emission times from a variable power-law source."""

    if n_packets <= 0:
        raise ValueError("n_packets must be positive")

    key_interval, key_time, key_energy = random.split(key, 3)
    u_interval = random.uniform(key_interval, shape=(n_packets,))
    u_time = random.uniform(key_time, shape=(n_packets,))

    time_index = jnp.searchsorted(source.time_cdf, u_interval, side="right")
    time_index = jnp.minimum(time_index, source.time_cdf.size - 1)
    t0 = source.time_edges_s[time_index]
    t1 = source.time_edges_s[time_index + 1]
    emission_time_s = t0 + (t1 - t0) * u_time

    energy_kev = _sample_powerlaw_energy(
        key_energy,
        source.energy_min_kev,
        source.energy_max_kev,
        source.photon_index,
        n_packets,
    )
    weight = jnp.full(
        (n_packets,),
        source.total_fluence / n_packets,
        dtype=source.total_fluence.dtype,
    )

    return SourcePackets(
        energy_kev=energy_kev,
        emission_time_s=emission_time_s,
        weight_observer_fluence=weight,
        time_index=time_index.astype(jnp.int32),
        spectral_bin_index=jnp.full((n_packets,), -1, dtype=jnp.int32),
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


def _sample_scattering_angle(key, angle_grid, phase_cdf):
    uniform = random.uniform(key, dtype=phase_cdf.dtype)
    upper = jnp.searchsorted(phase_cdf, uniform, side="right")
    upper = jnp.clip(upper, 1, phase_cdf.size - 1)
    lower = upper - 1
    cdf_low = phase_cdf[lower]
    cdf_high = phase_cdf[upper]
    fraction = (uniform - cdf_low) / jnp.maximum(cdf_high - cdf_low, 1.0e-30)
    return angle_grid[lower] + fraction * (angle_grid[upper] - angle_grid[lower])
