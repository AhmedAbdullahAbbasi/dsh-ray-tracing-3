"""Physical spectral-temporal source models for the JAX transport engine.

This module deliberately stops at the source boundary. It converts an input
light curve into Monte Carlo packet energies, emission times, and statistical
weights; source position and launch direction remain geometry concerns.

The production V1 model is a tabulated band light curve on the same energy
grid as the scattering and absorption tables. A post-peak exponential helper
constructs the current decay simulations. A continuous power-law sampler is
also retained as a supported library API for future dense energy tables; it
uses the same transport kernel and is not a separate scattering model.

Fluxes use the *unabsorbed observer-equivalent* convention. Absorption must be
applied later by the transport engine. Each packet carries an observer-fluence
weight in ph cm^-2, and sampling in proportion to fluence gives equal packet
weights while preserving the total input fluence.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import random


class TabulatedBandSource(NamedTuple):
    """JAX-ready piecewise-constant, band-integrated source table.

    Strings such as the time scale, reference epoch, and absorbed/unabsorbed
    convention are metadata and should remain outside JIT-compiled data. They
    will be added by the file-format adapter once the real MAXI schema is
    fixed.
    """

    time_edges_s: jnp.ndarray
    effective_energy_kev: jnp.ndarray
    band_flux: jnp.ndarray
    cell_fluence: jnp.ndarray
    flat_cdf: jnp.ndarray
    total_fluence: jnp.ndarray
    energy_edges_kev: jnp.ndarray | None = None
    photon_index: jnp.ndarray | None = None


class SourcePackets(NamedTuple):
    """Spectral-temporal properties sampled for a batch of photon packets."""

    energy_kev: jnp.ndarray
    emission_time_s: jnp.ndarray
    weight_observer_fluence: jnp.ndarray
    time_index: jnp.ndarray
    spectral_bin_index: jnp.ndarray


class VariablePowerLawSource(NamedTuple):
    """JAX-ready time-variable source with a fixed power-law spectrum.

    ``photon_flux`` is the photon flux integrated from ``energy_min_kev`` to
    ``energy_max_kev`` in each time interval. The spectrum inside that band is
    proportional to ``E**(-photon_index)`` and is sampled analytically rather
    than approximated with energy bins.
    """

    time_edges_s: jnp.ndarray
    photon_flux: jnp.ndarray
    time_bin_fluence: jnp.ndarray
    time_cdf: jnp.ndarray
    total_fluence: jnp.ndarray
    energy_min_kev: jnp.ndarray
    energy_max_kev: jnp.ndarray
    photon_index: jnp.ndarray


class ObservationWindow(NamedTuple):
    """Observer time interval, in seconds relative to the source epoch."""

    start_s: jnp.ndarray
    stop_s: jnp.ndarray


def build_tabulated_band_source(
    time_edges_s,
    band_flux,
    effective_energy_kev,
) -> TabulatedBandSource:
    """Validate and convert a band-integrated light curve into JAX arrays.

    ``band_flux`` has shape ``(n_time, n_band)`` and units
    ph cm^-2 s^-1. The light curve is piecewise constant inside each pair of
    ``time_edges_s``. Missing measurements must be resolved by the caller;
    NaNs are rejected rather than silently interpolated.
    """

    edges = np.asarray(time_edges_s)
    flux = np.asarray(band_flux)
    energies = np.asarray(effective_energy_kev)

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("time_edges_s must be a 1D array with at least two edges")
    if flux.ndim != 2:
        raise ValueError("band_flux must have shape (n_time, n_band)")
    if energies.ndim != 1 or energies.size == 0:
        raise ValueError("effective_energy_kev must be a non-empty 1D array")
    if flux.shape != (edges.size - 1, energies.size):
        raise ValueError(
            "band_flux shape must be (len(time_edges_s) - 1, len(effective_energy_kev))"
        )
    if not np.all(np.isfinite(edges)):
        raise ValueError("time_edges_s contains non-finite values")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("time_edges_s must be strictly increasing")
    if not np.all(np.isfinite(flux)):
        raise ValueError(
            "band_flux contains non-finite values; apply an explicit gap policy first"
        )
    if np.any(flux < 0.0):
        raise ValueError("band_flux cannot contain negative photon flux")
    if not np.all(np.isfinite(energies)) or np.any(energies <= 0.0):
        raise ValueError("effective_energy_kev must contain finite positive energies")

    dtype = np.result_type(edges.dtype, flux.dtype, energies.dtype, np.float32)
    edges = edges.astype(dtype, copy=False)
    flux = flux.astype(dtype, copy=False)
    energies = energies.astype(dtype, copy=False)

    durations = np.diff(edges)
    cell_fluence = flux * durations[:, None]
    total_fluence = cell_fluence.sum(dtype=dtype)
    if not np.isfinite(total_fluence) or total_fluence <= 0.0:
        raise ValueError("source has zero total fluence")

    flat_cdf = np.cumsum(cell_fluence.ravel(), dtype=dtype)
    flat_cdf /= total_fluence
    flat_cdf[-1] = 1.0

    return TabulatedBandSource(
        time_edges_s=jnp.asarray(edges),
        effective_energy_kev=jnp.asarray(energies),
        band_flux=jnp.asarray(flux),
        cell_fluence=jnp.asarray(cell_fluence),
        flat_cdf=jnp.asarray(flat_cdf),
        total_fluence=jnp.asarray(total_fluence),
    )


def build_powerlaw_band_source(
    time_edges_s,
    photon_flux,
    energy_edges_kev,
    photon_index,
) -> TabulatedBandSource:
    """Integrate a power law into bands while sampling energy within each band.

    ``photon_flux`` is the unabsorbed photon flux integrated over all energy
    bands in each time interval. Energies are drawn analytically from
    ``dN/dE proportional to E**(-photon_index)`` inside the selected band;
    they are not restricted to representative energy nodes.
    """

    edges = np.asarray(energy_edges_kev, dtype=np.float64)
    flux = np.asarray(photon_flux, dtype=np.float64)
    gamma = np.asarray(photon_index, dtype=np.float64)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or edges[0] <= 0.0
        or not np.all(np.diff(edges) > 0.0)
    ):
        raise ValueError("energy edges must be finite, positive, strictly increasing")
    if gamma.ndim != 0 or not np.isfinite(gamma):
        raise ValueError("photon_index must be one finite scalar")
    if flux.shape != (len(time_edges_s) - 1,):
        raise ValueError("photon_flux must have one value per source time interval")

    def integral(power):
        if abs(power + 1.0) < 1e-10:
            return np.log(edges[1:] / edges[:-1])
        exponent = power + 1.0
        return (edges[1:] ** exponent - edges[:-1] ** exponent) / exponent

    photons = integral(-float(gamma))
    if not np.all(photons > 0.0):
        raise ValueError("power-law bands must have positive photon weights")
    band_flux = flux[:, None] * (photons / photons.sum())[None, :]
    mean_energy = integral(1.0 - float(gamma)) / photons
    source = build_tabulated_band_source(time_edges_s, band_flux, mean_energy)
    return source._replace(
        energy_edges_kev=jnp.asarray(edges, dtype=source.effective_energy_kev.dtype),
        photon_index=jnp.asarray(gamma, dtype=source.effective_energy_kev.dtype),
    )


def build_post_peak_exponential_band_source(
    time_edges_s,
    effective_energy_kev,
    peak_band_flux,
    decay_time_s,
    *,
    baseline_band_flux=None,
    peak_time_s=0.0,
) -> TabulatedBandSource:
    """Build a band source following an exponential post-outburst decay.

    For every energy channel ``i``, the continuous light curve is

    ``F_i(t) = F_base,i + (F_peak,i - F_base,i) * exp(-(t-t_peak)/tau)``.

    All requested time edges must be at or after ``peak_time_s``.  The flux
    stored in each interval is its exact analytic average, so the integrated
    source fluence is independent of the chosen temporal bin width.  Packet
    emission times remain uniform inside each tabulated interval; choose bins
    short compared with ``decay_time_s`` when arrival-time structure matters.
    Fluxes are unabsorbed observer-equivalent band photon fluxes in
    ``ph cm^-2 s^-1``.
    """

    edges = np.asarray(time_edges_s, dtype=np.float64)
    energy = np.asarray(effective_energy_kev, dtype=np.float64)
    peak_flux = np.asarray(peak_band_flux, dtype=np.float64)
    if baseline_band_flux is None:
        baseline_flux = np.zeros_like(peak_flux)
    else:
        baseline_flux = np.asarray(baseline_band_flux, dtype=np.float64)
    decay_time = np.asarray(decay_time_s, dtype=np.float64)
    peak_time = np.asarray(peak_time_s, dtype=np.float64)

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("time_edges_s must be a 1D array with at least two edges")
    if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
        raise ValueError("time_edges_s must be finite and strictly increasing")
    if energy.ndim != 1 or energy.size == 0:
        raise ValueError("effective_energy_kev must be a nonempty 1D array")
    if not np.all(np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("effective_energy_kev must be finite and positive")
    if peak_flux.shape != energy.shape or baseline_flux.shape != energy.shape:
        raise ValueError(
            "peak and baseline band fluxes must match effective_energy_kev"
        )
    if (
        not np.all(np.isfinite(peak_flux))
        or not np.all(np.isfinite(baseline_flux))
        or np.any(peak_flux < 0.0)
        or np.any(baseline_flux < 0.0)
    ):
        raise ValueError("peak and baseline band fluxes must be finite and nonnegative")
    if np.any(peak_flux < baseline_flux):
        raise ValueError("peak_band_flux cannot be below baseline_band_flux")
    if decay_time.ndim != 0 or not np.isfinite(decay_time) or decay_time <= 0.0:
        raise ValueError("decay_time_s must be one finite positive scalar")
    if peak_time.ndim != 0 or not np.isfinite(peak_time):
        raise ValueError("peak_time_s must be one finite scalar")
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(float(peak_time)))
    if edges[0] < float(peak_time) - tolerance:
        raise ValueError("post-peak decay time edges cannot precede peak_time_s")

    left = edges[:-1] - float(peak_time)
    right = edges[1:] - float(peak_time)
    duration = right - left
    mean_decay = (
        float(decay_time)
        * (np.exp(-left / float(decay_time)) - np.exp(-right / float(decay_time)))
        / duration
    )
    band_flux = (
        baseline_flux[None, :]
        + (peak_flux - baseline_flux)[None, :] * mean_decay[:, None]
    )
    return build_tabulated_band_source(edges, band_flux, energy)


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


def build_variable_powerlaw_source(
    time_edges_s,
    photon_flux,
    energy_min_kev,
    energy_max_kev,
    photon_index,
) -> VariablePowerLawSource:
    """Build a generic time-variable power-law source.

    Parameters
    ----------
    time_edges_s
        Strictly increasing interval edges in seconds relative to a chosen
        reference epoch.
    photon_flux
        Unabsorbed observer-equivalent photon flux integrated over the full
        energy range, one value per time interval, in ph cm^-2 s^-1.
    energy_min_kev, energy_max_kev
        Positive bounds of the simulated energy range.
    photon_index
        Photon index ``Gamma`` in ``dN/dE proportional to E**(-Gamma)``.
    """

    edges = np.asarray(time_edges_s)
    flux = np.asarray(photon_flux)
    scalar_values = np.asarray(
        [energy_min_kev, energy_max_kev, photon_index], dtype=np.float64
    )

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("time_edges_s must be a 1D array with at least two edges")
    if flux.ndim != 1 or flux.shape != (edges.size - 1,):
        raise ValueError("photon_flux must have one value per time interval")
    if not np.all(np.isfinite(edges)):
        raise ValueError("time_edges_s contains non-finite values")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("time_edges_s must be strictly increasing")
    if not np.all(np.isfinite(flux)):
        raise ValueError(
            "photon_flux contains non-finite values; apply an explicit gap policy first"
        )
    if np.any(flux < 0.0):
        raise ValueError("photon_flux cannot contain negative values")
    if not np.all(np.isfinite(scalar_values)):
        raise ValueError("energy bounds and photon_index must be finite")
    if energy_min_kev <= 0.0 or energy_max_kev <= energy_min_kev:
        raise ValueError(
            "energy bounds must satisfy 0 < energy_min_kev < energy_max_kev"
        )

    dtype = np.result_type(edges.dtype, flux.dtype, np.float32)
    edges = edges.astype(dtype, copy=False)
    flux = flux.astype(dtype, copy=False)
    durations = np.diff(edges)
    time_bin_fluence = flux * durations
    total_fluence = time_bin_fluence.sum(dtype=dtype)
    if not np.isfinite(total_fluence) or total_fluence <= 0.0:
        raise ValueError("source has zero total fluence")

    time_cdf = np.cumsum(time_bin_fluence, dtype=dtype)
    time_cdf /= total_fluence
    time_cdf[-1] = 1.0

    return VariablePowerLawSource(
        time_edges_s=jnp.asarray(edges),
        photon_flux=jnp.asarray(flux),
        time_bin_fluence=jnp.asarray(time_bin_fluence),
        time_cdf=jnp.asarray(time_cdf),
        total_fluence=jnp.asarray(total_fluence),
        energy_min_kev=jnp.asarray(energy_min_kev, dtype=jnp.asarray(edges).dtype),
        energy_max_kev=jnp.asarray(energy_max_kev, dtype=jnp.asarray(edges).dtype),
        photon_index=jnp.asarray(photon_index, dtype=jnp.asarray(edges).dtype),
    )


def _sample_powerlaw_energy(
    key, energy_min_kev, energy_max_kev, photon_index, n_packets
):
    """Sample exactly from ``p(E) proportional to E**(-photon_index)``."""

    u = random.uniform(key, shape=(n_packets,))
    alpha = 1.0 - photon_index
    use_log_limit = jnp.abs(alpha) < 1.0e-6

    energy_log = energy_min_kev * (energy_max_kev / energy_min_kev) ** u

    alpha_safe = jnp.where(use_log_limit, 1.0, alpha)
    low_power = energy_min_kev**alpha_safe
    high_power = energy_max_kev**alpha_safe
    energy_power = (low_power + u * (high_power - low_power)) ** (1.0 / alpha_safe)
    return jnp.where(use_log_limit, energy_log, energy_power)


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


def fred_outburst_flux(
    time_edges_s,
    baseline_flux,
    peak_excess_flux,
    peak_time_s,
    rise_time_s,
    decay_time_s,
):
    r"""Create a fast-rise, exponential-decay transient light curve.

    The continuous model is

    .. math::

        F(t) = F_0 + A \exp[(t-t_p)/\tau_r],\quad t \le t_p,

        F(t) = F_0 + A \exp[-(t-t_p)/\tau_d],\quad t > t_p.

    Rather than evaluating this curve only at bin centers, the function
    integrates it analytically over every interval and returns the exact
    interval-average flux. Consequently, the total fluence does not depend on
    the chosen temporal bin width apart from floating-point roundoff.

    The returned array is intended for :func:`build_variable_powerlaw_source`
    and has the same units as ``baseline_flux`` and ``peak_excess_flux``.
    """

    edges = np.asarray(time_edges_s)
    parameters = np.asarray(
        [baseline_flux, peak_excess_flux, peak_time_s, rise_time_s, decay_time_s],
        dtype=np.float64,
    )
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
        raise ValueError(
            "time_edges_s must be a finite 1D array with at least two edges"
        )
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("time_edges_s must be strictly increasing")
    if not np.all(np.isfinite(parameters)):
        raise ValueError("FRED outburst parameters must be finite")
    if baseline_flux < 0.0 or peak_excess_flux < 0.0:
        raise ValueError("FRED outburst fluxes cannot be negative")
    if rise_time_s <= 0.0 or decay_time_s <= 0.0:
        raise ValueError("rise_time_s and decay_time_s must be positive")

    left = edges[:-1]
    right = edges[1:]
    duration = right - left

    rise_integral = (
        peak_excess_flux
        * rise_time_s
        * (
            np.exp((np.minimum(right, peak_time_s) - peak_time_s) / rise_time_s)
            - np.exp((np.minimum(left, peak_time_s) - peak_time_s) / rise_time_s)
        )
    )
    rise_integral = np.where(left < peak_time_s, rise_integral, 0.0)

    decay_integral = (
        peak_excess_flux
        * decay_time_s
        * (
            np.exp(-(np.maximum(left, peak_time_s) - peak_time_s) / decay_time_s)
            - np.exp(-(np.maximum(right, peak_time_s) - peak_time_s) / decay_time_s)
        )
    )
    decay_integral = np.where(right > peak_time_s, decay_integral, 0.0)

    return baseline_flux + (rise_integral + decay_integral) / duration


def build_decay_observation_window(
    start_s,
    stop_s,
    outburst_peak_s,
) -> ObservationWindow:
    """Validate an observation interval lying wholly on the outburst decay."""

    values = np.asarray([start_s, stop_s, outburst_peak_s], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("observation times and outburst_peak_s must be finite")
    if stop_s <= start_s:
        raise ValueError("observation stop_s must be later than start_s")
    if start_s <= outburst_peak_s:
        raise ValueError(
            "decay-phase observation must start strictly after the outburst peak"
        )

    return ObservationWindow(
        start_s=jnp.asarray(start_s),
        stop_s=jnp.asarray(stop_s),
    )
