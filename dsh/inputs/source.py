"""Explicit source adapters; flux-table values are integrated per-band photons."""

import numpy as np

from ..examples import DAY_S
from ..sources.histories import build_spectral_history_source, load_spectral_history
from ..sources.models import (
    build_post_peak_exponential_band_source,
    build_powerlaw_band_source,
    build_tabulated_band_source,
)


def validate_runtime_source(source):
    """Reject runtime precision collapse without changing existing samplers."""
    flux = np.asarray(source.band_flux, dtype=float)
    for name in ("time_edges_s", "energy_edges_kev"):
        array = getattr(source, name)
        if array is not None and (
            not np.all(np.isfinite(array)) or not np.all(np.diff(np.asarray(array)) > 0)
        ):
            raise ValueError(f"source.{name}: edges collapse at JAX precision")
    cells = np.asarray(source.cell_fluence)
    increments = np.diff(np.r_[0, np.asarray(source.flat_cdf, dtype=float)])
    if (
        not np.isfinite(float(source.total_fluence))
        or not np.all(np.isfinite(cells))
        or np.any((flux > 0) & (cells <= 0))
        or np.any((flux.ravel() > 0) & (increments <= 0))
    ):
        raise ValueError(
            "source: positive flux cells underflow or disappear at JAX sampling precision"
        )
    if source.photon_index is not None:
        gamma = np.asarray(source.photon_index)
        edges = np.asarray(source.energy_edges_kev, dtype=float)
        limit = np.log(np.finfo(np.asarray(source.band_flux).dtype).max) - 4
        if not np.all(np.isfinite(gamma)) or np.any(
            np.abs((1 - gamma) * np.log(edges[1:] / edges[:-1])) > limit
        ):
            raise ValueError(
                "source.photon_index: power-law dynamic range exceeds JAX sampling precision"
            )
    return source


def load_source(config):
    metadata = {
        "source_model": config.kind,
        "source_spectrum": config.kind,
        "source_flux_convention": config.flux_convention,
        "source_flux_unit": config.flux_unit,
    }
    times = np.asarray(config.time_edges_days, dtype=float) * DAY_S
    if config.kind == "spectral_history":
        source, history_metadata = load_spectral_history(config.file)
        metadata.update(history_metadata)
    elif config.kind == "flux_table":
        table = np.loadtxt(config.file, ndmin=2)
        columns = (config.time_column, *config.flux_columns)
        if max(columns) > table.shape[1]:
            raise ValueError(
                "source.flux_columns/time_column: exceed the full table width"
            )
        mjd = table[:, config.time_column - 1]
        flux = table[:, np.asarray(config.flux_columns) - 1] * config.flux_scale
        # Subtract epoch in host float64 BEFORE casting relative seconds to JAX.
        relative_days = mjd - config.epoch_mjd
        if (
            not np.all(np.isfinite(relative_days))
            or np.any(relative_days < 0)
            or not np.all(np.diff(relative_days) > 0)
        ):
            raise ValueError(
                "source.file: MJD times must be finite, ordered, distinct and not before epoch_mjd"
            )
        if (
            np.any(np.diff(relative_days) > config.max_interval_days + 1e-9)
            or config.last_interval_days > config.max_interval_days
        ):
            raise ValueError(
                "source.file: interval exceeds max_interval_days; gaps cannot be filled silently"
            )
        times = (
            np.r_[relative_days, relative_days[-1] + config.last_interval_days] * DAY_S
        )
        source = build_spectral_history_source(
            times, config.energy_edges_kev, flux, config.photon_index
        )
        metadata.update(
            source_epoch=f"MJD {config.epoch_mjd:.12g}",
            source_time_scale=config.time_scale,
            source_flux_columns=list(config.flux_columns),
            source_column_numbering=config.column_numbering,
        )
    elif config.kind == "band_powerlaw":
        source = build_spectral_history_source(
            times, config.energy_edges_kev, config.band_flux, config.photon_index
        )
    elif config.kind == "powerlaw":
        # Retain the existing scalar builder's representative labels and arrays.
        source = build_powerlaw_band_source(
            times, config.photon_flux, config.energy_edges_kev, config.photon_index
        )
    elif config.kind == "exponential_decay":
        source = build_post_peak_exponential_band_source(
            times,
            config.effective_energy_kev,
            config.peak_band_flux,
            config.decay_time_days * DAY_S,
            baseline_band_flux=config.baseline_band_flux or None,
        )
    else:
        source = build_tabulated_band_source(
            times, config.band_flux, config.effective_energy_kev
        )
    if np.asarray(source.time_edges_s)[0] < 0:
        raise ValueError(
            "source: emission times must begin at or after the declared epoch"
        )
    return validate_runtime_source(source), metadata
