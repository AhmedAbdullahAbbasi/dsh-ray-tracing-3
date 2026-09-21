"""Small built-in scenes and sources for smoke tests and demonstrations."""

from __future__ import annotations

import numpy as np

from .geometry.clouds import build_angular_distance_cloud
from .sources.models import (
    build_post_peak_exponential_band_source,
    build_tabulated_band_source,
)

DAY_S = 86_400.0


def build_synthetic_four_cloud_scene(source_distance_kpc=10.0):
    """Return a 500-arcsec four-cloud plus diffuse-H test scene."""

    x_arcsec = np.arange(-250.0, 250.0 + 10.0, 10.0)
    y_arcsec = np.arange(-250.0, 250.0 + 10.0, 10.0)
    z_kpc = np.arange(1.025, 8.975 + 0.025, 0.05)
    sky_x, sky_y = np.meshgrid(x_arcsec, y_arcsec, indexing="xy")
    delta_nh = np.zeros((z_kpc.size, y_arcsec.size, x_arcsec.size), dtype=np.float64)

    # distance, x0, y0, sigma_x, sigma_y, sigma_z, peak column [cm^-2]
    cloud_parameters = (
        (2.0, -90.0, 65.0, 70.0, 45.0, 0.10, 8.0e21),
        (3.7, 75.0, -55.0, 55.0, 85.0, 0.14, 1.0e22),
        (5.6, -25.0, -20.0, 105.0, 60.0, 0.18, 1.2e22),
        (7.8, 55.0, 80.0, 80.0, 100.0, 0.22, 9.0e21),
    )
    for distance, x0, y0, sx, sy, sz, peak_column in cloud_parameters:
        projected_column = peak_column * np.exp(
            -0.5 * ((sky_x - x0) / sx) ** 2 - 0.5 * ((sky_y - y0) / sy) ** 2
        )
        radial_weight = np.exp(-0.5 * ((z_kpc - distance) / sz) ** 2)
        radial_weight /= radial_weight.sum()
        delta_nh += radial_weight[:, None, None] * projected_column[None, :, :]

    diffuse_column = 2.0e21 * (1.0 + 0.12 * sky_x / 250.0 - 0.08 * sky_y / 250.0)
    delta_nh += diffuse_column[None, :, :] / z_kpc.size

    return build_angular_distance_cloud(
        delta_nh,
        x_centers_arcsec=x_arcsec,
        y_centers_arcsec=y_arcsec,
        z_centers_kpc=z_kpc,
        source_distance_kpc=source_distance_kpc,
    )


def build_v1_test_source(
    energy_kev,
    band_flux=(2.0e-2, 1.2e-2, 6.0e-3),
):
    """Return a one-hour flare at the tabulated Version-1 energies."""

    return build_tabulated_band_source(
        time_edges_s=[0.0, 3_600.0],
        band_flux=np.asarray([band_flux], dtype=np.float64),
        effective_energy_kev=energy_kev,
    )


def post_peak_time_edges_s(start_days, duration_days, bin_days):
    """Build validated post-peak time-bin edges in seconds."""

    values = np.asarray([start_days, duration_days, bin_days], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("decay timing parameters must be finite")
    if start_days < 0.0:
        raise ValueError("decay_start_days cannot be negative")
    if duration_days <= 0.0 or bin_days <= 0.0:
        raise ValueError("decay duration and source time-bin width must be positive")
    n_bins = int(np.ceil(duration_days / bin_days))
    edges_days = start_days + np.arange(n_bins + 1) * bin_days
    edges_days[-1] = start_days + duration_days
    return edges_days * DAY_S


def build_v1_decay_source(
    energy_kev,
    *,
    peak_band_flux,
    baseline_band_flux,
    decay_time_days,
    decay_start_days,
    decay_duration_days,
    source_time_bin_days,
):
    """Return the configurable post-outburst source used by the V1 runner."""

    time_edges_s = post_peak_time_edges_s(
        decay_start_days,
        decay_duration_days,
        source_time_bin_days,
    )
    return build_post_peak_exponential_band_source(
        time_edges_s,
        energy_kev,
        peak_band_flux,
        decay_time_s=decay_time_days * DAY_S,
        baseline_band_flux=baseline_band_flux,
        peak_time_s=0.0,
    )
