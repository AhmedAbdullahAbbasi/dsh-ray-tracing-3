"""Bin simulated observer events into a testable ideal-observer ring image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..observer.binning import bin_observer_events, build_observer_bin_geometry
from ..observer.scoring import ObserverEventResult
from .analytic import small_angle_ring_radius_arcsec


@dataclass(frozen=True)
class RingSliceMeasurement:
    """Median radius measured directly from one binned observer image."""

    time_bin_index: int
    time_start_s: float
    time_end_s: float
    event_count: int
    measured_median_radius_arcsec: float
    analytic_midpoint_radius_arcsec: float


@dataclass(frozen=True)
class ImageValidation:
    path: str
    binned_fluence: float
    unbinned_fluence: float
    ring_bins_checked: int
    maximum_ring_bound_violation_arcsec: float
    half_pixel_diagonal_arcsec: float
    fits_path: str | None = None
    ring_slices: tuple[RingSliceMeasurement, ...] = ()


def _write_optional_fits(
    output_path, cube, counts, sky_edges, time_edges, energy, distance_kpc, fraction
):
    """Write the same image with physical axes when Astropy is available."""

    try:
        from astropy.io import fits
    except ImportError:
        return None
    image = cube.astype(np.float32)
    primary = fits.PrimaryHDU(image.sum(axis=0, dtype=np.float64).astype(np.float32))
    primary.header["BUNIT"] = "ph cm-2"
    primary.header["IDEALOBS"] = True
    primary.header["ENERGY"] = (float(energy), "effective photon energy in keV")
    primary.header["DISTKPC"] = (float(distance_kpc), "observer-source distance in kpc")
    primary.header["DUSTX"] = (
        float(fraction),
        "observer-dust distance / source distance",
    )
    temporal = fits.ImageHDU(image, name="TIME_CUBE")
    temporal.header["BUNIT"] = "ph cm-2"
    temporal.header["CTYPE1"] = "XOFFSET"
    temporal.header["CTYPE2"] = "YOFFSET"
    temporal.header["CUNIT1"] = "arcsec"
    temporal.header["CUNIT2"] = "arcsec"
    temporal.header["CRPIX1"] = 1.0
    temporal.header["CRPIX2"] = 1.0
    temporal.header["CRVAL1"] = float((sky_edges[0] + sky_edges[1]) / 2.0)
    temporal.header["CRVAL2"] = float((sky_edges[0] + sky_edges[1]) / 2.0)
    temporal.header["CDELT1"] = float(sky_edges[1] - sky_edges[0])
    temporal.header["CDELT2"] = float(sky_edges[1] - sky_edges[0])
    temporal.header["CTYPE3"] = "TIME"
    temporal.header["CUNIT3"] = "s"
    temporal.header["CRPIX3"] = 1.0
    temporal.header["CRVAL3"] = float((time_edges[0] + time_edges[1]) / 2.0)
    temporal.header["CDELT3"] = float(time_edges[1] - time_edges[0])
    bins = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="LOW", format="D", unit="s", array=time_edges[:-1]),
            fits.Column(name="HIGH", format="D", unit="s", array=time_edges[1:]),
        ],
        name="TIME_BINS",
    )
    count_image = fits.ImageHDU(counts.astype(np.int32), name="EVENT_COUNT")
    fits_path = output_path.with_suffix(".fits")
    fits.HDUList([primary, temporal, count_image, bins]).writeto(
        fits_path, overwrite=True
    )
    return str(fits_path)


def bin_and_validate_ring_image(
    path,
    sky_x_arcsec,
    sky_y_arcsec,
    arrival_time_s,
    weight_observer_fluence,
    *,
    energy_kev,
    source_distance_kpc,
    fractional_distance,
    cloud_half_width_arcsec=2_000.0,
) -> ImageValidation:
    """Save a true scorer-event image cube and inspect its time-sliced rings.

    The finite, time-limited crop is deliberately a *ring diagnostic*;
    its unbinned fluence is reported separately and never equated to zero.
    """

    x = np.asarray(sky_x_arcsec, dtype=np.float32).reshape(-1)
    y = np.asarray(sky_y_arcsec, dtype=np.float32).reshape(-1)
    t = np.asarray(arrival_time_s, dtype=np.float32).reshape(-1)
    w = np.asarray(weight_observer_fluence, dtype=np.float32).reshape(-1)
    if not x.size or y.shape != x.shape or t.shape != x.shape or w.shape != x.shape:
        raise ValueError("scored image events must be nonempty matching arrays")

    ring_at_four_days = float(
        small_angle_ring_radius_arcsec(
            4.0 * 86_400.0, source_distance_kpc, fractional_distance
        )
    )
    half_width = min(cloud_half_width_arcsec, max(150.0, 1.3 * ring_at_four_days))
    sky_edges = np.linspace(-half_width, half_width, 129)
    time_edges = np.linspace(0.0, 8.0 * 86_400.0, 33)
    geometry = build_observer_bin_geometry(
        sky_edges, sky_edges, [energy_kev - 0.1, energy_kev + 0.1], time_edges
    )
    n = len(x)
    shape = (n, 1)
    zero = jnp.zeros(shape, dtype=jnp.float32)
    events = ObserverEventResult(
        valid=jnp.ones(shape, dtype=bool),
        sky_x_arcsec=jnp.asarray(x[:, None]),
        sky_y_arcsec=jnp.asarray(y[:, None]),
        energy_kev=jnp.full(shape, energy_kev, dtype=jnp.float32),
        arrival_time_s=jnp.asarray(t[:, None]),
        excess_path_length_pc=zero,
        scattering_angle_rad=zero,
        scattering_order=jnp.ones(shape, dtype=jnp.int32),
        escape_column_cm2=zero,
        escape_optical_depth=zero,
        transmission=jnp.ones(shape, dtype=jnp.float32),
        phase_pdf_per_sr=zero,
        weight_observer_fluence=jnp.asarray(w[:, None]),
        time_index=jnp.zeros(shape, dtype=jnp.int32),
        spectral_bin_index=jnp.zeros(shape, dtype=jnp.int32),
    )
    binned = bin_observer_events(events, geometry)
    cube = np.asarray(binned.total_fluence)[:, 0].astype(np.float64)
    counts = np.asarray(binned.event_count)[:, 0]
    centers = (sky_edges[1:] + sky_edges[:-1]) / 2.0
    yy, xx = np.meshgrid(centers, centers, indexing="ij")
    pixel_radii = np.hypot(xx, yy).ravel()
    half_pixel_diagonal = float(np.diff(sky_edges)[0] / np.sqrt(2.0))
    maximum_violation = 0.0
    ring_slices = []
    for i, image in enumerate(cube):
        low, high = time_edges[i : i + 2]
        if low < 86_400.0 or high > 4.0 * 86_400.0 or counts[i].sum() < 20:
            continue
        values = image.ravel()
        if values.sum() <= 0:
            continue
        order = np.argsort(pixel_radii)
        radius = pixel_radii[order]
        cumulated = np.cumsum(values[order], dtype=np.float64)
        measured = radius[np.searchsorted(cumulated, cumulated[-1] / 2.0)]
        bounds = small_angle_ring_radius_arcsec(
            np.asarray([low, high]), source_distance_kpc, fractional_distance
        )
        maximum_violation = max(
            maximum_violation,
            float(max(bounds[0] - measured, measured - bounds[1], 0.0)),
        )
        ring_slices.append(
            RingSliceMeasurement(
                time_bin_index=i,
                time_start_s=float(low),
                time_end_s=float(high),
                event_count=int(counts[i].sum()),
                measured_median_radius_arcsec=float(measured),
                analytic_midpoint_radius_arcsec=float(
                    small_angle_ring_radius_arcsec(
                        0.5 * (low + high), source_distance_kpc, fractional_distance
                    )
                ),
            )
        )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        image_schema_version=np.asarray(1),
        fluence_time_y_x=cube.astype(np.float32),
        event_count_time_y_x=counts,
        sky_x_edges_arcsec=sky_edges,
        sky_y_edges_arcsec=sky_edges,
        arrival_time_edges_s=time_edges,
        energy_kev=np.asarray(energy_kev),
        source_distance_kpc=np.asarray(source_distance_kpc),
        fractional_dust_distance=np.asarray(fractional_distance),
        fluence_unit=np.asarray("ph cm-2 per time and sky pixel"),
    )
    fits_path = _write_optional_fits(
        output_path,
        cube,
        counts,
        sky_edges,
        time_edges,
        energy_kev,
        source_distance_kpc,
        fractional_distance,
    )
    return ImageValidation(
        path=str(output_path),
        binned_fluence=float(np.asarray(binned.binned_weight_observer_fluence)),
        unbinned_fluence=float(np.asarray(binned.unbinned_weight_observer_fluence)),
        ring_bins_checked=len(ring_slices),
        maximum_ring_bound_violation_arcsec=maximum_violation,
        half_pixel_diagonal_arcsec=half_pixel_diagonal,
        fits_path=fits_path,
        ring_slices=tuple(ring_slices),
    )
