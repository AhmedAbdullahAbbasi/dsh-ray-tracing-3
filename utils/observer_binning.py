"""Bin peel-off observer events into physical DSH data products.

The canonical cube order is ``(time, energy, sky_y, sky_x)``.  This matches
NumPy/FITS image conventions in which the x coordinate is the final, fastest
varying array axis.  The input event weights are observer fluences in
``ph cm^-2``; binning only sums those weights and does not apply an instrument
response, exposure correction, or Poisson realization.

Histogram intervals are left-inclusive and right-exclusive, except that the
final edge of every axis is included in the final bin.  This is the same edge
convention used by NumPy histograms and prevents an event on the upper image
boundary from being lost.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .coordinates import ARCSEC_TO_RAD
from .observer import ObserverEventResult


class ObserverBinGeometry(NamedTuple):
    """Validated observer-product axes and exact sky-pixel solid angles."""

    sky_x_edges_arcsec: jnp.ndarray
    sky_y_edges_arcsec: jnp.ndarray
    energy_edges_kev: jnp.ndarray
    arrival_time_edges_s: jnp.ndarray
    sky_pixel_solid_angle_sr: jnp.ndarray


class BinnedObserverProducts(NamedTuple):
    """Weighted DSH cubes and Monte Carlo closure diagnostics.

    The three fluence arrays and ``event_count`` have shape
    ``(n_time, n_energy, n_sky_y, n_sky_x)``.  Fluence units are
    ``ph cm^-2`` per four-dimensional bin.  ``valid_*`` totals refer to all
    input events marked valid; ``binned_*`` totals include only events inside
    every requested axis.  Their difference is reported as ``unbinned_*``.
    """

    total_fluence: jnp.ndarray
    first_scatter_fluence: jnp.ndarray
    multiple_scatter_fluence: jnp.ndarray
    event_count: jnp.ndarray
    valid_event_count: jnp.ndarray
    binned_event_count: jnp.ndarray
    unbinned_event_count: jnp.ndarray
    valid_weight_observer_fluence: jnp.ndarray
    binned_weight_observer_fluence: jnp.ndarray
    unbinned_weight_observer_fluence: jnp.ndarray


def _validated_edges(values, name, *, positive=False):
    edges = np.asarray(values, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(f"{name} must be one-dimensional with at least two edges")
    if not np.all(np.isfinite(edges)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    if positive and np.any(edges <= 0.0):
        raise ValueError(f"{name} must contain only positive values")
    return edges


def _solid_angle_primitive(slope_x, slope_y):
    return np.arctan2(
        slope_x * slope_y,
        np.sqrt(1.0 + slope_x * slope_x + slope_y * slope_y),
    )


def _sky_pixel_solid_angles_sr(x_edges_arcsec, y_edges_arcsec):
    x_slope = np.tan(x_edges_arcsec * ARCSEC_TO_RAD)
    y_slope = np.tan(y_edges_arcsec * ARCSEC_TO_RAD)
    x0 = x_slope[:-1][None, :]
    x1 = x_slope[1:][None, :]
    y0 = y_slope[:-1][:, None]
    y1 = y_slope[1:][:, None]
    return (
        _solid_angle_primitive(x1, y1)
        - _solid_angle_primitive(x0, y1)
        - _solid_angle_primitive(x1, y0)
        + _solid_angle_primitive(x0, y0)
    )


def build_observer_bin_geometry(
    sky_x_edges_arcsec,
    sky_y_edges_arcsec,
    energy_edges_kev,
    arrival_time_edges_s,
) -> ObserverBinGeometry:
    """Validate bin edges and calculate exact spherical pixel solid angles."""

    x_edges = _validated_edges(sky_x_edges_arcsec, "sky_x_edges_arcsec")
    y_edges = _validated_edges(sky_y_edges_arcsec, "sky_y_edges_arcsec")
    energy_edges = _validated_edges(
        energy_edges_kev, "energy_edges_kev", positive=True
    )
    time_edges = _validated_edges(
        arrival_time_edges_s, "arrival_time_edges_s"
    )
    angular_limit_arcsec = (0.5 * np.pi) / ARCSEC_TO_RAD
    if (
        x_edges[0] <= -angular_limit_arcsec
        or x_edges[-1] >= angular_limit_arcsec
        or y_edges[0] <= -angular_limit_arcsec
        or y_edges[-1] >= angular_limit_arcsec
    ):
        raise ValueError("sky bin edges must lie strictly within +/-90 degrees")

    solid_angle = _sky_pixel_solid_angles_sr(x_edges, y_edges)
    if not np.all(np.isfinite(solid_angle)) or np.any(solid_angle <= 0.0):
        raise ValueError("sky pixels must have finite positive solid angles")

    dtype = np.float32
    return ObserverBinGeometry(
        sky_x_edges_arcsec=jnp.asarray(x_edges, dtype=dtype),
        sky_y_edges_arcsec=jnp.asarray(y_edges, dtype=dtype),
        energy_edges_kev=jnp.asarray(energy_edges, dtype=dtype),
        arrival_time_edges_s=jnp.asarray(time_edges, dtype=dtype),
        sky_pixel_solid_angle_sr=jnp.asarray(solid_angle, dtype=dtype),
    )


def _validate_event_fields(events: ObserverEventResult):
    if events.valid.ndim != 2:
        raise ValueError(
            "observer event fields must have shape "
            "(n_packets, max_interactions)"
        )
    event_shape = events.valid.shape
    required_fields = (
        events.sky_x_arcsec,
        events.sky_y_arcsec,
        events.energy_kev,
        events.arrival_time_s,
        events.scattering_order,
        events.weight_observer_fluence,
    )
    if any(field.shape != event_shape for field in required_fields):
        raise ValueError("observer event fields have inconsistent shapes")
    return event_shape


def _histogram_index(edges, values):
    n_bins = edges.size - 1
    index = jnp.searchsorted(edges, values, side="right") - 1
    index = jnp.where(values == edges[-1], n_bins - 1, index)
    inside = (
        jnp.isfinite(values)
        & (values >= edges[0])
        & (values <= edges[-1])
    )
    return jnp.clip(index, 0, n_bins - 1), inside


def _scatter_sum(flat_index, values, size):
    return jnp.zeros(size, dtype=values.dtype).at[flat_index].add(values)


def bin_observer_events(
    events: ObserverEventResult,
    geometry: ObserverBinGeometry,
) -> BinnedObserverProducts:
    """Accumulate weighted observer events into ``(t, E, y, x)`` cubes.

    The function is deterministic and JAX-jittable.  Events marked valid but
    lying outside at least one requested axis are excluded from the cubes and
    reported by the unbinned closure diagnostics.  A valid event must also
    have a finite nonnegative weight and a scattering order of at least one.
    """

    _validate_event_fields(events)
    event_valid = jnp.asarray(events.valid).reshape(-1)
    sky_x = jnp.asarray(events.sky_x_arcsec).reshape(-1)
    sky_y = jnp.asarray(events.sky_y_arcsec).reshape(-1)
    energy = jnp.asarray(events.energy_kev).reshape(-1)
    arrival_time = jnp.asarray(events.arrival_time_s).reshape(-1)
    scattering_order = jnp.asarray(events.scattering_order).reshape(-1)
    weight = jnp.asarray(events.weight_observer_fluence).reshape(-1)

    x_index, inside_x = _histogram_index(
        geometry.sky_x_edges_arcsec, sky_x
    )
    y_index, inside_y = _histogram_index(
        geometry.sky_y_edges_arcsec, sky_y
    )
    energy_index, inside_energy = _histogram_index(
        geometry.energy_edges_kev, energy
    )
    time_index, inside_time = _histogram_index(
        geometry.arrival_time_edges_s, arrival_time
    )

    finite_nonnegative_weight = jnp.isfinite(weight) & (weight >= 0.0)
    eligible = (
        event_valid
        & finite_nonnegative_weight
        & (scattering_order >= 1)
    )
    binned = eligible & inside_x & inside_y & inside_energy & inside_time
    first_scatter = binned & (scattering_order == 1)
    multiple_scatter = binned & (scattering_order >= 2)

    n_x = geometry.sky_x_edges_arcsec.size - 1
    n_y = geometry.sky_y_edges_arcsec.size - 1
    n_energy = geometry.energy_edges_kev.size - 1
    n_time = geometry.arrival_time_edges_s.size - 1
    cube_shape = (n_time, n_energy, n_y, n_x)
    cube_size = n_time * n_energy * n_y * n_x
    flat_index = (
        ((time_index * n_energy + energy_index) * n_y + y_index) * n_x
        + x_index
    )
    safe_index = jnp.where(binned, flat_index, 0)

    first_weight = jnp.where(first_scatter, weight, 0.0)
    multiple_weight = jnp.where(multiple_scatter, weight, 0.0)
    first_cube = _scatter_sum(safe_index, first_weight, cube_size).reshape(
        cube_shape
    )
    multiple_cube = _scatter_sum(
        safe_index, multiple_weight, cube_size
    ).reshape(cube_shape)
    total_cube = first_cube + multiple_cube
    count_cube = _scatter_sum(
        safe_index,
        binned.astype(jnp.int32),
        cube_size,
    ).reshape(cube_shape)

    valid_weight_mask = event_valid & finite_nonnegative_weight
    unbinned_weight_mask = valid_weight_mask & ~binned
    valid_weight = jnp.sum(jnp.where(valid_weight_mask, weight, 0.0))
    binned_weight = jnp.sum(jnp.where(binned, weight, 0.0))
    unbinned_weight = jnp.sum(jnp.where(unbinned_weight_mask, weight, 0.0))
    valid_event_count = jnp.sum(event_valid, dtype=jnp.int32)
    binned_event_count = jnp.sum(binned, dtype=jnp.int32)

    return BinnedObserverProducts(
        total_fluence=total_cube,
        first_scatter_fluence=first_cube,
        multiple_scatter_fluence=multiple_cube,
        event_count=count_cube,
        valid_event_count=valid_event_count,
        binned_event_count=binned_event_count,
        unbinned_event_count=valid_event_count - binned_event_count,
        valid_weight_observer_fluence=valid_weight,
        binned_weight_observer_fluence=binned_weight,
        unbinned_weight_observer_fluence=unbinned_weight,
    )


def add_binned_observer_products(
    left: BinnedObserverProducts,
    right: BinnedObserverProducts,
) -> BinnedObserverProducts:
    """Add products from independently simulated packet chunks."""

    if left.total_fluence.shape != right.total_fluence.shape:
        raise ValueError("binned observer product cube shapes must match")
    return BinnedObserverProducts(
        total_fluence=left.total_fluence + right.total_fluence,
        first_scatter_fluence=(
            left.first_scatter_fluence + right.first_scatter_fluence
        ),
        multiple_scatter_fluence=(
            left.multiple_scatter_fluence + right.multiple_scatter_fluence
        ),
        event_count=left.event_count + right.event_count,
        valid_event_count=left.valid_event_count + right.valid_event_count,
        binned_event_count=left.binned_event_count + right.binned_event_count,
        unbinned_event_count=(
            left.unbinned_event_count + right.unbinned_event_count
        ),
        valid_weight_observer_fluence=(
            left.valid_weight_observer_fluence
            + right.valid_weight_observer_fluence
        ),
        binned_weight_observer_fluence=(
            left.binned_weight_observer_fluence
            + right.binned_weight_observer_fluence
        ),
        unbinned_weight_observer_fluence=(
            left.unbinned_weight_observer_fluence
            + right.unbinned_weight_observer_fluence
        ),
    )


def fluence_surface_brightness_per_sr(
    fluence_cube,
    geometry: ObserverBinGeometry,
):
    """Convert a ``(t, E, y, x)`` fluence cube to ``ph cm^-2 sr^-1``."""

    fluence = jnp.asarray(fluence_cube)
    expected_shape = (
        geometry.arrival_time_edges_s.size - 1,
        geometry.energy_edges_kev.size - 1,
        geometry.sky_y_edges_arcsec.size - 1,
        geometry.sky_x_edges_arcsec.size - 1,
    )
    if fluence.shape != expected_shape:
        raise ValueError("fluence_cube shape does not match observer bin geometry")
    return fluence / geometry.sky_pixel_solid_angle_sr[None, None, :, :]


def mean_flux_surface_brightness_per_sr_s(
    fluence_cube,
    geometry: ObserverBinGeometry,
):
    """Return time-bin mean brightness in ``ph cm^-2 s^-1 sr^-1``."""

    brightness = fluence_surface_brightness_per_sr(fluence_cube, geometry)
    duration = jnp.diff(geometry.arrival_time_edges_s)
    return brightness / duration[:, None, None, None]
