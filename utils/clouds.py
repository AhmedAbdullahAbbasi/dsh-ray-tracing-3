"""Physical input model for angular--distance hydrogen-column cubes.

The native FITS cube is a frustum described by sky-angle pixels and radial
distance bins.  Its voxel values are increments of hydrogen column,
``delta_NH`` [cm^-2], not a Cartesian density and not an interaction
coefficient.  This module preserves that distinction and performs only the
unit conversion that is unambiguous before a scattering model is selected::

    n_H [cm^-3] = delta_NH [cm^-2] / delta_r [cm]

No arbitrary rescaling or cubic downsampling is performed.  A later transport
adapter can query this native frustum or conservatively remap it, but either
route must retain the column-closure checks provided here.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

import jax.numpy as jnp
import numpy as np


KPC_TO_CM = 3.0856775814913673e21


class AngularDistanceCloud(NamedTuple):
    """JAX-ready physical cloud scene on a native ``(z, y, x)`` grid.

    All axes are stored in increasing order even when the input FITS WCS has a
    negative increment.  ``delta_nh_cm2`` is the column carried by each radial
    voxel.  ``n_h_cm3`` is the corresponding radial-average volume density.
    """

    delta_nh_cm2: jnp.ndarray
    n_h_cm3: jnp.ndarray
    x_edges_arcsec: jnp.ndarray
    y_edges_arcsec: jnp.ndarray
    z_edges_kpc: jnp.ndarray
    radial_bin_width_cm: jnp.ndarray
    source_distance_kpc: jnp.ndarray


def _axis_in_increasing_order(values, name):
    """Validate a center-coordinate axis and return it increasing plus a flip flag."""

    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"{name} must be a one-dimensional axis with at least two centers")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must contain only finite values")

    differences = np.diff(axis)
    if np.all(differences > 0.0):
        return axis, False
    if np.all(differences < 0.0):
        return axis[::-1], True
    raise ValueError(f"{name} must be strictly monotonic")


def centers_to_edges(centers):
    """Construct cell edges from a strictly increasing center-coordinate axis."""

    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("centers must be one-dimensional with at least two values")
    differences = np.diff(centers)
    if not np.all(np.isfinite(centers)) or not np.all(differences > 0.0):
        raise ValueError("centers must be finite and strictly increasing")

    interior = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - 0.5 * differences[0]
    last = centers[-1] + 0.5 * differences[-1]
    return np.concatenate(([first], interior, [last]))


def _validate_column_unit(unit):
    normalized = str(unit).lower().replace(" ", "").replace("^", "")
    accepted = {"cm-2", "cm**-2", "1/cm2", "cm^-2"}
    if normalized not in accepted:
        raise ValueError(
            "cloud voxel values must have hydrogen-column units cm^-2; "
            f"received {unit!r}"
        )


def build_angular_distance_cloud(
    delta_nh_cm2,
    x_centers_arcsec,
    y_centers_arcsec,
    z_centers_kpc,
    source_distance_kpc,
    *,
    unit="cm-2",
) -> AngularDistanceCloud:
    """Validate and build a physical cloud scene without changing its column.

    ``delta_nh_cm2`` must have shape ``(n_z, n_y, n_x)``.  Decreasing input
    axes are normalized to increasing order and the data are flipped along the
    corresponding array dimension.  Radial cell edges must lie between the
    observer and the source.
    """

    _validate_column_unit(unit)
    delta_nh = np.asarray(delta_nh_cm2, dtype=np.float64)
    if delta_nh.ndim != 3 or any(size == 0 for size in delta_nh.shape):
        raise ValueError("delta_nh_cm2 must be a nonempty three-dimensional array")
    if not np.all(np.isfinite(delta_nh)):
        raise ValueError("delta_nh_cm2 must contain only finite values")
    if np.any(delta_nh < 0.0):
        raise ValueError("delta_nh_cm2 cannot contain negative column density")

    source_distance = np.asarray(source_distance_kpc, dtype=np.float64)
    if source_distance.ndim != 0 or not np.isfinite(source_distance):
        raise ValueError("source_distance_kpc must be one finite scalar")
    if source_distance <= 0.0:
        raise ValueError("source_distance_kpc must be positive")

    x_centers, reverse_x = _axis_in_increasing_order(x_centers_arcsec, "x_centers_arcsec")
    y_centers, reverse_y = _axis_in_increasing_order(y_centers_arcsec, "y_centers_arcsec")
    z_centers, reverse_z = _axis_in_increasing_order(z_centers_kpc, "z_centers_kpc")

    expected_shape = (z_centers.size, y_centers.size, x_centers.size)
    if delta_nh.shape != expected_shape:
        raise ValueError(
            "delta_nh_cm2 shape must be (len(z), len(y), len(x)); "
            f"expected {expected_shape}, received {delta_nh.shape}"
        )

    if reverse_z:
        delta_nh = np.flip(delta_nh, axis=0)
    if reverse_y:
        delta_nh = np.flip(delta_nh, axis=1)
    if reverse_x:
        delta_nh = np.flip(delta_nh, axis=2)

    x_edges = centers_to_edges(x_centers)
    y_edges = centers_to_edges(y_centers)
    z_edges = centers_to_edges(z_centers)
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, float(source_distance))
    if z_edges[0] < -tolerance:
        raise ValueError("the first radial cell extends behind the observer")
    if z_edges[-1] > float(source_distance) + tolerance:
        raise ValueError("the cloud cube extends beyond the source distance")

    radial_bin_width_cm = np.diff(z_edges) * KPC_TO_CM
    n_h_cm3 = delta_nh / radial_bin_width_cm[:, None, None]

    # Explicit float32 keeps behavior consistent on default JAX installations;
    # relative closure remains much tighter than the Monte Carlo uncertainty.
    array_dtype = np.float32
    return AngularDistanceCloud(
        delta_nh_cm2=jnp.asarray(delta_nh, dtype=array_dtype),
        n_h_cm3=jnp.asarray(n_h_cm3, dtype=array_dtype),
        x_edges_arcsec=jnp.asarray(x_edges, dtype=array_dtype),
        y_edges_arcsec=jnp.asarray(y_edges, dtype=array_dtype),
        z_edges_kpc=jnp.asarray(z_edges, dtype=array_dtype),
        radial_bin_width_cm=jnp.asarray(radial_bin_width_cm, dtype=array_dtype),
        source_distance_kpc=jnp.asarray(source_distance, dtype=array_dtype),
    )


def cloud_from_loaded_fits(cube: Mapping, source_distance_kpc) -> AngularDistanceCloud:
    """Build a cloud scene from :func:`utils.fits_cube.load_cube` output."""

    delta_nh = cube.get("delta_nh_cm2", cube.get("density"))
    if delta_nh is None:
        raise ValueError("loaded FITS cube has no delta_nh_cm2 data")
    return build_angular_distance_cloud(
        delta_nh,
        cube["x_arcsec"],
        cube["y_arcsec"],
        cube["z_kpc"],
        source_distance_kpc,
        unit=cube.get("unit", ""),
    )


def total_column_map_cm2(cloud: AngularDistanceCloud):
    """Integrate per-voxel column increments along the radial axis."""

    return jnp.sum(cloud.delta_nh_cm2, axis=0)


def recovered_delta_column_cm2(cloud: AngularDistanceCloud):
    """Recover each voxel column from ``n_H * delta_r`` for closure tests."""

    return cloud.n_h_cm3 * cloud.radial_bin_width_cm[:, None, None]


def optical_depth_map(cloud: AngularDistanceCloud, cross_section_cm2_per_h):
    """Return ``tau = sigma_H * NH`` for scalar or vector cross-sections.

    A vector input of shape ``(n_energy,)`` returns ``(n_energy, n_y, n_x)``;
    a scalar input returns ``(n_y, n_x)``.  The function is JAX-jittable.
    """

    cross_section = jnp.asarray(cross_section_cm2_per_h)
    return cross_section[..., None, None] * total_column_map_cm2(cloud)


def transmission_map(cloud: AngularDistanceCloud, cross_section_cm2_per_h):
    """Beer--Lambert transmission for the full observer--source column."""

    return jnp.exp(-optical_depth_map(cloud, cross_section_cm2_per_h))
