"""Exact straight-ray column integrals through an angular--distance cloud.

The native cloud cells are bounded by radial spheres and by constant
tangent-plane angle surfaces.  In the local coordinate convention those
angular surfaces are planes through the observer::

    sky_x_pc = line_of_sight_pc * tan(theta_x)
    sky_y_pc = line_of_sight_pc * tan(theta_y)

A straight ray therefore has analytic intersections with every possible cell
boundary.  Sorting those intersections partitions the ray into segments that
each lie in one voxel.  Summing ``n_H * segment_length`` then gives a column
without a fixed-step approximation.  The implementation uses fixed-size JAX
arrays and is compatible with ``jax.jit`` and ``jax.vmap``.
"""

from __future__ import annotations

import jax.numpy as jnp

from .clouds import AngularDistanceCloud, KPC_TO_CM
from .coordinates import ARCSEC_TO_RAD, PC_PER_KPC


PC_TO_CM = KPC_TO_CM / PC_PER_KPC


def _plane_crossing_distances(origin_pc, direction, angle_edges_arcsec, transverse_axis):
    """Ray parameters where a ray crosses constant sky-angle planes."""

    slopes = jnp.tan(jnp.asarray(angle_edges_arcsec) * ARCSEC_TO_RAD)
    numerator = slopes * origin_pc[0] - origin_pc[transverse_axis]
    denominator = direction[transverse_axis] - slopes * direction[0]
    safe_denominator = jnp.where(jnp.abs(denominator) > 1.0e-20, denominator, 1.0)
    crossing = numerator / safe_denominator
    return jnp.where(jnp.abs(denominator) > 1.0e-20, crossing, jnp.nan)


def _sphere_crossing_distances(origin_pc, direction, radial_edges_pc):
    """Both ray parameters for intersections with each radial sphere."""

    projection = jnp.dot(origin_pc, direction)
    constant = jnp.dot(origin_pc, origin_pc) - radial_edges_pc**2
    discriminant = projection**2 - constant
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    near = -projection - root
    far = -projection + root
    roots = jnp.stack((near, far), axis=-1).reshape(-1)
    valid = jnp.repeat(discriminant >= 0.0, 2)
    return jnp.where(valid, roots, jnp.nan)


def ray_boundary_distances_pc(
    cloud: AngularDistanceCloud,
    origin_pc,
    direction,
    max_distance_pc,
):
    """Sorted ray parameters for all cloud boundaries inside a ray segment.

    Invalid or out-of-segment intersections are represented by repeated
    ``max_distance_pc`` values.  Repeated values create zero-length intervals
    and therefore do not affect the integral.
    """

    origin = jnp.asarray(origin_pc)
    ray_direction = jnp.asarray(direction)
    ray_direction = ray_direction / jnp.linalg.norm(ray_direction)
    maximum = jnp.asarray(max_distance_pc, dtype=origin.dtype)

    x_crossings = _plane_crossing_distances(
        origin, ray_direction, cloud.x_edges_arcsec, transverse_axis=1
    )
    y_crossings = _plane_crossing_distances(
        origin, ray_direction, cloud.y_edges_arcsec, transverse_axis=2
    )
    radial_crossings = _sphere_crossing_distances(
        origin, ray_direction, cloud.z_edges_kpc * PC_PER_KPC
    )
    candidates = jnp.concatenate(
        (jnp.array([0.0, maximum], dtype=origin.dtype),
         x_crossings, y_crossings, radial_crossings)
    )
    inside_segment = jnp.isfinite(candidates) & (candidates > 0.0) & (candidates < maximum)
    candidates = jnp.where(inside_segment, candidates, maximum)
    candidates = candidates.at[0].set(0.0)
    return jnp.sort(candidates)


def integrate_ray_column_cm2(
    cloud: AngularDistanceCloud,
    origin_pc,
    direction,
    max_distance_pc,
):
    """Integrate hydrogen column along one finite straight ray segment."""

    origin = jnp.asarray(origin_pc)
    ray_direction = jnp.asarray(direction)
    ray_direction = ray_direction / jnp.linalg.norm(ray_direction)
    boundaries = ray_boundary_distances_pc(
        cloud, origin, ray_direction, max_distance_pc
    )

    segment_start = boundaries[:-1]
    segment_stop = boundaries[1:]
    segment_length_pc = jnp.maximum(segment_stop - segment_start, 0.0)
    midpoint_distance = 0.5 * (segment_start + segment_stop)
    midpoint = origin[None, :] + midpoint_distance[:, None] * ray_direction[None, :]

    radial_kpc = jnp.linalg.norm(midpoint, axis=-1) / PC_PER_KPC
    sky_x_arcsec = jnp.arctan2(midpoint[:, 1], midpoint[:, 0]) / ARCSEC_TO_RAD
    sky_y_arcsec = jnp.arctan2(midpoint[:, 2], midpoint[:, 0]) / ARCSEC_TO_RAD

    z_index = jnp.searchsorted(cloud.z_edges_kpc, radial_kpc, side="right") - 1
    y_index = jnp.searchsorted(cloud.y_edges_arcsec, sky_y_arcsec, side="right") - 1
    x_index = jnp.searchsorted(cloud.x_edges_arcsec, sky_x_arcsec, side="right") - 1

    n_z, n_y, n_x = cloud.n_h_cm3.shape
    inside_cloud = (
        (z_index >= 0) & (z_index < n_z)
        & (y_index >= 0) & (y_index < n_y)
        & (x_index >= 0) & (x_index < n_x)
    )
    safe_z = jnp.clip(z_index, 0, n_z - 1)
    safe_y = jnp.clip(y_index, 0, n_y - 1)
    safe_x = jnp.clip(x_index, 0, n_x - 1)
    local_n_h_cm3 = cloud.n_h_cm3[safe_z, safe_y, safe_x]

    segment_column_cm2 = jnp.where(
        inside_cloud,
        local_n_h_cm3 * segment_length_pc * PC_TO_CM,
        0.0,
    )
    return jnp.sum(segment_column_cm2)


def integrate_ray_optical_depth(
    cloud: AngularDistanceCloud,
    origin_pc,
    direction,
    max_distance_pc,
    cross_section_cm2_per_h,
):
    """Return ``tau = sigma_H * integral(n_H ds)`` along one ray segment."""

    column_cm2 = integrate_ray_column_cm2(
        cloud, origin_pc, direction, max_distance_pc
    )
    return jnp.asarray(cross_section_cm2_per_h) * column_cm2
