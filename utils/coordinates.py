"""Physical coordinate conventions for source, cloud, and observer geometry.

The canonical Cartesian vector order is::

    (line_of_sight_pc, sky_x_pc, sky_y_pc)

The observer is at the origin and the central source lies on the positive
line-of-sight axis. Photons therefore travel approximately in the negative
line-of-sight direction on their way to the observer. All Cartesian lengths
are in parsecs, angular offsets are in arcseconds, and directions are
dimensionless unit vectors.

The FITS cloud cubes use angular sky axes plus radial distance. Those voxels
form an angular frustum rather than a rectangular Cartesian box. The helper
functions here convert individual sightlines exactly; resampling a full cube
onto a Cartesian transport grid is deliberately left to a separate adapter.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
PC_PER_KPC = 1000.0


class SightlineGeometry(NamedTuple):
    """JAX-ready central source and observer geometry in parsecs."""

    observer_position_pc: jnp.ndarray
    source_position_pc: jnp.ndarray
    source_distance_pc: jnp.ndarray
    source_to_observer_direction: jnp.ndarray


def angular_offset_direction(sky_x_arcsec, sky_y_arcsec):
    """Return unit sightline vectors for tangent-plane angular offsets.

    Inputs may be scalars or broadcast-compatible JAX arrays. The result has
    shape ``broadcast_shape + (3,)`` in canonical
    ``(line_of_sight, sky_x, sky_y)`` order.
    """

    sky_x, sky_y = jnp.broadcast_arrays(
        jnp.asarray(sky_x_arcsec), jnp.asarray(sky_y_arcsec)
    )
    angle_x_rad = sky_x * ARCSEC_TO_RAD
    angle_y_rad = sky_y * ARCSEC_TO_RAD
    unnormalized = jnp.stack(
        [jnp.ones_like(angle_x_rad), jnp.tan(angle_x_rad), jnp.tan(angle_y_rad)],
        axis=-1,
    )
    return unnormalized / jnp.linalg.norm(unnormalized, axis=-1, keepdims=True)


def sky_position_pc(distance_kpc, sky_x_arcsec=0.0, sky_y_arcsec=0.0):
    """Convert radial distance and sky offsets to Cartesian parsecs.

    ``distance_kpc`` is the radial observer-to-point distance, not merely the
    first Cartesian component. Inputs may be scalars or broadcast-compatible
    arrays and are safe to use under :func:`jax.jit`.
    """

    distance, sky_x, sky_y = jnp.broadcast_arrays(
        jnp.asarray(distance_kpc),
        jnp.asarray(sky_x_arcsec),
        jnp.asarray(sky_y_arcsec),
    )
    direction = angular_offset_direction(sky_x, sky_y)
    return direction * (distance * PC_PER_KPC)[..., None]


def cartesian_to_sky(position_pc):
    """Convert canonical Cartesian positions to distance and sky offsets.

    Returns ``(distance_kpc, sky_x_arcsec, sky_y_arcsec)``. This is the inverse
    of :func:`sky_position_pc` for points in front of the observer.
    """

    position = jnp.asarray(position_pc)
    if position.ndim < 1 or position.shape[-1] != 3:
        raise ValueError("position_pc must have final dimension 3")

    distance_kpc = jnp.linalg.norm(position, axis=-1) / PC_PER_KPC
    sky_x_arcsec = (
        jnp.arctan2(position[..., 1], position[..., 0]) / ARCSEC_TO_RAD
    )
    sky_y_arcsec = (
        jnp.arctan2(position[..., 2], position[..., 0]) / ARCSEC_TO_RAD
    )
    return distance_kpc, sky_x_arcsec, sky_y_arcsec


def build_sightline_geometry(source_distance_kpc) -> SightlineGeometry:
    """Validate and build the centered source-observer geometry.

    The source is placed at ``(source_distance_pc, 0, 0)`` and the observer at
    ``(0, 0, 0)``. A separate sky-origin/WCS adapter can later rotate this
    local tangent frame onto a celestial coordinate system without changing
    the transport convention.
    """

    distance = np.asarray(source_distance_kpc)
    if distance.ndim != 0 or not np.isfinite(distance):
        raise ValueError("source_distance_kpc must be one finite scalar")
    if distance <= 0.0:
        raise ValueError("source_distance_kpc must be positive")

    dtype = np.result_type(distance.dtype, np.float32)
    source_distance_pc = np.asarray(distance * PC_PER_KPC, dtype=dtype)
    observer_position_pc = jnp.zeros(3, dtype=jnp.asarray(source_distance_pc).dtype)
    source_position_pc = jnp.array(
        [source_distance_pc, 0.0, 0.0], dtype=observer_position_pc.dtype
    )
    source_to_observer = observer_position_pc - source_position_pc
    source_to_observer /= jnp.linalg.norm(source_to_observer)

    return SightlineGeometry(
        observer_position_pc=observer_position_pc,
        source_position_pc=source_position_pc,
        source_distance_pc=jnp.asarray(source_distance_pc),
        source_to_observer_direction=source_to_observer,
    )
