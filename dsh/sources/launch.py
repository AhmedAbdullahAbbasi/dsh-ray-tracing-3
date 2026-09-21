"""Launch physical source packets toward a native DSH cloud frustum.

The source samplers in :mod:`dsh.sources.models` assign energy, emission time, and
observer-equivalent fluence.  This module supplies the missing geometric
state required by :func:`dsh.transport.kernel.transport_photon_batch`:

* a source position;
* a null photon four-momentum; and
* the probability density used to sample the launch direction.

Version 1 assumes an isotropic point source on the central line of sight.  To
avoid wasting nearly every packet on the rest of the sphere, directions are
importance sampled inside a rectangular cone that covers the cloud frustum.
The sampler is uniform in tangent-plane slopes ``(u, v)`` about the negative
line-of-sight axis, not uniform in solid angle.  Its exact density is

``q(Omega) = (1 + u**2 + v**2)**(3/2) / slope_area``.

Each result therefore carries the isotropic-source importance factor
``1 / (4*pi*q)`` separately from the physical observer-fluence weight.  A
future peel-off estimator can use both without changing the source-flux
convention or silently making the result depend on the chosen launch cone.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import random

from ..geometry.clouds import AngularDistanceCloud
from ..geometry.coordinates import ARCSEC_TO_RAD, PC_PER_KPC
from .models import SourcePackets


class SourceLaunchGeometry(NamedTuple):
    """Fixed rectangular launch cone for a centered point source.

    ``slope_x_bounds`` and ``slope_y_bounds`` describe directions
    ``normalize((-1, u, v))``.  ``launch_solid_angle_sr`` is the exact solid
    angle of that spherical rectangle; it is metadata for validation and
    normalization checks rather than a small-angle approximation.
    """

    source_position_pc: jnp.ndarray
    source_distance_pc: jnp.ndarray
    slope_x_bounds: jnp.ndarray
    slope_y_bounds: jnp.ndarray
    slope_area: jnp.ndarray
    launch_solid_angle_sr: jnp.ndarray


class LaunchedSourcePackets(NamedTuple):
    """Source packet metadata plus JAX-ready initial transport states.

    ``launch_pdf_per_sr`` is normalized over the launch cone.
    ``isotropic_importance`` is the ratio of the physical isotropic emission
    density to that sampling density, ``1 / (4*pi*q)``.  Neither quantity is
    folded into ``weight_observer_fluence``.
    """

    position_pc: jnp.ndarray
    momentum_kev: jnp.ndarray
    launch_pdf_per_sr: jnp.ndarray
    isotropic_importance: jnp.ndarray
    emission_time_s: jnp.ndarray
    weight_observer_fluence: jnp.ndarray
    time_index: jnp.ndarray
    spectral_bin_index: jnp.ndarray


def _rectangular_slope_solid_angle(slope_x_bounds, slope_y_bounds):
    """Return the exact solid angle subtended by a rectangle in slopes."""

    u0, u1 = slope_x_bounds
    v0, v1 = slope_y_bounds

    def primitive(u, v):
        return np.arctan2(u * v, np.sqrt(1.0 + u * u + v * v))

    return primitive(u1, v1) - primitive(u0, v1) - primitive(u1, v0) + primitive(u0, v0)


def build_rectangular_launch_geometry(
    source_distance_kpc,
    slope_x_bounds,
    slope_y_bounds,
) -> SourceLaunchGeometry:
    """Validate and build an explicitly specified rectangular launch cone.

    Slopes are dimensionless tangent-plane coordinates around the direction
    from the source to the observer.  This lower-level constructor is useful
    for controlled experiments; normal DSH simulations should use
    :func:`build_cloud_launch_geometry`.
    """

    source_distance = np.asarray(source_distance_kpc, dtype=np.float64)
    x_bounds = np.asarray(slope_x_bounds, dtype=np.float64)
    y_bounds = np.asarray(slope_y_bounds, dtype=np.float64)
    if source_distance.ndim != 0 or not np.isfinite(source_distance):
        raise ValueError("source_distance_kpc must be one finite scalar")
    if source_distance <= 0.0:
        raise ValueError("source_distance_kpc must be positive")
    for name, bounds in (
        ("slope_x_bounds", x_bounds),
        ("slope_y_bounds", y_bounds),
    ):
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
            raise ValueError(f"{name} must contain two finite values")
        if bounds[0] >= bounds[1]:
            raise ValueError(f"{name} must be strictly increasing")

    slope_area = np.diff(x_bounds)[0] * np.diff(y_bounds)[0]
    solid_angle = _rectangular_slope_solid_angle(x_bounds, y_bounds)
    if not np.isfinite(solid_angle) or solid_angle <= 0.0:
        raise ValueError("launch cone must have a finite positive solid angle")

    source_distance_pc = source_distance * PC_PER_KPC
    dtype = np.float32
    return SourceLaunchGeometry(
        source_position_pc=jnp.asarray([source_distance_pc, 0.0, 0.0], dtype=dtype),
        source_distance_pc=jnp.asarray(source_distance_pc, dtype=dtype),
        slope_x_bounds=jnp.asarray(x_bounds, dtype=dtype),
        slope_y_bounds=jnp.asarray(y_bounds, dtype=dtype),
        slope_area=jnp.asarray(slope_area, dtype=dtype),
        launch_solid_angle_sr=jnp.asarray(solid_angle, dtype=dtype),
    )


def build_cloud_launch_geometry(
    cloud: AngularDistanceCloud,
    *,
    padding_arcsec: float = 0.0,
) -> SourceLaunchGeometry:
    """Build a conservative launch cone covering the complete cloud frustum.

    The transverse bounds use the largest cloud radius and the smallest
    possible source--cloud line-of-sight separation.  Consequently every
    frustum point is covered, although some sampled rays may miss an offset
    or strongly tapered cloud.  That affects efficiency, not normalization.

    The cloud's outer radial edge must lie strictly in front of the source.
    A frustum touching the point source has no finite rectangular slope bound
    and should be trimmed or handled by a later full-sphere launch model.
    """

    padding = np.asarray(padding_arcsec, dtype=np.float64)
    if padding.ndim != 0 or not np.isfinite(padding) or padding < 0.0:
        raise ValueError("padding_arcsec must be one finite nonnegative scalar")

    source_distance = float(np.asarray(cloud.source_distance_kpc))
    z_edges = np.asarray(cloud.z_edges_kpc, dtype=np.float64)
    x_edges = np.asarray(cloud.x_edges_arcsec, dtype=np.float64)
    y_edges = np.asarray(cloud.y_edges_arcsec, dtype=np.float64)
    radial_max = float(z_edges[-1])
    separation_min = source_distance - radial_max
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, source_distance)
    if separation_min <= tolerance:
        raise ValueError(
            "cloud outer radial edge must lie strictly in front of the source "
            "to define a finite launch cone"
        )

    x_min = float(x_edges[0] - padding)
    x_max = float(x_edges[-1] + padding)
    y_min = float(y_edges[0] - padding)
    y_max = float(y_edges[-1] + padding)
    angular_limit_arcsec = (0.5 * np.pi) / ARCSEC_TO_RAD
    if (
        x_min <= -angular_limit_arcsec
        or x_max >= angular_limit_arcsec
        or y_min <= -angular_limit_arcsec
        or y_max >= angular_limit_arcsec
    ):
        raise ValueError("cloud angular launch bounds must lie within +/-90 degrees")

    scale = radial_max / separation_min
    x_tangent_min = np.tan(x_min * ARCSEC_TO_RAD)
    x_tangent_max = np.tan(x_max * ARCSEC_TO_RAD)
    y_tangent_min = np.tan(y_min * ARCSEC_TO_RAD)
    y_tangent_max = np.tan(y_max * ARCSEC_TO_RAD)

    # Including zero produces a simple guaranteed bound for offset fields:
    # |D - r*n_los| >= D - r_max and |r*n_sky| <= r_max*|tan(angle)|.
    slope_x_bounds = [
        min(0.0, scale * x_tangent_min),
        max(0.0, scale * x_tangent_max),
    ]
    slope_y_bounds = [
        min(0.0, scale * y_tangent_min),
        max(0.0, scale * y_tangent_max),
    ]
    return build_rectangular_launch_geometry(
        source_distance,
        slope_x_bounds,
        slope_y_bounds,
    )


def _validate_source_packet_shapes(packets: SourcePackets):
    packet_fields = (
        packets.energy_kev,
        packets.emission_time_s,
        packets.weight_observer_fluence,
        packets.time_index,
        packets.spectral_bin_index,
    )
    if any(field.ndim != 1 for field in packet_fields):
        raise ValueError("all SourcePackets fields must be one-dimensional")
    n_packets = packets.energy_kev.shape[0]
    if n_packets <= 0:
        raise ValueError("SourcePackets cannot be empty")
    if any(field.shape != (n_packets,) for field in packet_fields[1:]):
        raise ValueError("all SourcePackets fields must have the same length")
    return n_packets


def sample_source_launches(
    key,
    packets: SourcePackets,
    geometry: SourceLaunchGeometry,
) -> LaunchedSourcePackets:
    """Sample initial positions and four-momenta for physical source packets.

    The function is JAX-jittable.  Sampling is uniform in ``(u, v)`` inside
    the geometry's rectangle, while ``launch_pdf_per_sr`` accounts exactly
    for the tangent-plane Jacobian.  The returned momentum convention is
    ``(E, p_los, p_sky_x, p_sky_y)`` and satisfies ``|p| = E``.
    """

    n_packets = _validate_source_packet_shapes(packets)
    if geometry.source_position_pc.shape != (3,):
        raise ValueError("geometry.source_position_pc must have shape (3,)")
    if geometry.slope_x_bounds.shape != (2,) or geometry.slope_y_bounds.shape != (2,):
        raise ValueError("geometry slope bounds must each have shape (2,)")

    key_x, key_y = random.split(key)
    u = random.uniform(
        key_x,
        shape=(n_packets,),
        minval=geometry.slope_x_bounds[0],
        maxval=geometry.slope_x_bounds[1],
    )
    v = random.uniform(
        key_y,
        shape=(n_packets,),
        minval=geometry.slope_y_bounds[0],
        maxval=geometry.slope_y_bounds[1],
    )
    norm = jnp.sqrt(1.0 + u * u + v * v)
    direction = jnp.stack((-1.0 / norm, u / norm, v / norm), axis=1)

    energy = jnp.asarray(packets.energy_kev)
    momentum = jnp.concatenate([energy[:, None], energy[:, None] * direction], axis=1)
    launch_pdf = norm**3 / geometry.slope_area
    isotropic_importance = 1.0 / (4.0 * jnp.pi * launch_pdf)
    positions = jnp.broadcast_to(geometry.source_position_pc, (n_packets, 3))

    return LaunchedSourcePackets(
        position_pc=positions,
        momentum_kev=momentum,
        launch_pdf_per_sr=launch_pdf,
        isotropic_importance=isotropic_importance,
        emission_time_s=packets.emission_time_s,
        weight_observer_fluence=packets.weight_observer_fluence,
        time_index=packets.time_index,
        spectral_bin_index=packets.spectral_bin_index,
    )
