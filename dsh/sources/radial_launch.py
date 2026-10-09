"""Smooth, energy-dependent source launches with exact importance weights.

Each component is circular in tangent-plane slopes (u,v), with density
    p(u,v | s) = s**2 / [pi * (s**2 + u**2 + v**2)**2].
Its radial CDF is r**2/(s**2+r**2). The solid-angle density includes the exact
tangent-plane Jacobian. A uniform forward-hemisphere component guarantees
support and bounds the isotropic importance weight. No inner boxes or angular
cutoffs are introduced, and physical scattering is still sampled by transport.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from .launch import LaunchedSourcePackets, SourceLaunchGeometry
from .models import SourcePackets


class RadialLaunchGeometry(NamedTuple):
    """JAX proposal arrays; the rectangle is coverage metadata, not a cutoff."""

    source_position_pc: jnp.ndarray
    source_distance_pc: jnp.ndarray
    slope_x_bounds: jnp.ndarray
    slope_y_bounds: jnp.ndarray
    slope_area: jnp.ndarray
    launch_solid_angle_sr: jnp.ndarray
    energy_kev: jnp.ndarray
    component_scale: jnp.ndarray
    component_probability: jnp.ndarray
    hemisphere_probability: jnp.ndarray


def build_radial_launch_geometry(
    coverage: SourceLaunchGeometry,
    energy_kev,
    component_scale,
    component_probability,
    *,
    hemisphere_probability: float = 0.05,
) -> RadialLaunchGeometry:
    """Build normalized circular components; scales have shape (energy,component).

    Scales interpolate linearly in log-energy/log-scale at each packet energy.
    This interpolation only selects a proposal: it does not alter cross sections.
    The forward hemisphere covers every cloud in front of the centered source.
    Its probability must be positive so the importance factor is bounded by
    1/(2*hemisphere_probability), including directions near the horizon.
    """
    energy = np.asarray(energy_kev, dtype=np.float64)
    scales = np.asarray(component_scale, dtype=np.float64)
    weights = np.asarray(component_probability, dtype=np.float64)
    if (
        energy.ndim != 1
        or energy.size < 2
        or not np.all(np.isfinite(energy))
        or np.any(energy <= 0)
        or np.any(np.diff(energy) <= 0)
    ):
        raise ValueError("proposal energies must be finite, positive, and increasing")
    if (
        weights.ndim != 1
        or weights.size == 0
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise ValueError("component probabilities must be finite and positive")
    if scales.shape != (energy.size, weights.size):
        raise ValueError("component scales must have shape (energy, component)")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("component scales must be finite and positive")
    if not np.isfinite(hemisphere_probability) or not 0 < hemisphere_probability < 1:
        raise ValueError(
            "hemisphere_probability must lie strictly between zero and one"
        )
    weights = weights / weights.sum()
    return RadialLaunchGeometry(
        *coverage[:5],
        jnp.asarray(2 * np.pi, dtype=jnp.float32),
        jnp.asarray(energy, dtype=jnp.float32),
        jnp.asarray(scales, dtype=jnp.float32),
        jnp.asarray(weights, dtype=jnp.float32),
        jnp.asarray(hemisphere_probability, dtype=jnp.float32),
    )


def build_radial_cloud_launch_geometry(cloud, physics, *, max_components: int = 16):
    """Use column-weighted cloud distances and table phase medians for efficiency.

    The shape is only a proposal. For heterogeneous cubes, mean column per
    radial slab is used; clouds with many active slabs are combined in at most
    max_components consecutive, equal-index groups. Each group's distance is
    column weighted. This bounds hot-path memory while preserving exact proposal
    normalization. The full cloud and its original columns remain in transport.
    """
    from .launch import build_cloud_launch_geometry

    if not isinstance(max_components, int) or max_components < 1:
        raise ValueError("max_components must be a positive integer")
    coverage = build_cloud_launch_geometry(cloud)
    columns = np.asarray(cloud.delta_nh_cm2, dtype=np.float64).mean(axis=(1, 2))
    centers = np.asarray(cloud.z_edges_kpc, dtype=np.float64)
    fractions = (centers[1:] + centers[:-1]) / (2 * float(cloud.source_distance_kpc))
    active = np.flatnonzero(columns > 0)
    if active.size:
        groups = np.array_split(active, min(max_components, active.size))
        weights = np.array([columns[g].sum() for g in groups])
        fractions = np.array(
            [np.average(fractions[g], weights=columns[g]) for g in groups]
        )
    else:
        # Vacuum still has a valid proposal; no halo score can be produced.
        weights = np.ones(1)
        fractions = np.array([np.mean(fractions)])
    angles = np.asarray(physics.scattering_angle_rad, dtype=np.float64)
    cdf = np.asarray(physics.scattering_angle_cdf, dtype=np.float64)
    median = np.array([np.interp(0.5, row, angles) for row in cdf])
    # Typical beta=f*alpha is a small-angle efficiency heuristic. The capped
    # scale remains valid for broader synthetic phase laws, without changing them.
    scales = np.tan(np.minimum(median[:, None] * fractions[None, :], np.pi / 4))
    return build_radial_launch_geometry(coverage, physics.energy_kev, scales, weights)


def _packet_scales(energy_kev, geometry):
    def interpolate(column):
        return jnp.exp(
            jnp.interp(
                jnp.log(energy_kev), jnp.log(geometry.energy_kev), jnp.log(column)
            )
        )

    return jax.vmap(interpolate, in_axes=1, out_axes=1)(geometry.component_scale)


def radial_launch_pdf_per_sr(
    cos_beta, sin_squared_beta, scales, probabilities, defensive
):
    """Exact density on the forward hemisphere, stable on axis and at its edge.

    Written in cos(beta) to avoid tan(beta) overflow. Compute sin_squared_beta
    from the two transverse direction components, not 1-cos(beta)**2, which
    loses small-angle precision in float32.
    """
    mu = jnp.asarray(cos_beta)
    transverse = jnp.asarray(sin_squared_beta)
    scale2 = scales**2
    denominator = scale2 * mu[..., None] ** 2 + transverse[..., None]
    component_pdf = scale2 * mu[..., None] / (jnp.pi * denominator**2)
    mixture = jnp.sum(component_pdf * probabilities, axis=-1)
    return jnp.where(mu >= 0, (1 - defensive) * mixture + defensive / (2 * jnp.pi), 0)


def sample_radial_source_launches(
    key, packets: SourcePackets, geometry: RadialLaunchGeometry
):
    """Draw smooth circular directions with the full mixture PDF for every packet."""
    from .launch import _validate_source_packet_shapes

    count = _validate_source_packet_shapes(packets)
    choose_key, radius_key, azimuth_key, defense_key = random.split(key, 4)
    scales = _packet_scales(packets.energy_kev, geometry)
    choice = random.categorical(
        choose_key, jnp.log(geometry.component_probability), shape=(count,)
    )
    selected = jnp.take_along_axis(scales, choice[:, None], axis=1)[:, 0]
    uniform = random.uniform(radius_key, (count,), dtype=jnp.float32)
    radius = selected * jnp.sqrt(uniform / (1 - uniform))
    inverse_norm = 1 / jnp.sqrt(1 + radius**2)
    use_defense = (
        random.uniform(defense_key, (count,)) < geometry.hemisphere_probability
    )
    # The same independent radial variate is uniform in cos(beta) for defense.
    mu = jnp.where(use_defense, uniform, inverse_norm)
    transverse = jnp.where(use_defense, jnp.sqrt(1 - uniform**2), radius * inverse_norm)
    azimuth = 2 * jnp.pi * random.uniform(azimuth_key, (count,), dtype=jnp.float32)
    direction = jnp.stack(
        (-mu, transverse * jnp.cos(azimuth), transverse * jnp.sin(azimuth)), axis=1
    )
    pdf = radial_launch_pdf_per_sr(
        mu,
        direction[:, 1] ** 2 + direction[:, 2] ** 2,
        scales,
        geometry.component_probability,
        geometry.hemisphere_probability,
    )
    energy = jnp.asarray(packets.energy_kev)
    momentum = jnp.concatenate([energy[:, None], energy[:, None] * direction], axis=1)
    return LaunchedSourcePackets(
        jnp.broadcast_to(geometry.source_position_pc, (count, 3)),
        momentum,
        pdf,
        1 / (4 * jnp.pi * pdf),
        packets.emission_time_s,
        packets.weight_observer_fluence,
        packets.time_index,
        packets.spectral_bin_index,
    )
