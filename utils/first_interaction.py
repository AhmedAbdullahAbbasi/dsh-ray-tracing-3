"""One-event Monte Carlo transport through a physical cloud frustum.

This module is an integration checkpoint between source packets, exact ray
columns, and stochastic interaction physics.  It deliberately permits at
most one interaction per packet.  For each input ray it:

1. integrates the available hydrogen column exactly;
2. draws an exponential target optical depth ``-log(xi)``;
3. locates the interaction position by inverting cumulative column;
4. chooses scattering or absorption from their opacity ratio; and
5. draws an isotropic post-scatter direction for the temporary test model.

The isotropic phase function is not a dust model.  It exists so the geometry
and Monte Carlo probability laws can be validated before a physical
differential dust-scattering cross-section is introduced.
"""

from __future__ import annotations

from typing import NamedTuple

from jax import random, vmap
import jax.numpy as jnp

from .clouds import AngularDistanceCloud
from .ray_integrals import integrate_ray_column_cm2, locate_ray_column_depth_pc
from .raytracing import direction_from_axis_mu_phi
from .source import SourcePackets


NO_INTERACTION = 0
SCATTERED = 1
ABSORBED = 2

STATUS_NAMES = {
    NO_INTERACTION: "no interaction before ray endpoint",
    SCATTERED: "scattered once",
    ABSORBED: "absorbed",
}


class FirstInteractionResult(NamedTuple):
    """One-event transport result with physical source metadata preserved."""

    position_pc: jnp.ndarray
    direction_before: jnp.ndarray
    direction_after: jnp.ndarray
    distance_pc: jnp.ndarray
    target_tau: jnp.ndarray
    available_tau: jnp.ndarray
    available_column_cm2: jnp.ndarray
    status: jnp.ndarray
    energy_kev: jnp.ndarray
    emission_time_s: jnp.ndarray
    weight_observer_fluence: jnp.ndarray
    time_index: jnp.ndarray
    spectral_bin_index: jnp.ndarray


def _packet_scalar(values, n_packets, name):
    array = jnp.asarray(values)
    if array.ndim == 0:
        return jnp.broadcast_to(array, (n_packets,))
    if array.shape != (n_packets,):
        raise ValueError(f"{name} must be scalar or have shape (n_packets,)")
    return array


def _packet_vectors(values, n_packets, name):
    array = jnp.asarray(values)
    if array.shape == (3,):
        return jnp.broadcast_to(array, (n_packets, 3))
    if array.shape != (n_packets, 3):
        raise ValueError(f"{name} must have shape (3,) or (n_packets, 3)")
    return array


def simulate_first_interactions(
    key,
    packets: SourcePackets,
    cloud: AngularDistanceCloud,
    origin_pc,
    direction,
    max_distance_pc,
    scattering_cross_section_cm2_per_h,
    absorption_cross_section_cm2_per_h,
):
    """Run a single-interaction Monte Carlo experiment for source packets.

    Cross-sections may be scalars or one value per packet, allowing callers to
    evaluate energy-dependent physics before entering this function.  Packet
    weights are not changed: the result describes analog Monte Carlo outcomes
    whose weighted frequencies can be compared with analytic probabilities.
    """

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

    origins = _packet_vectors(origin_pc, n_packets, "origin_pc")
    directions = _packet_vectors(direction, n_packets, "direction")
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
    maximum_distances = _packet_scalar(max_distance_pc, n_packets, "max_distance_pc")
    sigma_scattering = _packet_scalar(
        scattering_cross_section_cm2_per_h, n_packets,
        "scattering_cross_section_cm2_per_h",
    )
    sigma_absorption = _packet_scalar(
        absorption_cross_section_cm2_per_h, n_packets,
        "absorption_cross_section_cm2_per_h",
    )
    sigma_total = sigma_scattering + sigma_absorption

    available_column = vmap(
        lambda ray_origin, ray_direction, maximum: integrate_ray_column_cm2(
            cloud, ray_origin, ray_direction, maximum
        )
    )(origins, directions, maximum_distances)
    available_tau = sigma_total * available_column

    key_tau, key_process, key_mu, key_phi = random.split(key, 4)
    uniform_tau = random.uniform(
        key_tau,
        shape=(n_packets,),
        minval=jnp.finfo(available_tau.dtype).tiny,
        maxval=1.0,
    )
    target_tau = -jnp.log(uniform_tau)
    target_column = target_tau / jnp.maximum(sigma_total, 1.0e-30)

    reached, interaction_distance, interaction_position = vmap(
        lambda ray_origin, ray_direction, maximum, target: locate_ray_column_depth_pc(
            cloud, ray_origin, ray_direction, maximum, target
        )
    )(origins, directions, maximum_distances, target_column)
    interacted = (sigma_total > 0.0) & (target_tau < available_tau) & reached

    scattering_probability = sigma_scattering / jnp.maximum(sigma_total, 1.0e-30)
    scattered = interacted & (
        random.uniform(key_process, shape=(n_packets,)) < scattering_probability
    )
    absorbed = interacted & ~scattered
    status = jnp.where(
        scattered,
        jnp.asarray(SCATTERED, dtype=jnp.int32),
        jnp.where(
            absorbed,
            jnp.asarray(ABSORBED, dtype=jnp.int32),
            jnp.asarray(NO_INTERACTION, dtype=jnp.int32),
        ),
    )

    mu = 2.0 * random.uniform(key_mu, shape=(n_packets,)) - 1.0
    phi = 2.0 * jnp.pi * random.uniform(key_phi, shape=(n_packets,))
    scattered_directions = vmap(direction_from_axis_mu_phi)(directions, mu, phi)
    directions_after = jnp.where(scattered[:, None], scattered_directions, directions)

    endpoint_positions = origins + maximum_distances[:, None] * directions
    positions = jnp.where(interacted[:, None], interaction_position, endpoint_positions)
    distances = jnp.where(interacted, interaction_distance, maximum_distances)

    return FirstInteractionResult(
        position_pc=positions,
        direction_before=directions,
        direction_after=directions_after,
        distance_pc=distances,
        target_tau=jnp.where(interacted, target_tau, jnp.nan),
        available_tau=available_tau,
        available_column_cm2=available_column,
        status=status,
        energy_kev=packets.energy_kev,
        emission_time_s=packets.emission_time_s,
        weight_observer_fluence=packets.weight_observer_fluence,
        time_index=packets.time_index,
        spectral_bin_index=packets.spectral_bin_index,
    )
