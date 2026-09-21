"""Next-event (peel-off) observer scoring for DSH scattering histories.

The voxel transport follows one stochastic outgoing direction after every
scattering so that multiple scattering can be simulated.  A point observer,
however, subtends an effectively zero solid angle and would almost never be
hit by those analog directions.  The standard next-event estimator therefore
creates a *virtual* observer contribution at every physical scattering event.

For each recorded scattering this module:

1. aims a virtual ray from the event to the observer;
2. evaluates the normalized NewDust phase density in that direction;
3. integrates the remaining hydrogen column exactly through the native
   angular--distance voxels;
4. applies scattering-plus-absorption extinction along the virtual ray; and
5. returns its sky position, arrival time, scattering order, and fluence.

The stochastic outgoing four-momentum stored by transport is intentionally
not used for the peel-off direction.  It controls the packet's later physical
history; the incoming momentum controls the probability of scattering toward
the observer.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import vmap

from ..geometry.clouds import AngularDistanceCloud
from ..geometry.coordinates import ARCSEC_TO_RAD, PC_PER_KPC
from ..geometry.rays import PC_TO_CM, integrate_ray_column_cm2
from ..physics.dust import DustPhysicsTable
from ..sources.launch import LaunchedSourcePackets
from ..transport.kernel import (
    DUST_SCATTERING,
    PhotonTransportResult,
)

SPEED_OF_LIGHT_CM_S = 2.99792458e10
PC_LIGHT_TRAVEL_TIME_S = PC_TO_CM / SPEED_OF_LIGHT_CM_S


class ObserverEventResult(NamedTuple):
    """Fixed-shape virtual photons contributed by scattering events.

    Every field except ``valid`` has zero in unused or non-scattering slots.
    Scalar event fields have shape ``(n_packets, max_interactions)``.
    ``weight_observer_fluence`` is the contribution to an image bin in
    ``ph cm^-2``; division by the bin's sky solid angle produces surface
    brightness per steradian.
    """

    valid: jnp.ndarray
    sky_x_arcsec: jnp.ndarray
    sky_y_arcsec: jnp.ndarray
    energy_kev: jnp.ndarray
    arrival_time_s: jnp.ndarray
    excess_path_length_pc: jnp.ndarray
    scattering_angle_rad: jnp.ndarray
    scattering_order: jnp.ndarray
    escape_column_cm2: jnp.ndarray
    escape_optical_depth: jnp.ndarray
    transmission: jnp.ndarray
    phase_pdf_per_sr: jnp.ndarray
    weight_observer_fluence: jnp.ndarray
    time_index: jnp.ndarray
    spectral_bin_index: jnp.ndarray


def _validate_differential_table(physics: DustPhysicsTable):
    expected = (
        physics.energy_kev.size,
        physics.scattering_angle_rad.size,
    )
    if physics.differential_cross_section_cm2_per_sr_per_h.shape != expected:
        raise ValueError(
            "observer scoring requires a differential cross-section table "
            "with shape (n_energy, n_angle)"
        )


def _interpolation_bracket(grid, value):
    upper = jnp.searchsorted(grid, value, side="right")
    upper = jnp.clip(upper, 1, grid.size - 1)
    return upper - 1, upper


def _interpolate_nonnegative_pair(low, high, fraction):
    """Log-interpolate positive values and linearly handle exact zeros."""

    linear = low + fraction * (high - low)
    dtype = jnp.result_type(low, high, fraction)
    tiny = jnp.finfo(dtype).tiny
    logarithmic = jnp.exp(
        jnp.log(jnp.maximum(low, tiny))
        + fraction
        * (jnp.log(jnp.maximum(high, tiny)) - jnp.log(jnp.maximum(low, tiny)))
    )
    return jnp.where((low > 0.0) & (high > 0.0), logarithmic, linear)


def _energy_fraction(energy_grid, energy):
    lower, upper = _interpolation_bracket(energy_grid, energy)
    log_grid = jnp.log(energy_grid)
    fraction = (jnp.log(energy) - log_grid[lower]) / (log_grid[upper] - log_grid[lower])
    return lower, upper, jnp.clip(fraction, 0.0, 1.0)


def _angle_fraction(angle_grid, angle):
    lower, upper = _interpolation_bracket(angle_grid, angle)
    low_angle = angle_grid[lower]
    high_angle = angle_grid[upper]
    linear_fraction = (angle - low_angle) / (high_angle - low_angle)
    tiny = jnp.finfo(angle_grid.dtype).tiny
    log_fraction = (
        jnp.log(jnp.maximum(angle, tiny)) - jnp.log(jnp.maximum(low_angle, tiny))
    ) / (jnp.log(jnp.maximum(high_angle, tiny)) - jnp.log(jnp.maximum(low_angle, tiny)))
    fraction = jnp.where(low_angle > 0.0, log_fraction, linear_fraction)
    return lower, upper, jnp.clip(fraction, 0.0, 1.0)


def scattering_phase_pdf_per_sr(
    physics: DustPhysicsTable,
    energy_kev,
    scattering_angle_rad,
):
    """Evaluate the normalized physical scattering phase density.

    The energy interpolation is the same convex mixture of normalized phase
    functions used by the transport CDF.  Within each energy row the
    differential cross-section is interpolated in log angle and log value
    wherever both tabulated values are positive.  Unsupported or non-finite
    energy/angle inputs return zero.
    """

    _validate_differential_table(physics)
    energy, angle = jnp.broadcast_arrays(
        jnp.asarray(energy_kev),
        jnp.asarray(scattering_angle_rad),
    )
    supported = (
        jnp.isfinite(energy)
        & jnp.isfinite(angle)
        & (energy >= physics.energy_kev[0])
        & (energy <= physics.energy_kev[-1])
        & (angle >= physics.scattering_angle_rad[0])
        & (angle <= physics.scattering_angle_rad[-1])
    )
    safe_energy = jnp.clip(energy, physics.energy_kev[0], physics.energy_kev[-1])
    safe_angle = jnp.clip(
        angle,
        physics.scattering_angle_rad[0],
        physics.scattering_angle_rad[-1],
    )
    energy_lower, energy_upper, energy_fraction = _energy_fraction(
        physics.energy_kev, safe_energy
    )
    angle_lower, angle_upper, angle_fraction = _angle_fraction(
        physics.scattering_angle_rad, safe_angle
    )

    differential = physics.differential_cross_section_cm2_per_sr_per_h
    lower_energy_differential = _interpolate_nonnegative_pair(
        differential[energy_lower, angle_lower],
        differential[energy_lower, angle_upper],
        angle_fraction,
    )
    upper_energy_differential = _interpolate_nonnegative_pair(
        differential[energy_upper, angle_lower],
        differential[energy_upper, angle_upper],
        angle_fraction,
    )
    lower_phase = lower_energy_differential / jnp.maximum(
        physics.scattering_cross_section_cm2_per_h[energy_lower],
        jnp.finfo(differential.dtype).tiny,
    )
    upper_phase = upper_energy_differential / jnp.maximum(
        physics.scattering_cross_section_cm2_per_h[energy_upper],
        jnp.finfo(differential.dtype).tiny,
    )
    phase_pdf = lower_phase + energy_fraction * (upper_phase - lower_phase)
    return jnp.where(supported, phase_pdf, 0.0)


def _cross_sections_at_energy(physics: DustPhysicsTable, energy_kev):
    energy = jnp.asarray(energy_kev)
    supported = (
        jnp.isfinite(energy)
        & (energy >= physics.energy_kev[0])
        & (energy <= physics.energy_kev[-1])
    )
    safe_energy = jnp.clip(energy, physics.energy_kev[0], physics.energy_kev[-1])
    lower, upper, fraction = _energy_fraction(physics.energy_kev, safe_energy)
    sigma_scattering = _interpolate_nonnegative_pair(
        physics.scattering_cross_section_cm2_per_h[lower],
        physics.scattering_cross_section_cm2_per_h[upper],
        fraction,
    )
    sigma_absorption = _interpolate_nonnegative_pair(
        physics.absorption_cross_section_cm2_per_h[lower],
        physics.absorption_cross_section_cm2_per_h[upper],
        fraction,
    )
    return supported, sigma_scattering, sigma_absorption


def _validate_event_shapes(
    launched: LaunchedSourcePackets,
    transported: PhotonTransportResult,
):
    launch_fields = (
        launched.launch_pdf_per_sr,
        launched.isotropic_importance,
        launched.emission_time_s,
        launched.weight_observer_fluence,
        launched.time_index,
        launched.spectral_bin_index,
    )
    if launched.position_pc.ndim != 2 or launched.position_pc.shape[1] != 3:
        raise ValueError("launched.position_pc must have shape (n_packets, 3)")
    n_packets = launched.position_pc.shape[0]
    if launched.momentum_kev.shape != (n_packets, 4):
        raise ValueError("launched.momentum_kev must have shape (n_packets, 4)")
    if any(field.shape != (n_packets,) for field in launch_fields):
        raise ValueError("launched packet metadata must have shape (n_packets,)")

    records = transported.interactions
    if records.valid.ndim != 2 or records.valid.shape[0] != n_packets:
        raise ValueError(
            "interaction records must have shape (n_packets, max_interactions)"
        )
    n_interactions = records.valid.shape[1]
    scalar_shape = (n_packets, n_interactions)
    vector_shape = scalar_shape + (3,)
    momentum_shape = scalar_shape + (4,)
    scalar_fields = (
        records.interaction_type,
        records.cumulative_path_length_pc,
        records.cumulative_excess_path_length_pc,
        records.scattering_order,
    )
    if any(field.shape != scalar_shape for field in scalar_fields):
        raise ValueError("interaction scalar fields have inconsistent shapes")
    if records.position_pc.shape != vector_shape:
        raise ValueError("interaction positions have an inconsistent shape")
    if (
        records.incoming_momentum_kev.shape != momentum_shape
        or records.outgoing_momentum_kev.shape != momentum_shape
    ):
        raise ValueError("interaction momenta have inconsistent shapes")
    return n_packets, n_interactions


def score_peeloff_events(
    launched: LaunchedSourcePackets,
    transported: PhotonTransportResult,
    cloud: AngularDistanceCloud,
    physics: DustPhysicsTable,
) -> ObserverEventResult:
    """Score one virtual observer photon per recorded dust scattering.

    This function is deterministic and JAX-jittable.  The analog transport
    has already sampled the probability of reaching each scattering event.
    Consequently the event scorer uses the normalized phase density
    ``(d sigma/d Omega) / sigma_sca`` and must not multiply by the scattering
    cross-section a second time.
    """

    _validate_differential_table(physics)
    n_packets, n_interactions = _validate_event_shapes(launched, transported)
    records = transported.interactions
    position = jnp.asarray(records.position_pc)
    incoming_momentum = jnp.asarray(records.incoming_momentum_kev)
    energy = incoming_momentum[..., 0]
    incoming_spatial = incoming_momentum[..., 1:]
    incoming_norm = jnp.linalg.norm(incoming_spatial, axis=-1)
    safe_incoming_norm = jnp.maximum(
        incoming_norm, jnp.finfo(incoming_spatial.dtype).tiny
    )
    incoming_direction = incoming_spatial / safe_incoming_norm[..., None]

    observer_distance = jnp.linalg.norm(position, axis=-1)
    safe_observer_distance = jnp.maximum(
        observer_distance, jnp.finfo(position.dtype).tiny
    )
    observer_direction = -position / safe_observer_distance[..., None]
    fallback_direction = jnp.asarray([-1.0, 0.0, 0.0], dtype=position.dtype)

    cosine = jnp.clip(
        jnp.sum(incoming_direction * observer_direction, axis=-1),
        -1.0,
        1.0,
    )
    sine = jnp.linalg.norm(jnp.cross(incoming_direction, observer_direction), axis=-1)
    scattering_angle = jnp.arctan2(sine, cosine)
    energy_supported, sigma_scattering, sigma_absorption = _cross_sections_at_energy(
        physics, energy
    )
    phase_pdf = scattering_phase_pdf_per_sr(physics, energy, scattering_angle)

    valid = (
        records.valid
        & (records.interaction_type == DUST_SCATTERING)
        & jnp.isfinite(position).all(axis=-1)
        & jnp.isfinite(incoming_momentum).all(axis=-1)
        & (observer_distance > 0.0)
        & (incoming_norm > 0.0)
        & energy_supported
        & jnp.isfinite(launched.isotropic_importance)[:, None]
        & (launched.isotropic_importance[:, None] > 0.0)
    )

    escape_origins = jnp.where(valid[..., None], position, 0.0)
    escape_directions = jnp.where(
        valid[..., None], observer_direction, fallback_direction
    )
    escape_distances = jnp.where(valid, observer_distance, 0.0)
    flat_column = vmap(
        lambda origin, direction, distance: integrate_ray_column_cm2(
            cloud, origin, direction, distance
        )
    )(
        escape_origins.reshape((-1, 3)),
        escape_directions.reshape((-1, 3)),
        escape_distances.reshape((-1,)),
    )
    escape_column = flat_column.reshape((n_packets, n_interactions))
    escape_optical_depth = escape_column * (sigma_scattering + sigma_absorption)
    transmission = jnp.exp(-escape_optical_depth)

    sky_x_arcsec = jnp.arctan2(position[..., 1], position[..., 0]) / ARCSEC_TO_RAD
    sky_y_arcsec = jnp.arctan2(position[..., 2], position[..., 0]) / ARCSEC_TO_RAD
    transverse_squared = jnp.sum(position[..., 1:] ** 2, axis=-1)
    radial_minus_los = transverse_squared / jnp.maximum(
        observer_distance + position[..., 0],
        jnp.finfo(position.dtype).tiny,
    )
    excess_path_length = records.cumulative_excess_path_length_pc + radial_minus_los
    arrival_time = (
        launched.emission_time_s[:, None] + excess_path_length * PC_LIGHT_TRAVEL_TIME_S
    )

    source_distance_pc = cloud.source_distance_kpc * PC_PER_KPC
    geometric_dilution = (source_distance_pc / safe_observer_distance) ** 2
    event_weight = (
        launched.weight_observer_fluence[:, None]
        * geometric_dilution
        * (4.0 * jnp.pi * launched.isotropic_importance[:, None])
        * phase_pdf
        * transmission
    )

    def masked(values):
        return jnp.where(valid, values, jnp.zeros_like(values))

    time_index = jnp.broadcast_to(
        launched.time_index[:, None], (n_packets, n_interactions)
    )
    spectral_bin_index = jnp.broadcast_to(
        launched.spectral_bin_index[:, None], (n_packets, n_interactions)
    )
    return ObserverEventResult(
        valid=valid,
        sky_x_arcsec=masked(sky_x_arcsec),
        sky_y_arcsec=masked(sky_y_arcsec),
        energy_kev=masked(energy),
        arrival_time_s=masked(arrival_time),
        excess_path_length_pc=masked(excess_path_length),
        scattering_angle_rad=masked(scattering_angle),
        scattering_order=jnp.where(
            valid, records.scattering_order, jnp.zeros_like(records.scattering_order)
        ),
        escape_column_cm2=masked(escape_column),
        escape_optical_depth=masked(escape_optical_depth),
        transmission=masked(transmission),
        phase_pdf_per_sr=masked(phase_pdf),
        weight_observer_fluence=masked(event_weight),
        time_index=jnp.where(valid, time_index, jnp.zeros_like(time_index)),
        spectral_bin_index=jnp.where(
            valid, spectral_bin_index, jnp.zeros_like(spectral_bin_index)
        ),
    )
