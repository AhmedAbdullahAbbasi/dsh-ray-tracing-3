"""JAX photon transport through the native angular-distance DSH voxels.

This is the Version-1 transport kernel.  It is intentionally restricted to
elastic dust scattering plus absorption.  Source sampling, observer scoring,
detector response, and FITS I/O remain outside this module.

The function integrates each straight flight through the native frustum
exactly using :mod:`dsh.geometry.rays`; it does not take fixed spatial
substeps.  One exponential optical-depth budget is therefore carried across
every voxel crossed by a flight.  A new budget is drawn only after a physical
scattering event.

Coordinate and four-momentum conventions
----------------------------------------
Cartesian vectors use ``(line_of_sight, sky_x, sky_y)`` in parsecs.  The
observer is at the origin and the source is on the positive line-of-sight
axis.  ``momentum_kev`` is ``(E, p_los, p_sky_x, p_sky_y)`` with ``c=1``;
for a photon, the spatial norm equals ``E``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax, random, vmap

from ..geometry.clouds import AngularDistanceCloud
from ..geometry.coordinates import PC_PER_KPC
from ..geometry.rays import locate_ray_column_depth_pc
from ..physics.dust import DustPhysicsTable
from .directions import orthonormal_basis

ACTIVE = 0
REACHED_OBSERVER_PLANE = 1
ESCAPED_OUTER_BOUNDARY = 2
ABSORBED = 3
MAX_INTERACTIONS = 4
INVALID_ENERGY = 5
INVALID_STATE = 6

NO_INTERACTION = 0
DUST_SCATTERING = 1
PHOTOELECTRIC_ABSORPTION = 2

STATUS_NAMES = {
    ACTIVE: "active",
    REACHED_OBSERVER_PLANE: "reached the observer plane",
    ESCAPED_OUTER_BOUNDARY: "escaped through the outer transport boundary",
    ABSORBED: "absorbed",
    MAX_INTERACTIONS: "reached the numerical interaction safety limit",
    INVALID_ENERGY: "energy lies outside the dust table",
    INVALID_STATE: "position or photon four-momentum is invalid",
}

INTERACTION_NAMES = {
    NO_INTERACTION: "unused record slot",
    DUST_SCATTERING: "dust scattering",
    PHOTOELECTRIC_ABSORPTION: "photoelectric absorption",
}


class PhotonInteractionRecord(NamedTuple):
    """Fixed-size history emitted by one photon transport.

    Every field has leading dimension ``max_interactions``.  Slots for which
    ``valid`` is false are exactly zero and must not be interpreted as
    physical events.  ``scattering_order`` is one-based for scattering
    events.  For a terminal absorption it is the number of scatterings that
    occurred before absorption.

    The incoming and outgoing momenta use the same
    ``(E, p_los, p_sky_x, p_sky_y)`` convention as the main transport API.
    The outgoing momentum of an absorption event is zero.
    """

    valid: jnp.ndarray
    interaction_type: jnp.ndarray
    position_pc: jnp.ndarray
    incoming_momentum_kev: jnp.ndarray
    outgoing_momentum_kev: jnp.ndarray
    cumulative_path_length_pc: jnp.ndarray
    cumulative_excess_path_length_pc: jnp.ndarray
    scattering_order: jnp.ndarray


class PhotonTransportResult(NamedTuple):
    """Fixed-shape result for one transported photon."""

    position_pc: jnp.ndarray
    momentum_kev: jnp.ndarray
    path_length_pc: jnp.ndarray
    excess_path_length_pc: jnp.ndarray
    deposited_energy_kev: jnp.ndarray
    n_interactions: jnp.ndarray
    n_scatter: jnp.ndarray
    status: jnp.ndarray
    interactions: PhotonInteractionRecord


def _interpolation_bracket(grid, value):
    upper = jnp.searchsorted(grid, value, side="right")
    upper = jnp.clip(upper, 1, grid.size - 1)
    lower = upper - 1
    log_grid = jnp.log(grid)
    fraction = (jnp.log(value) - log_grid[lower]) / (log_grid[upper] - log_grid[lower])
    return lower, upper, jnp.clip(fraction, 0.0, 1.0)


def _interpolate_nonnegative(values, lower, upper, fraction):
    low = values[lower]
    high = values[upper]
    linear = low + fraction * (high - low)
    both_positive = (low > 0.0) & (high > 0.0)
    log_value = jnp.exp(
        jnp.log(jnp.maximum(low, jnp.finfo(values.dtype).tiny))
        + fraction
        * (
            jnp.log(jnp.maximum(high, jnp.finfo(values.dtype).tiny))
            - jnp.log(jnp.maximum(low, jnp.finfo(values.dtype).tiny))
        )
    )
    return jnp.where(both_positive, log_value, linear)


def _physics_at_energy(physics: DustPhysicsTable, energy_kev):
    supported = (
        (energy_kev >= physics.energy_kev[0])
        & (energy_kev <= physics.energy_kev[-1])
        & jnp.isfinite(energy_kev)
    )
    safe_energy = jnp.clip(energy_kev, physics.energy_kev[0], physics.energy_kev[-1])
    lower, upper, fraction = _interpolation_bracket(physics.energy_kev, safe_energy)
    sigma_sca = _interpolate_nonnegative(
        physics.scattering_cross_section_cm2_per_h,
        lower,
        upper,
        fraction,
    )
    sigma_abs = _interpolate_nonnegative(
        physics.absorption_cross_section_cm2_per_h,
        lower,
        upper,
        fraction,
    )
    phase_cdf = physics.scattering_angle_cdf[lower] + fraction * (
        physics.scattering_angle_cdf[upper] - physics.scattering_angle_cdf[lower]
    )
    # A convex interpolation of two nondecreasing CDF rows is itself
    # nondecreasing.  Avoid ``ufunc.accumulate`` here because it is not
    # available in every supported JAX release (the reference project pins
    # JAX 0.4.34).
    phase_cdf = phase_cdf.at[0].set(0.0)
    phase_cdf = phase_cdf.at[-1].set(1.0)
    return supported, sigma_sca, sigma_abs, phase_cdf


def _sample_scattering_angle(key, angle_grid, phase_cdf):
    uniform = random.uniform(key, dtype=phase_cdf.dtype)
    upper = jnp.searchsorted(phase_cdf, uniform, side="right")
    upper = jnp.clip(upper, 1, phase_cdf.size - 1)
    lower = upper - 1
    cdf_low = phase_cdf[lower]
    cdf_high = phase_cdf[upper]
    fraction = (uniform - cdf_low) / jnp.maximum(cdf_high - cdf_low, 1.0e-30)
    return angle_grid[lower] + fraction * (angle_grid[upper] - angle_grid[lower])


def _direction_from_axis_theta_phi(axis, theta, phi):
    """Rotate ``axis`` without losing arcsecond angles in float32.

    Forming ``mu = cos(theta)`` first and then recovering ``sin(theta)`` as
    ``sqrt(1-mu**2)`` rounds DSH-scale angles to zero in float32.  Evaluating
    sine directly preserves their first-order transverse component.
    """

    tangent_1, tangent_2 = orthonormal_basis(axis)
    transverse = jnp.cos(phi) * tangent_1 + jnp.sin(phi) * tangent_2
    return jnp.cos(theta) * axis + jnp.sin(theta) * transverse


def _line_of_sight_excess_factor(direction):
    """Return ``1 + direction[0]`` stably for observer-bound rays.

    For a unit vector with negative line-of-sight component,
    ``1 + mu = (dy**2 + dz**2) / (1 - mu)``.  The latter retains the
    second-order geometric delay when float32 rounds ``mu`` to ``-1``.
    """

    mu = direction[0]
    transverse_squared = jnp.sum(direction[1:] ** 2)
    stable_forward = transverse_squared / jnp.maximum(1.0 - mu, 1.0e-30)
    return jnp.where(mu < 0.0, stable_forward, 1.0 + mu)


def _transport_boundary_pc(position_pc, direction, source_distance_pc):
    """Distance and outcome at the observer plane or outer sphere.

    The Version-1 world is the half-sphere between the observer plane
    ``line_of_sight=0`` and the sphere through the source.  Dust occupies an
    arbitrary subset of that domain through ``AngularDistanceCloud``.
    """

    tiny = jnp.asarray(1.0e-12, dtype=position_pc.dtype)
    to_observer_plane = jnp.where(
        direction[0] < 0.0,
        -position_pc[0] / jnp.minimum(direction[0], -tiny),
        jnp.inf,
    )
    projection = jnp.dot(position_pc, direction)
    constant = jnp.dot(position_pc, position_pc) - source_distance_pc**2
    discriminant = jnp.maximum(projection**2 - constant, 0.0)
    to_outer_sphere = -projection + jnp.sqrt(discriminant)
    to_observer_plane = jnp.maximum(to_observer_plane, 0.0)
    to_outer_sphere = jnp.maximum(to_outer_sphere, 0.0)
    reached_observer = to_observer_plane <= to_outer_sphere
    boundary_status = jnp.where(
        reached_observer,
        jnp.asarray(REACHED_OBSERVER_PLANE, dtype=jnp.int32),
        jnp.asarray(ESCAPED_OUTER_BOUNDARY, dtype=jnp.int32),
    )
    return jnp.minimum(to_observer_plane, to_outer_sphere), boundary_status


def transport_photon_voxels(
    key,
    position_pc,
    momentum_kev,
    cloud: AngularDistanceCloud,
    physics: DustPhysicsTable,
    *,
    max_interactions: int = 64,
) -> PhotonTransportResult:
    """Transport one X-ray photon through the DSH voxel field.

    Parameters
    ----------
    key
        JAX PRNG key dedicated to this photon.
    position_pc
        Initial Cartesian position ``(los, sky_x, sky_y)`` in parsecs.
    momentum_kev
        Photon four-momentum ``(E, p_los, p_sky_x, p_sky_y)`` in keV.
    cloud
        Native angular-distance hydrogen-column voxels.
    physics
        Intrinsic energy-dependent scattering/absorption table.
    max_interactions
        Numerical safety limit.  It is not a physical scattering-order cut:
        a photon that reaches it is reported as ``MAX_INTERACTIONS`` and must
        not be counted as escaped or absorbed.

    Notes
    -----
    ``max_interactions`` controls the fixed JAX scan length and must be static
    when this function is jitted.  It is also the fixed leading length of every
    array in ``result.interactions``; unused slots have ``valid=False`` and
    zero-valued data.  For ``ABSORBED``, the returned four-momentum is zero and
    the incident energy is reported in ``deposited_energy_kev``.
    For ``REACHED_OBSERVER_PLANE``, ``excess_path_length_pc`` is the geometric
    path excess relative to an unscattered photon moving along negative LOS;
    it is accumulated in a form that retains small DSH angles in float32.
    """

    if max_interactions <= 0:
        raise ValueError("max_interactions must be positive")

    raw_position = jnp.asarray(position_pc)
    raw_momentum = jnp.asarray(momentum_kev)
    if raw_position.shape != (3,):
        raise ValueError("position_pc must have shape (3,)")
    if raw_momentum.shape != (4,):
        raise ValueError("momentum_kev must have shape (4,)")
    state_dtype = jnp.result_type(raw_position, raw_momentum, jnp.float32)
    position = raw_position.astype(state_dtype)
    momentum = raw_momentum.astype(state_dtype)
    energy = momentum[0]
    spatial_momentum = momentum[1:]
    momentum_norm = jnp.linalg.norm(spatial_momentum)
    direction = spatial_momentum / jnp.maximum(momentum_norm, 1.0e-30)
    supported, sigma_sca, sigma_abs, phase_cdf = _physics_at_energy(physics, energy)
    valid_state = (
        jnp.isfinite(position).all()
        & jnp.isfinite(momentum).all()
        & (energy > 0.0)
        & (momentum_norm > 0.0)
        & jnp.isclose(momentum_norm, energy, rtol=2.0e-5, atol=1.0e-8)
    )
    initially_valid = supported & valid_state
    initial_status = jnp.where(
        ~valid_state,
        jnp.asarray(INVALID_STATE, dtype=jnp.int32),
        jnp.where(
            supported,
            jnp.asarray(ACTIVE, dtype=jnp.int32),
            jnp.asarray(INVALID_ENERGY, dtype=jnp.int32),
        ),
    )
    source_distance_pc = cloud.source_distance_kpc * PC_PER_KPC

    initial_carry = (
        position,
        direction,
        key,
        jnp.asarray(0.0, dtype=position.dtype),
        jnp.asarray(0.0, dtype=position.dtype),
        jnp.asarray(0.0, dtype=energy.dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        initial_status,
    )

    def one_interaction(carry, _):
        (
            current_position,
            current_direction,
            current_key,
            path_length,
            excess_path_length,
            deposited_energy,
            n_interactions,
            n_scatter,
            status,
        ) = carry
        active = status == ACTIVE
        next_key, key_tau, key_process, key_theta, key_phi = random.split(
            current_key, 5
        )

        distance_to_boundary, boundary_status = _transport_boundary_pc(
            current_position, current_direction, source_distance_pc
        )
        sigma_total = sigma_sca + sigma_abs
        uniform_tau = random.uniform(
            key_tau,
            dtype=position.dtype,
            minval=jnp.finfo(position.dtype).tiny,
            maxval=1.0,
        )
        target_tau = -jnp.log(uniform_tau)
        positive_opacity = sigma_total > 0.0
        safe_sigma_total = jnp.where(positive_opacity, sigma_total, 1.0)
        target_column = jnp.where(
            positive_opacity,
            target_tau / safe_sigma_total,
            jnp.inf,
        )
        reached, interaction_distance, interaction_position = (
            locate_ray_column_depth_pc(
                cloud,
                current_position,
                current_direction,
                distance_to_boundary,
                target_column,
            )
        )
        interacted = active & positive_opacity & reached
        escaped = active & ~interacted

        scattering_probability = jnp.where(
            positive_opacity, sigma_sca / safe_sigma_total, 0.0
        )
        scattered = interacted & (
            random.uniform(key_process, dtype=energy.dtype) < scattering_probability
        )
        absorbed = interacted & ~scattered

        theta = _sample_scattering_angle(
            key_theta, physics.scattering_angle_rad, phase_cdf
        )
        phi = 2.0 * jnp.pi * random.uniform(key_phi, dtype=position.dtype)
        scattered_direction = _direction_from_axis_theta_phi(
            current_direction, theta, phi
        )
        scattered_direction /= jnp.linalg.norm(scattered_direction)

        endpoint = current_position + distance_to_boundary * current_direction
        next_position = jnp.where(
            interacted,
            interaction_position,
            jnp.where(escaped, endpoint, current_position),
        )
        travelled = jnp.where(
            interacted,
            interaction_distance,
            jnp.where(escaped, distance_to_boundary, 0.0),
        )
        next_direction = jnp.where(scattered, scattered_direction, current_direction)
        next_path_length = path_length + travelled
        next_excess_path_length = (
            excess_path_length
            + travelled * _line_of_sight_excess_factor(current_direction)
        )
        next_status = jnp.where(
            absorbed,
            jnp.asarray(ABSORBED, dtype=jnp.int32),
            jnp.where(
                escaped,
                boundary_status,
                status,
            ),
        )

        incoming_momentum = jnp.concatenate([energy[None], energy * current_direction])
        scattered_momentum = jnp.concatenate(
            [energy[None], energy * scattered_direction]
        )
        outgoing_momentum = jnp.where(
            scattered, scattered_momentum, jnp.zeros_like(scattered_momentum)
        )
        interaction_type = jnp.where(
            scattered,
            jnp.asarray(DUST_SCATTERING, dtype=jnp.int32),
            jnp.where(
                absorbed,
                jnp.asarray(PHOTOELECTRIC_ABSORPTION, dtype=jnp.int32),
                jnp.asarray(NO_INTERACTION, dtype=jnp.int32),
            ),
        )
        zero_position = jnp.zeros_like(current_position)
        zero_momentum = jnp.zeros_like(incoming_momentum)
        record = PhotonInteractionRecord(
            valid=interacted,
            interaction_type=interaction_type,
            position_pc=jnp.where(interacted, interaction_position, zero_position),
            incoming_momentum_kev=jnp.where(
                interacted, incoming_momentum, zero_momentum
            ),
            outgoing_momentum_kev=jnp.where(
                interacted, outgoing_momentum, zero_momentum
            ),
            cumulative_path_length_pc=jnp.where(interacted, next_path_length, 0.0),
            cumulative_excess_path_length_pc=jnp.where(
                interacted, next_excess_path_length, 0.0
            ),
            scattering_order=jnp.where(
                interacted,
                n_scatter + scattered.astype(jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )
        return (
            next_position,
            next_direction,
            next_key,
            next_path_length,
            next_excess_path_length,
            deposited_energy + jnp.where(absorbed, energy, 0.0),
            n_interactions + interacted.astype(jnp.int32),
            n_scatter + scattered.astype(jnp.int32),
            next_status,
        ), record

    final_carry, interactions = lax.scan(
        one_interaction, initial_carry, xs=None, length=max_interactions
    )
    (
        final_position,
        final_direction,
        _,
        path_length,
        excess_path_length,
        deposited_energy,
        n_interactions,
        n_scatter,
        final_status,
    ) = final_carry
    final_status = jnp.where(
        final_status == ACTIVE,
        jnp.asarray(MAX_INTERACTIONS, dtype=jnp.int32),
        final_status,
    )
    final_momentum = jnp.concatenate([energy[None], energy * final_direction])
    final_momentum = jnp.where(
        final_status == ABSORBED, jnp.zeros_like(final_momentum), final_momentum
    )
    final_momentum = jnp.where(initially_valid, final_momentum, momentum)
    return PhotonTransportResult(
        position_pc=final_position,
        momentum_kev=final_momentum,
        path_length_pc=path_length,
        excess_path_length_pc=excess_path_length,
        deposited_energy_kev=deposited_energy,
        n_interactions=n_interactions,
        n_scatter=n_scatter,
        status=final_status,
        interactions=interactions,
    )


def transport_photon_batch(
    key,
    position_pc,
    momentum_kev,
    cloud: AngularDistanceCloud,
    physics: DustPhysicsTable,
    *,
    max_interactions: int = 64,
) -> PhotonTransportResult:
    """Vectorize :func:`transport_photon_voxels` over a photon batch."""

    positions = jnp.asarray(position_pc)
    momenta = jnp.asarray(momentum_kev)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("position_pc must have shape (n_photons, 3)")
    if momenta.shape != (positions.shape[0], 4):
        raise ValueError("momentum_kev must have shape (n_photons, 4)")
    if positions.shape[0] <= 0:
        raise ValueError("photon batch cannot be empty")
    keys = random.split(key, positions.shape[0])
    return vmap(
        lambda photon_key, photon_position, photon_momentum: transport_photon_voxels(
            photon_key,
            photon_position,
            photon_momentum,
            cloud,
            physics,
            max_interactions=max_interactions,
        )
    )(keys, positions, momenta)
