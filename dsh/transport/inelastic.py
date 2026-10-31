"""Coupled dust/Compton/photoabsorption transport through native voxels.

Explicit low-level API for the Compton extension. The configured runner and
elastic archive format remain unchanged. With compton=None the accepted
elastic kernel is called directly, including its exact random streams.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax, random, vmap

from ..geometry.coordinates import PC_PER_KPC
from ..geometry.rays import locate_ray_column_depth_pc
from ..physics.compton import evaluate_compton, sample_compton_scattering
from .kernel import (
    ABSORBED,
    ACTIVE,
    DUST_SCATTERING,
    INVALID_ENERGY,
    INVALID_STATE,
    MAX_INTERACTIONS,
    NO_INTERACTION,
    PHOTOELECTRIC_ABSORPTION,
    _direction_from_axis_theta_phi,
    _interpolate_nonnegative,
    _interpolation_bracket,
    _line_of_sight_excess_factor,
    _physics_at_energy,
    _sample_scattering_angle,
    _transport_boundary_pc,
    transport_photon_voxels,
)

COMPTON_SCATTERING = 3
ENERGY_CUTOFF = 7
UNSUPPORTED_CONTINUATION = 8
INELASTIC_STATUS_NAMES = {
    ENERGY_CUTOFF: "continuation below the declared energy cutoff",
    UNSUPPORTED_CONTINUATION: "continuation outside common material support",
}


class InelasticInteractionRecord(NamedTuple):
    valid: jnp.ndarray
    interaction_type: jnp.ndarray
    position_pc: jnp.ndarray
    incoming_momentum_kev: jnp.ndarray
    outgoing_momentum_kev: jnp.ndarray
    cumulative_path_length_pc: jnp.ndarray
    cumulative_excess_path_length_pc: jnp.ndarray
    scattering_order: jnp.ndarray
    dust_order: jnp.ndarray
    compton_order: jnp.ndarray
    recoil_energy_kev: jnp.ndarray
    deposited_energy_kev: jnp.ndarray


class InelasticTransportResult(NamedTuple):
    position_pc: jnp.ndarray
    momentum_kev: jnp.ndarray
    path_length_pc: jnp.ndarray
    excess_path_length_pc: jnp.ndarray
    deposited_energy_kev: jnp.ndarray
    n_interactions: jnp.ndarray
    n_scatter: jnp.ndarray
    status: jnp.ndarray
    interactions: InelasticInteractionRecord
    n_dust: jnp.ndarray
    n_compton: jnp.ndarray
    recoil_energy_kev: jnp.ndarray
    absorbed_energy_kev: jnp.ndarray
    source_energy_kev: jnp.ndarray
    cutoff_energy_kev: jnp.ndarray
    unsupported_energy_kev: jnp.ndarray
    transport_energy_cutoff_kev: jnp.ndarray


def interaction_opacities(physics, compton, energy_kev):
    """Support and (dust, Compton, absorption) totals at the CURRENT energy.

    Unsupported requests return zero opacities, never clipped physical data.
    The dust helper internally uses safe indices, which do not grant support.
    """
    energy = jnp.asarray(energy_kev)
    supported = (
        jnp.isfinite(energy)
        & (energy >= physics.energy_kev[0])
        & (energy <= physics.energy_kev[-1])
    )
    safe = jnp.clip(
        jnp.where(jnp.isfinite(energy), energy, physics.energy_kev[0]),
        physics.energy_kev[0],
        physics.energy_kev[-1],
    )
    low, high, fraction = _interpolation_bracket(physics.energy_kev, safe)
    if physics.energy_edge_kev is not None:
        edge = physics.energy_edge_kev[low]
        fraction = jnp.where(edge > 0, (safe >= edge).astype(fraction.dtype), fraction)
    dust = _interpolate_nonnegative(
        physics.scattering_cross_section_cm2_per_h, low, high, fraction
    )
    absorption = _interpolate_nonnegative(
        physics.absorption_cross_section_cm2_per_h, low, high, fraction
    )
    if compton is None:
        incoherent = jnp.zeros_like(dust)
    else:
        evaluated = evaluate_compton(compton, energy_kev, jnp.zeros_like(energy_kev))
        supported &= evaluated.supported
        incoherent = evaluated.cross_section_cm2_per_h
    return (
        supported,
        jnp.where(supported, dust, 0),
        jnp.where(supported, incoherent, 0),
        jnp.where(supported, absorption, 0),
    )


def _legacy_result(result, source_energy, cutoff):
    records = result.interactions
    zero = jnp.zeros_like(records.scattering_order)
    recoil = jnp.zeros_like(records.cumulative_path_length_pc)
    deposit = jnp.where(
        records.valid & (records.interaction_type == PHOTOELECTRIC_ABSORPTION),
        records.incoming_momentum_kev[..., 0],
        0,
    )
    enriched = InelasticInteractionRecord(
        *records, records.scattering_order, zero, recoil, deposit
    )
    return InelasticTransportResult(
        *result[:-1],
        enriched,
        result.n_scatter,
        jnp.zeros_like(result.n_scatter),
        jnp.zeros_like(result.deposited_energy_kev),
        result.deposited_energy_kev,
        source_energy,
        jnp.zeros_like(source_energy),
        jnp.zeros_like(source_energy),
        jnp.asarray(cutoff, dtype=source_energy.dtype),
    )


def transport_photon_inelastic(
    key,
    position_pc,
    momentum_kev,
    cloud,
    physics,
    compton,
    *,
    energy_cutoff_kev,
    max_interactions=64,
):
    """Transport one photon, updating energy and all opacities after recoil.

    energy_cutoff_kev is an explicit numerical CONTINUATION cutoff. Below
    it, retain the residual photon energy as a diagnosed loss, not heat or
    escaped radiation. A continuation above the cutoff but outside table
    support has a distinct incomplete status. Earlier records survive both.

    For every valid packet, source energy = deposited energy + final photon
    energy, including truncated/safety-limited packets. Deposited energy is
    analog recoil plus photoabsorption only, never virtual peel-off recoil.
    compton=None dispatches to the unchanged elastic kernel; its cutoff is
    intentionally not applied to that legacy path.
    """
    if (
        isinstance(max_interactions, bool)
        or not isinstance(max_interactions, int)
        or max_interactions < 1
    ):
        raise ValueError("max_interactions must be a positive static integer")
    raw_position, raw_momentum = jnp.asarray(position_pc), jnp.asarray(momentum_kev)
    if raw_position.shape != (3,) or raw_momentum.shape != (4,):
        raise ValueError("position/momentum must have shapes (3,) and (4,)")
    dtype = jnp.result_type(raw_position, raw_momentum, jnp.float32)
    if compton is not None:
        # Real input loaders may combine float32 packets/clouds with float64
        # tables. Promote the complete evolving carry, rather than allowing
        # one scatter to change a lax.scan state's dtype.
        dtype = jnp.result_type(
            dtype, physics.energy_kev, compton.energy_kev, cloud.n_h_cm3
        )
    position, momentum = raw_position.astype(dtype), raw_momentum.astype(dtype)
    cutoff = jnp.asarray(energy_cutoff_kev, dtype=dtype)
    if cutoff.ndim != 0:
        raise ValueError("energy_cutoff_kev must be scalar")
    if compton is None:
        return _legacy_result(
            transport_photon_voxels(
                key,
                position_pc,
                momentum_kev,
                cloud,
                physics,
                max_interactions=max_interactions,
            ),
            momentum[0],
            cutoff,
        )
    energy = momentum[0]
    norm = jnp.linalg.norm(momentum[1:])
    direction = momentum[1:] / jnp.maximum(norm, 1e-30)
    source_distance = cloud.source_distance_kpc * PC_PER_KPC
    valid = (
        jnp.isfinite(position).all()
        & jnp.isfinite(momentum).all()
        & (energy > 0)
        & (norm > 0)
        & jnp.isclose(norm, energy, rtol=2e-5, atol=1e-8)
        & (position[0] >= 0)
        & (jnp.linalg.norm(position) <= source_distance * (1 + 2e-6))
        & jnp.isfinite(cutoff)
        & (cutoff > 0)
    )
    supported, _, _, _ = interaction_opacities(physics, compton, energy)
    initial_status = jnp.where(
        ~valid,
        INVALID_STATE,
        jnp.where(
            ~supported,
            INVALID_ENERGY,
            jnp.where(energy < cutoff, ENERGY_CUTOFF, ACTIVE),
        ),
    ).astype(jnp.int32)
    # Safe inactive geometry prevents NaN/inf work in padded scan iterations.
    safe_position = jnp.where(
        valid, position, jnp.array([source_distance, 0.0, 0.0], dtype=dtype)
    )
    safe_direction = jnp.where(
        valid, direction, jnp.array([-1.0, 0.0, 0.0], dtype=dtype)
    )
    safe_energy = jnp.where(valid, energy, physics.energy_kev[0]).astype(dtype)
    zero = jnp.asarray(0.0, dtype=dtype)
    izero = jnp.asarray(0, dtype=jnp.int32)
    initial = (
        safe_position,
        safe_direction,
        safe_energy,
        key,
        zero,
        zero,
        zero,
        zero,
        izero,
        izero,
        izero,
        initial_status,
    )

    def step(carry, _):
        (
            pos,
            axis,
            incident,
            rng,
            path,
            excess,
            recoil_sum,
            absorbed_sum,
            count,
            nd,
            nc,
            status,
        ) = carry
        active = status == ACTIVE
        rng_next, tau_key, process_key, theta_key, phi_key, compton_key = random.split(
            rng, 6
        )
        support, sd, sc, sa = interaction_opacities(physics, compton, incident)
        _, _, _, dust_cdf = _physics_at_energy(physics, incident)
        total = sd + sc + sa
        positive = total > 0
        safe_total = jnp.where(positive, total, 1.0)
        tau = -jnp.log(
            random.uniform(
                tau_key, dtype=dtype, minval=jnp.finfo(dtype).tiny, maxval=1.0
            )
        )
        column = jnp.where(positive, tau / safe_total, jnp.inf)
        boundary, boundary_status = _transport_boundary_pc(pos, axis, source_distance)
        reached, distance, event_position = locate_ray_column_depth_pc(
            cloud, pos, axis, boundary, column
        )
        event = active & support & positive & reached
        escaped = active & support & ~event
        choice = random.uniform(process_key, dtype=dtype)
        dust = event & (choice < sd / safe_total)
        incoherent = event & ~dust & (choice < (sd + sc) / safe_total)
        absorption = event & ~(dust | incoherent)
        theta = _sample_scattering_angle(
            theta_key, physics.scattering_angle_rad, dust_cdf
        )
        phi = 2 * jnp.pi * random.uniform(phi_key, dtype=dtype)
        dust_axis = _direction_from_axis_theta_phi(axis, theta, phi)
        sampled = sample_compton_scattering(compton_key, compton, incident)
        compton_axis = _direction_from_axis_theta_phi(
            axis, jnp.arccos(jnp.clip(sampled.cosine, -1, 1)), sampled.azimuth_rad
        )
        next_axis = jnp.where(
            dust, dust_axis, jnp.where(incoherent, compton_axis, axis)
        )
        next_axis /= jnp.maximum(jnp.linalg.norm(next_axis), 1e-30)
        outgoing = jnp.where(incoherent, sampled.outgoing_energy_kev, incident).astype(
            dtype
        )
        recoil = jnp.where(incoherent, sampled.recoil_energy_kev, 0.0).astype(dtype)
        absorbed_energy = jnp.where(absorption, incident, 0.0)
        next_energy = jnp.where(absorption, 0.0, outgoing)
        travel = jnp.where(event, distance, jnp.where(escaped, boundary, 0.0))
        next_position = jnp.where(
            event, event_position, jnp.where(escaped, pos + boundary * axis, pos)
        )
        next_path = path + travel
        next_excess = excess + travel * _line_of_sight_excess_factor(axis)
        nd_next = nd + dust.astype(jnp.int32)
        nc_next = nc + incoherent.astype(jnp.int32)
        next_supported, _, _, _ = interaction_opacities(
            physics, compton, jnp.where(absorption, incident, next_energy)
        )
        next_status = jnp.where(
            absorption, ABSORBED, jnp.where(escaped, boundary_status, status)
        )
        next_status = jnp.where(
            incoherent & (outgoing < cutoff), ENERGY_CUTOFF, next_status
        )
        next_status = jnp.where(
            incoherent & (outgoing >= cutoff) & ~next_supported,
            UNSUPPORTED_CONTINUATION,
            next_status,
        )
        next_status = jnp.where(
            active & ~support, UNSUPPORTED_CONTINUATION, next_status
        ).astype(jnp.int32)
        incoming_momentum = jnp.r_[incident, incident * axis]
        outgoing_momentum = jnp.where(
            absorption,
            jnp.zeros(4, dtype=dtype),
            jnp.r_[outgoing, outgoing * next_axis],
        )
        kind = jnp.where(
            dust,
            DUST_SCATTERING,
            jnp.where(
                incoherent,
                COMPTON_SCATTERING,
                jnp.where(absorption, PHOTOELECTRIC_ABSORPTION, NO_INTERACTION),
            ),
        ).astype(jnp.int32)
        record = InelasticInteractionRecord(
            event,
            kind,
            jnp.where(event, event_position, 0.0),
            jnp.where(event, incoming_momentum, 0.0),
            jnp.where(event, outgoing_momentum, 0.0),
            jnp.where(event, next_path, 0.0),
            jnp.where(event, next_excess, 0.0),
            jnp.where(event, nd_next + nc_next, 0),
            jnp.where(event, nd_next, 0),
            jnp.where(event, nc_next, 0),
            recoil,
            recoil + absorbed_energy,
        )
        return (
            next_position,
            next_axis,
            next_energy,
            rng_next,
            next_path,
            next_excess,
            recoil_sum + recoil,
            absorbed_sum + absorbed_energy,
            count + event.astype(jnp.int32),
            nd_next,
            nc_next,
            next_status,
        ), record

    final, records = lax.scan(step, initial, xs=None, length=max_interactions)
    (
        pos,
        axis,
        final_energy,
        _,
        path,
        excess,
        recoil,
        absorbed,
        count,
        nd,
        nc,
        status,
    ) = final
    status = jnp.where(status == ACTIVE, MAX_INTERACTIONS, status)
    final_momentum = jnp.r_[final_energy, final_energy * axis]
    final_momentum = jnp.where(valid & supported, final_momentum, momentum)
    return InelasticTransportResult(
        pos,
        final_momentum,
        path,
        excess,
        recoil + absorbed,
        count,
        nd + nc,
        status,
        records,
        nd,
        nc,
        recoil,
        absorbed,
        energy,
        jnp.where(status == ENERGY_CUTOFF, final_energy, 0.0),
        jnp.where(status == UNSUPPORTED_CONTINUATION, final_energy, 0.0),
        cutoff,
    )


def transport_inelastic_batch(
    key,
    position_pc,
    momentum_kev,
    cloud,
    physics,
    compton,
    *,
    energy_cutoff_kev,
    max_interactions=64,
):
    """Vectorize evolving-energy transport; max_interactions is JIT-static."""
    positions, momenta = jnp.asarray(position_pc), jnp.asarray(momentum_kev)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 1:
        raise ValueError("positions must have nonempty shape (n_packets,3)")
    if momenta.shape != (positions.shape[0], 4):
        raise ValueError("momenta must have shape (n_packets,4)")
    return vmap(
        lambda rng, pos, mom: transport_photon_inelastic(
            rng,
            pos,
            mom,
            cloud,
            physics,
            compton,
            energy_cutoff_kev=energy_cutoff_kev,
            max_interactions=max_interactions,
        )
    )(random.split(key, positions.shape[0]), positions, momenta)
