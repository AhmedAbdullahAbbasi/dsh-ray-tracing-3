"""Channel-specific peel-off scoring for evolving-energy photon histories."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import vmap

from ..geometry.coordinates import ARCSEC_TO_RAD, PC_PER_KPC
from ..geometry.rays import integrate_ray_column_cm2
from ..physics.compton import ELECTRON_REST_ENERGY_KEV, _energy_bracket
from ..transport.inelastic import COMPTON_SCATTERING, interaction_opacities
from ..transport.kernel import DUST_SCATTERING
from .binning import BinnedObserverProducts, bin_observer_events
from .scoring import (
    PC_LIGHT_TRAVEL_TIME_S,
    ObserverEventResult,
    _validate_event_shapes,
    scattering_phase_pdf_per_sr,
    score_peeloff_events,
)

NO_COMPONENT = 0
DUST_ONLY = 1
COMPTON_ONLY = 2
MIXED = 3


class InelasticObserverEvents(NamedTuple):
    """ObserverEventResult uses OUTGOING energy; source energy stays separate.

    Classification uses history up to each scored interaction, never terminal
    packet counters. Unsupported virtual escape energies are flagged rather
    than extrapolated or silently accepted. Numerical continuation cutoff
    does not erase an earlier supported virtual score.
    """

    events: ObserverEventResult
    source_energy_kev: jnp.ndarray
    incident_energy_kev: jnp.ndarray
    dust_order: jnp.ndarray
    compton_order: jnp.ndarray
    interaction_type: jnp.ndarray
    component: jnp.ndarray
    candidate: jnp.ndarray
    unsupported_escape: jnp.ndarray
    unsupported_escape_energy_kev: jnp.ndarray
    unsupported_phase: jnp.ndarray


def compton_observer_phase_and_energy(table, energy_kev, scattering_angle_rad):
    """The same linear-in-cosine phase as Stage 1, stable at tiny angles.

    Compute distance FROM the right cosine node using 1-cos(theta), preserving
    the forward binding suppression and tiny recoil when float32 cos=1.
    """
    energy, angle = jnp.broadcast_arrays(
        jnp.asarray(energy_kev), jnp.asarray(scattering_angle_rad)
    )
    support, low, high, fraction = _energy_bracket(table, energy)
    support &= jnp.isfinite(angle) & (angle >= 0) & (angle <= jnp.pi)
    safe_angle = jnp.clip(jnp.where(jnp.isfinite(angle), angle, 0.0), 0, jnp.pi)
    gap = 2 * jnp.sin(safe_angle / 2) ** 2
    cosine = jnp.cos(safe_angle)
    right = jnp.clip(
        jnp.searchsorted(table.cosine, cosine, side="right"), 1, table.cosine.size - 1
    )
    left = right - 1
    from_right = jnp.clip(
        ((table.cosine[right] - 1) + gap) / (table.cosine[right] - table.cosine[left]),
        0.0,
        1.0,
    )
    phase_low = (
        from_right * table.phase_pdf_per_sr[low, left]
        + (1 - from_right) * table.phase_pdf_per_sr[low, right]
    )
    phase_high = (
        from_right * table.phase_pdf_per_sr[high, left]
        + (1 - from_right) * table.phase_pdf_per_sr[high, right]
    )
    phase = (1 - fraction) * phase_low + fraction * phase_high
    transfer = energy / ELECTRON_REST_ENERGY_KEV * gap
    outgoing = energy / (1 + transfer)
    return support, jnp.where(support, phase, 0.0), jnp.where(support, outgoing, 0.0)


def _components(dust, compton):
    return jnp.where(
        (dust > 0) & (compton > 0),
        MIXED,
        jnp.where(
            dust > 0, DUST_ONLY, jnp.where(compton > 0, COMPTON_ONLY, NO_COMPONENT)
        ),
    ).astype(jnp.int32)


def score_inelastic_peeloff(launched, transported, cloud, physics, compton):
    """Score dust and Compton events using their own observer outgoing energy.

    The real interaction probability was already sampled in transport. Use
    only the selected channel's normalized phase, not another sigma factor.
    All three opacities attenuate the virtual ray at its OUTGOING energy.
    No recoil deposition or later continuation outcome enters the weight.
    """
    n_packets, n_records = _validate_event_shapes(launched, transported)
    records = transported.interactions
    shape = (n_packets, n_records)
    source_energy = jnp.broadcast_to(launched.momentum_kev[:, 0, None], shape)
    if compton is None:
        events = score_peeloff_events(launched, transported, cloud, physics)
        zero = jnp.zeros(shape, dtype=jnp.int32)
        return InelasticObserverEvents(
            events,
            jnp.where(events.valid, source_energy, 0.0),
            events.energy_kev,
            jnp.where(events.valid, records.dust_order, 0),
            zero,
            jnp.where(events.valid, records.interaction_type, 0),
            jnp.where(events.valid, DUST_ONLY, NO_COMPONENT).astype(jnp.int32),
            events.valid,
            jnp.zeros(shape, dtype=bool),
            jnp.zeros_like(events.energy_kev),
            jnp.zeros(shape, dtype=bool),
        )
    position = records.position_pc
    incoming = records.incoming_momentum_kev
    energy = incoming[..., 0]
    norm = jnp.linalg.norm(incoming[..., 1:], axis=-1)
    direction = incoming[..., 1:] / jnp.maximum(norm[..., None], 1e-30)
    distance = jnp.linalg.norm(position, axis=-1)
    safe_distance = jnp.maximum(distance, 1e-30)
    observer_axis = -position / safe_distance[..., None]
    cosine = jnp.clip(jnp.sum(direction * observer_axis, axis=-1), -1, 1)
    sine = jnp.linalg.norm(jnp.cross(direction, observer_axis), axis=-1)
    angle = jnp.arctan2(sine, cosine)
    incident_support, sigma_dust, _, _ = interaction_opacities(physics, compton, energy)
    c_support, c_phase, c_outgoing = compton_observer_phase_and_energy(
        compton, energy, angle
    )
    dust = records.interaction_type == DUST_SCATTERING
    incoherent = records.interaction_type == COMPTON_SCATTERING
    expected = (physics.energy_kev.size, physics.scattering_angle_rad.size)
    if physics.differential_cross_section_cm2_per_sr_per_h.shape == expected:
        dust_phase = scattering_phase_pdf_per_sr(physics, energy, angle)
        dust_phase_supported = jnp.ones(shape, dtype=bool)
    elif physics.differential_cross_section_cm2_per_sr_per_h.size == 0:
        # Valid Compton-only tables can have identically zero dust opacity.
        dust_phase = jnp.zeros_like(energy)
        dust_phase_supported = sigma_dust <= 0
    else:
        raise ValueError("invalid dust differential shape for inelastic scoring")
    outgoing = jnp.where(incoherent, c_outgoing, energy)
    phase = jnp.where(incoherent, c_phase, dust_phase)
    candidate = (
        records.valid
        & (dust | incoherent)
        & incident_support
        & jnp.isfinite(position).all(axis=-1)
        & (position[..., 0] > 0)
        & jnp.isfinite(incoming).all(axis=-1)
        & (norm > 0)
        & (distance > 0)
        & jnp.isfinite(launched.isotropic_importance[:, None])
        & (launched.isotropic_importance[:, None] > 0)
        & jnp.isfinite(launched.weight_observer_fluence[:, None])
        & (launched.weight_observer_fluence[:, None] >= 0)
    )
    unsupported_phase = candidate & ~jnp.where(
        incoherent, c_support, dust_phase_supported
    )
    escape_support, sd, sc, sa = interaction_opacities(physics, compton, outgoing)
    unsupported = candidate & ~escape_support
    valid = candidate & escape_support & ~unsupported_phase
    fallback = jnp.array([-1.0, 0.0, 0.0], dtype=position.dtype)
    origins = jnp.where(valid[..., None], position, 0.0)
    axes = jnp.where(valid[..., None], observer_axis, fallback)
    lengths = jnp.where(valid, distance, 0.0)
    columns = vmap(
        lambda origin, axis, length: integrate_ray_column_cm2(
            cloud, origin, axis, length
        )
    )(origins.reshape(-1, 3), axes.reshape(-1, 3), lengths.reshape(-1)).reshape(shape)
    tau = columns * (sd + sc + sa)
    transmission = jnp.exp(-tau)
    transverse_squared = jnp.sum(position[..., 1:] ** 2, axis=-1)
    radial_minus_los = transverse_squared / jnp.maximum(
        distance + position[..., 0], 1e-30
    )
    excess = records.cumulative_excess_path_length_pc + radial_minus_los
    arrival = launched.emission_time_s[:, None] + excess * PC_LIGHT_TRAVEL_TIME_S
    weight = (
        launched.weight_observer_fluence[:, None]
        * (cloud.source_distance_kpc * PC_PER_KPC / safe_distance) ** 2
        * (4 * jnp.pi * launched.isotropic_importance[:, None])
        * phase
        * transmission
    )

    def mask(values):
        return jnp.where(valid, values, jnp.zeros_like(values))

    events = ObserverEventResult(
        valid,
        mask(jnp.arctan2(position[..., 1], position[..., 0]) / ARCSEC_TO_RAD),
        mask(jnp.arctan2(position[..., 2], position[..., 0]) / ARCSEC_TO_RAD),
        mask(outgoing),
        mask(arrival),
        mask(excess),
        mask(angle),
        mask(records.scattering_order),
        mask(columns),
        mask(tau),
        mask(transmission),
        mask(phase),
        mask(weight),
        mask(jnp.broadcast_to(launched.time_index[:, None], shape)),
        mask(jnp.broadcast_to(launched.spectral_bin_index[:, None], shape)),
    )
    return InelasticObserverEvents(
        events,
        mask(source_energy),
        mask(energy),
        mask(records.dust_order),
        mask(records.compton_order),
        mask(records.interaction_type),
        mask(_components(records.dust_order, records.compton_order)),
        candidate,
        unsupported,
        jnp.where(unsupported, outgoing, 0.0),
        unsupported_phase,
    )


class InelasticBinnedProducts(NamedTuple):
    total: BinnedObserverProducts
    dust_only: BinnedObserverProducts
    compton_only: BinnedObserverProducts
    mixed: BinnedObserverProducts


def bin_inelastic_events(scored, geometry):
    """Reuse physical binning at observer energy, preserving history moments.

    Total second moments are calculated from complete histories, including
    cross terms between components. Do not sum component variances to obtain
    the total variance; cross-component covariance is not exported here.
    """
    return InelasticBinnedProducts(
        bin_observer_events(scored.events, geometry),
        *(
            bin_observer_events(
                scored.events._replace(
                    valid=scored.events.valid & (scored.component == component)
                ),
                geometry,
            )
            for component in (DUST_ONLY, COMPTON_ONLY, MIXED)
        ),
    )
