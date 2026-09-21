"""High-statistics Monte Carlo experiments for the DSH validation ladder.

Unlike :mod:`dsh.validation.analytic`, these functions execute the production
source-launch, voxel-transport, and peel-off scorer. They are intended for
explicit local validation runs rather than the routine unit-test suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from ..geometry.clouds import build_angular_distance_cloud
from ..geometry.coordinates import ARCSEC_TO_RAD, PC_PER_KPC
from ..observer.scoring import score_peeloff_events
from ..physics.newdust import (
    NewDustScatteringTable,
    build_dust_physics_from_newdust,
)
from ..sources.launch import build_cloud_launch_geometry
from ..sources.models import SourcePackets
from ..transport.kernel import transport_photon_batch
from .analytic import (
    azimuthal_harmonic_amplitudes,
    exact_single_scatter_delay_s,
    phase_containment_angle_rad,
)
from .launch import nested_screen_launch_geometries, sample_mixture_source_launches


@dataclass(frozen=True)
class UniformScreenValidation:
    """Summary from one energy through a low-opacity uniform screen."""

    energy_kev: float
    packet_count: int
    scored_event_count: int
    target_scattering_optical_depth: float
    expected_interaction_probability: float
    analog_scattered_fraction: float
    analog_binomial_z: float
    scored_observer_fluence: float
    scored_fluence_to_tau: float
    phase_probability_inside_inscribed_aperture: float
    weighted_median_radius_arcsec: float
    expected_median_radius_arcsec: float
    maximum_delay_error_s: float
    maximum_relative_delay_error: float
    center_screen_delay_residual_rms_fraction: float
    azimuthal_effective_sample_size: float
    observer_fluence_relative_standard_error: float
    azimuthal_harmonic_amplitudes: tuple[float, ...]


def _uniform_screen_cloud(
    source_distance_kpc,
    fractional_distance,
    thickness_kpc,
    total_column_cm2,
    half_width_arcsec,
    sky_pixels,
    radial_cells,
):
    center = fractional_distance * source_distance_kpc
    radial_width = thickness_kpc / radial_cells
    z_centers = (
        center
        - 0.5 * thickness_kpc
        + (np.arange(radial_cells, dtype=np.float64) + 0.5) * radial_width
    )
    sky_width = 2.0 * half_width_arcsec / sky_pixels
    sky_centers = (
        -half_width_arcsec + (np.arange(sky_pixels, dtype=np.float64) + 0.5) * sky_width
    )
    delta_nh = np.full(
        (radial_cells, sky_pixels, sky_pixels),
        total_column_cm2 / radial_cells,
        dtype=np.float64,
    )
    return build_angular_distance_cloud(
        delta_nh,
        x_centers_arcsec=sky_centers,
        y_centers_arcsec=sky_centers,
        z_centers_kpc=z_centers,
        source_distance_kpc=source_distance_kpc,
    )


def _weighted_median(values, weights):
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    midpoint = 0.5 * sorted_weights.sum(dtype=np.float64)
    index = np.searchsorted(np.cumsum(sorted_weights, dtype=np.float64), midpoint)
    return float(sorted_values[min(index, sorted_values.size - 1)])


def run_uniform_screen_validation(
    key,
    scattering: NewDustScatteringTable,
    *,
    energy_index: int,
    packet_count: int,
    chunk_size: int,
    source_distance_kpc: float = 10.0,
    fractional_distance: float = 0.5,
    thickness_kpc: float = 0.01,
    target_scattering_optical_depth: float = 0.01,
    half_width_arcsec: float = 2_000.0,
    sky_pixels: int = 32,
    radial_cells: int = 4,
) -> UniformScreenValidation:
    """Run a delta flare through one low-opacity, laterally uniform screen.

    Absorption is intentionally disabled. The experiment jointly probes
    interaction normalization, exact delay geometry, radial energy scaling,
    launch-importance weights, peel-off normalization, and azimuthal symmetry.
    A finite square screen truncates the extreme phase-function tail; the
    report includes the phase probability within an inscribed circular
    aperture so this limitation remains explicit.
    """

    if not isinstance(energy_index, int) or not 0 <= energy_index < len(
        scattering.energy_kev
    ):
        raise ValueError("energy_index lies outside the scattering table")
    for value, name in ((packet_count, "packet_count"), (chunk_size, "chunk_size")):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    scalar_values = np.asarray(
        [
            source_distance_kpc,
            fractional_distance,
            thickness_kpc,
            target_scattering_optical_depth,
            half_width_arcsec,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(scalar_values)) or np.any(scalar_values <= 0.0):
        raise ValueError("screen parameters must be finite and positive")
    if fractional_distance >= 1.0:
        raise ValueError("fractional_distance must be less than one")
    if not isinstance(sky_pixels, int) or sky_pixels < 2:
        raise ValueError("sky_pixels must be an integer of at least two")
    if not isinstance(radial_cells, int) or radial_cells < 2:
        raise ValueError("radial_cells must be an integer of at least two")

    energy = float(scattering.energy_kev[energy_index])
    sigma_scattering = float(
        scattering.scattering_cross_section_cm2_per_h[energy_index]
    )
    total_column = target_scattering_optical_depth / sigma_scattering
    cloud = _uniform_screen_cloud(
        source_distance_kpc,
        fractional_distance,
        thickness_kpc,
        total_column,
        half_width_arcsec,
        sky_pixels,
        radial_cells,
    )
    physics = build_dust_physics_from_newdust(
        scattering,
        np.zeros_like(scattering.energy_kev),
    )
    launch_geometry = build_cloud_launch_geometry(cloud)
    median_scattering_angle = float(
        phase_containment_angle_rad(
            scattering.scattering_angle_rad,
            scattering.scattering_angle_cdf[energy_index],
            0.5,
        )
    )
    proposals = nested_screen_launch_geometries(
        launch_geometry,
        source_distance_kpc=source_distance_kpc,
        fractional_distance=fractional_distance,
        median_scattering_angle_rad=median_scattering_angle,
    )
    launch_jit = jax.jit(sample_mixture_source_launches)
    transport_jit = jax.jit(
        transport_photon_batch,
        static_argnames=("max_interactions",),
    )
    score_jit = jax.jit(score_peeloff_events)

    scattered_count = 0
    event_weights = []
    event_radius_arcsec = []
    event_azimuth_rad = []
    event_delay_s = []
    event_position_pc = []
    completed = 0
    chunk_index = 0
    while completed < packet_count:
        current = min(chunk_size, packet_count - completed)
        packets = SourcePackets(
            energy_kev=jnp.full((current,), energy, dtype=jnp.float32),
            emission_time_s=jnp.zeros(current, dtype=jnp.float32),
            weight_observer_fluence=jnp.full(
                (current,), 1.0 / packet_count, dtype=jnp.float32
            ),
            time_index=jnp.zeros(current, dtype=jnp.int32),
            spectral_bin_index=jnp.full((current,), energy_index, dtype=jnp.int32),
        )
        chunk_key = random.fold_in(key, chunk_index)
        launch_key, transport_key = random.split(chunk_key)
        launched = launch_jit(launch_key, packets, proposals)
        transported = transport_jit(
            transport_key,
            launched.position_pc,
            launched.momentum_kev,
            cloud,
            physics,
            max_interactions=1,
        )
        events = score_jit(launched, transported, cloud, physics)
        valid = np.asarray(events.valid)[:, 0]
        scattered_count += int(np.asarray(transported.n_scatter).sum())
        if np.any(valid):
            sky_x = np.asarray(events.sky_x_arcsec)[valid, 0].astype(np.float64)
            sky_y = np.asarray(events.sky_y_arcsec)[valid, 0].astype(np.float64)
            event_weights.append(
                np.asarray(events.weight_observer_fluence)[valid, 0].astype(np.float64)
            )
            event_radius_arcsec.append(np.hypot(sky_x, sky_y))
            event_azimuth_rad.append(np.arctan2(sky_y, sky_x))
            event_delay_s.append(
                np.asarray(events.arrival_time_s)[valid, 0].astype(np.float64)
            )
            event_position_pc.append(
                np.asarray(transported.interactions.position_pc)[valid, 0].astype(
                    np.float64
                )
            )
        completed += current
        chunk_index += 1

    if not event_weights:
        raise RuntimeError("uniform-screen experiment produced no scattering events")
    weights = np.concatenate(event_weights)
    radius_arcsec = np.concatenate(event_radius_arcsec)
    azimuth_rad = np.concatenate(event_azimuth_rad)
    delay_s = np.concatenate(event_delay_s)
    position_pc = np.concatenate(event_position_pc)

    analog_fraction = scattered_count / packet_count
    expected_probability = -np.expm1(-target_scattering_optical_depth)
    standard_error = np.sqrt(
        expected_probability * (1.0 - expected_probability) / packet_count
    )
    analog_z = (analog_fraction - expected_probability) / standard_error
    scored_fluence = float(weights.sum(dtype=np.float64))

    source_distance_pc = source_distance_kpc * PC_PER_KPC
    event_fraction = np.linalg.norm(position_pc, axis=1) / source_distance_pc
    event_theta = np.arctan2(
        np.linalg.norm(position_pc[:, 1:], axis=1), position_pc[:, 0]
    )
    expected_delay = exact_single_scatter_delay_s(
        source_distance_kpc,
        event_fraction,
        event_theta,
    )
    delay_error = delay_s - expected_delay
    delay_scale = np.maximum(np.abs(expected_delay), 1.0)
    center_screen_delay = exact_single_scatter_delay_s(
        source_distance_kpc,
        fractional_distance,
        event_theta,
    )

    expected_median_radius = (
        (1.0 - fractional_distance) * median_scattering_angle / ARCSEC_TO_RAD
    )
    weighted_median_radius = _weighted_median(radius_arcsec, weights)

    aperture = (radius_arcsec > 0.0) & (radius_arcsec <= 0.8 * half_width_arcsec)
    aperture_weights = weights[aperture]
    harmonics = azimuthal_harmonic_amplitudes(
        azimuth_rad[aperture], aperture_weights, max_order=4
    )
    effective_size = float(
        aperture_weights.sum(dtype=np.float64) ** 2
        / np.sum(aperture_weights**2, dtype=np.float64)
    )
    fluence_relative_se = float(
        np.sqrt(
            np.sum(weights**2, dtype=np.float64) / scored_fluence**2
            - 1.0 / packet_count
        )
    )
    delay_residual_fraction = (
        delay_s[aperture] - center_screen_delay[aperture]
    ) / np.maximum(center_screen_delay[aperture], 1.0)
    residual_mean = np.average(delay_residual_fraction, weights=aperture_weights)
    residual_rms = float(
        np.sqrt(
            np.average(
                (delay_residual_fraction - residual_mean) ** 2,
                weights=aperture_weights,
            )
        )
    )
    inscribed_physical_angle = (
        0.8 * half_width_arcsec * ARCSEC_TO_RAD / (1.0 - fractional_distance)
    )
    phase_probability = float(
        np.interp(
            inscribed_physical_angle,
            scattering.scattering_angle_rad,
            scattering.scattering_angle_cdf[energy_index],
        )
    )

    return UniformScreenValidation(
        energy_kev=energy,
        packet_count=packet_count,
        scored_event_count=weights.size,
        target_scattering_optical_depth=target_scattering_optical_depth,
        expected_interaction_probability=float(expected_probability),
        analog_scattered_fraction=float(analog_fraction),
        analog_binomial_z=float(analog_z),
        scored_observer_fluence=scored_fluence,
        scored_fluence_to_tau=scored_fluence / target_scattering_optical_depth,
        phase_probability_inside_inscribed_aperture=phase_probability,
        weighted_median_radius_arcsec=weighted_median_radius,
        expected_median_radius_arcsec=float(expected_median_radius),
        maximum_delay_error_s=float(np.max(np.abs(delay_error))),
        maximum_relative_delay_error=float(np.max(np.abs(delay_error) / delay_scale)),
        center_screen_delay_residual_rms_fraction=residual_rms,
        azimuthal_effective_sample_size=effective_size,
        observer_fluence_relative_standard_error=fluence_relative_se,
        azimuthal_harmonic_amplitudes=tuple(float(value) for value in harmonics),
    )
