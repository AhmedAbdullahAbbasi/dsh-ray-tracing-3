"""Stage 9A: independent first-event references for actual material tables.

The shell has constant physical density. Its intersections with a straight
source ray are computed here with NumPy sphere chords, independently of the
production voxel ray integrator and inverse-column locator. Scattering is
stopped after the first event; a scattered packet's cap status is expected.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from ..geometry.clouds import build_angular_distance_cloud
from ..geometry.rays import integrate_ray_column_cm2
from ..physics.dust import build_dust_physics_table
from ..transport.kernel import (
    ABSORBED,
    DUST_SCATTERING,
    MAX_INTERACTIONS,
    PHOTOELECTRIC_ABSORPTION,
    REACHED_OBSERVER_PLANE,
    transport_photon_batch,
)

SOURCE_KPC = 10.0
INNER_KPC = 4.0
OUTER_KPC = 5.0
RAY_ANGLES_RAD = (0.0, 0.2)
LOCATION_FRACTIONS = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class FirstEventReference:
    column_cm2: float
    first_scatter: float
    first_absorb: float
    no_event: float
    location_cdf: tuple[float, ...]


def sphere_chord(source_kpc: float, angle_rad: float) -> tuple[float, float]:
    """Distances from the source to the outer and inner radial surfaces."""

    impact = source_kpc * math.sin(angle_rad)
    if not 0.0 <= angle_rad < math.pi / 2 or impact >= INNER_KPC:
        raise ValueError("ray must pass through both shell boundaries")
    projected = source_kpc * math.cos(angle_rad)
    return (
        projected - math.sqrt(OUTER_KPC**2 - impact**2),
        projected - math.sqrt(INNER_KPC**2 - impact**2),
    )


def first_event_reference(
    central_column_cm2: float,
    sigma_sca: float,
    sigma_abs: float,
    angle_rad: float,
) -> FirstEventReference:
    """First-event category probabilities and conditional interaction CDF.

    The two half-kpc radial cells have identical physical density. The
    oblique ray therefore carries central NH times its shell path length in
    kpc, independently of the code under test.
    """

    if min(central_column_cm2, sigma_sca, sigma_abs) < 0.0:
        raise ValueError("column and cross sections must be nonnegative")
    entrance, exit_ = sphere_chord(SOURCE_KPC, angle_rad)
    column = central_column_cm2 * (exit_ - entrance) / (OUTER_KPC - INNER_KPC)
    tau = column * (sigma_sca + sigma_abs)
    interaction = -math.expm1(-tau)
    if tau == 0.0:
        return FirstEventReference(column, 0.0, 0.0, 1.0, ())
    scatter = interaction * sigma_sca / (sigma_sca + sigma_abs)
    absorb = interaction * sigma_abs / (sigma_sca + sigma_abs)
    cdf = tuple(-math.expm1(-tau * f) / interaction for f in LOCATION_FRACTIONS)
    return FirstEventReference(column, scatter, absorb, math.exp(-tau), cdf)


def _cloud(column_cm2: float):
    # Centers 4.25,4.75 kpc produce exact radial shell edges at 4,4.5,5.
    # Wide angular coverage keeps both controlled rays inside the shell.
    return build_angular_distance_cloud(
        np.full((2, 2, 2), column_cm2 / 2.0),
        x_centers_arcsec=[-120_000.0, 120_000.0],
        y_centers_arcsec=[-120_000.0, 120_000.0],
        z_centers_kpc=[4.25, 4.75],
        source_distance_kpc=SOURCE_KPC,
    )


def _z(observed: int, probability: float, trials: int) -> float:
    variance = trials * probability * (1.0 - probability)
    if variance == 0.0:
        return 0.0 if observed == trials * probability else math.inf
    return (observed - trials * probability) / math.sqrt(variance)


def run_first_event_case(
    key,
    scattering,
    absorption,
    *,
    energy_index: int,
    mode: str,
    target_tau_sca_3p3: float,
    packets_per_ray: int,
    chunk_size: int,
    sigma_limit: float = 5.0,
) -> dict:
    """Run two fixed source rays; compare first events with sphere chords.

    ``mode`` is a controlled zero/pure-absorption/pure-scattering/both switch.
    In the ``both`` case the supplied scattering and TBabs coefficients are
    used without changing their ratio. NH is set by 3.3-keV scattering tau.
    Event location tests use the CDF conditioned on *any* first event.
    """

    if mode not in {"zero", "absorption", "scattering", "both"}:
        raise ValueError("unknown material mode")
    if not 0 <= energy_index < len(scattering.energy_kev):
        raise ValueError("energy_index out of range")
    if not np.array_equal(scattering.energy_kev, absorption.energy_kev):
        raise ValueError("material energy grids differ")
    if target_tau_sca_3p3 < 0 or not math.isfinite(target_tau_sca_3p3):
        raise ValueError("target optical depth must be finite and nonnegative")
    if packets_per_ray <= 0 or chunk_size <= 0 or sigma_limit <= 0:
        raise ValueError("packet counts and sigma limit must be positive")

    reference_index = np.flatnonzero(scattering.energy_kev == 3.3)
    if reference_index.size != 1:
        raise ValueError("material table must contain exactly one 3.3 keV reference")
    central_column = (
        target_tau_sca_3p3
        / scattering.scattering_cross_section_cm2_per_h[reference_index[0]]
    )
    sca = np.asarray(scattering.scattering_cross_section_cm2_per_h).copy()
    absorb = np.asarray(absorption.absorption_cross_section_cm2_per_h).copy()
    if mode in {"zero", "absorption"}:
        sca[:] = 0.0
    if mode in {"zero", "scattering"}:
        absorb[:] = 0.0
    physics = build_dust_physics_table(
        scattering.energy_kev,
        sca,
        absorb,
        scattering.scattering_angle_rad,
        scattering.scattering_angle_cdf,
    )
    cloud = _cloud(central_column)
    transport = jax.jit(transport_photon_batch, static_argnames=("max_interactions",))
    energy = float(scattering.energy_kev[energy_index])
    ray_results = []
    for ray_index, angle in enumerate(RAY_ANGLES_RAD):
        entrance, exit_ = sphere_chord(SOURCE_KPC, angle)
        direction = np.array([-math.cos(angle), math.sin(angle), 0.0])
        source = np.array([SOURCE_KPC * 1_000.0, 0.0, 0.0])
        reference = first_event_reference(
            central_column, sca[energy_index], absorb[energy_index], angle
        )
        # This is a separate column check; probability references above never
        # call the production ray integrator.
        integrated = float(
            integrate_ray_column_cm2(
                cloud, source, direction, SOURCE_KPC * 1_000.0 / math.cos(angle)
            )
        )
        column_relative_error = (
            abs(integrated - reference.column_cm2) / reference.column_cm2
            if reference.column_cm2
            else abs(integrated)
        )
        counts = {"no_event": 0, "first_scatter": 0, "first_absorb": 0}
        location_counts = np.zeros(len(LOCATION_FRACTIONS), dtype=int)
        invalid_records = 0
        completed = 0
        chunk_index = 0
        while completed < packets_per_ray:
            current = min(chunk_size, packets_per_ray - completed)
            positions = jnp.broadcast_to(
                jnp.asarray(source, dtype=jnp.float32), (current, 3)
            )
            momentum = jnp.asarray([energy, *(energy * direction)], dtype=jnp.float32)
            momenta = jnp.broadcast_to(momentum, (current, 4))
            result = transport(
                random.fold_in(random.fold_in(key, ray_index), chunk_index),
                positions,
                momenta,
                cloud,
                physics,
                max_interactions=1,
            )
            kind = np.asarray(result.interactions.interaction_type)[:, 0]
            valid = np.asarray(result.interactions.valid)[:, 0]
            status = np.asarray(result.status)
            no_event = ~valid
            scattered = valid & (kind == DUST_SCATTERING)
            absorbed = valid & (kind == PHOTOELECTRIC_ABSORPTION)
            counts["no_event"] += int(no_event.sum())
            counts["first_scatter"] += int(scattered.sum())
            counts["first_absorb"] += int(absorbed.sum())
            path = (
                np.asarray(result.interactions.cumulative_path_length_pc)[:, 0]
                / 1_000.0
            )
            for i, fraction in enumerate(LOCATION_FRACTIONS):
                location_counts[i] += int(
                    (valid & (path <= entrance + fraction * (exit_ - entrance))).sum()
                )
            final_energy = np.asarray(result.momentum_kev)[:, 0]
            deposited = np.asarray(result.deposited_energy_kev)
            event_position = np.asarray(result.interactions.position_pc)[:, 0]
            incoming = np.asarray(result.interactions.incoming_momentum_kev)[:, 0]
            outgoing = np.asarray(result.interactions.outgoing_momentum_kev)[:, 0]
            final_momentum = np.asarray(result.momentum_kev)
            expected_position = source + (path * 1_000.0)[:, None] * direction
            invalid_records += int(
                np.count_nonzero(
                    (no_event & (status != REACHED_OBSERVER_PLANE))
                    | (scattered & (status != MAX_INTERACTIONS))
                    | (absorbed & (status != ABSORBED))
                    | (np.asarray(result.n_interactions) != valid.astype(int))
                    | (np.asarray(result.n_scatter) != scattered.astype(int))
                    | (np.abs(final_energy + deposited - energy) > 1e-5)
                    | (absorbed & (np.abs(deposited - energy) > 1e-5))
                    | (~absorbed & (deposited != 0))
                    | (valid & ((path < entrance - 1e-3) | (path > exit_ + 1e-3)))
                    | (
                        valid
                        & (
                            np.max(np.abs(event_position - expected_position), axis=1)
                            > 0.004
                        )
                    )
                    | (
                        valid
                        & (
                            np.max(np.abs(incoming - np.asarray(momentum)), axis=1)
                            > 1e-5
                        )
                    )
                    | (
                        absorbed
                        & (
                            np.any(outgoing != 0.0, axis=1)
                            | np.any(final_momentum != 0.0, axis=1)
                        )
                    )
                    | (scattered & (np.abs(outgoing[:, 0] - energy) > 1e-5))
                    | (
                        scattered
                        & (
                            np.abs(np.linalg.norm(outgoing[:, 1:], axis=1) - energy)
                            > 1e-4
                        )
                    )
                )
            )
            completed += current
            chunk_index += 1

        expected = asdict(reference)
        expected.pop("column_cm2")
        expected.pop("location_cdf")
        category_z = {
            name: _z(counts[name], probability, packets_per_ray)
            for name, probability in expected.items()
        }
        events = counts["first_scatter"] + counts["first_absorb"]
        location_z = (
            [
                _z(int(c), p, events)
                for c, p in zip(location_counts, reference.location_cdf, strict=True)
            ]
            if events
            else []
        )
        checks = {
            "column": bool(column_relative_error < 2e-4),
            "counts_close": sum(counts.values()) == packets_per_ray,
            "event_records": invalid_records == 0,
            "categories": all(abs(z) < sigma_limit for z in category_z.values()),
            "locations": all(abs(z) < sigma_limit for z in location_z),
        }
        ray_results.append(
            {
                "angle_rad": angle,
                "reference_column_cm2": reference.column_cm2,
                "integrated_column_cm2": integrated,
                "column_relative_error": column_relative_error,
                "reference_probabilities": expected,
                "counts": counts,
                "category_z": category_z,
                "location_fractions": LOCATION_FRACTIONS,
                "reference_location_cdf": reference.location_cdf,
                "location_counts": location_counts.tolist(),
                "location_z": location_z,
                "invalid_records": invalid_records,
                "checks": checks,
                "all_passed": all(checks.values()),
            }
        )
    return {
        "energy_kev": energy,
        "mode": mode,
        "target_tau_sca_3p3": target_tau_sca_3p3,
        "central_column_cm2": central_column,
        "sigma_sca_cm2_per_h": float(sca[energy_index]),
        "sigma_abs_cm2_per_h": float(absorb[energy_index]),
        "packets_per_ray": packets_per_ray,
        "max_interactions": 1,
        "rays": ray_results,
        "all_passed": all(row["all_passed"] for row in ray_results),
    }
