"""Stage 9C: pure-scattering flight hazards against independent sphere chords.

Conditional on each *actual* sampled scattering position and outgoing direction,
the next flight must have an exponential column-depth budget. The reference
calculates the intervening column with NumPy sphere chords and a separate
observer-plane/outer-sphere intersection; it does not call the ray integrator.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from ..geometry.clouds import build_angular_distance_cloud
from ..physics.dust import build_dust_physics_table
from ..transport.kernel import (
    ABSORBED,
    DUST_SCATTERING,
    ESCAPED_OUTER_BOUNDARY,
    INVALID_ENERGY,
    INVALID_STATE,
    MAX_INTERACTIONS,
    REACHED_OBSERVER_PLANE,
    transport_photon_batch,
)

SOURCE_KPC = 10.0
INNER_KPC = 4.0
OUTER_KPC = 5.0
PC_TO_CM = 3.0856775814913673e18
PHASE_QUANTILES = (0.5, 0.9)
LOCATION_QUANTILES = (0.25, 0.5, 0.75)


def _inside_sphere_length(origin, direction, radius_pc, maximum_pc):
    """Exact length of the positive ray segment inside a centered sphere."""

    projection = float(np.dot(origin, direction))
    discriminant = projection**2 + radius_pc**2 - float(np.dot(origin, origin))
    if discriminant <= 0:
        return 0.0
    root = math.sqrt(discriminant)
    left = max(0.0, -projection - root)
    right = min(maximum_pc, -projection + root)
    return max(0.0, right - left)


def analytic_shell_column(origin_pc, direction, maximum_pc, central_column_cm2):
    """Independent NumPy uniform-shell column along a finite straight flight."""

    origin = np.asarray(origin_pc, dtype=np.float64)
    unit = np.asarray(direction, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    length = _inside_sphere_length(origin, unit, OUTER_KPC * 1_000, maximum_pc)
    length -= _inside_sphere_length(origin, unit, INNER_KPC * 1_000, maximum_pc)
    return max(0.0, length) * central_column_cm2 / 1_000.0


def analytic_world_boundary(origin_pc, direction):
    """Distance to the observer plane or source-radius outer sphere."""

    origin = np.asarray(origin_pc, dtype=np.float64)
    unit = np.asarray(direction, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    projection = float(np.dot(origin, unit))
    discriminant = (
        projection**2 + (SOURCE_KPC * 1_000) ** 2 - float(np.dot(origin, origin))
    )
    sphere_distance = -projection + math.sqrt(max(0.0, discriminant))
    observer_distance = -origin[0] / unit[0] if unit[0] < 0 else math.inf
    if observer_distance <= sphere_distance:
        return observer_distance, REACHED_OBSERVER_PLANE
    return sphere_distance, ESCAPED_OUTER_BOUNDARY


def _z(observed, expected, variance):
    if variance <= 0:
        return 0.0 if abs(observed - expected) < 1e-9 else math.inf
    return (observed - expected) / math.sqrt(variance)


def _model(scattering, energy_kev, central_column_cm2):
    energy_grid = np.asarray(scattering.energy_kev)
    if not energy_grid[0] <= energy_kev <= energy_grid[-1]:
        raise ValueError("energy is outside the scattering table")
    sigma = float(
        np.exp(
            np.interp(
                math.log(energy_kev),
                np.log(energy_grid),
                np.log(scattering.scattering_cross_section_cm2_per_h),
            )
        )
    )
    upper = int(np.searchsorted(energy_grid, energy_kev, side="right"))
    upper = min(max(upper, 1), len(energy_grid) - 1)
    lower = upper - 1
    fraction = (math.log(energy_kev) - math.log(energy_grid[lower])) / (
        math.log(energy_grid[upper]) - math.log(energy_grid[lower])
    )
    phase = (1 - fraction) * scattering.scattering_angle_cdf[lower] + (
        fraction * scattering.scattering_angle_cdf[upper]
    )
    # The angular domain includes the entire physically populated phase tail.
    cloud = build_angular_distance_cloud(
        np.full((2, 2, 2), central_column_cm2 / 2.0),
        x_centers_arcsec=[-150_000.0, 150_000.0],
        y_centers_arcsec=[-150_000.0, 150_000.0],
        z_centers_kpc=[4.25, 4.75],
        source_distance_kpc=SOURCE_KPC,
    )
    physics = build_dust_physics_table(
        scattering.energy_kev,
        scattering.scattering_cross_section_cm2_per_h,
        np.zeros_like(scattering.energy_kev),
        scattering.scattering_angle_rad,
        scattering.scattering_angle_cdf,
    )
    return sigma, phase, cloud, physics


def run_multiple_scattering_case(
    key,
    scattering,
    *,
    energy_kev: float,
    target_tau: float,
    packets: int,
    chunk_size: int,
    max_interactions: int = 4,
    sigma_limit: float = 5.0,
    min_expected_events: int = 20,
):
    """Validate up to four analog scattering flights at actual table opacity."""

    if (
        packets <= 0
        or chunk_size <= 0
        or max_interactions < 3
        or target_tau <= 0
        or sigma_limit <= 0
        or min_expected_events < 0
    ):
        raise ValueError("invalid Stage 9C packet, optical-depth, or limit setting")
    energy = float(energy_kev)
    grid = np.asarray(scattering.energy_kev)
    if not grid[0] <= energy <= grid[-1]:
        raise ValueError("energy is outside the scattering table")
    sigma = float(
        np.exp(
            np.interp(
                math.log(energy),
                np.log(grid),
                np.log(scattering.scattering_cross_section_cm2_per_h),
            )
        )
    )
    central_column = target_tau / sigma
    sigma, phase, cloud, physics = _model(scattering, energy, central_column)
    phase_angles = np.asarray(scattering.scattering_angle_rad)
    source = np.array([SOURCE_KPC * 1_000, 0.0, 0.0])
    direction = np.array([-1.0, 0.0, 0.0])
    orders = [
        {
            "at_risk": 0,
            "observed": 0,
            "expected": 0.0,
            "variance": 0.0,
            "pit_counts": np.zeros(len(LOCATION_QUANTILES), dtype=np.int64),
        }
        for _ in range(max_interactions)
    ]
    phase_counts = np.zeros(len(PHASE_QUANTILES), dtype=np.int64)
    scatter_count = 0
    invalid_histories = 0
    statuses = np.zeros(7, dtype=np.int64)
    transport = jax.jit(transport_photon_batch, static_argnames=("max_interactions",))
    for offset in range(0, packets, chunk_size):
        n = min(chunk_size, packets - offset)
        positions = jnp.broadcast_to(jnp.asarray(source, dtype=jnp.float32), (n, 3))
        four_momentum = jnp.asarray([energy, -energy, 0.0, 0.0], dtype=jnp.float32)
        momenta = jnp.broadcast_to(four_momentum, (n, 4))
        result = transport(
            random.fold_in(key, offset // chunk_size),
            positions,
            momenta,
            cloud,
            physics,
            max_interactions=max_interactions,
        )
        valid = np.asarray(result.interactions.valid)
        kinds = np.asarray(result.interactions.interaction_type)
        points = np.asarray(result.interactions.position_pc, dtype=np.float64)
        incoming = np.asarray(
            result.interactions.incoming_momentum_kev, dtype=np.float64
        )
        outgoing = np.asarray(
            result.interactions.outgoing_momentum_kev, dtype=np.float64
        )
        lengths = np.asarray(result.interactions.cumulative_path_length_pc)
        recorded_order = np.asarray(result.interactions.scattering_order)
        ns = np.asarray(result.n_scatter)
        ni = np.asarray(result.n_interactions)
        status = np.asarray(result.status)
        statuses += np.bincount(status, minlength=len(statuses))
        invalid_histories += int(
            np.count_nonzero(
                (ni != ns)
                | (ns != valid.sum(axis=1))
                | (status == ABSORBED)
                | (status == INVALID_ENERGY)
                | (status == INVALID_STATE)
                | ((status == MAX_INTERACTIONS) != (ns == max_interactions))
                | np.any(valid & (kinds != DUST_SCATTERING), axis=1)
                | np.any(~valid & (kinds != 0), axis=1)
                | np.any(
                    valid
                    & (recorded_order != np.arange(1, max_interactions + 1)[None, :]),
                    axis=1,
                )
                | np.any(~valid & (recorded_order != 0), axis=1)
                | (np.abs(np.asarray(result.momentum_kev)[:, 0] - energy) > 1e-5)
                | (np.asarray(result.deposited_energy_kev) != 0)
            )
        )
        for photon in range(n):
            current_position = source
            current_direction = direction
            previous_path = 0.0
            for j, order in enumerate(orders):
                # No subsequent flight exists after a terminal observer/outer exit.
                if j and not valid[photon, j - 1]:
                    break
                boundary, expected_status = analytic_world_boundary(
                    current_position, current_direction
                )
                column = analytic_shell_column(
                    current_position, current_direction, boundary, central_column
                )
                p = -math.expm1(-sigma * column)
                order["at_risk"] += 1
                order["expected"] += p
                order["variance"] += p * (1.0 - p)
                if not valid[photon, j]:
                    if status[photon] != expected_status:
                        invalid_histories += 1
                    break
                order["observed"] += 1
                point = points[photon, j]
                flight = float(np.dot(point - current_position, current_direction))
                event_column = analytic_shell_column(
                    current_position, current_direction, flight, central_column
                )
                pit = -math.expm1(-sigma * event_column) / max(p, 1e-100)
                order["pit_counts"] += np.asarray(
                    [pit <= q for q in LOCATION_QUANTILES], dtype=np.int64
                )
                incoming_direction = incoming[photon, j, 1:] / energy
                outgoing_direction = outgoing[photon, j, 1:] / energy
                theta = math.atan2(
                    float(
                        np.linalg.norm(np.cross(incoming_direction, outgoing_direction))
                    ),
                    float(np.dot(incoming_direction, outgoing_direction)),
                )
                phase_u = float(np.interp(theta, phase_angles, phase))
                phase_counts += np.asarray(
                    [phase_u <= q for q in PHASE_QUANTILES], dtype=np.int64
                )
                scatter_count += 1
                previous_flight = float(lengths[photon, j] - previous_path)
                if (
                    flight < -0.02
                    or flight > boundary + 0.02
                    or abs(previous_flight - flight) > 0.03
                    or np.linalg.norm(incoming_direction - current_direction) > 3e-5
                    or abs(np.linalg.norm(outgoing_direction) - 1.0) > 3e-5
                    or event_column <= 0
                    or pit < -1e-3
                    or pit > 1.001
                ):
                    invalid_histories += 1
                current_position = point
                current_direction = outgoing_direction / np.linalg.norm(
                    outgoing_direction
                )
                previous_path = float(lengths[photon, j])
    summary = []
    for j, order in enumerate(orders):
        z = _z(order["observed"], order["expected"], order["variance"])
        pit_z = {
            str(q): _z(
                int(count), order["observed"] * q, order["observed"] * q * (1 - q)
            )
            for q, count in zip(LOCATION_QUANTILES, order["pit_counts"], strict=True)
        }
        summary.append(
            {
                "flight": j + 1,
                "at_risk": order["at_risk"],
                "observed_scatters": order["observed"],
                "expected_scatters": order["expected"],
                "hazard_z": z,
                "location_pit_counts": order["pit_counts"].tolist(),
                "location_pit_z": pit_z,
                "powered": order["expected"] >= min_expected_events,
                "passed": (
                    order["expected"] >= min_expected_events
                    and abs(z) < sigma_limit
                    and all(abs(value) < sigma_limit for value in pit_z.values())
                ),
            }
        )
    phase_z = {
        str(q): _z(int(count), scatter_count * q, scatter_count * q * (1 - q))
        for q, count in zip(PHASE_QUANTILES, phase_counts, strict=True)
    }
    checks = {
        "analog_history_records": invalid_histories == 0,
        "all_flight_hazards_and_locations": all(row["passed"] for row in summary),
        "phase_quantiles": all(abs(value) < sigma_limit for value in phase_z.values()),
        "no_absorption": bool(statuses[ABSORBED] == 0),
        "has_repeated_scattering": summary[2]["observed_scatters"] > 0,
    }
    return {
        "energy_kev": energy,
        "target_tau_scattering": target_tau,
        "central_column_cm2": central_column,
        "sigma_scattering_cm2_per_h": sigma,
        "packets": packets,
        "max_interactions": max_interactions,
        "flight_rows": summary,
        "phase_quantile_counts": phase_counts.tolist(),
        "phase_quantile_z": phase_z,
        "scatter_count": scatter_count,
        "invalid_histories": invalid_histories,
        "status_counts": statuses.tolist(),
        "checks": checks,
        "all_passed": all(checks.values()),
    }
