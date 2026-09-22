"""Stage 9D: absorption-enabled analog histories versus independent flight risks."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from ..geometry.clouds import build_angular_distance_cloud
from ..physics.newdust import build_dust_physics_from_tables
from ..transport.kernel import (
    ABSORBED,
    DUST_SCATTERING,
    INVALID_ENERGY,
    INVALID_STATE,
    MAX_INTERACTIONS,
    PHOTOELECTRIC_ABSORPTION,
    transport_photon_batch,
)
from .multiple_scattering import (
    LOCATION_QUANTILES,
    PHASE_QUANTILES,
    SOURCE_KPC,
    _z,
    analytic_shell_column,
    analytic_world_boundary,
)

CATEGORIES = ("scatter", "absorb", "escape")


def competing_flight_probabilities(column_cm2, sigma_scattering, sigma_absorption):
    """Independent two-process exponential risks with a finite escape path."""

    if min(column_cm2, sigma_scattering, sigma_absorption) < 0:
        raise ValueError("column and cross sections must be nonnegative")
    total = sigma_scattering + sigma_absorption
    if total == 0:
        return {"scatter": 0.0, "absorb": 0.0, "escape": 1.0}
    event = -math.expm1(-column_cm2 * total)
    return {
        "scatter": event * sigma_scattering / total,
        "absorb": event * sigma_absorption / total,
        "escape": math.exp(-column_cm2 * total),
    }


def _host_sigma(energy_grid, values, energy):
    return float(
        math.exp(
            np.interp(
                math.log(energy),
                np.log(energy_grid),
                np.log(values),
            )
        )
    )


def run_absorbing_multiple_scattering_case(
    key,
    scattering,
    absorption,
    *,
    energy_kev: float,
    target_tau_scattering: float,
    packets: int,
    chunk_size: int,
    max_interactions: int = 4,
    sigma_limit: float = 5.0,
    min_expected_category: float = 20.0,
):
    """Compare each sampled flight's three outcomes and interaction position."""

    if (
        packets <= 0
        or chunk_size <= 0
        or max_interactions < 3
        or target_tau_scattering <= 0
        or sigma_limit <= 0
        or min_expected_category < 0
    ):
        raise ValueError("invalid Stage 9D packet, optical-depth, or limit setting")
    energy = float(energy_kev)
    grid = np.asarray(scattering.energy_kev)
    if not np.array_equal(grid, absorption.energy_kev):
        raise ValueError("scattering and absorption grids differ")
    if not grid[0] <= energy <= grid[-1]:
        raise ValueError("energy is outside the registered material range")
    sigma_sca = _host_sigma(grid, scattering.scattering_cross_section_cm2_per_h, energy)
    sigma_abs = _host_sigma(grid, absorption.absorption_cross_section_cm2_per_h, energy)
    sigma_total = sigma_sca + sigma_abs
    central_column = target_tau_scattering / sigma_sca
    cloud = build_angular_distance_cloud(
        np.full((2, 2, 2), central_column / 2.0),
        x_centers_arcsec=[-150_000.0, 150_000.0],
        y_centers_arcsec=[-150_000.0, 150_000.0],
        z_centers_kpc=[4.25, 4.75],
        source_distance_kpc=SOURCE_KPC,
    )
    physics = build_dust_physics_from_tables(scattering, absorption)
    upper = int(np.searchsorted(grid, energy, side="right"))
    upper = min(max(upper, 1), len(grid) - 1)
    lower = upper - 1
    f = (math.log(energy) - math.log(grid[lower])) / (
        math.log(grid[upper]) - math.log(grid[lower])
    )
    phase = (1 - f) * scattering.scattering_angle_cdf[lower] + (
        f * scattering.scattering_angle_cdf[upper]
    )
    phase_angles = np.asarray(scattering.scattering_angle_rad)

    source = np.array([SOURCE_KPC * 1_000, 0.0, 0.0])
    direction = np.array([-1.0, 0.0, 0.0])
    rows = [
        {
            "at_risk": 0,
            "observed": dict.fromkeys(CATEGORIES, 0),
            "expected": dict.fromkeys(CATEGORIES, 0.0),
            "variance": dict.fromkeys(CATEGORIES, 0.0),
            "pit_counts": np.zeros(len(LOCATION_QUANTILES), dtype=np.int64),
            "event_count": 0,
        }
        for _ in range(max_interactions)
    ]
    phases = np.zeros(len(PHASE_QUANTILES), dtype=np.int64)
    total_scatters = 0
    total_events = 0
    invalid_histories = 0
    statuses = np.zeros(7, dtype=np.int64)
    absorbed_after_scattering = 0
    absorbed_after_two_scatters = 0
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
        records = result.interactions
        valid = np.asarray(records.valid)
        kind = np.asarray(records.interaction_type)
        points = np.asarray(records.position_pc, dtype=np.float64)
        incoming = np.asarray(records.incoming_momentum_kev, dtype=np.float64)
        outgoing = np.asarray(records.outgoing_momentum_kev, dtype=np.float64)
        lengths = np.asarray(records.cumulative_path_length_pc)
        scatter_order = np.asarray(records.scattering_order)
        status = np.asarray(result.status)
        ns = np.asarray(result.n_scatter)
        ni = np.asarray(result.n_interactions)
        deposit = np.asarray(result.deposited_energy_kev)
        final_momentum = np.asarray(result.momentum_kev)
        final_energy = final_momentum[:, 0]
        statuses += np.bincount(status, minlength=len(statuses))
        invalid_histories += int(
            np.count_nonzero(
                (ni != valid.sum(axis=1))
                | (ns != (valid & (kind == DUST_SCATTERING)).sum(axis=1))
                | ((status == ABSORBED) != (ni == ns + 1))
                | ((status == MAX_INTERACTIONS) != (ns == max_interactions))
                | (status == INVALID_ENERGY)
                | (status == INVALID_STATE)
                | np.any(~valid & ((kind != 0) | (scatter_order != 0)), axis=1)
                | (np.abs(final_energy + deposit - energy) > 1e-5)
                | ((status == ABSORBED) & (np.abs(deposit - energy) > 1e-5))
                | ((status == ABSORBED) & np.any(final_momentum != 0, axis=1))
                | ((status != ABSORBED) & (deposit != 0))
            )
        )
        for photon in range(n):
            current_position = source
            current_direction = direction
            previous_path = 0.0
            for j, row in enumerate(rows):
                if j and kind[photon, j - 1] != DUST_SCATTERING:
                    break
                boundary, expected_status = analytic_world_boundary(
                    current_position, current_direction
                )
                column = analytic_shell_column(
                    current_position, current_direction, boundary, central_column
                )
                probabilities = competing_flight_probabilities(
                    column, sigma_sca, sigma_abs
                )
                row["at_risk"] += 1
                for category in CATEGORIES:
                    p = probabilities[category]
                    row["expected"][category] += p
                    row["variance"][category] += p * (1 - p)
                if not valid[photon, j]:
                    row["observed"]["escape"] += 1
                    if status[photon] != expected_status:
                        invalid_histories += 1
                    break
                current_kind = kind[photon, j]
                if current_kind == DUST_SCATTERING:
                    category = "scatter"
                elif current_kind == PHOTOELECTRIC_ABSORPTION:
                    category = "absorb"
                else:
                    invalid_histories += 1
                    break
                row["observed"][category] += 1
                row["event_count"] += 1
                total_events += 1
                point = points[photon, j]
                flight = float(np.dot(point - current_position, current_direction))
                event_column = analytic_shell_column(
                    current_position, current_direction, flight, central_column
                )
                interaction_p = probabilities["scatter"] + probabilities["absorb"]
                pit = -math.expm1(-sigma_total * event_column) / max(
                    interaction_p, 1e-100
                )
                row["pit_counts"] += np.asarray(
                    [pit <= q for q in LOCATION_QUANTILES], dtype=np.int64
                )
                input_direction = incoming[photon, j, 1:] / energy
                last_flight = float(lengths[photon, j] - previous_path)
                if (
                    flight < -0.02
                    or flight > boundary + 0.02
                    or abs(last_flight - flight) > 0.03
                    or np.linalg.norm(input_direction - current_direction) > 3e-5
                    or event_column <= 0
                    or pit < -1e-3
                    or pit > 1.001
                    or np.any(valid[photon, j + 1 :])
                    and category == "absorb"
                ):
                    invalid_histories += 1
                if category == "absorb":
                    if (
                        scatter_order[photon, j] != j
                        or status[photon] != ABSORBED
                        or np.any(outgoing[photon, j] != 0)
                    ):
                        invalid_histories += 1
                    absorbed_after_scattering += int(j >= 1)
                    absorbed_after_two_scatters += int(j >= 2)
                    break
                if scatter_order[photon, j] != j + 1 or (
                    j == max_interactions - 1 and status[photon] != MAX_INTERACTIONS
                ):
                    invalid_histories += 1
                outgoing_direction = outgoing[photon, j, 1:] / energy
                theta = math.atan2(
                    float(
                        np.linalg.norm(np.cross(input_direction, outgoing_direction))
                    ),
                    float(np.dot(input_direction, outgoing_direction)),
                )
                phase_u = float(np.interp(theta, phase_angles, phase))
                phases += np.asarray(
                    [phase_u <= q for q in PHASE_QUANTILES], dtype=np.int64
                )
                total_scatters += 1
                if abs(np.linalg.norm(outgoing_direction) - 1.0) > 3e-5:
                    invalid_histories += 1
                current_position = point
                current_direction = outgoing_direction / np.linalg.norm(
                    outgoing_direction
                )
                previous_path = float(lengths[photon, j])
    summary = []
    for j, row in enumerate(rows):
        z = {
            name: _z(
                row["observed"][name], row["expected"][name], row["variance"][name]
            )
            for name in CATEGORIES
        }
        location_z = {
            str(q): _z(
                int(count), row["event_count"] * q, row["event_count"] * q * (1 - q)
            )
            for q, count in zip(LOCATION_QUANTILES, row["pit_counts"], strict=True)
        }
        powered = all(
            row["expected"][name] >= min_expected_category for name in CATEGORIES
        )
        summary.append(
            {
                "flight": j + 1,
                "at_risk": row["at_risk"],
                "observed": row["observed"],
                "expected": row["expected"],
                "category_variance": row["variance"],
                "category_z": z,
                "location_pit_counts": row["pit_counts"].tolist(),
                "location_pit_z": location_z,
                "powered": powered,
                "passed": powered
                and all(abs(value) < sigma_limit for value in z.values())
                and all(abs(value) < sigma_limit for value in location_z.values()),
            }
        )
    phase_z = {
        str(q): _z(int(count), total_scatters * q, total_scatters * q * (1 - q))
        for q, count in zip(PHASE_QUANTILES, phases, strict=True)
    }
    process_z = _z(
        total_scatters,
        total_events * sigma_sca / sigma_total,
        total_events * sigma_sca / sigma_total * sigma_abs / sigma_total,
    )
    checks = {
        "analog_history_records": invalid_histories == 0,
        "all_flight_competing_risks_and_locations": all(
            row["passed"] for row in summary
        ),
        "phase_quantiles": all(abs(value) < sigma_limit for value in phase_z.values()),
        "pooled_process_branching": abs(process_z) < sigma_limit,
        "absorption_after_two_scatters": absorbed_after_two_scatters > 0,
        "no_invalid_terminals": statuses[INVALID_ENERGY] == 0
        and statuses[INVALID_STATE] == 0,
    }
    return {
        "energy_kev": energy,
        "target_tau_scattering": target_tau_scattering,
        "central_column_cm2": central_column,
        "sigma_scattering_cm2_per_h": sigma_sca,
        "sigma_absorption_cm2_per_h": sigma_abs,
        "packets": packets,
        "max_interactions": max_interactions,
        "flight_rows": summary,
        "scatter_count": total_scatters,
        "absorption_after_scattering": absorbed_after_scattering,
        "absorption_after_two_scatters": absorbed_after_two_scatters,
        "phase_quantile_counts": phases.tolist(),
        "phase_quantile_z": phase_z,
        "process_branching_z": process_z,
        "invalid_histories": invalid_histories,
        "status_counts": statuses.tolist(),
        "checks": {key: bool(value) for key, value in checks.items()},
        "all_passed": all(checks.values()),
    }
