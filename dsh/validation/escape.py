"""Stage 9B: independent radial-column references for virtual observer rays.

Controlled scattering histories are kept fixed while foreground columns and
absorption are changed. The reference observer ray travels from an event
radially inward through spherical shell cells at a fixed sky angle. Its column
is the sum of complete near shells plus the occupied fraction of the event
shell, calculated with NumPy rather than the production ray integrator.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry.clouds import build_angular_distance_cloud
from ..observer.scoring import score_peeloff_events
from ..sources.launch import LaunchedSourcePackets
from ..transport.kernel import (
    ABSORBED,
    DUST_SCATTERING,
    PHOTOELECTRIC_ABSORPTION,
    PhotonInteractionRecord,
    PhotonTransportResult,
)

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
SOURCE_DISTANCE_PC = 10_000.0
RADIAL_COLUMNS_CM2 = np.array([5.0e20, 8.0e20, 1.1e21, 0.0])
FOREGROUND_INCREMENT_CM2 = 1.0e22
BEHIND_INCREMENT_CM2 = 1.0e22
COLUMN_TOLERANCE = 8.0e-6
WEIGHT_TOLERANCE = 2.0e-4
RATIO_TOLERANCE = 2.0e-4


def controlled_columns(kind: str) -> np.ndarray:
    """Make four two-kpc shells and spatially distinct foreground sightlines."""
    if kind not in {"vacuum", "base", "foreground", "behind"}:
        raise ValueError("unknown controlled cloud kind")
    columns = np.zeros((4, 3, 3), dtype=np.float64)
    if kind == "vacuum":
        return columns
    for y in range(3):
        for x in range(3):
            columns[:, y, x] = RADIAL_COLUMNS_CM2 * (1.0 + 0.25 * (x + y))
    if kind == "foreground":
        columns[0, 0, 0] += FOREGROUND_INCREMENT_CM2
    if kind == "behind":
        columns[3, :, :] += BEHIND_INCREMENT_CM2
    return columns


def controlled_cloud(columns: np.ndarray):
    return build_angular_distance_cloud(
        columns,
        x_centers_arcsec=[-120.0, 0.0, 120.0],
        y_centers_arcsec=[-120.0, 0.0, 120.0],
        z_centers_kpc=[1.0, 3.0, 5.0, 7.0],
        source_distance_kpc=SOURCE_DISTANCE_PC / 1_000.0,
    )


def radial_reference_column_cm2(
    event_position_pc: np.ndarray,
    radial_columns_cm2: np.ndarray,
    *,
    x_edges_arcsec: np.ndarray,
    y_edges_arcsec: np.ndarray,
    z_edges_kpc: np.ndarray,
) -> float:
    """Analytic observer-ray column; no production ray integration is called."""
    position = np.asarray(event_position_pc, dtype=np.float64)
    if position.shape != (3,) or position[0] <= 0 or not np.all(np.isfinite(position)):
        raise ValueError("event must be a finite positive-LOS three-vector")
    radius = float(np.linalg.norm(position) / 1_000.0)
    sky_x = math.atan2(position[1], position[0]) / ARCSEC_TO_RAD
    sky_y = math.atan2(position[2], position[0]) / ARCSEC_TO_RAD
    x_edges = np.asarray(x_edges_arcsec, dtype=np.float64)
    y_edges = np.asarray(y_edges_arcsec, dtype=np.float64)
    z_edges = np.asarray(z_edges_kpc, dtype=np.float64)
    columns = np.asarray(radial_columns_cm2, dtype=np.float64)
    if columns.shape != (z_edges.size - 1, y_edges.size - 1, x_edges.size - 1):
        raise ValueError("radial columns do not match sky and shell edges")
    x = int(np.searchsorted(x_edges, sky_x, side="right") - 1)
    y = int(np.searchsorted(y_edges, sky_y, side="right") - 1)
    if (
        not 0 <= x < columns.shape[2]
        or not 0 <= y < columns.shape[1]
        or not 0 <= radius <= z_edges[-1]
    ):
        raise ValueError("event lies outside the controlled cloud")
    occupied_fraction = np.clip((radius - z_edges[:-1]) / np.diff(z_edges), 0.0, 1.0)
    return float(np.dot(columns[:, y, x], occupied_fraction))


def _source_to_event_position(radius_kpc: float, x_arcsec: float, y_arcsec: float):
    """Construct a known off-axis radius without calling the production helper."""
    slopes = np.tan(np.array([x_arcsec, y_arcsec]) * ARCSEC_TO_RAD)
    direction = np.r_[1.0, slopes]
    return 1_000.0 * radius_kpc * direction / np.linalg.norm(direction)


def controlled_history(energy_kev: float):
    """Two fixed scattering events followed by absorption in each of two rays."""
    energy = float(energy_kev)
    if not math.isfinite(energy) or energy <= 0.0:
        raise ValueError("energy must be finite and positive")
    positions = np.asarray(
        [
            [
                _source_to_event_position(5.0, -100.0, -100.0),
                _source_to_event_position(3.0, 100.0, 100.0),
                _source_to_event_position(1.0, 100.0, 100.0),
            ],
            [
                _source_to_event_position(5.0, 100.0, 100.0),
                _source_to_event_position(3.0, -100.0, -100.0),
                _source_to_event_position(1.0, -100.0, -100.0),
            ],
        ],
        dtype=np.float32,
    )
    source_position = np.asarray([SOURCE_DISTANCE_PC, 0.0, 0.0], dtype=np.float32)
    starts = np.concatenate(
        (np.broadcast_to(source_position, (2, 1, 3)), positions[:, :-1]), axis=1
    )
    segment = positions - starts
    distance = np.linalg.norm(segment, axis=-1)
    direction = segment / distance[..., None]
    momenta = np.concatenate(
        (np.full((2, 3, 1), energy, dtype=np.float32), energy * direction), axis=-1
    )
    outgoing = np.zeros_like(momenta)
    outgoing[:, :2] = momenta[:, 1:]
    excess = np.cumsum(distance * (1.0 + direction[..., 0]), axis=1)
    records = PhotonInteractionRecord(
        valid=jnp.ones((2, 3), dtype=bool),
        interaction_type=jnp.asarray(
            np.broadcast_to(
                [DUST_SCATTERING, DUST_SCATTERING, PHOTOELECTRIC_ABSORPTION],
                (2, 3),
            ),
            dtype=jnp.int32,
        ),
        position_pc=jnp.asarray(positions),
        incoming_momentum_kev=jnp.asarray(momenta),
        outgoing_momentum_kev=jnp.asarray(outgoing),
        cumulative_path_length_pc=jnp.asarray(np.cumsum(distance, axis=1)),
        cumulative_excess_path_length_pc=jnp.asarray(excess),
        scattering_order=jnp.asarray([[1, 2, 2], [1, 2, 2]], dtype=jnp.int32),
    )
    launched = LaunchedSourcePackets(
        position_pc=jnp.asarray(np.broadcast_to(source_position, (2, 3))),
        momentum_kev=jnp.asarray(momenta[:, 0]),
        launch_pdf_per_sr=jnp.asarray([2.0e5, 5.0e5], dtype=jnp.float32),
        isotropic_importance=jnp.asarray(
            [1.0 / (4.0 * np.pi * 2.0e5), 1.0 / (4.0 * np.pi * 5.0e5)],
            dtype=jnp.float32,
        ),
        emission_time_s=jnp.asarray([0.0, 100.0], dtype=jnp.float32),
        weight_observer_fluence=jnp.asarray([1.0, 0.7], dtype=jnp.float32),
        time_index=jnp.asarray([0, 1], dtype=jnp.int32),
        spectral_bin_index=jnp.asarray([0, 1], dtype=jnp.int32),
    )
    transported = PhotonTransportResult(
        position_pc=jnp.asarray(positions[:, -1]),
        momentum_kev=jnp.zeros((2, 4), dtype=jnp.float32),
        path_length_pc=jnp.asarray(np.sum(distance, axis=1)),
        excess_path_length_pc=jnp.asarray(excess[:, -1]),
        deposited_energy_kev=jnp.full((2,), energy, dtype=jnp.float32),
        n_interactions=jnp.full((2,), 3, dtype=jnp.int32),
        n_scatter=jnp.full((2,), 2, dtype=jnp.int32),
        status=jnp.full((2,), ABSORBED, dtype=jnp.int32),
        interactions=records,
    )
    return launched, transported


def _interpolated_cross_section(energy_kev, node_energies, node_cross_sections):
    """Independent NumPy log-log opacity interpolation for the validation."""
    energy = np.asarray(node_energies, dtype=np.float64)
    values = np.asarray(node_cross_sections, dtype=np.float64)
    if not energy[0] <= energy_kev <= energy[-1] or np.any(values <= 0.0):
        raise ValueError("reference energy or opacity is unsupported")
    return float(np.exp(np.interp(np.log(energy_kev), np.log(energy), np.log(values))))


def _max_relative_error(actual, expected):
    observed = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(expected, dtype=np.float64)
    if np.any(predicted <= 0) or not np.all(np.isfinite(observed)):
        return float("inf")
    return float(np.max(np.abs(observed / predicted - 1.0)))


def run_escape_attenuation_case(scattering, absorption, energy_kev: float) -> dict:
    """Test full observer weights, fixed-history ratios and masked absorption."""
    if not np.array_equal(scattering.energy_kev, absorption.energy_kev):
        raise ValueError("material energy grids differ")
    from ..physics.newdust import build_dust_physics_from_tables

    physics = build_dust_physics_from_tables(scattering, absorption)
    no_absorption_physics = physics._replace(
        absorption_cross_section_cm2_per_h=jnp.zeros_like(
            physics.absorption_cross_section_cm2_per_h
        )
    )
    launched, transported = controlled_history(energy_kev)
    scorer = jax.jit(score_peeloff_events)
    scattering_sigma = _interpolated_cross_section(
        energy_kev,
        scattering.energy_kev,
        scattering.scattering_cross_section_cm2_per_h,
    )
    absorption_sigma = _interpolated_cross_section(
        energy_kev,
        absorption.energy_kev,
        absorption.absorption_cross_section_cm2_per_h,
    )
    results = {}
    diagnostics = []
    for kind in ("vacuum", "base", "foreground", "behind"):
        columns = controlled_columns(kind)
        cloud = controlled_cloud(columns)
        for absorption_on, selected_physics in (
            (False, no_absorption_physics),
            (True, physics),
        ):
            events = scorer(launched, transported, cloud, selected_physics)
            label = f"{kind}_{'both' if absorption_on else 'scattering_only'}"
            valid = np.asarray(events.valid)
            order = np.asarray(events.scattering_order)
            observed_column = np.asarray(events.escape_column_cm2)[:, :2]
            position = np.asarray(transported.interactions.position_pc)[:, :2]
            reference_column = np.asarray(
                [
                    [
                        radial_reference_column_cm2(
                            point,
                            columns,
                            x_edges_arcsec=np.asarray(cloud.x_edges_arcsec),
                            y_edges_arcsec=np.asarray(cloud.y_edges_arcsec),
                            z_edges_kpc=np.asarray(cloud.z_edges_kpc),
                        )
                        for point in packet
                    ]
                    for packet in position
                ]
            )
            reference_tau = reference_column * (
                scattering_sigma + (absorption_sigma if absorption_on else 0.0)
            )
            reference_transmission = np.exp(-reference_tau)
            observed_weight = np.asarray(events.weight_observer_fluence)[:, :2]
            # Phase is an unchanged nuisance here: the independent column and
            # transmission law are tested in the absolute weight and ratios.
            phase = np.asarray(events.phase_pdf_per_sr)[:, :2]
            packet_weight = np.asarray(launched.weight_observer_fluence)[:, None]
            launch_pdf = np.asarray(launched.launch_pdf_per_sr)[:, None]
            event_radius_pc = np.linalg.norm(position.astype(np.float64), axis=-1)
            unattenuated_weight = (
                packet_weight
                * (SOURCE_DISTANCE_PC / event_radius_pc) ** 2
                * phase
                / launch_pdf
            )
            expected_weight = unattenuated_weight * reference_transmission
            errors = {
                "column_relative": _max_relative_error(
                    observed_column + 1.0e-30, reference_column + 1.0e-30
                ),
                "optical_depth_absolute": float(
                    np.max(
                        np.abs(
                            np.asarray(events.escape_optical_depth)[:, :2]
                            - reference_tau
                        )
                    )
                ),
                "transmission_relative": _max_relative_error(
                    np.asarray(events.transmission)[:, :2], reference_transmission
                ),
                "weight_relative": _max_relative_error(
                    observed_weight, expected_weight
                ),
            }
            checks = {
                "valid_only_scatters": bool(
                    np.array_equal(valid, [[True, True, False], [True, True, False]])
                    and np.array_equal(order, [[1, 2, 0], [1, 2, 0]])
                ),
                "absorption_slot_masked": bool(
                    np.all(np.asarray(events.weight_observer_fluence)[:, 2] == 0.0)
                    and np.all(np.asarray(events.transmission)[:, 2] == 0.0)
                ),
                "analytic_column": errors["column_relative"] < COLUMN_TOLERANCE,
                "analytic_transmission": (
                    errors["optical_depth_absolute"] < 5.0e-6
                    and errors["transmission_relative"] < WEIGHT_TOLERANCE
                ),
                "absolute_escape_weight": errors["weight_relative"] < WEIGHT_TOLERANCE,
            }
            diagnostics.append(
                {
                    "configuration": label,
                    "columns_cm2": reference_column.tolist(),
                    "errors": errors,
                    "checks": checks,
                }
            )
            results[label] = (observed_weight, reference_column, phase)

    paired_checks = {}
    for absorption_on in (False, True):
        tag = "both" if absorption_on else "scattering_only"
        baseline_weight, baseline_column, baseline_phase = results[f"base_{tag}"]
        foreground_weight, foreground_column, foreground_phase = results[
            f"foreground_{tag}"
        ]
        behind_weight, behind_column, behind_phase = results[f"behind_{tag}"]
        extra_column = foreground_column - baseline_column
        expected_foreground_ratio = np.exp(
            -extra_column
            * (scattering_sigma + (absorption_sigma if absorption_on else 0.0))
        )
        paired_checks[f"foreground_ratio_{tag}"] = _max_relative_error(
            foreground_weight / baseline_weight, expected_foreground_ratio
        )
        paired_checks[f"behind_source_path_{tag}"] = _max_relative_error(
            behind_weight, baseline_weight
        )
        paired_checks[f"unchanged_phase_{tag}"] = bool(
            np.array_equal(foreground_phase, baseline_phase)
            and np.array_equal(behind_phase, baseline_phase)
            and np.array_equal(behind_column, baseline_column)
        )
    for kind in ("base", "foreground"):
        full_weight, column, _ = results[f"{kind}_both"]
        scatter_weight, scatter_column, _ = results[f"{kind}_scattering_only"]
        paired_checks[f"absorption_ratio_{kind}"] = _max_relative_error(
            full_weight / scatter_weight,
            np.exp(-column * absorption_sigma),
        )
        paired_checks[f"same_histories_{kind}"] = bool(
            np.array_equal(column, scatter_column)
        )

    passed_ratios = all(
        value if isinstance(value, bool) else value < RATIO_TOLERANCE
        for value in paired_checks.values()
    )
    return {
        "energy_kev": float(energy_kev),
        "sigma_scattering_cm2_per_h": scattering_sigma,
        "sigma_absorption_cm2_per_h": absorption_sigma,
        "scenarios": diagnostics,
        "paired_checks": paired_checks,
        "all_passed": passed_ratios
        and all(all(entry["checks"].values()) for entry in diagnostics),
    }
