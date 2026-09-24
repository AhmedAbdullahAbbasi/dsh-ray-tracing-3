"""Stage 9F asymmetric-cloud observer fluence and absorbed multiple scattering.

Run at a fixed energy with the bundled materials. The four first-order sky
quadrants have an independent absolute quadrature reference; scattering-only
histories supply a separate, explicitly absorbed repeated-order reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from dsh.geometry.clouds import build_angular_distance_cloud
from dsh.observer.binning import bin_observer_events, build_observer_bin_geometry
from dsh.observer.scoring import score_peeloff_events
from dsh.physics.materials import (
    DEFAULT_2_10_ABSORPTION,
    DEFAULT_2_10_SCATTERING,
    load_2_10_material_tables,
)
from dsh.sources.launch import build_rectangular_launch_geometry, sample_source_launches
from dsh.sources.models import SourcePackets
from dsh.transport.kernel import transport_photon_batch
from dsh.validation.absorbed_observer import host_material
from dsh.validation.heterogeneous_observer import (
    LAUNCH_BOUNDS,
    RADIAL_EDGES_KPC,
    SKY_EDGES,
    TIME_EDGES_S,
    first_order_quadrants,
    host_ray_column,
    reference_history_weights,
    scene_columns,
)


def _git(*args):
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _history_weights(events):
    """Build independent (photon, y, x, order) moments, including zero scores."""
    valid = np.asarray(events.valid)
    x = np.asarray(events.sky_x_arcsec, dtype=np.float64)
    y = np.asarray(events.sky_y_arcsec, dtype=np.float64)
    t = np.asarray(events.arrival_time_s, dtype=np.float64)
    w = np.asarray(events.weight_observer_fluence, dtype=np.float64)
    order = np.asarray(events.scattering_order)
    selected = valid & (t >= TIME_EDGES_S[0]) & (t <= TIME_EDGES_S[-1])
    history = np.zeros((len(valid), 2, 2, 3), dtype=np.float64)
    for iy in range(2):
        for ix in range(2):
            spatial = selected & (x >= SKY_EDGES[ix]) & (y >= SKY_EDGES[iy])
            spatial &= x <= SKY_EDGES[ix + 1] if ix == 1 else x < SKY_EDGES[ix + 1]
            spatial &= y <= SKY_EDGES[iy + 1] if iy == 1 else y < SKY_EDGES[iy + 1]
            for group in range(3):
                history[:, iy, ix, group] = np.where(
                    spatial & ((order == group + 1) if group < 2 else (order >= 3)),
                    w, 0.0,
                ).sum(axis=1)
    return history


def _moment_row(
    analog, reference, a_q, b_q, n, *, sigma_limit,
    minimum_effective_histories, maximum_relative_error, quadrature_error=0.0,
):
    a_var = n / (n - 1) * max(a_q - analog**2 / n, 0.0)
    b_var = n / (n - 1) * max(b_q - reference**2 / n, 0.0)
    se_a, se_b = math.sqrt(a_var), math.sqrt(b_var)
    total_error = math.hypot(math.hypot(se_a, se_b), quadrature_error)
    z = (analog - reference) / total_error if total_error > 0 else None
    effective_a = analog**2 / a_q if a_q > 0 else 0.0
    effective_b = reference**2 / b_q if b_q > 0 else 0.0
    relative_a = se_a / analog if analog > 0 else None
    relative_b = se_b / reference if b_q > 0 and reference > 0 else None
    quadrature_relative = quadrature_error / max(reference, 1e-100)
    passed = bool(
        z is not None and abs(z) <= sigma_limit
        and effective_a >= minimum_effective_histories
        and (effective_b >= minimum_effective_histories if b_q > 0 else True)
        and relative_a is not None and relative_a <= maximum_relative_error
        and (relative_b is not None and relative_b <= maximum_relative_error
             if b_q > 0 else True)
        and quadrature_relative <= 0.02
    )
    return {
        "analog": float(analog), "reference": float(reference),
        "analog_photon_standard_error": se_a,
        "reference_photon_standard_error": se_b if b_q > 0 else None,
        "analog_effective_histories": float(effective_a),
        "reference_effective_histories": float(effective_b) if b_q > 0 else None,
        "analog_relative_standard_error": relative_a,
        "reference_relative_standard_error": relative_b,
        "quadrature_relative_change": quadrature_relative,
        "z": float(z) if z is not None else None, "passed": passed,
    }


def _summed_moment(s, q, mask):
    flat = s.reshape(-1)
    idx = np.flatnonzero(mask.reshape(-1))
    return float(flat[idx].sum()), float(q[np.ix_(idx, idx)].sum())


def run_case(physics, *, energy, target_tau, packets, chunk_size, seeds,
             max_interactions, sigma_limit, minimum_effective_histories,
             maximum_relative_error):
    _, _, _, sigma_scattering, _ = host_material(physics, energy)
    columns = scene_columns(target_tau / sigma_scattering)
    cloud = build_angular_distance_cloud(
        columns, [-900.0, 900.0], [-900.0, 900.0],
        (RADIAL_EDGES_KPC[:-1] + RADIAL_EDGES_KPC[1:]) / 2, 10.0,
    )
    launch = build_rectangular_launch_geometry(10.0, *LAUNCH_BOUNDS)
    bins = build_observer_bin_geometry(
        SKY_EDGES, SKY_EDGES, [energy - 0.1, energy + 0.1], TIME_EDGES_S,
    )
    pure = physics._replace(
        absorption_cross_section_cm2_per_h=jnp.zeros_like(
            physics.absorption_cross_section_cm2_per_h
        )
    )
    print("Computing coarse independent quadrant quadrature...", flush=True)
    coarse = first_order_quadrants(
        physics, energy, columns, n_slope=18, n_depth=16,
    )
    print("Computing fine independent quadrant quadrature...", flush=True)
    fine = first_order_quadrants(
        physics, energy, columns, n_slope=42, n_depth=36,
    )

    def analog_batch(key, source):
        launch_key, transport_key = random.split(key)
        launched = sample_source_launches(launch_key, source, launch)
        transported = transport_photon_batch(
            transport_key, launched.position_pc, launched.momentum_kev,
            cloud, physics, max_interactions=max_interactions,
        )
        events = score_peeloff_events(launched, transported, cloud, physics)
        product = bin_observer_events(events, bins)
        return (events, transported.status, transported.interactions.position_pc,
                product.first_scatter_fluence, product.multiple_scatter_fluence)

    analog_jit = jax.jit(analog_batch)
    pure_jit = jax.jit(
        lambda key, launched: transport_photon_batch(
            key, launched.position_pc, launched.momentum_kev,
            cloud, pure, max_interactions=max_interactions,
        )
    )
    a_sum = np.zeros((2, 2, 3), dtype=np.float64)
    b_sum = np.zeros_like(a_sum)
    a_q = np.zeros((12, 12), dtype=np.float64)
    b_q = np.zeros_like(a_q)
    statuses_a = np.zeros(7, dtype=np.int64)
    statuses_b = np.zeros(7, dtype=np.int64)
    column_max_relative_error = 0.0
    column_samples = 0
    multiple_column_samples = 0
    per_seed = []
    for seed in seeds:
        seed_a = np.zeros_like(a_sum)
        seed_b = np.zeros_like(b_sum)
        for index, offset in enumerate(range(0, packets, chunk_size)):
            n = min(chunk_size, packets - offset)
            source = SourcePackets(
                energy_kev=jnp.full((n,), energy),
                emission_time_s=jnp.zeros((n,)),
                weight_observer_fluence=jnp.full((n,), 1.0 / packets),
                time_index=jnp.zeros((n,), dtype=jnp.int32),
                spectral_bin_index=jnp.zeros((n,), dtype=jnp.int32),
            )
            analog_key, launch_key, pure_key = random.split(
                random.fold_in(random.PRNGKey(seed), index), 3
            )
            events, status, positions, first, multiple = jax.block_until_ready(
                analog_jit(analog_key, source)
            )
            a_history = _history_weights(events)
            first_image = np.asarray(first, dtype=np.float64)[0, 0]
            multiple_image = np.asarray(multiple, dtype=np.float64)[0, 0]
            if not np.allclose(
                a_history[..., 0].sum(axis=0), first_image,
                rtol=3e-4, atol=1e-10,
            ):
                raise RuntimeError(
                    "first-order quadrant scores do not close to production image"
                )
            if not np.allclose(
                a_history[..., 1:].sum(axis=(0, 3)), multiple_image,
                rtol=3e-4, atol=1e-10,
            ):
                raise RuntimeError(
                    "multiple-order quadrant scores do not close to production image"
                )
            # Probe scorer escape columns against independent host intersections.
            # Take the first valid event from the first few chunks of every seed.
            if index < 5:
                event_valid = np.asarray(events.valid)
                choice = np.argwhere(event_valid)
                event_order = np.asarray(events.scattering_order)
                multiple = np.argwhere(
                    event_valid & (event_order >= 2)
                )
                positions_host = np.asarray(positions, dtype=np.float64)
                escape_host = np.asarray(events.escape_column_cm2, dtype=np.float64)
                for photon, interaction in np.concatenate(
                    (choice[:4], multiple[:4]), axis=0
                ):
                    point = positions_host[photon, interaction]
                    distance = float(np.linalg.norm(point))
                    reference_column = host_ray_column(
                        point, -point / distance, distance, columns,
                    )
                    actual_column = escape_host[photon, interaction]
                    error = abs(actual_column - reference_column) / max(
                        reference_column, columns[2].min() * 1e-6
                    )
                    column_max_relative_error = max(column_max_relative_error, error)
                    column_samples += 1
                    multiple_column_samples += int(
                        event_order[photon, interaction] >= 2
                    )
            launched = sample_source_launches(launch_key, source, launch)
            transported = jax.block_until_ready(pure_jit(pure_key, launched))
            b_history = reference_history_weights(
                launched, transported, physics, energy, columns,
            )
            a_sum += a_history.sum(axis=0)
            b_sum += b_history.sum(axis=0)
            seed_a += a_history.sum(axis=0)
            seed_b += b_history.sum(axis=0)
            a_flat = a_history.reshape(n, -1)
            b_flat = b_history.reshape(n, -1)
            a_q += a_flat.T @ a_flat
            b_q += b_flat.T @ b_flat
            statuses_a += np.bincount(np.asarray(status), minlength=7)
            statuses_b += np.bincount(np.asarray(transported.status), minlength=7)
            if index % 10 == 0:
                print(f"seed {seed}: {offset+n:,}/{packets:,}", flush=True)
        per_seed.append({
            "seed": int(seed), "analog_order_fluence": seed_a.sum(axis=(0, 1)).tolist(),
            "explicit_absorption_order_fluence": seed_b.sum(axis=(0, 1)).tolist(),
        })
    n_total = packets * len(seeds)
    a_sum /= len(seeds)
    b_sum /= len(seeds)
    a_q /= len(seeds)**2
    b_q /= len(seeds)**2
    quadrants = []
    for iy in range(2):
        for ix in range(2):
            mask = np.zeros((2, 2, 3), dtype=bool)
            mask[iy, ix, 0] = True
            analog, aq = _summed_moment(a_sum, a_q, mask)
            row = _moment_row(
                analog, float(fine[iy, ix]), aq, 0.0, n_total,
                sigma_limit=sigma_limit,
                minimum_effective_histories=minimum_effective_histories,
                maximum_relative_error=maximum_relative_error,
                quadrature_error=float(abs(fine[iy, ix] - coarse[iy, ix])),
            )
            row["sky_quadrant_yx"] = [iy, ix]
            quadrants.append(row)
    comparisons = {}
    for name, group, max_error in (
        ("first", [0], maximum_relative_error),
        ("second", [1], 0.20),
        ("third_plus", [2], 0.20),
        ("total", [0, 1, 2], maximum_relative_error),
    ):
        mask = np.zeros((2, 2, 3), dtype=bool)
        mask[..., group] = True
        left, left_q = _summed_moment(a_sum, a_q, mask)
        right, right_q = _summed_moment(b_sum, b_q, mask)
        comparisons[name] = _moment_row(
            left, right, left_q, right_q, n_total,
            sigma_limit=sigma_limit,
            minimum_effective_histories=minimum_effective_histories,
            maximum_relative_error=max_error,
        )
    multiple_quadrants = []
    for iy in range(2):
        for ix in range(2):
            mask = np.zeros((2, 2, 3), dtype=bool)
            mask[iy, ix, 1:] = True
            left, left_q = _summed_moment(a_sum, a_q, mask)
            right, right_q = _summed_moment(b_sum, b_q, mask)
            row = _moment_row(
                left, right, left_q, right_q, n_total,
                sigma_limit=sigma_limit,
                minimum_effective_histories=minimum_effective_histories,
                maximum_relative_error=0.20,
            )
            row["sky_quadrant_yx"] = [iy, ix]
            multiple_quadrants.append(row)
    checks = {
        "independent_first_order_quadrants": all(r["passed"] for r in quadrants),
        "explicit_absorption_by_order": all(r["passed"] for r in comparisons.values()),
        "explicit_absorption_multiple_quadrants": all(
            r["passed"] for r in multiple_quadrants
        ),
        "independent_escape_columns": (
            column_samples >= 12
            and multiple_column_samples >= 4
            and column_max_relative_error <= 5e-4
        ),
        "no_cap_or_invalid_analog_terminal": not any(
            statuses_a[i] for i in (0, 4, 5, 6)
        ),
        "no_cap_or_invalid_pure_terminal": not any(statuses_b[i] for i in (0, 4, 5, 6)),
        "status_count_closure": statuses_a.sum() == statuses_b.sum() == n_total,
    }
    return {
        "energy_kev": energy, "central_tau_scattering": target_tau,
        "columns_cm2": columns.tolist(), "radial_edges_kpc": RADIAL_EDGES_KPC.tolist(),
        "sky_edges_arcsec": SKY_EDGES.tolist(), "launch_slope_bounds": LAUNCH_BOUNDS,
        "time_edges_s": TIME_EDGES_S,
        "reference_coarse": coarse.tolist(), "reference_fine": fine.tolist(),
        "first_order_quadrants": quadrants, "order_comparisons": comparisons,
        "multiple_scatter_quadrants": multiple_quadrants,
        "analog_statuses": statuses_a.tolist(),
        "pure_scattering_statuses": statuses_b.tolist(),
        "escape_column_samples": column_samples,
        "multiple_escape_column_samples": multiple_column_samples,
        "escape_column_max_relative_error": column_max_relative_error,
        "per_seed": per_seed, "checks": {k: bool(v) for k, v in checks.items()},
        "all_passed": bool(all(checks.values())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packets", type=int, default=100_000)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[912, 319, 141])
    parser.add_argument("--energy", type=float, default=5.35)
    parser.add_argument("--tau-scattering", type=float, default=1.5)
    parser.add_argument("--max-interactions", type=int, default=32)
    parser.add_argument("--sigma-limit", type=float, default=5.0)
    parser.add_argument("--minimum-effective-histories", type=int, default=30)
    parser.add_argument("--maximum-relative-error", type=float, default=0.10)
    args = parser.parse_args()
    if (args.packets < 2 or args.chunk_size < 1 or len(args.seeds) < 3
        or len(set(args.seeds)) != len(args.seeds)
        or not 2.0 <= args.energy <= 10.0
        or not 0 < args.tau_scattering or args.max_interactions < 4
        or not 0 < args.sigma_limit or args.minimum_effective_histories < 1
        or not 0 < args.maximum_relative_error < 1):
        parser.error("invalid validation settings")
    _, _, physics = load_2_10_material_tables()
    case = run_case(
        physics, energy=args.energy, target_tau=args.tau_scattering,
        packets=args.packets, chunk_size=args.chunk_size,
        seeds=args.seeds, max_interactions=args.max_interactions,
        sigma_limit=args.sigma_limit,
        minimum_effective_histories=args.minimum_effective_histories,
        maximum_relative_error=args.maximum_relative_error,
    )
    report = {
        "stage": "9F_heterogeneous_observer", "git_head": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "python": platform.python_version(), "numpy": np.__version__,
        "jax": jax.__version__, "backend": jax.default_backend(),
        "scattering_table_sha256": hashlib.sha256(
            DEFAULT_2_10_SCATTERING.read_bytes()
        ).hexdigest(),
        "absorption_table_sha256": hashlib.sha256(
            DEFAULT_2_10_ABSORPTION.read_bytes()
        ).hexdigest(),
        "config": {
            "packets_per_seed": args.packets, "chunk_size": args.chunk_size,
            "seeds": args.seeds, "max_interactions": args.max_interactions,
            "sigma_limit": args.sigma_limit,
            "minimum_effective_histories": args.minimum_effective_histories,
            "maximum_relative_error": args.maximum_relative_error,
        },
        "case": case, "all_passed": case["all_passed"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f'Saved {args.output}; all_passed={report["all_passed"]}', flush=True)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
