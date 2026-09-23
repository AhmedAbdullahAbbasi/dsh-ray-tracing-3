"""Stage 9F: end-to-end absorbed observer comparison in a controlled shell.

Run locally with the bundled 2--10 keV material tables. This is an ideal
observer benchmark, not a four-cloud production simulation or a detector image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from dsh.geometry.clouds import build_angular_distance_cloud
from dsh.observer.binning import build_observer_bin_geometry
from dsh.physics.materials import (
    DEFAULT_2_10_ABSORPTION,
    DEFAULT_2_10_SCATTERING,
    load_2_10_material_tables,
)
from dsh.pipeline import simulate_source_packets_to_observer
from dsh.sources.launch import build_rectangular_launch_geometry, sample_source_launches
from dsh.sources.models import SourcePackets
from dsh.transport.kernel import transport_photon_batch
from dsh.validation.absorbed_observer import (
    DAY_S,
    first_order_quadrature,
    first_order_radial_quadrature,
    host_material,
    score_scattering_only_histories,
)


def _git(*args):
    value = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return value.stdout.strip() if value.returncode == 0 else "unavailable"


def _compare(left, right, se_left, se_right, *, sigma_limit):
    combined = float(np.hypot(se_left, se_right))
    z = (left - right) / combined if combined > 0 else None
    return {
        "analog": float(left),
        "reference": float(right),
        "standard_error": combined,
        "z": z,
        "passed": bool(z is not None and abs(z) <= sigma_limit),
    }


def _compare_first_order_time_bins(
    analog_sum,
    analog_cross,
    analog_cov,
    coarse,
    fine,
    *,
    minimum_effective_histories,
    maximum_relative_standard_error,
    quadrature_rtol,
    sigma_limit,
):
    """Gate each time bin using per-photon moments, including zero-score histories."""
    rows = []
    for t, (reference_coarse, reference_fine) in enumerate(
        zip(coarse, fine, strict=True)
    ):
        index = 3 * t  # first-scatter component of (time, order) covariance
        analog = float(analog_sum[t, 0])
        squared = float(analog_cross[index, index])
        photon_se = float(np.sqrt(max(analog_cov[index, index], 0.0)))
        quadrature_error = float(abs(reference_fine - reference_coarse))
        result = _compare(
            analog,
            reference_fine,
            photon_se,
            quadrature_error,
            sigma_limit=sigma_limit,
        )
        result.update(
            {
                "time_index": t,
                "photon_standard_error": photon_se,
                "quadrature_difference": quadrature_error,
                "quadrature_relative_change": quadrature_error
                / max(reference_fine, 1e-100),
                "effective_histories": analog**2 / squared if squared > 0 else 0.0,
                "relative_photon_standard_error": photon_se / analog
                if analog > 0
                else None,
            }
        )
        result["powered"] = bool(
            result["effective_histories"] >= minimum_effective_histories
        )
        result["precise"] = bool(
            result["relative_photon_standard_error"] is not None
            and result["relative_photon_standard_error"]
            <= maximum_relative_standard_error
        )
        result["quadrature_converged"] = bool(
            result["quadrature_relative_change"] <= quadrature_rtol
        )
        result["passed"] = bool(
            result["passed"]
            and result["powered"]
            and result["precise"]
            and result["quadrature_converged"]
        )
        rows.append(result)
    return rows


def run_case(
    physics,
    *,
    energy,
    target_tau,
    packets,
    chunk_size,
    seeds,
    max_interactions,
    sigma_limit,
    minimum_effective_histories,
    quadrature_rtol,
    maximum_time_bin_relative_error=0.10,
):
    _, _, _, sigma_sca, _ = host_material(physics, energy)
    column = target_tau / sigma_sca
    cloud = build_angular_distance_cloud(
        np.full((2, 2, 2), column / 2.0),
        [-15000.0, 15000.0],
        [-15000.0, 15000.0],
        [4.25, 4.75],
        10.0,
    )
    bounds = ((-0.003, 0.003), (-0.003, 0.003))
    launch = build_rectangular_launch_geometry(10.0, *bounds)
    time_edges = np.array([0, 0.5, 2, 6, 30]) * DAY_S
    bins = build_observer_bin_geometry(
        [-1800.0, 1800.0], [-1800.0, 1800.0], [2.0, 10.0], time_edges
    )
    pure = physics._replace(
        absorption_cross_section_cm2_per_h=jnp.zeros_like(
            physics.absorption_cross_section_cm2_per_h
        )
    )
    coarse = first_order_radial_quadrature(
        physics, energy, column, bounds, time_edges, n_radius=24, n_depth=24
    )
    fine = first_order_radial_quadrature(
        physics, energy, column, bounds, time_edges, n_radius=64, n_depth=56
    )
    cartesian = first_order_quadrature(
        physics, energy, column, bounds, time_edges, n_slope=96, n_depth=42
    )
    cartesian_integrated_difference = float(
        abs(fine.sum() - cartesian.sum()) / max(fine.sum(), 1e-100)
    )
    quad_error = float(abs(fine.sum() - coarse.sum()) / max(fine.sum(), 1e-100))
    early_quad_error = float(
        abs(fine[:2].sum() - coarse[:2].sum()) / max(fine[:2].sum(), 1e-100)
    )
    analog_jit = jax.jit(
        simulate_source_packets_to_observer, static_argnames=("max_interactions",)
    )
    scatter_jit = jax.jit(transport_photon_batch, static_argnames=("max_interactions",))
    score_rows = []
    for seed in seeds:
        analog_s = np.zeros((len(time_edges) - 1, 3), dtype=np.float64)
        analog_q = np.zeros((analog_s.size, analog_s.size), dtype=np.float64)
        pure_s = analog_s.copy()
        pure_q = analog_q.copy()
        statuses_analog = np.zeros(7, dtype=np.int64)
        statuses_pure = statuses_analog.copy()
        done = 0
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
            result = analog_jit(
                analog_key,
                source,
                launch,
                cloud,
                physics,
                bins,
                max_interactions=max_interactions,
            )
            result = jax.block_until_ready(result)
            s = np.asarray(result.products.time_order_fluence_sum, dtype=np.float64)
            q = np.asarray(result.products.time_order_fluence_cross, dtype=np.float64)
            analog_s += s
            analog_q += q.reshape(analog_q.shape)
            statuses_analog += np.asarray(result.diagnostics.transport_status_count)

            launched = sample_source_launches(launch_key, source, launch)
            transported = scatter_jit(
                pure_key,
                launched.position_pc,
                launched.momentum_kev,
                cloud,
                pure,
                max_interactions=max_interactions,
            )
            reference = score_scattering_only_histories(
                launched,
                transported,
                physics,
                energy,
                column,
                time_edges,
                np.asarray(bins.sky_x_edges_arcsec),
                np.asarray(bins.sky_y_edges_arcsec),
            )
            pure_s += reference["sum"]
            pure_q += reference["cross"]
            statuses_pure += reference["statuses"]
            done += n
            if index % 10 == 0:
                print(f"seed {seed}: {done:,}/{packets:,}", flush=True)
        score_rows.append(
            {
                "seed": seed,
                "analog_sum": analog_s,
                "analog_q": analog_q,
                "pure_sum": pure_s,
                "pure_q": pure_q,
                "analog_status": statuses_analog,
                "pure_status": statuses_pure,
            }
        )

    # Independent seeds are additional checks; photon-count moments use all
    # launched histories, including zero-score histories.
    n_total = packets * len(seeds)
    a = sum(row["analog_sum"] for row in score_rows)
    b = sum(row["pure_sum"] for row in score_rows)
    a_q = sum(row["analog_q"] for row in score_rows)
    b_q = sum(row["pure_q"] for row in score_rows)
    # Each run simulated unit total fluence. Average to compare with the unit
    # source quadrature; second moments therefore scale as 1/seeds**2.
    scale = len(seeds)
    a, b, a_q, b_q = a / scale, b / scale, a_q / scale**2, b_q / scale**2
    a_flat, b_flat = a.reshape(-1), b.reshape(-1)
    a_cov = n_total / (n_total - 1) * (a_q - np.outer(a_flat, a_flat) / n_total)
    b_cov = n_total / (n_total - 1) * (b_q - np.outer(b_flat, b_flat) / n_total)
    first_order_time_bins = _compare_first_order_time_bins(
        a,
        a_q,
        a_cov,
        coarse,
        fine,
        minimum_effective_histories=minimum_effective_histories,
        maximum_relative_standard_error=maximum_time_bin_relative_error,
        quadrature_rtol=quadrature_rtol,
        sigma_limit=sigma_limit,
    )

    comparisons = {}
    for label, tsel, osel in (
        ("all_first", range(4), (0,)),
        ("all_second", range(4), (1,)),
        ("all_third_plus", range(4), (2,)),
        ("all_total", range(4), (0, 1, 2)),
        ("early_first", range(2), (0,)),
    ):
        idx = [3 * t + o for t in tsel for o in osel]
        left, right = float(a_flat[idx].sum()), float(b_flat[idx].sum())
        va = float(np.maximum(a_cov[np.ix_(idx, idx)].sum(), 0.0))
        vb = float(np.maximum(b_cov[np.ix_(idx, idx)].sum(), 0.0))
        comp = _compare(left, right, np.sqrt(va), np.sqrt(vb), sigma_limit=sigma_limit)
        qa = float(a_q[np.ix_(idx, idx)].sum())
        qb = float(b_q[np.ix_(idx, idx)].sum())
        comp["effective_histories"] = [
            left**2 / qa if qa > 0 else 0,
            right**2 / qb if qb > 0 else 0,
        ]
        comp["powered"] = bool(
            min(comp["effective_histories"]) >= minimum_effective_histories
        )
        comp["passed"] &= comp["powered"]
        comparisons[label] = comp

    first = a_flat[[0, 3, 6, 9]].sum()
    var_first = max(a_cov[np.ix_([0, 3, 6, 9], [0, 3, 6, 9])].sum(), 0.0)
    first_reference = _compare(
        first,
        float(fine.sum()),
        np.sqrt(var_first),
        float(abs(fine.sum() - coarse.sum())),
        sigma_limit=sigma_limit,
    )
    first_reference["quadrature_converged"] = quad_error <= quadrature_rtol
    first_reference["powered"] = comparisons["all_first"]["powered"]
    first_reference["passed"] &= (
        first_reference["quadrature_converged"] and first_reference["powered"]
    )
    early_indices = [0, 3]
    early_fluence = float(a_flat[early_indices].sum())
    early_variance = max(float(a_cov[np.ix_(early_indices, early_indices)].sum()), 0.0)
    early_reference = _compare(
        early_fluence,
        float(fine[:2].sum()),
        np.sqrt(early_variance),
        float(abs(fine[:2].sum() - coarse[:2].sum())),
        sigma_limit=sigma_limit,
    )
    early_reference["quadrature_converged"] = early_quad_error <= quadrature_rtol
    early_reference["powered"] = comparisons["early_first"]["powered"]
    early_reference["passed"] &= (
        early_reference["quadrature_converged"] and early_reference["powered"]
    )
    statuses_a = sum(row["analog_status"] for row in score_rows).tolist()
    statuses_b = sum(row["pure_status"] for row in score_rows).tolist()
    checks = {
        "first_order_absolute": first_reference["passed"],
        "first_order_early_window": early_reference["passed"],
        "first_order_each_time_bin": all(
            row["passed"] for row in first_order_time_bins
        ),
        "radial_vs_cartesian_integral": cartesian_integrated_difference
        <= quadrature_rtol,
        "analog_vs_explicit_absorption": all(c["passed"] for c in comparisons.values()),
        "no_invalid_analog_terminals": not any(statuses_a[i] for i in (0, 4, 5, 6)),
        "no_invalid_pure_terminals": not any(statuses_b[i] for i in (0, 4, 5, 6)),
        "status_count_closure": sum(statuses_a) == sum(statuses_b) == n_total,
        "independent_seed_ensemble": len(seeds) >= 3,
    }
    # Precision is separate from agreement: 5 sigma agreement can be
    # uninformative when one photon dominates the score.
    targets = {
        "all_first": 0.05,
        "all_second": 0.10,
        "all_third_plus": 0.10,
        "all_total": 0.05,
        "early_first": 0.10,
    }
    achieved = {}
    for name, maximum in targets.items():
        idx = {
            "all_first": [0, 3, 6, 9],
            "all_second": [1, 4, 7, 10],
            "all_third_plus": [2, 5, 8, 11],
            "all_total": list(range(12)),
            "early_first": [0, 3],
        }[name]
        measured = float(a_flat[idx].sum())
        relative = (
            float(np.sqrt(max(a_cov[np.ix_(idx, idx)].sum(), 0.0)) / measured)
            if measured > 0
            else None
        )
        achieved[name] = {
            "relative_standard_error": relative,
            "maximum": maximum,
            "passed": bool(relative is not None and relative <= maximum),
        }
    checks["observer_precision_screen"] = all(
        row["passed"] for row in achieved.values()
    )
    return {
        "energy_kev": energy,
        "tau_scattering": target_tau,
        "column_cm2": column,
        "time_edges_s": time_edges.tolist(),
        "quadrature": {
            "method": "axisymmetric radial shell quadrature with exact rectangular azimuth",
            "coarse": coarse.tolist(),
            "fine": fine.tolist(),
            "cartesian_fine": cartesian.tolist(),
            "cartesian_integrated_relative_difference": cartesian_integrated_difference,
            "relative_integrated_error_estimate": quad_error,
            "relative_early_window_error_estimate": early_quad_error,
            "relative_each_time_bin_changes": (
                abs(fine - coarse) / np.maximum(fine, 1e-100)
            ).tolist(),
        },
        "first_order_reference": first_reference,
        "first_order_early_reference": early_reference,
        "first_order_time_bins": first_order_time_bins,
        "achieved_precision": achieved,
        "comparisons": comparisons,
        "analog_statuses": statuses_a,
        "pure_scattering_statuses": statuses_b,
        "per_seed_integrated": [
            {
                "seed": row["seed"],
                "analog": row["analog_sum"].sum(axis=0).tolist(),
                "explicit_absorption": row["pure_sum"].sum(axis=0).tolist(),
                "first_order_time_bins": row["analog_sum"][:, 0].tolist(),
                "first_order_time_bin_standard_errors": np.sqrt(
                    np.maximum(
                        packets
                        / (packets - 1)
                        * (
                            np.diag(row["analog_q"])[::3]
                            - row["analog_sum"][:, 0] ** 2 / packets
                        ),
                        0.0,
                    )
                ).tolist(),
            }
            for row in score_rows
        ],
        "checks": checks,
        "all_passed": bool(all(checks.values())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--packets", type=int, default=20_000, help="photons per independent seed"
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[912, 319, 141])
    parser.add_argument("--energy", type=float, default=5.35)
    parser.add_argument("--tau-scattering", type=float, default=1.5)
    parser.add_argument("--max-interactions", type=int, default=16)
    parser.add_argument("--sigma-limit", type=float, default=5.0)
    parser.add_argument("--minimum-effective-histories", type=int, default=30)
    parser.add_argument("--quadrature-rtol", type=float, default=0.02)
    parser.add_argument(
        "--max-time-bin-relative-error",
        type=float,
        default=0.10,
        help="maximum first-order per-bin photon standard error divided by its fluence",
    )
    args = parser.parse_args()
    if (
        args.packets < 2
        or args.chunk_size <= 0
        or args.max_interactions < 4
        or len(set(args.seeds)) != len(args.seeds)
        or args.tau_scattering <= 0
        or not 0 < args.max_time_bin_relative_error < 1
    ):
        parser.error("invalid photon, seed, optical-depth or interaction-cap settings")
    _, _, physics = load_2_10_material_tables()
    report = {
        "stage": "9F_absorbed_observer",
        "git_head": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "scattering_table_sha256": hashlib.sha256(
            DEFAULT_2_10_SCATTERING.read_bytes()
        ).hexdigest(),
        "absorption_table_sha256": hashlib.sha256(
            DEFAULT_2_10_ABSORPTION.read_bytes()
        ).hexdigest(),
        "config": {
            "packets_per_seed": args.packets,
            "chunk_size": args.chunk_size,
            "seeds": args.seeds,
            "max_interactions": args.max_interactions,
            "sigma_limit": args.sigma_limit,
            "minimum_effective_histories": args.minimum_effective_histories,
            "quadrature_rtol": args.quadrature_rtol,
            "max_time_bin_relative_error": args.max_time_bin_relative_error,
        },
        "case": run_case(
            physics,
            energy=args.energy,
            target_tau=args.tau_scattering,
            packets=args.packets,
            chunk_size=args.chunk_size,
            seeds=args.seeds,
            max_interactions=args.max_interactions,
            sigma_limit=args.sigma_limit,
            minimum_effective_histories=args.minimum_effective_histories,
            quadrature_rtol=args.quadrature_rtol,
            maximum_time_bin_relative_error=args.max_time_bin_relative_error,
        ),
    }
    report["all_passed"] = report["case"]["all_passed"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}; all_passed={report['all_passed']}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
