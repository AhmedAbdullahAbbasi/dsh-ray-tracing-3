"""Stage 9F continuous-spectrum absorbed observer benchmark.

Run on the local JAX machine. The centered uniform shell emits an instantaneous
unit-fluence 2--10 keV power law (Gamma=1.7 by default). Energy bands are
stratified, with exact power-law probabilities; photons remain continuous
within each band. This is an ideal observer experiment, not a detector image.
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
from dsh.sources.launch import (
    build_rectangular_launch_geometry,
    sample_source_launches,
)
from dsh.sources.models import SourcePackets, _sample_powerlaw_energy
from dsh.transport.kernel import transport_photon_batch
from dsh.validation.absorbed_observer import (
    DAY_S,
    first_order_radial_quadrature,
    host_material,
    score_scattering_only_histories,
)
from scripts.run_absorbed_observer_validation import (
    _compare,
    _compare_first_order_time_bins,
)

BANDS = (2.0, 4.0, 6.0, 10.0)
BOUNDS = ((-0.003, 0.003), (-0.003, 0.003))
TIME_EDGES = np.array([0.0, 0.5, 2.0, 6.0, 30.0]) * DAY_S


def _powerlaw_integral(low, high, gamma):
    exponent = 1.0 - gamma
    if abs(exponent) < 1e-10:
        return float(np.log(high / low))
    return float(low**exponent * np.expm1(exponent * np.log(high / low)) / exponent)


def _edge_breaks(physics, low, high):
    """Split short material-grid intervals so absorption edges are resolved."""
    grid = np.asarray(physics.energy_kev, dtype=np.float64)
    short = np.flatnonzero(np.diff(grid) < 0.001)
    boundaries = np.concatenate((short, short + 1))
    candidates = grid[boundaries]
    interior = candidates[(candidates > low) & (candidates < high)]
    return np.unique(
        np.r_[low, interior, high]
    )


def spectral_first_order_quadrature(
    physics,
    column,
    gamma,
    bands=BANDS,
    *,
    n_energy=6,
    n_radius=32,
    n_depth=28,
):
    """Integrate the independent absorbed shell reference over the power law.

    Piecewise energy integration resolves narrow absorption features in the
    frozen material grid; the source density is normalized over the full band.
    The same physical column applies at every energy.
    """
    if n_energy < 2 or n_radius < 2 or n_depth < 2:
        raise ValueError("quadrature orders must be at least two")
    if not np.isfinite(gamma) or not np.all(np.diff(bands) > 0):
        raise ValueError("invalid spectral shape or energy bands")
    grid = np.asarray(physics.energy_kev, dtype=np.float64)
    if bands[0] < grid[0] or bands[-1] > grid[-1]:
        raise ValueError("spectral bands exceed the material table")
    normalizer = _powerlaw_integral(bands[0], bands[-1], gamma)
    roots, weights = np.polynomial.legendre.leggauss(n_energy)
    output = np.zeros((len(bands) - 1, len(TIME_EDGES) - 1), np.float64)
    for band in range(len(bands) - 1):
        breaks = _edge_breaks(physics, bands[band], bands[band + 1])
        for low, high in zip(breaks[:-1], breaks[1:], strict=True):
            center, half = (low + high) / 2, (high - low) / 2
            for root, weight in zip(roots, weights, strict=True):
                energy = center + half * root
                output[band] += (
                    weight
                    * half
                    * energy ** (-gamma)
                    / normalizer
                    * first_order_radial_quadrature(
                        physics,
                        energy,
                        column,
                        BOUNDS,
                        TIME_EDGES,
                        n_radius=n_radius,
                        n_depth=n_depth,
                    )
                )
    return output


def _covariance(sum_vector, cross, n):
    if n < 2:
        raise ValueError("at least two histories are needed per band")
    flattened = sum_vector.reshape(-1)
    return n / (n - 1) * (cross - np.outer(flattened, flattened) / n)


def _compare_order_group(
    analog,
    pure,
    analog_cross,
    pure_cross,
    analog_cov,
    pure_cov,
    order,
    minimum_histories,
    sigma_limit,
):
    index = [3 * time + order for time in range(len(TIME_EDGES) - 1)]
    a = float(analog.reshape(-1)[index].sum())
    b = float(pure.reshape(-1)[index].sum())
    va = float(max(analog_cov[np.ix_(index, index)].sum(), 0.0))
    vb = float(max(pure_cov[np.ix_(index, index)].sum(), 0.0))
    qa = float(analog_cross[np.ix_(index, index)].sum())
    qb = float(pure_cross[np.ix_(index, index)].sum())
    result = _compare(a, b, np.sqrt(va), np.sqrt(vb), sigma_limit=sigma_limit)
    result["effective_histories"] = [
        a * a / qa if qa > 0 else 0.0,
        b * b / qb if qb > 0 else 0.0,
    ]
    result["relative_analog_standard_error"] = np.sqrt(va) / a if a > 0 else None
    result["powered"] = min(result["effective_histories"]) >= minimum_histories
    relative = result["relative_analog_standard_error"]
    result["precise"] = relative is not None and relative <= (
        0.05 if order == 0 else 0.10
    )
    result["passed"] = bool(
        result["passed"] and result["powered"] and result["precise"]
    )
    return result


def _git(*args):
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def run_case(
    physics,
    *,
    packets,
    chunk_size,
    seeds,
    max_interactions,
    gamma,
    pivot_energy,
    target_tau,
    sigma_limit,
    minimum_histories,
    quadrature_rtol,
    maximum_time_bin_relative_error,
    n_energy,
):
    _, _, _, pivot_sigma, _ = host_material(physics, pivot_energy)
    column = target_tau / pivot_sigma
    cloud = build_angular_distance_cloud(
        np.full((2, 2, 2), column / 2.0),
        [-15000.0, 15000.0],
        [-15000.0, 15000.0],
        [4.25, 4.75],
        10.0,
    )
    launch = build_rectangular_launch_geometry(10.0, *BOUNDS)
    bins = build_observer_bin_geometry(
        [-1800.0, 1800.0], [-1800.0, 1800.0], BANDS, TIME_EDGES
    )
    pure = physics._replace(
        absorption_cross_section_cm2_per_h=jnp.zeros_like(
            physics.absorption_cross_section_cm2_per_h
        )
    )
    print("Computing coarse energy and shell quadrature...", flush=True)
    coarse = spectral_first_order_quadrature(
        physics, column, gamma, n_energy=n_energy, n_radius=24, n_depth=24
    )
    print("Computing fine energy and shell quadrature...", flush=True)
    fine = spectral_first_order_quadrature(
        physics, column, gamma, n_energy=2 * n_energy, n_radius=64, n_depth=56
    )
    normalizer = _powerlaw_integral(BANDS[0], BANDS[-1], gamma)
    probabilities = np.array(
        [
            _powerlaw_integral(low, high, gamma) / normalizer
            for low, high in zip(BANDS[:-1], BANDS[1:], strict=True)
        ]
    )
    allocation = np.floor(packets * probabilities).astype(int)
    # Allocate residual photons deterministically to the largest remainders.
    residual = packets - int(allocation.sum())
    allocation[np.argsort(-(packets * probabilities - allocation))[:residual]] += 1
    if np.any(allocation < 2):
        raise ValueError("packets must allocate at least two photons per energy band")
    analog_jit = jax.jit(
        simulate_source_packets_to_observer, static_argnames=("max_interactions",)
    )
    scatter_jit = jax.jit(
        transport_photon_batch, static_argnames=("max_interactions",)
    )
    reports = []
    total_analog_status = np.zeros(7, np.int64)
    total_pure_status = np.zeros(7, np.int64)
    for band, (low, high) in enumerate(zip(BANDS[:-1], BANDS[1:], strict=True)):
        n_band = int(allocation[band])
        n_histories = n_band * len(seeds)
        analog_sum = np.zeros((len(TIME_EDGES) - 1, 3), np.float64)
        pure_sum = analog_sum.copy()
        analog_cross = np.zeros((analog_sum.size, analog_sum.size), np.float64)
        pure_cross = analog_cross.copy()
        analog_status = np.zeros(7, np.int64)
        pure_status = np.zeros(7, np.int64)
        per_seed = []
        for seed in seeds:
            seed_analog = np.zeros_like(analog_sum)
            seed_pure = np.zeros_like(pure_sum)
            for index, offset in enumerate(range(0, n_band, chunk_size)):
                size = min(chunk_size, n_band - offset)
                energy_key, analog_key, launch_key, pure_key = random.split(
                    random.fold_in(random.fold_in(random.PRNGKey(seed), band), index), 4
                )
                energies = _sample_powerlaw_energy(
                    energy_key,
                    jnp.float32(low),
                    jnp.float32(high),
                    jnp.float32(gamma),
                    size,
                )
                if band < len(BANDS) - 2:
                    energies = jnp.minimum(
                        energies, jnp.nextafter(jnp.float32(high), jnp.float32(low))
                    )
                source = SourcePackets(
                    energy_kev=energies,
                    emission_time_s=jnp.zeros((size,), jnp.float32),
                    weight_observer_fluence=jnp.full(
                        (size,), probabilities[band] / n_band, jnp.float32
                    ),
                    time_index=jnp.zeros((size,), jnp.int32),
                    spectral_bin_index=jnp.full((size,), band, jnp.int32),
                )
                result = jax.block_until_ready(
                    analog_jit(
                        analog_key,
                        source,
                        launch,
                        cloud,
                        physics,
                        bins,
                        max_interactions=max_interactions,
                    )
                )
                values = np.asarray(result.products.time_order_fluence_sum, np.float64)
                q = np.asarray(
                    result.products.time_order_fluence_cross, np.float64
                ).reshape(analog_cross.shape)
                analog_sum += values
                seed_analog += values
                analog_cross += q
                analog_status += np.asarray(result.diagnostics.transport_status_count)
                cube = np.asarray(result.products.total_fluence)
                if (
                    not np.allclose(
                        cube[:, band, 0, 0],
                        values.sum(axis=1),
                        rtol=1e-4,
                        atol=1e-9,
                    )
                    or np.any(np.delete(cube, band, axis=1))
                ):
                    raise RuntimeError(
                        "spectral band and time-order observer scores disagree"
                    )
                launched = sample_source_launches(launch_key, source, launch)
                history = scatter_jit(
                    pure_key,
                    launched.position_pc,
                    launched.momentum_kev,
                    cloud,
                    pure,
                    max_interactions=max_interactions,
                )
                reference = score_scattering_only_histories(
                    launched,
                    history,
                    physics,
                    np.asarray(energies, np.float64),
                    column,
                    TIME_EDGES,
                    np.asarray(bins.sky_x_edges_arcsec),
                    np.asarray(bins.sky_y_edges_arcsec),
                )
                pure_sum += reference["sum"]
                seed_pure += reference["sum"]
                pure_cross += reference["cross"]
                pure_status += reference["statuses"]
                if index % 10 == 0:
                    print(
                        f"band {low:g}-{high:g} keV, seed {seed}: "
                        f"{offset+size:,}/{n_band:,}",
                        flush=True,
                    )
            per_seed.append(
                {
                    "seed": seed,
                    "analog": seed_analog.tolist(),
                    "explicit_absorption": seed_pure.tolist(),
                }
            )
        total_analog_status += analog_status
        total_pure_status += pure_status
        scale = len(seeds)
        analog_sum /= scale
        pure_sum /= scale
        analog_cross /= scale**2
        pure_cross /= scale**2
        analog_cov = _covariance(analog_sum, analog_cross, n_histories)
        pure_cov = _covariance(pure_sum, pure_cross, n_histories)
        time_bins = _compare_first_order_time_bins(
            analog_sum,
            analog_cross,
            analog_cov,
            coarse[band],
            fine[band],
            minimum_effective_histories=minimum_histories,
            maximum_relative_standard_error=maximum_time_bin_relative_error,
            quadrature_rtol=quadrature_rtol,
            sigma_limit=sigma_limit,
        )
        orders = {
            name: _compare_order_group(
                analog_sum,
                pure_sum,
                analog_cross,
                pure_cross,
                analog_cov,
                pure_cov,
                order,
                minimum_histories,
                sigma_limit,
            )
            for order, name in enumerate(("first", "second", "third_plus"))
        }
        checks = {
            "first_order_each_time_bin": all(item["passed"] for item in time_bins),
            "analog_vs_explicit_absorption_each_order": all(
                item["passed"] for item in orders.values()
            ),
            "no_interaction_limit_or_invalid_analog_terminal": not any(
                analog_status[i] for i in (0, 4, 5, 6)
            ),
            "no_interaction_limit_or_invalid_pure_terminal": not any(
                pure_status[i] for i in (0, 4, 5, 6)
            ),
            "status_count_closure": int(analog_status.sum())
            == int(pure_status.sum())
            == n_histories,
        }
        reports.append(
            {
                "band_kev": [low, high],
                "source_probability": float(probabilities[band]),
                "packets_per_seed": n_band,
                "quadrature_coarse": coarse[band].tolist(),
                "quadrature_fine": fine[band].tolist(),
                "first_order_time_bins": time_bins,
                "order_comparisons": orders,
                "analog_statuses": analog_status.tolist(),
                "pure_scattering_statuses": pure_status.tolist(),
                "per_seed": per_seed,
                "checks": checks,
                "all_passed": bool(all(checks.values())),
            }
        )
    checks = {
        "all_band_checks": all(band["all_passed"] for band in reports),
        "source_probabilities_sum_to_one": bool(
            np.isclose(probabilities.sum(), 1.0, atol=1e-14)
        ),
        "photon_allocation_closure": bool(allocation.sum() == packets),
        "ensemble_has_independent_seeds": len(seeds) >= 3
        and len(seeds) == len(set(seeds)),
    }
    return {
        "spectrum": {
            "type": "instantaneous_powerlaw",
            "energy_edges_kev": list(BANDS),
            "photon_index": gamma,
            "total_source_fluence": 1.0,
        },
        "pivot_energy_kev": pivot_energy,
        "pivot_tau_scattering": target_tau,
        "column_cm2": column,
        "time_edges_s": TIME_EDGES.tolist(),
        "quadrature": {
            "method": "energy-split Gauss-Legendre times axisymmetric radial shell",
            "coarse_energy_nodes_per_segment": n_energy,
            "fine_energy_nodes_per_segment": 2 * n_energy,
        },
        "bands": reports,
        "analog_statuses": total_analog_status.tolist(),
        "pure_scattering_statuses": total_pure_status.tolist(),
        "checks": checks,
        "all_passed": bool(all(checks.values())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--packets",
        type=int,
        default=100_000,
        help="total photons per seed across all three bands",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[912, 319, 141])
    parser.add_argument("--photon-index", type=float, default=1.7)
    parser.add_argument("--pivot-energy", type=float, default=5.35)
    parser.add_argument("--tau-scattering", type=float, default=1.5)
    parser.add_argument("--max-interactions", type=int, default=16)
    parser.add_argument("--sigma-limit", type=float, default=5.0)
    parser.add_argument("--minimum-effective-histories", type=int, default=30)
    parser.add_argument("--quadrature-rtol", type=float, default=0.02)
    parser.add_argument("--max-time-bin-relative-error", type=float, default=0.10)
    parser.add_argument(
        "--spectral-nodes",
        type=int,
        default=6,
        help="coarse nodes per smooth spectral segment; fine uses twice this",
    )
    args = parser.parse_args()
    if (
        args.packets < 6
        or args.chunk_size < 1
        or len(args.seeds) < 3
        or len(set(args.seeds)) != len(args.seeds)
        or args.max_interactions < 4
        or args.spectral_nodes < 2
        or not np.isfinite(args.photon_index)
        or not (2.0 <= args.pivot_energy <= 10.0)
        or not np.isfinite(args.tau_scattering)
        or args.tau_scattering <= 0
        or not np.isfinite(args.sigma_limit)
        or args.sigma_limit <= 0
        or args.minimum_effective_histories < 1
        or not (0 < args.quadrature_rtol < 1)
        or not (0 < args.max_time_bin_relative_error < 1)
    ):
        parser.error(
            "invalid spectrum, photon count, quadrature or validation thresholds"
        )
    _, _, physics = load_2_10_material_tables()
    report = {
        "stage": "9F_continuous_spectrum_absorbed_observer",
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
            "maximum_time_bin_relative_error": args.max_time_bin_relative_error,
            "spectral_nodes": args.spectral_nodes,
        },
        "case": run_case(
            physics,
            packets=args.packets,
            chunk_size=args.chunk_size,
            seeds=args.seeds,
            max_interactions=args.max_interactions,
            gamma=args.photon_index,
            pivot_energy=args.pivot_energy,
            target_tau=args.tau_scattering,
            sigma_limit=args.sigma_limit,
            minimum_histories=args.minimum_effective_histories,
            quadrature_rtol=args.quadrature_rtol,
            maximum_time_bin_relative_error=args.max_time_bin_relative_error,
            n_energy=args.spectral_nodes,
        ),
    }
    report["all_passed"] = report["case"]["all_passed"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f'Saved {args.output}; all_passed={report["all_passed"]}')
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
