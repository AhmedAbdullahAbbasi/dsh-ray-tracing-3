"""Stage 9F first-order spatial fluence in an absorbing spherical shell.

Compare production launch, transport, peel-off scoring and observer binning
with an independent radial shell integral for three disjoint sky annuli. All
Monte Carlo moments are accumulated by launched photon, including zero scores.
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
from dsh.validation.absorbed_observer import (
    DAY_S,
    first_order_radial_quadrature,
    host_material,
)

ANNULUS_EDGES_ARCSEC = (0.0, 45.0, 90.0, 1800.0)
LAUNCH_BOUNDS = ((-0.003, 0.003), (-0.003, 0.003))
TIME_EDGES_S = (0.0, 30.0 * DAY_S)


def _git(*args):
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _annular_photon_weights(events, annulus_edges, time_edges):
    """Group all scored events by original launched-photon history."""
    valid = np.asarray(events.valid)
    order = np.asarray(events.scattering_order)
    x = np.asarray(events.sky_x_arcsec, dtype=np.float64)
    y = np.asarray(events.sky_y_arcsec, dtype=np.float64)
    arrival = np.asarray(events.arrival_time_s, dtype=np.float64)
    weight = np.asarray(events.weight_observer_fluence, dtype=np.float64)
    selected = valid & (order == 1) & (arrival >= time_edges[0]) & (
        arrival <= time_edges[-1]
    )
    radius = np.hypot(x, y)
    per_history = np.zeros((len(valid), len(annulus_edges) - 1), np.float64)
    for annulus, (low, high) in enumerate(
        zip(annulus_edges[:-1], annulus_edges[1:], strict=True)
    ):
        inside = selected & (radius >= low) & (
            (radius < high) if annulus < len(annulus_edges) - 2 else (radius <= high)
        )
        per_history[:, annulus] = np.where(inside, weight, 0.0).sum(axis=1)
    return per_history


def run_case(
    physics,
    *,
    energy,
    packets,
    chunk_size,
    seeds,
    max_interactions,
    tau_scattering,
    sigma_limit,
    maximum_relative_error,
    minimum_effective_histories,
    quadrature_rtol,
):
    _, _, _, sigma_scattering, _ = host_material(physics, energy)
    column = tau_scattering / sigma_scattering
    cloud = build_angular_distance_cloud(
        np.full((2, 2, 2), column / 2.0),
        [-15000.0, 15000.0],
        [-15000.0, 15000.0],
        [4.25, 4.75],
        10.0,
    )
    launch = build_rectangular_launch_geometry(10.0, *LAUNCH_BOUNDS)
    bins = build_observer_bin_geometry(
        [-1800.0, 1800.0], [-1800.0, 1800.0],
        [energy - 0.1, energy + 0.1], TIME_EDGES_S,
    )
    references = []
    for n_radius, n_depth in ((24, 24), (64, 56)):
        print(f"Computing radial reference {n_radius} x {n_depth}...", flush=True)
        references.append(
            np.asarray(
                [
                    first_order_radial_quadrature(
                        physics, energy, column, LAUNCH_BOUNDS, TIME_EDGES_S,
                        n_radius=n_radius, n_depth=n_depth,
                        annulus_arcsec=(low, high),
                    ).sum()
                    for low, high in zip(
                        ANNULUS_EDGES_ARCSEC[:-1], ANNULUS_EDGES_ARCSEC[1:],
                        strict=True,
                    )
                ],
                dtype=np.float64,
            )
        )
    coarse, fine = references
    full = float(first_order_radial_quadrature(
        physics, energy, column, LAUNCH_BOUNDS, TIME_EDGES_S,
        n_radius=64, n_depth=56,
    ).sum())

    def batch(key, packets_chunk):
        key_launch, key_transport = random.split(key)
        launched = sample_source_launches(key_launch, packets_chunk, launch)
        transported = transport_photon_batch(
            key_transport, launched.position_pc, launched.momentum_kev,
            cloud, physics, max_interactions=max_interactions,
        )
        events = score_peeloff_events(launched, transported, cloud, physics)
        products = bin_observer_events(events, bins)
        return events, transported.status, products.first_scatter_fluence

    simulate = jax.jit(batch)
    total_sum = np.zeros(len(ANNULUS_EDGES_ARCSEC) - 1, np.float64)
    total_cross = np.zeros_like(total_sum)
    statuses = np.zeros(7, np.int64)
    per_seed = []
    for seed in seeds:
        seed_sum = np.zeros_like(total_sum)
        for index, offset in enumerate(range(0, packets, chunk_size)):
            size = min(chunk_size, packets - offset)
            source = SourcePackets(
                energy_kev=jnp.full((size,), energy),
                emission_time_s=jnp.zeros((size,)),
                weight_observer_fluence=jnp.full((size,), 1.0 / packets),
                time_index=jnp.zeros((size,), jnp.int32),
                spectral_bin_index=jnp.zeros((size,), jnp.int32),
            )
            events, status, first_image = jax.block_until_ready(
                simulate(random.fold_in(random.PRNGKey(seed), index), source)
            )
            history = _annular_photon_weights(
                events, ANNULUS_EDGES_ARCSEC, TIME_EDGES_S
            )
            chunk_sum = history.sum(axis=0)
            if not np.isclose(
                chunk_sum.sum(), float(np.asarray(first_image).sum()),
                rtol=3e-4, atol=1e-10,
            ):
                raise RuntimeError("annular scores do not close to production image")
            total_sum += chunk_sum
            seed_sum += chunk_sum
            total_cross += np.square(history).sum(axis=0)
            statuses += np.bincount(np.asarray(status), minlength=7)
            if index % 10 == 0:
                print(f"seed {seed}: {offset + size:,}/{packets:,}", flush=True)
        per_seed.append(
            {"seed": seed, "annular_first_order_fluence": seed_sum.tolist()}
        )
    n_total = packets * len(seeds)
    total_sum /= len(seeds)
    total_cross /= len(seeds) ** 2
    variance = n_total / (n_total - 1) * np.maximum(
        total_cross - total_sum**2 / n_total, 0.0
    )
    rows = []
    for index, (low, high) in enumerate(
        zip(ANNULUS_EDGES_ARCSEC[:-1], ANNULUS_EDGES_ARCSEC[1:], strict=True)
    ):
        analog = float(total_sum[index])
        photon_se = math.sqrt(float(variance[index]))
        quadrature_se = float(abs(fine[index] - coarse[index]))
        se = math.hypot(photon_se, quadrature_se)
        z = (analog - fine[index]) / se if se > 0 else None
        effective = analog**2 / total_cross[index] if total_cross[index] > 0 else 0.0
        relative = photon_se / analog if analog > 0 else None
        relative_quad = quadrature_se / max(fine[index], 1e-100)
        row = {
            "annulus_arcsec": [low, high],
            "analog": analog,
            "reference": float(fine[index]),
            "photon_standard_error": photon_se,
            "effective_histories": float(effective),
            "relative_photon_standard_error": relative,
            "quadrature_relative_change": relative_quad,
            "z": float(z) if z is not None else None,
            "passed": bool(
                z is not None and abs(z) <= sigma_limit
                and effective >= minimum_effective_histories
                and relative is not None and relative <= maximum_relative_error
                and relative_quad <= quadrature_rtol
            ),
        }
        rows.append(row)
    closure = float(abs(fine.sum() - full) / max(full, 1e-100))
    checks = {
        "annular_reference_closure": closure <= quadrature_rtol,
        "annular_first_order_fluence": all(row["passed"] for row in rows),
        "no_cap_or_invalid_terminal": not any(statuses[i] for i in (0, 4, 5, 6)),
        "status_count_closure": int(statuses.sum()) == n_total,
    }
    return {
        "energy_kev": energy,
        "column_cm2": column,
        "source_fluence": 1.0,
        "time_edges_s": list(TIME_EDGES_S),
        "annulus_edges_arcsec": list(ANNULUS_EDGES_ARCSEC),
        "quadrature_coarse": coarse.tolist(),
        "quadrature_fine": fine.tolist(),
        "full_sky_reference": full,
        "annular_reference_closure_relative": closure,
        "annuli": rows,
        "per_seed": per_seed,
        "analog_statuses": statuses.tolist(),
        "checks": checks,
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
    parser.add_argument("--maximum-relative-error", type=float, default=0.10)
    parser.add_argument("--minimum-effective-histories", type=int, default=30)
    parser.add_argument("--quadrature-rtol", type=float, default=0.02)
    args = parser.parse_args()
    if (
        args.packets < 2 or args.chunk_size < 1 or len(args.seeds) < 3
        or len(set(args.seeds)) != len(args.seeds)
        or not 2.0 <= args.energy <= 10.0 or args.max_interactions < 4
        or not np.isfinite(args.tau_scattering) or args.tau_scattering <= 0
        or not np.isfinite(args.sigma_limit) or args.sigma_limit <= 0
        or not 0 < args.maximum_relative_error < 1
        or args.minimum_effective_histories < 1
        or not 0 < args.quadrature_rtol < 1
    ):
        parser.error("invalid photon, material, or validation settings")
    _, _, physics = load_2_10_material_tables()
    report = {
        "stage": "9F_absorbed_first_order_annuli",
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
            "maximum_relative_error": args.maximum_relative_error,
            "minimum_effective_histories": args.minimum_effective_histories,
            "quadrature_rtol": args.quadrature_rtol,
        },
        "case": run_case(
            physics, energy=args.energy, packets=args.packets,
            chunk_size=args.chunk_size, seeds=args.seeds,
            max_interactions=args.max_interactions,
            tau_scattering=args.tau_scattering, sigma_limit=args.sigma_limit,
            maximum_relative_error=args.maximum_relative_error,
            minimum_effective_histories=args.minimum_effective_histories,
            quadrature_rtol=args.quadrature_rtol,
        ),
    }
    report["all_passed"] = report["case"]["all_passed"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f'Saved {args.output}; all_passed={report["all_passed"]}')
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
