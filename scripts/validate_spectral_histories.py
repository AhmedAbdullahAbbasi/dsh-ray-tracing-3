"""Independent direct transport versus reuse of an energy/sky/delay response.

The frozen experiment uses an absorbing WD01 screen, hard-to-soft intrinsic
spectra with a dark time interval, continuous energies, and bins on both sides
of the native Fe K edge. Response and direct runs have independent random keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from scipy.stats import chi2

from dsh.observer.binning import bin_observer_events, build_observer_bin_geometry
from dsh.observer.response import (
    fold_spectral_response,
    load_spectral_response,
    write_spectral_response,
)
from dsh.observer.response_transport import collect_spectral_response
from dsh.observer.scoring import score_peeloff_events
from dsh.physics.wd01_ism import load_wd01_ism
from dsh.sources.histories import FLUX_CONVENTION, HISTORY_UNITS
from dsh.sources.launch import sample_source_launches
from dsh.sources.models import (
    build_powerlaw_band_source,
    build_tabulated_band_source,
    sample_tabulated_band_source,
)
from dsh.sources.radial_launch import build_radial_cloud_launch_geometry
from dsh.transport.kernel import transport_photon_batch
from dsh.validation.experiments import _uniform_screen_cloud

DAY = 86400.0
ENERGIES = np.array([2, 4, 6.9, 7.124, 7.4, 10], dtype=np.float32)
ARRIVAL = np.array([0, 0.05, 0.15, 0.35, 1.5, 12], dtype=np.float32) * DAY
RADII = np.array([0, 90, 600], dtype=np.float32)


def compare(expected, actual, covariance):
    e, a = expected.ravel(), actual.ravel()
    sigma = np.sqrt(np.maximum(np.diag(covariance), 0))
    well = (sigma > 0) & (e > 5 * sigma)
    z = np.divide(a - e, sigma, out=np.zeros_like(e), where=sigma > 0)
    total_sigma = np.sqrt(max(float(covariance.sum()), 0))
    total_z = float((a.sum() - e.sum()) / total_sigma) if total_sigma > 0 else 0.0
    block = covariance[np.ix_(well, well)]
    delta = (a - e)[well]
    rank = int(np.linalg.matrix_rank(block))
    statistic = float(delta @ np.linalg.pinv(block) @ delta)
    probability = float(chi2.sf(statistic, rank)) if rank else 0.0
    return {
        "response_fluence": float(e.sum()),
        "direct_fluence": float(a.sum()),
        "window_total_z": total_z,
        "well_sampled_bins": int(well.sum()),
        "maximum_absolute_bin_z": float(np.max(abs(z[well]), initial=0)),
        "global_chi2": statistic,
        "global_degrees_of_freedom": rank,
        "global_pvalue": probability,
        "passes": bool(
            well.sum() >= 12
            and abs(total_z) < 5
            and np.all(abs(z[well]) < 5)
            and probability > 0.001
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, default=4194304)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--max-interactions", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.packets < 2 or args.chunk_size < 2 or args.packets % args.chunk_size:
        raise ValueError(
            "validation requires complete chunks of at least two histories"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    definition, _, _, _, physics = load_wd01_ism(
        "dsh/data/materials/wd01_wilm_c50_si90_v1"
    )
    cloud = _uniform_screen_cloud(10, 0.5, 0.03, 1e23, 1800, 8, 2)
    launch = build_radial_cloud_launch_geometry(cloud, physics)
    proposal = build_powerlaw_band_source([0, 1], [1], ENERGIES, 1.0)
    # Oversample the Fe-edge neighborhood while preserving support at all E.
    proposal = build_tabulated_band_source(
        [0, 1], [[0.25, 0.2, 0.15, 0.2, 0.2]], np.sqrt(ENERGIES[:-1] * ENERGIES[1:])
    )._replace(
        energy_edges_kev=proposal.energy_edges_kev,
        photon_index=proposal.photon_index,
    )
    source = build_powerlaw_band_source(
        [0, 1800, 5400, 21600], [0.04, 0, 0.008], ENERGIES, [1.0, 2.7, 2.2]
    )
    history = {
        "schema_version": 1,
        "flux_convention": FLUX_CONVENTION,
        "units": HISTORY_UNITS,
        "epoch": "t=0 of controlled screen experiment",
        "time_scale": "simulation-relative",
        "time_edges_s": np.asarray(source.time_edges_s).tolist(),
        "energy_edges_kev": np.asarray(source.energy_edges_kev).tolist(),
        "band_flux": np.asarray(source.band_flux).tolist(),
        "photon_index": np.asarray(source.photon_index).tolist(),
    }
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print("Collecting independent response histories", flush=True)
    response = collect_spectral_response(
        proposal,
        cloud,
        physics,
        launch,
        n_packets=args.packets,
        chunk_size=args.chunk_size,
        max_interactions=args.max_interactions,
        seed=args.seed,
        progress=lambda n, total: (
            print(f"Response: {n:,}/{total:,}", flush=True)
            if n % (8 * args.chunk_size) == 0 or n == total
            else None
        ),
    )
    response.provenance["material_definition_sha256"] = definition.sha256
    response_path = write_spectral_response(args.output / "response.npz", response)
    response = load_spectral_response(response_path)
    folded = fold_spectral_response(response, source, ARRIVAL, ENERGIES, RADII)
    first = fold_spectral_response(
        response, source, ARRIVAL, ENERGIES, RADII, scattering_orders=[1]
    )
    multiple = fold_spectral_response(
        response,
        source,
        ARRIVAL,
        ENERGIES,
        RADII,
        scattering_orders=np.arange(2, args.max_interactions + 1),
    )
    geometry = build_observer_bin_geometry([-600, 600], [-600, 600], ENERGIES, ARRIVAL)

    @jax.jit
    def batch(key):
        source_key, launch_key, transport_key = random.split(key, 3)
        packets = sample_tabulated_band_source(
            source_key, source, args.chunk_size
        )._replace(
            weight_observer_fluence=jnp.full(
                (args.chunk_size,), source.total_fluence / args.packets
            ),
        )
        launched = sample_source_launches(launch_key, packets, launch)
        transported = transport_photon_batch(
            transport_key,
            launched.position_pc,
            launched.momentum_kev,
            cloud,
            physics,
            max_interactions=args.max_interactions,
        )
        events = score_peeloff_events(launched, transported, cloud, physics)
        return packets, transported.status, events

    bin_jit = jax.jit(bin_observer_events)
    shape = folded.fluence.shape
    n_bins = folded.fluence.size
    sums = np.zeros((3, n_bins))
    second = np.zeros((3, n_bins, n_bins))
    statuses = np.zeros(7, dtype=int)
    joint_counts = np.zeros(source.band_flux.shape, dtype=int)
    maximum_binning_relative_difference = 0.0
    direct_multiple_events = 0
    for chunk, start in enumerate(range(0, args.packets, args.chunk_size)):
        packets, status, events = batch(
            random.fold_in(random.PRNGKey(args.seed + 1), chunk)
        )
        statuses += np.bincount(np.asarray(status), minlength=7)
        np.add.at(
            joint_counts,
            (np.asarray(packets.time_index), np.asarray(packets.spectral_bin_index)),
            1,
        )
        valid = np.asarray(events.valid)
        rows = np.broadcast_to(np.arange(args.chunk_size)[:, None], valid.shape)[valid]
        time = np.asarray(events.arrival_time_s)[valid]
        energy = np.asarray(events.energy_kev)[valid]
        radius = np.hypot(
            np.asarray(events.sky_x_arcsec)[valid],
            np.asarray(events.sky_y_arcsec)[valid],
        )
        order = np.asarray(events.scattering_order)[valid]
        weight = np.asarray(events.weight_observer_fluence)[valid].astype(float)
        indices = [
            np.searchsorted(axis, value, side="right") - 1
            for axis, value in ((ARRIVAL, time), (ENERGIES, energy), (RADII, radius))
        ]
        for axis, value, index in zip(
            (ARRIVAL, ENERGIES, RADII), (time, energy, radius), indices, strict=True
        ):
            index[value == axis[-1]] = len(axis) - 2
        keep = np.ones(rows.size, dtype=bool)
        for index, size in zip(indices, shape, strict=True):
            keep &= (index >= 0) & (index < size)
        flat = (indices[0] * shape[1] + indices[1]) * shape[2] + indices[2]
        direct_multiple_events += int(np.sum(order[keep] > 1))
        group_all = None
        for k, select in enumerate((keep, keep & (order == 1), keep & (order > 1))):
            grouped = np.zeros((args.chunk_size, n_bins))
            np.add.at(grouped, (rows[select], flat[select]), weight[select])
            sums[k] += grouped.sum(axis=0)
            second[k] += grouped.T @ grouped
            if k == 0:
                group_all = grouped.sum(axis=0).reshape(shape)
        # Check the production JAX observer binning independently of the host
        # per-history grouping used for the comparison covariance.
        event_radius = jnp.hypot(events.sky_x_arcsec, events.sky_y_arcsec)
        for r, (low, high) in enumerate(zip(RADII[:-1], RADII[1:], strict=True)):
            selected = (event_radius >= low) & (
                event_radius < high if r < len(RADII) - 2 else event_radius <= high
            )
            product = bin_jit(events._replace(valid=events.valid & selected), geometry)
            actual = np.asarray(product.total_fluence)[:, :, 0, 0]
            reference = group_all[:, :, r]
            denominator = np.maximum(abs(reference), 1e-12)
            maximum_binning_relative_difference = max(
                maximum_binning_relative_difference,
                float(np.max(abs(actual - reference) / denominator)),
            )
        if (chunk + 1) % 8 == 0 or start + args.chunk_size == args.packets:
            print(f"Direct: {start + args.chunk_size:,}/{args.packets:,}", flush=True)
    covariance = np.array(
        [
            args.packets
            / (args.packets - 1)
            * (moment - np.outer(total, total) / args.packets)
            for moment, total in zip(second, sums, strict=True)
        ]
    )
    controls = {}
    for k, (name, fold) in enumerate(
        (("total", folded), ("first", first), ("multiple", multiple))
    ):
        controls[name] = compare(
            fold.fluence, sums[k].reshape(shape), fold.covariance + covariance[k]
        )
    expected_cells = np.asarray(source.cell_fluence, dtype=float)
    probabilities = expected_cells / expected_cells.sum()
    count_sigma = np.sqrt(args.packets * probabilities * (1 - probabilities))
    count_z = np.divide(
        joint_counts - args.packets * probabilities,
        count_sigma,
        out=np.zeros_like(probabilities),
        where=count_sigma > 0,
    )
    gates = {
        "independent_total_response": controls["total"]["passes"],
        "independent_first_order_response": controls["first"]["passes"],
        "independent_multiple_order_response": controls["multiple"]["passes"],
        "joint_source_sampling": bool(
            np.all(abs(count_z) < 5) and np.all(joint_counts[probabilities == 0] == 0)
        ),
        "production_binning": maximum_binning_relative_difference < 2e-5,
        "transport_complete": bool(
            statuses[[0, 4, 5, 6]].sum() == 0 and statuses.sum() == args.packets
        ),
        "multiple_orders_exercised": direct_multiple_events >= 100,
        "response_order_closure": bool(
            np.allclose(
                folded.fluence, first.fluence + multiple.fluence, rtol=1e-12, atol=1e-13
            )
        ),
    }
    summary = {
        "all_passed": all(gates.values()),
        "gates": gates,
        "controls": controls,
        "histories_each_run": args.packets,
        "chunk_size": args.chunk_size,
        "max_interactions": args.max_interactions,
        "seed": args.seed,
        "source_total_fluence": float(source.total_fluence),
        "response_events": len(response.energy_kev),
        "response_multiple_events": int(np.sum(response.scattering_order > 1)),
        "direct_multiple_events_in_window": direct_multiple_events,
        "direct_transport_status_counts": statuses.tolist(),
        "maximum_joint_cell_absolute_z": float(np.max(abs(count_z))),
        "maximum_production_binning_relative_difference": maximum_binning_relative_difference,
        "geometry": {
            "source_distance_kpc": 10,
            "screen_fraction": 0.5,
            "thickness_kpc": 0.03,
            "column_cm2": 1e23,
            "half_width_arcsec": 1800,
        },
        "response_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
        "response_provenance": response.provenance,
        "scope": "Independent Monte Carlo check of joint spectral histories and physical halo-response reuse; no new external halo or instrument calibration reference",
        "xspec_status": "user reported checks pass; raw XSPEC report not available in this environment",
    }
    np.savez_compressed(
        args.output / "comparison_arrays.npz",
        response_fluence=folded.fluence,
        direct_fluence=sums[0].reshape(shape),
        response_covariance=folded.covariance,
        direct_covariance=covariance[0],
        arrival_time_edges_s=ARRIVAL,
        energy_edges_kev=ENERGIES,
        radial_edges_arcsec=RADII,
        source_joint_counts=joint_counts,
        source_cell_probabilities=probabilities,
    )
    (args.output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {"all_passed": summary["all_passed"], "gates": gates, "controls": controls},
            indent=2,
        )
    )
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
