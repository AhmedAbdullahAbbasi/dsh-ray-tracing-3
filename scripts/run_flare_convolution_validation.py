"""Compare a transported DSH flare with its continuous-time impulse convolution.

The same simulated scattering histories supply both sides of this conditional
test. The production source sampler draws independent emission times for the
flare; an analytic time integral of every scored impulse predicts the binned
light curve. Only source-time sampling and observer binning can contribute to
their difference. Run the high-statistics experiment locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from dsh.observer.binning import bin_observer_events, build_observer_bin_geometry
from dsh.observer.scoring import PC_LIGHT_TRAVEL_TIME_S, score_peeloff_events
from dsh.physics.newdust import (
    build_dust_physics_from_newdust,
    load_newdust_scattering_table,
)
from dsh.sources.launch import build_cloud_launch_geometry
from dsh.sources.models import (
    build_post_peak_exponential_band_source,
    sample_tabulated_band_source,
)
from dsh.transport.kernel import transport_photon_batch
from dsh.validation.analytic import phase_containment_angle_rad
from dsh.validation.convolution import convolve_scored_impulses
from dsh.validation.experiments import _uniform_screen_cloud
from dsh.validation.launch import (
    nested_screen_launch_geometries,
    sample_mixture_source_launches,
)

DAY_S = 86_400.0
ANNULI_ARCSEC = ((50.0, 110.0), (110.0, 170.0))


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=966)
    parser.add_argument("--source-distance-kpc", type=float, default=10.0)
    parser.add_argument("--screen-fraction", type=float, default=0.5)
    parser.add_argument("--screen-thickness-kpc", type=float, default=0.01)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--decay-days", type=float, default=0.8)
    parser.add_argument("--source-duration-days", type=float, default=2.0)
    parser.add_argument("--source-bin-days", type=float, default=0.125)
    parser.add_argument("--arrival-bin-days", type=float, default=0.25)
    parser.add_argument("--arrival-duration-days", type=float, default=8.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_outputs/rigorous_flare_convolution.json"),
    )
    return parser.parse_args()


def _validated_edges(duration_days, bin_days):
    if not np.isfinite(duration_days) or not np.isfinite(bin_days):
        raise ValueError("bin duration and width must be finite")
    if duration_days <= 0.0 or bin_days <= 0.0:
        raise ValueError("bin duration and width must be positive")
    edges = np.arange(np.ceil(duration_days / bin_days) + 1) * bin_days
    edges[-1] = duration_days
    return edges * DAY_S


def _source_and_cdf(args, energy):
    edges = _validated_edges(args.source_duration_days, args.source_bin_days)
    tau = args.decay_days * DAY_S
    duration = args.source_duration_days * DAY_S
    peak_flux = 1.0 / (tau * -np.expm1(-duration / tau))
    source = build_post_peak_exponential_band_source(
        edges, [energy], [peak_flux], decay_time_s=tau
    )
    source_edges = np.asarray(source.time_edges_s, dtype=np.float64)
    cell = np.asarray(source.cell_fluence, dtype=np.float64)[:, 0]
    normalized_cdf = np.r_[0.0, np.cumsum(cell)] / cell.sum()
    grid = np.linspace(0.0, duration, 1_001)
    exact_cdf = -np.expm1(-grid / tau) / -np.expm1(-duration / tau)
    max_tabulation_error = float(
        np.max(np.abs(np.interp(grid, source_edges, normalized_cdf) - exact_cdf))
    )
    return source, source_edges, cell, max_tabulation_error


def _annulus_result(actual, statistics, scored_event_count):
    expected = statistics.expected_flare_fluence
    sigma = np.sqrt(statistics.conditional_variance)
    well_sampled = (sigma > 0.0) & (expected >= 5.0 * sigma)
    z = np.divide(
        actual - expected, sigma, out=np.zeros_like(expected), where=sigma > 0
    )
    window_sigma = np.sqrt(statistics.conditional_window_variance)
    total_z = (
        (actual.sum() - statistics.expected_window_fluence) / window_sigma
        if window_sigma > 0.0
        else 0.0
    )
    deterministic_bins_match = np.allclose(
        actual[sigma == 0.0], expected[sigma == 0.0], rtol=1.0e-5, atol=1.0e-12
    )
    deterministic_window_matches = window_sigma > 0.0 or np.isclose(
        actual.sum(), statistics.expected_window_fluence, rtol=1.0e-5, atol=1.0e-12
    )
    return {
        "scored_event_count": int(scored_event_count),
        "well_sampled_time_bins": int(well_sampled.sum()),
        "maximum_absolute_bin_z": float(np.max(np.abs(z[well_sampled]), initial=0.0)),
        "window_total_z": float(total_z),
        "deterministic_bins_match": bool(deterministic_bins_match),
        "deterministic_window_matches": bool(deterministic_window_matches),
        "expected_window_fluence": float(statistics.expected_window_fluence),
        "measured_window_fluence": float(actual.sum()),
        "passes": bool(
            scored_event_count >= 100
            and well_sampled.sum() >= 6
            and np.all(np.abs(z[well_sampled]) < 5.0)
            and abs(total_z) < 5.0
            and deterministic_bins_match
            and deterministic_window_matches
        ),
    }


def main():
    args = _arguments()
    if args.packets <= 0 or args.chunk_size <= 0:
        raise ValueError("packet count and chunk size must be positive")
    if not 0.0 < args.screen_fraction < 1.0 or not np.isfinite(args.decay_days):
        raise ValueError("screen fraction and decay time must be physical")
    if (
        args.decay_days <= 0.0
        or not np.isfinite(args.target_tau)
        or args.target_tau <= 0.0
    ):
        raise ValueError("decay time and scattering optical depth must be positive")
    if args.arrival_duration_days <= args.source_duration_days:
        raise ValueError("arrival window must extend past the source emission window")

    scattering = load_newdust_scattering_table()
    energy = float(scattering.energy_kev[0])
    source, source_edges, source_cell, max_tabulation_error = _source_and_cdf(
        args, energy
    )
    cloud = _uniform_screen_cloud(
        args.source_distance_kpc,
        args.screen_fraction,
        args.screen_thickness_kpc,
        args.target_tau / scattering.scattering_cross_section_cm2_per_h[0],
        2_000.0,
        32,
        4,
    )
    physics = build_dust_physics_from_newdust(
        scattering, np.zeros_like(scattering.energy_kev)
    )
    median_angle = float(
        phase_containment_angle_rad(
            scattering.scattering_angle_rad,
            scattering.scattering_angle_cdf[0],
            0.5,
        )
    )
    proposals = nested_screen_launch_geometries(
        build_cloud_launch_geometry(cloud),
        source_distance_kpc=args.source_distance_kpc,
        fractional_distance=args.screen_fraction,
        median_scattering_angle_rad=median_angle,
    )
    arrival_edges = _validated_edges(args.arrival_duration_days, args.arrival_bin_days)
    bin_geometry = build_observer_bin_geometry(
        [-2_000.0, 2_000.0],
        [-2_000.0, 2_000.0],
        [energy - 0.1, energy + 0.1],
        arrival_edges,
    )
    sample_jit = jax.jit(sample_tabulated_band_source, static_argnames=("n_packets",))
    launch_jit = jax.jit(sample_mixture_source_launches)
    transport_jit = jax.jit(
        transport_photon_batch, static_argnames=("max_interactions",)
    )
    score_jit = jax.jit(score_peeloff_events)
    bin_jit = jax.jit(bin_observer_events)

    observed = np.zeros((len(ANNULI_ARCSEC), len(arrival_edges) - 1), dtype=np.float64)
    event_delays = [[] for _ in ANNULI_ARCSEC]
    event_weights = [[] for _ in ANNULI_ARCSEC]
    source_bin_counts = np.zeros(len(source_edges) - 1, dtype=np.int64)
    key = random.PRNGKey(args.seed)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"3.3-keV post-peak exponential: {args.packets:,} packets")
    for chunk_index, start in enumerate(range(0, args.packets, args.chunk_size)):
        n = min(args.chunk_size, args.packets - start)
        source_key, launch_key, transport_key = random.split(
            random.fold_in(key, chunk_index), 3
        )
        packets = sample_jit(source_key, source, n_packets=n)._replace(
            weight_observer_fluence=jnp.full(
                (n,), source.total_fluence / args.packets, dtype=jnp.float32
            )
        )
        source_bin_counts += np.bincount(
            np.asarray(packets.time_index), minlength=len(source_bin_counts)
        )
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
        sky_radius_sq = events.sky_x_arcsec**2 + events.sky_y_arcsec**2
        valid = np.asarray(events.valid)[:, 0]
        r = np.sqrt(np.asarray(sky_radius_sq)[:, 0])
        for index, (low, high) in enumerate(ANNULI_ARCSEC):
            selected = valid & (r >= low) & (r < high)
            if np.any(selected):
                event_delays[index].append(
                    np.asarray(events.excess_path_length_pc)[selected, 0].astype(
                        np.float64
                    )
                    * PC_LIGHT_TRAVEL_TIME_S
                )
                event_weights[index].append(
                    np.asarray(events.weight_observer_fluence)[selected, 0].astype(
                        np.float64
                    )
                )
            annular_events = events._replace(
                valid=events.valid
                & (sky_radius_sq >= low**2)
                & (sky_radius_sq < high**2)
            )
            observed[index] += np.asarray(
                bin_jit(annular_events, bin_geometry).total_fluence
            )[:, 0, 0, 0].astype(np.float64)
        print(f"  completed {start + n:,}/{args.packets:,}", flush=True)

    impulse = np.zeros_like(observed)
    expected = np.zeros_like(observed)
    standard_error = np.zeros_like(observed)
    annulus_results = []
    for index, (low, high) in enumerate(ANNULI_ARCSEC):
        if not event_delays[index]:
            raise RuntimeError(
                f"no scored scattering in annulus {low:g}–{high:g} arcsec"
            )
        delays = np.concatenate(event_delays[index])
        weights = np.concatenate(event_weights[index])
        stats = convolve_scored_impulses(
            delays, weights, source_edges, source_cell, arrival_edges
        )
        impulse[index] = stats.impulse_fluence
        expected[index] = stats.expected_flare_fluence
        standard_error[index] = np.sqrt(stats.conditional_variance)
        outcome = _annulus_result(observed[index], stats, len(delays))
        outcome["radius_arcsec"] = [low, high]
        annulus_results.append(outcome)
        print(
            f"  {low:g}–{high:g} arcsec: {outcome['well_sampled_time_bins']} usable "
            f"bins; maximum |z|={outcome['maximum_absolute_bin_z']:.2f}; "
            f"window z={outcome['window_total_z']:+.2f}",
            flush=True,
        )

    source_probability = source_cell / source_cell.sum()
    n_expected = args.packets * source_probability
    source_z = (source_bin_counts - n_expected) / np.sqrt(
        n_expected * (1.0 - source_probability)
    )
    source_gate = bool(np.max(np.abs(source_z)) < 5.0)
    report = {
        "schema_version": 1,
        "configuration": vars(args) | {"output": str(args.output)},
        "physics": {
            "energy_kev": energy,
            "absorption_enabled": False,
            "max_interactions": 1,
            "source_fluence_ph_cm2": float(np.asarray(source.total_fluence)),
            "source_description": "finite post-peak exponential sampled from tabulated cell averages",
        },
        "simulation": {
            "source_bin_counts": source_bin_counts.tolist(),
            "maximum_absolute_source_bin_z": float(np.max(np.abs(source_z))),
            "maximum_source_cdf_tabulation_error": max_tabulation_error,
            "annuli": annulus_results,
        },
        "checks": {
            "source_time_sampling": source_gate,
            "continuous_decay_approximation": max_tabulation_error < 0.01,
            "monte_carlo_flare_convolution": all(
                item["passes"] for item in annulus_results
            ),
        },
        "scope": (
            "single-scattering, absorption-free, uniform screen; the exact "
            "conditional convolution isolates time sampling and observer binning"
        ),
    }
    report["checks"]["all_passed"] = all(report["checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    npz_path = args.output.with_name(args.output.stem + "_lightcurves.npz")
    np.savez_compressed(
        npz_path,
        arrival_time_edges_s=arrival_edges,
        source_time_edges_s=source_edges,
        source_cell_fluence_ph_cm2=source_cell,
        annuli_arcsec=np.asarray(ANNULI_ARCSEC),
        impulse_fluence_ph_cm2=impulse,
        expected_flare_fluence_ph_cm2=expected,
        simulated_flare_fluence_ph_cm2=observed,
        conditional_standard_error_ph_cm2=standard_error,
    )
    report["simulation"]["lightcurve_npz"] = str(npz_path)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved: {args.output.resolve()}")
    print(f"Saved: {npz_path.resolve()}")
    print("PASS" if report["checks"]["all_passed"] else "FAIL")
    if not report["checks"]["all_passed"]:
        raise SystemExit(
            "validation checks failed: "
            + ", ".join(name for name, good in report["checks"].items() if not good)
        )


if __name__ == "__main__":
    main()
