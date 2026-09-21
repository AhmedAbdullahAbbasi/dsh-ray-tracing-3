"""Run high-statistics DSH validation experiments and write a JSON report."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np
from jax import random

from dsh.geometry.coordinates import ARCSEC_TO_RAD
from dsh.physics.newdust import load_newdust_scattering_table
from dsh.validation import (
    finite_screen_ring_bounds_arcsec,
    log_log_power_law_slope,
    phase_containment_angle_rad,
    small_angle_ring_radius_arcsec,
)
from dsh.validation.experiments import run_uniform_screen_validation


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=806)
    parser.add_argument("--source-distance-kpc", type=float, default=10.0)
    parser.add_argument("--screen-fraction", type=float, default=0.5)
    parser.add_argument("--screen-thickness-kpc", type=float, default=0.01)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--half-width-arcsec", type=float, default=2_000.0)
    parser.add_argument("--sky-pixels", type=int, default=32)
    parser.add_argument("--radial-cells", type=int, default=4)
    parser.add_argument(
        "--thickness-sweep-kpc",
        type=float,
        nargs="+",
        default=(0.001, 0.01, 0.1),
        help="increasing screen thicknesses for the 3.3-keV smearing test",
    )
    parser.add_argument("--sweep-packets", type=int, default=300_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_outputs/rigorous_dsh_validation.json"),
    )
    return parser.parse_args()


def _pass_fail_summary(
    results, angle_slope, cross_section_slope, scored_fluence_slope, thickness_sweep
):
    checks = {
        "geometry_delay": all(
            result.maximum_relative_delay_error < 1.0e-3 for result in results
        ),
        "analog_flux_conservation": all(
            abs(result.analog_binomial_z) < 5.0 for result in results
        ),
        "peeloff_flux_conservation": all(
            0.90 < result.scored_fluence_to_tau < 1.10 for result in results
        ),
        "energy_width_scaling": abs(angle_slope + 1.0) < 0.03,
        "energy_opacity_scaling": abs(cross_section_slope + 2.0) < 0.03,
        "simulated_energy_fluence_scaling": (
            abs(scored_fluence_slope - cross_section_slope) < 0.20
        ),
        "azimuthal_symmetry": all(
            max(result.azimuthal_harmonic_amplitudes)
            < 5.0 / np.sqrt(result.azimuthal_effective_sample_size)
            for result in results
        ),
        "thin_screen_convergence": all(
            right.center_screen_delay_residual_rms_fraction
            > left.center_screen_delay_residual_rms_fraction
            for left, right in zip(
                thickness_sweep[:-1], thickness_sweep[1:], strict=True
            )
        ),
    }
    return checks | {"all_passed": all(checks.values())}


def main():
    args = parse_arguments()
    if args.packets <= 0 or args.chunk_size <= 0 or args.sweep_packets <= 0:
        raise ValueError("packet and chunk counts must be positive")
    thicknesses = np.asarray(args.thickness_sweep_kpc, dtype=np.float64)
    if (
        thicknesses.ndim != 1
        or thicknesses.size < 2
        or np.any(~np.isfinite(thicknesses))
        or np.any(thicknesses <= 0.0)
        or np.any(np.diff(thicknesses) <= 0.0)
    ):
        raise ValueError("thickness-sweep-kpc must be finite, positive, and increasing")
    scattering = load_newdust_scattering_table()
    # Hold the physical hydrogen column fixed across the energy sweep.  The
    # resulting optical depths then follow the tabulated sigma_sca(E).
    reference_column_cm2 = (
        args.target_tau / scattering.scattering_cross_section_cm2_per_h[0]
    )
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(
        f"Running {len(scattering.energy_kev)} energies with "
        f"{args.packets:,} packets each"
    )

    results = []
    root_key = random.PRNGKey(args.seed)
    for energy_index, energy in enumerate(scattering.energy_kev):
        print(f"  {energy:.1f} keV ...", flush=True)
        result = run_uniform_screen_validation(
            random.fold_in(root_key, energy_index),
            scattering,
            energy_index=energy_index,
            packet_count=args.packets,
            chunk_size=args.chunk_size,
            source_distance_kpc=args.source_distance_kpc,
            fractional_distance=args.screen_fraction,
            thickness_kpc=args.screen_thickness_kpc,
            target_scattering_optical_depth=float(
                reference_column_cm2
                * scattering.scattering_cross_section_cm2_per_h[energy_index]
            ),
            half_width_arcsec=args.half_width_arcsec,
            sky_pixels=args.sky_pixels,
            radial_cells=args.radial_cells,
        )
        results.append(result)
        print(
            "    analog fraction / expected: "
            f"{result.analog_scattered_fraction:.7g} / "
            f"{result.expected_interaction_probability:.7g}; "
            f"z={result.analog_binomial_z:+.2f}"
        )
        print(
            "    scored fluence / tau: "
            f"{result.scored_observer_fluence:.7g} / "
            f"{result.target_scattering_optical_depth:.7g}; "
            f"ratio={result.scored_fluence_to_tau:.4f}"
        )

    median_angles = phase_containment_angle_rad(
        scattering.scattering_angle_rad,
        scattering.scattering_angle_cdf,
        0.5,
    )
    analytic_angle_slope = log_log_power_law_slope(scattering.energy_kev, median_angles)
    analytic_cross_section_slope = log_log_power_law_slope(
        scattering.energy_kev,
        scattering.scattering_cross_section_cm2_per_h,
    )
    simulated_radius_slope = log_log_power_law_slope(
        scattering.energy_kev,
        [result.weighted_median_radius_arcsec for result in results],
    )
    simulated_fluence_slope = log_log_power_law_slope(
        scattering.energy_kev,
        [result.scored_observer_fluence for result in results],
    )

    print("Running 3.3-keV thickness convergence:")
    thickness_results = []
    for sweep_index, thickness in enumerate(thicknesses):
        print(f"  thickness={thickness:g} kpc ...", flush=True)
        result = run_uniform_screen_validation(
            random.fold_in(root_key, 10_000 + sweep_index),
            scattering,
            energy_index=0,
            packet_count=args.sweep_packets,
            chunk_size=min(args.chunk_size, args.sweep_packets),
            source_distance_kpc=args.source_distance_kpc,
            fractional_distance=args.screen_fraction,
            thickness_kpc=float(thickness),
            target_scattering_optical_depth=args.target_tau,
            half_width_arcsec=args.half_width_arcsec,
            sky_pixels=args.sky_pixels,
            radial_cells=args.radial_cells,
        )
        thickness_results.append(result)
        print(
            "    center-screen fractional delay width: "
            f"{result.center_screen_delay_residual_rms_fraction:.7g}"
        )

    example_delay_s = 10.0 * 86_400.0
    thickness_fraction = args.screen_thickness_kpc / args.source_distance_kpc
    screen_bounds = (
        args.screen_fraction - 0.5 * thickness_fraction,
        args.screen_fraction + 0.5 * thickness_fraction,
    )
    thickness_reference = {
        "delay_s": example_delay_s,
        "zero_thickness_radius_arcsec": float(
            small_angle_ring_radius_arcsec(
                example_delay_s,
                args.source_distance_kpc,
                args.screen_fraction,
            )
        ),
        "finite_screen_inner_outer_arcsec": finite_screen_ring_bounds_arcsec(
            example_delay_s,
            args.source_distance_kpc,
            screen_bounds,
        ).tolist(),
    }
    checks = _pass_fail_summary(
        results,
        simulated_radius_slope,
        analytic_cross_section_slope,
        simulated_fluence_slope,
        thickness_results,
    )
    report = {
        "schema_version": 1,
        "configuration": vars(args) | {"output": str(args.output)},
        "hydrogen_column_cm2": float(reference_column_cm2),
        "table_energy_kev": scattering.energy_kev.tolist(),
        "analytic": {
            "median_scattering_angle_arcsec": (median_angles / ARCSEC_TO_RAD).tolist(),
            "median_angle_energy_slope": analytic_angle_slope,
            "scattering_cross_section_energy_slope": analytic_cross_section_slope,
            "finite_thickness_reference": thickness_reference,
        },
        "simulation": {
            "median_observed_radius_energy_slope": simulated_radius_slope,
            "scored_observer_fluence_energy_slope": simulated_fluence_slope,
            "uniform_screen": [asdict(result) for result in results],
            "thickness_sweep_3p3_kev": [asdict(result) for result in thickness_results],
        },
        "checks": checks,
        "scope": {
            "single_grain": (
                "not evaluated: the v1 table is already integrated over the MRN "
                "grain-size distribution"
            ),
            "literature_cross_validation": (
                "not evaluated: requires a frozen external reference model and "
                "matched dust/geometry conventions"
            ),
            "instrument_response": "outside Version 1",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Saved: {args.output.resolve()}")
    print("PASS" if checks["all_passed"] else "FAIL")
    if not checks["all_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"validation checks failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
