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
        "--image-screen-fractions",
        type=float,
        nargs="+",
        default=(0.1, 0.5, 0.9),
        help="observer-to-dust distance fractions for simulated ring images",
    )
    parser.add_argument(
        "--image-sweep-packets",
        type=int,
        default=300_000,
        help="packets for each extra image (the main screen reuses its energy run)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_outputs/rigorous_dsh_validation.json"),
    )
    return parser.parse_args()


def _ring_image_passes(image):
    return (
        image is not None
        and image.ring_bins_checked >= 6
        and image.maximum_ring_bound_violation_arcsec
        <= image.half_pixel_diagonal_arcsec
    )


def _screen_radius_order_passes(image_screen_sweep):
    """Compare image radii in at least six shared time bins across screens."""

    if len(image_screen_sweep) < 3:
        return False
    if any(result.image is None for _, result in image_screen_sweep):
        return False
    radii = [
        {
            slice_.time_bin_index: slice_.measured_median_radius_arcsec
            for slice_ in result.image.ring_slices
        }
        for _, result in image_screen_sweep
    ]
    shared_bins = set.intersection(*(set(measurements) for measurements in radii))
    return len(shared_bins) >= 6 and all(
        all(
            left[bin_index] > right[bin_index]
            for left, right in zip(radii[:-1], radii[1:], strict=True)
        )
        for bin_index in shared_bins
    )


def _pass_fail_summary(
    results,
    angle_slope,
    analytic_angle_slope,
    angle_slope_standard_error,
    cross_section_slope,
    scored_fluence_slope,
    thickness_sweep,
    image_screen_sweep,
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
        "energy_width_scaling": (
            abs(angle_slope - analytic_angle_slope) < 3.0 * angle_slope_standard_error
        ),
        "energy_width_precision": angle_slope_standard_error < 0.03,
        "energy_opacity_scaling": abs(cross_section_slope + 2.0) < 0.03,
        "simulated_energy_fluence_scaling": (
            abs(scored_fluence_slope - cross_section_slope) < 0.20
        ),
        "observer_mc_precision": all(
            result.observer_fluence_relative_standard_error < 0.05
            and result.azimuthal_effective_sample_size >= 400.0
            for result in results
        ),
        "image_ring_geometry": _ring_image_passes(results[0].image),
        "image_screen_geometry": all(
            _ring_image_passes(result.image)
            and result.maximum_relative_delay_error < 1.0e-3
            for _, result in image_screen_sweep
        ),
        "image_screen_fraction_order": _screen_radius_order_passes(image_screen_sweep),
        "image_screen_fluence_closure": all(
            result.image is not None
            and abs(
                result.image.binned_fluence
                + result.image.unbinned_fluence
                - result.scored_observer_fluence
            )
            < 1.0e-5 * result.scored_observer_fluence
            for _, result in image_screen_sweep
        ),
        "image_fluence_closure": (
            results[0].image is not None
            and abs(
                results[0].image.binned_fluence
                + results[0].image.unbinned_fluence
                - results[0].scored_observer_fluence
            )
            < 1.0e-5 * results[0].scored_observer_fluence
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
    if (
        args.packets <= 0
        or args.chunk_size <= 0
        or args.sweep_packets <= 0
        or args.image_sweep_packets <= 0
    ):
        raise ValueError("packet and chunk counts must be positive")
    image_fractions = np.asarray(args.image_screen_fractions, dtype=np.float64)
    if (
        image_fractions.ndim != 1
        or image_fractions.size < 3
        or not np.all(np.isfinite(image_fractions))
        or np.any(image_fractions <= 0.0)
        or np.any(image_fractions >= 1.0)
        or np.any(np.diff(image_fractions) <= 0.0)
    ):
        raise ValueError(
            "image-screen-fractions must have three or more distinct increasing values in (0, 1)"
        )
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
            image_output_path=(
                str(args.output.with_name(args.output.stem + "_3p3_image.npz"))
                if energy_index == 0
                else None
            ),
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
            f"ratio={result.scored_fluence_to_tau:.4f}; "
            f"relative MC SE={result.observer_fluence_relative_standard_error:.4f}; "
            f"aperture N_eff={result.azimuthal_effective_sample_size:.0f}"
        )
        if result.image is not None:
            print(
                f"    image ring slices: {result.image.ring_bins_checked}; "
                "largest radius outside analytic time-bin bounds: "
                f"{result.image.maximum_ring_bound_violation_arcsec:.2f} arcsec"
            )
            print(f"    saved simulated image: {result.image.path}")
            if result.image.fits_path is not None:
                print(f"    saved FITS image: {result.image.fits_path}")

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
    log_energies = np.log(scattering.energy_kev)
    slope_gradient = (log_energies - np.mean(log_energies)) / np.sum(
        (log_energies - np.mean(log_energies)) ** 2
    )
    relative_median_errors = np.asarray(
        [
            result.median_radius_standard_error_arcsec
            / result.weighted_median_radius_arcsec
            for result in results
        ]
    )
    simulated_radius_slope_se = float(
        np.linalg.norm(slope_gradient * relative_median_errors)
    )
    print(
        "Observed halo-width energy slope: "
        f"{simulated_radius_slope:.4f} +/- {simulated_radius_slope_se:.4f} "
        f"(table: {analytic_angle_slope:.4f})"
    )
    simulated_fluence_slope = log_log_power_law_slope(
        scattering.energy_kev,
        [result.scored_observer_fluence for result in results],
    )

    print("Running 3.3-keV screen-distance image checks:")
    image_screen_sweep = []
    for sweep_index, fraction in enumerate(image_fractions):
        if fraction == args.screen_fraction:
            result = results[0]
            print(f"  x={fraction:g}: reusing main image", flush=True)
        else:
            fraction_label = f"{fraction:.9g}".replace(".", "p")
            output_path = args.output.with_name(
                args.output.stem + f"_x{fraction_label}_3p3_image.npz"
            )
            print(f"  x={fraction:g} ...", flush=True)
            result = run_uniform_screen_validation(
                random.fold_in(root_key, 20_000 + sweep_index),
                scattering,
                energy_index=0,
                packet_count=args.image_sweep_packets,
                chunk_size=min(args.chunk_size, args.image_sweep_packets),
                source_distance_kpc=args.source_distance_kpc,
                fractional_distance=float(fraction),
                thickness_kpc=args.screen_thickness_kpc,
                target_scattering_optical_depth=args.target_tau,
                half_width_arcsec=args.half_width_arcsec,
                sky_pixels=args.sky_pixels,
                radial_cells=args.radial_cells,
                image_output_path=str(output_path),
            )
            print(f"    saved simulated image: {result.image.path}")
            if result.image.fits_path is not None:
                print(f"    saved FITS image: {result.image.fits_path}")
        print(
            f"    ring slices: {result.image.ring_bins_checked}; "
            "largest radius outside analytic time-bin bounds: "
            f"{result.image.maximum_ring_bound_violation_arcsec:.2f} arcsec",
            flush=True,
        )
        image_screen_sweep.append((float(fraction), result))

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
        analytic_angle_slope,
        simulated_radius_slope_se,
        analytic_cross_section_slope,
        simulated_fluence_slope,
        thickness_results,
        image_screen_sweep,
    )
    report = {
        "schema_version": 3,
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
            "median_observed_radius_energy_slope_standard_error": (
                simulated_radius_slope_se
            ),
            "scored_observer_fluence_energy_slope": simulated_fluence_slope,
            "uniform_screen": [asdict(result) for result in results],
            "image_screen_sweep_3p3_kev": [
                {
                    "fractional_distance": fraction,
                    "packet_count": result.packet_count,
                    "scored_event_count": result.scored_event_count,
                    "maximum_relative_delay_error": result.maximum_relative_delay_error,
                    "image": asdict(result.image),
                }
                for fraction, result in image_screen_sweep
            ],
            "thickness_sweep_3p3_kev": [asdict(result) for result in thickness_results],
        },
        "checks": checks,
        "scope": {
            "simulated_image": (
                "3.3-keV uniform-screen images at the configured screen "
                "fractions; instrument response remains untested"
            ),
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
