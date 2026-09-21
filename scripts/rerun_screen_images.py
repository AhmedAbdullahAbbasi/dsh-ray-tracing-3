"""Increase screen-image statistics while retaining a completed energy sweep.

This resamples the non-primary screen fractions in a Version-3 validation
report. Its original energy and thickness results are kept unchanged; the
screen-image checks are recalculated using newly transported photons.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from jax import random

from dsh.physics.newdust import load_newdust_scattering_table
from dsh.validation.experiments import run_uniform_screen_validation
from scripts.run_validation_ladder import (
    _ring_image_passes,
    _screen_radius_order_passes,
)


def _saved_primary_result(report):
    """Restore only the fields needed to recheck the saved primary image."""

    saved = report["simulation"]["uniform_screen"][0]
    image = saved["image"]
    if image is None:
        raise ValueError("the previous validation report has no primary image")
    restored_image = SimpleNamespace(
        **{
            **image,
            "ring_slices": [SimpleNamespace(**item) for item in image["ring_slices"]],
        }
    )
    return SimpleNamespace(
        image=restored_image,
        packet_count=saved["packet_count"],
        scored_event_count=saved["scored_event_count"],
        scored_observer_fluence=saved["scored_observer_fluence"],
        maximum_relative_delay_error=saved["maximum_relative_delay_error"],
    )


def _update_screen_checks(report, screen_results):
    """Replace image results and gates without changing older physics gates."""

    checks = report["checks"].copy()
    checks["image_ring_geometry"] = _ring_image_passes(
        _saved_primary_result(report).image
    )
    checks["image_screen_geometry"] = all(
        _ring_image_passes(result.image)
        and result.maximum_relative_delay_error < 1.0e-3
        for _, result in screen_results
    )
    checks["image_screen_fraction_order"] = _screen_radius_order_passes(screen_results)
    checks["image_screen_fluence_closure"] = all(
        abs(
            result.image.binned_fluence
            + result.image.unbinned_fluence
            - result.scored_observer_fluence
        )
        < 1.0e-5 * result.scored_observer_fluence
        for _, result in screen_results
    )
    checks["all_passed"] = all(
        value for name, value in checks.items() if name != "all_passed"
    )
    report["checks"] = checks
    report["simulation"]["image_screen_sweep_3p3_kev"] = [
        {
            "fractional_distance": fraction,
            "packet_count": result.packet_count,
            "scored_event_count": result.scored_event_count,
            "maximum_relative_delay_error": result.maximum_relative_delay_error,
            "image": (
                asdict(result.image)
                if not isinstance(result.image, SimpleNamespace)
                else {
                    **vars(result.image),
                    "ring_slices": [
                        vars(slice_) for slice_ in result.image.ring_slices
                    ],
                }
            ),
        }
        for fraction, result in screen_results
    ]
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-report", type=Path, required=True)
    parser.add_argument("--packets", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--seed", type=int, default=None, help="defaults to the previous seed plus one"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.packets <= 0 or args.chunk_size <= 0 or args.output == args.previous_report:
        raise ValueError(
            "packets and chunk size must be positive, and output must be new"
        )
    report = json.loads(args.previous_report.read_text(encoding="utf-8"))
    if report.get("schema_version") != 3:
        raise ValueError(
            "resuming screen images requires a Version-3 validation report"
        )
    config = report["configuration"]
    fractions = config["image_screen_fractions"]
    if len(fractions) < 3 or config["screen_fraction"] not in fractions:
        raise ValueError("the saved primary screen must occur in the image sweep")
    if any(
        not passed
        for name, passed in report["checks"].items()
        if name
        not in {
            "all_passed",
            "image_screen_geometry",
            "image_screen_fraction_order",
            "image_screen_fluence_closure",
        }
    ):
        raise ValueError("other validation checks failed; rerun the complete ladder")
    scattering = load_newdust_scattering_table()
    if not np.array_equal(scattering.energy_kev, report["table_energy_kev"]):
        raise ValueError("the scattering energy grid changed since the original run")

    seed = config["seed"] + 1 if args.seed is None else args.seed
    key = random.PRNGKey(seed)
    screen_results = []
    for index, fraction in enumerate(fractions):
        if fraction == config["screen_fraction"]:
            result = _saved_primary_result(report)
            print(
                f"x={fraction:g}: reusing {result.packet_count:,}-packet primary image"
            )
        else:
            fraction_label = f"{fraction:.9g}".replace(".", "p")
            path = args.output.with_name(
                args.output.stem + f"_x{fraction_label}_3p3_image.npz"
            )
            print(
                f"x={fraction:g}: transporting {args.packets:,} packets ...", flush=True
            )
            result = run_uniform_screen_validation(
                random.fold_in(key, index),
                scattering,
                energy_index=0,
                packet_count=args.packets,
                chunk_size=args.chunk_size,
                source_distance_kpc=config["source_distance_kpc"],
                fractional_distance=float(fraction),
                thickness_kpc=config["screen_thickness_kpc"],
                target_scattering_optical_depth=float(
                    report["hydrogen_column_cm2"]
                    * scattering.scattering_cross_section_cm2_per_h[0]
                ),
                half_width_arcsec=config["half_width_arcsec"],
                sky_pixels=config["sky_pixels"],
                radial_cells=config["radial_cells"],
                image_output_path=str(path),
            )
            print(f"  saved NPZ: {result.image.path}")
            if result.image.fits_path is not None:
                print(f"  saved FITS: {result.image.fits_path}")
        print(
            f"  ring slices: {result.image.ring_bins_checked}; "
            "largest radius outside analytic time-bin bounds: "
            f"{result.image.maximum_ring_bound_violation_arcsec:.2f} arcsec"
        )
        screen_results.append((float(fraction), result))

    checks = _update_screen_checks(report, screen_results)
    report["configuration"].update(
        {
            "output": str(args.output),
            "screen_rerun_previous_report": str(args.previous_report),
            "screen_rerun_packets": args.packets,
            "screen_rerun_seed": seed,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved: {args.output.resolve()}")
    print("PASS" if checks["all_passed"] else "FAIL")
    if not checks["all_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"validation checks failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
