"""Run the first complete Version-1 ideal-observer DSH simulation locally.

The output is physical Monte Carlo fluence, not detector counts.  No PSF,
effective area, exposure map, energy redistribution, background, or Poisson
realization is applied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np
from jax import random

from .examples import (
    DAY_S,
    build_synthetic_four_cloud_scene,
    build_v1_decay_source,
    build_v1_test_source,
)
from .geometry.clouds import centers_to_edges, cloud_from_loaded_fits
from .io.fits_output import write_ideal_observer_fits
from .io.npz_output import write_ideal_observer_npz
from .observer.binning import build_observer_bin_geometry
from .physics.absorption import load_photoelectric_absorption_table
from .physics.materials import load_2_10_material_tables
from .physics.newdust import (
    build_dust_physics_from_tables,
    load_newdust_scattering_table,
)
from .pipeline import (
    TRANSPORT_STATUS_LABELS,
    run_tabulated_source_to_observer_chunked,
)
from .sources.launch import build_cloud_launch_geometry


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cloud-fits",
        type=Path,
        help=(
            "optional physical delta_NH FITS cube; the source distance must "
            "lie beyond its outer radial cell edge"
        ),
    )
    parser.add_argument("--source-distance-kpc", type=float, default=10.0)
    parser.add_argument(
        "--materials",
        choices=("v1", "2-10"),
        default="v1",
        help="frozen three-energy tables (default) or checked 2–10 keV tables",
    )
    parser.add_argument(
        "--source-model",
        choices=("constant-flare", "exponential-decay"),
        default="constant-flare",
    )
    parser.add_argument(
        "--peak-band-fluxes",
        type=float,
        nargs=3,
        metavar=("F3P3", "F4P9", "F6P9"),
        default=(2.0e-2, 1.2e-2, 6.0e-3),
        help="peak band fluxes at 3.3/4.9/6.9 keV in ph cm^-2 s^-1",
    )
    parser.add_argument(
        "--baseline-band-fluxes",
        type=float,
        nargs=3,
        metavar=("F3P3", "F4P9", "F6P9"),
        default=(0.0, 0.0, 0.0),
        help="asymptotic band fluxes for exponential-decay mode",
    )
    parser.add_argument("--decay-time-days", type=float, default=25.0)
    parser.add_argument("--decay-start-days", type=float, default=0.0)
    parser.add_argument("--decay-duration-days", type=float, default=120.0)
    parser.add_argument("--source-time-bin-days", type=float, default=0.25)
    parser.add_argument("--packets", type=int, default=4_096)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-interactions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--arrival-days", type=float, default=60.0)
    parser.add_argument("--time-bin-days", type=float, default=1.0)
    parser.add_argument(
        "--arrival-time-edges-days",
        type=float,
        nargs="+",
        metavar="DAY",
        help=(
            "optional explicit observer arrival-bin edges in days, beginning "
            "at zero; overrides --arrival-days and --time-bin-days"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dsh_v1_ideal_observer.npz"),
    )
    parser.add_argument(
        "--fits-output",
        type=Path,
        help="FITS path; defaults to the NPZ output path with suffix .fits",
    )
    return parser.parse_args()


def _arrival_edges(arrival_days, time_bin_days, explicit_edges_days=None):
    if explicit_edges_days is not None:
        edges = np.asarray(explicit_edges_days, dtype=np.float64)
        if (
            edges.ndim != 1
            or edges.size < 2
            or not np.all(np.isfinite(edges))
            or edges[0] != 0.0
            or not np.all(np.diff(edges) > 0.0)
        ):
            raise ValueError(
                "arrival-time-edges-days must start at zero and be finite, "
                "strictly increasing values"
            )
        return edges * DAY_S
    if not np.isfinite(arrival_days) or arrival_days <= 0.0:
        raise ValueError("arrival_days must be finite and positive")
    if not np.isfinite(time_bin_days) or time_bin_days <= 0.0:
        raise ValueError("time_bin_days must be finite and positive")
    n_bins = int(np.ceil(arrival_days / time_bin_days))
    return np.linspace(0.0, arrival_days * DAY_S, n_bins + 1)


def _report_progress(completed, total):
    print(
        f"\rCompleted packets: {completed:,} / {total:,}",
        end="\n" if completed == total else "",
        flush=True,
    )


def main():
    args = parse_arguments()
    fits_output = (
        args.fits_output
        if args.fits_output is not None
        else args.output.with_suffix(".fits")
    )
    try:
        import astropy  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "FITS output requires Astropy. Install it before this run with "
            "'python -m pip install astropy'."
        ) from error

    if args.materials == "2-10":
        scattering, absorption, physics = load_2_10_material_tables()
    else:
        scattering = load_newdust_scattering_table()
        absorption = load_photoelectric_absorption_table()
        physics = build_dust_physics_from_tables(scattering, absorption)
    # Source remains the three representative bands in both material modes.
    # A continuous source spectrum is a separate change to the source model.
    source_energies = np.asarray([3.3, 4.9, 6.9], dtype=np.float64)
    if not np.all(np.isin(source_energies, scattering.energy_kev)):
        raise ValueError("material tables must include all source-band energies")

    if args.cloud_fits is None:
        cloud = build_synthetic_four_cloud_scene(args.source_distance_kpc)
        cloud_description = "built-in four-cloud synthetic scene"
    else:
        from .io.cloud_fits import load_cube

        cloud = cloud_from_loaded_fits(
            load_cube(args.cloud_fits),
            source_distance_kpc=args.source_distance_kpc,
        )
        cloud_description = str(args.cloud_fits)

    if args.source_model == "constant-flare":
        source = build_v1_test_source(
            source_energies,
            band_flux=args.peak_band_fluxes,
        )
    else:
        source = build_v1_decay_source(
            source_energies,
            peak_band_flux=args.peak_band_fluxes,
            baseline_band_flux=args.baseline_band_fluxes,
            decay_time_days=args.decay_time_days,
            decay_start_days=args.decay_start_days,
            decay_duration_days=args.decay_duration_days,
            source_time_bin_days=args.source_time_bin_days,
        )
    arrival_edges_s = _arrival_edges(
        args.arrival_days, args.time_bin_days, args.arrival_time_edges_days
    )
    if float(np.asarray(source.time_edges_s)[-1]) >= arrival_edges_s[-1]:
        raise ValueError(
            "arrival_days must extend beyond the final source-emission time "
            "to leave room for dust-scattering delays"
        )
    launch_geometry = build_cloud_launch_geometry(cloud)
    bin_geometry = build_observer_bin_geometry(
        sky_x_edges_arcsec=np.asarray(cloud.x_edges_arcsec),
        sky_y_edges_arcsec=np.asarray(cloud.y_edges_arcsec),
        energy_edges_kev=centers_to_edges(source_energies),
        arrival_time_edges_s=arrival_edges_s,
    )

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Material tables: {args.materials} ({scattering.energy_kev.size} nodes)")
    print(f"Devices: {jax.devices()}")
    print(f"Cloud: {cloud_description}")
    print(f"Cloud shape (z, y, x): {tuple(cloud.delta_nh_cm2.shape)}")
    print(f"Packets: {args.packets:,} in chunks of {args.chunk_size:,}")
    if args.source_model == "constant-flare":
        print("Input source: constant one-hour unabsorbed observer-equivalent flare")
    else:
        print("Input source: unabsorbed observer-equivalent exponential decay")
        print(
            "  post-peak interval: "
            f"{args.decay_start_days:g}--"
            f"{args.decay_start_days + args.decay_duration_days:g} days; "
            f"tau={args.decay_time_days:g} days; "
            f"source bins={args.source_time_bin_days:g} days"
        )
    for energy, flux in zip(
        np.asarray(source.effective_energy_kev),
        args.peak_band_fluxes,
        strict=True,
    ):
        print(f"  {float(energy):.1f} keV peak: {float(flux):.7g} ph cm^-2 s^-1")
    print(
        "  peak total: "
        f"{float(np.sum(args.peak_band_fluxes)):.7g} ph cm^-2 s^-1; "
        f"simulated fluence={float(np.asarray(source.total_fluence)):.7g} "
        "ph cm^-2"
    )
    print("Starting ideal-observer simulation...")
    result = run_tabulated_source_to_observer_chunked(
        random.PRNGKey(args.seed),
        source,
        launch_geometry,
        cloud,
        physics,
        bin_geometry,
        total_packets=args.packets,
        chunk_size=args.chunk_size,
        max_interactions=args.max_interactions,
        progress_callback=_report_progress,
    )

    product_arrays = {
        field: np.asarray(getattr(result.products, field))
        for field in result.products._fields
    }
    diagnostic_arrays = {
        field: np.asarray(getattr(result.diagnostics, field))
        for field in result.diagnostics._fields
    }
    status_count = diagnostic_arrays["transport_status_count"]
    print("Transport terminal states:")
    for label, count in zip(TRANSPORT_STATUS_LABELS, status_count, strict=True):
        print(f"  {label}: {int(count):,}")
    print(
        "Analog interactions/scatterings: "
        f"{int(diagnostic_arrays['analog_interaction_count']):,} / "
        f"{int(diagnostic_arrays['analog_scattering_count']):,}"
    )
    print(
        "Scored/binned observer events: "
        f"{int(diagnostic_arrays['scored_observer_event_count']):,} / "
        f"{int(product_arrays['binned_event_count']):,}"
    )
    print(
        "Scored/binned observer fluence [ph cm^-2]: "
        f"{float(diagnostic_arrays['scored_observer_fluence']):.7g} / "
        f"{float(product_arrays['binned_weight_observer_fluence']):.7g}"
    )
    print("Out-of-range observer events (marginal diagnostics):")
    print(
        "  sky / energy / arrival time: "
        f"{int(product_arrays['outside_sky_event_count']):,} / "
        f"{int(product_arrays['outside_energy_event_count']):,} / "
        f"{int(product_arrays['outside_arrival_time_event_count']):,}"
    )

    decay_mode = args.source_model == "exponential-decay"
    run_metadata = {
        "material_tables": args.materials,
        "scattering_table_sha256": scattering.metadata["table_sha256"],
        "absorption_table_sha256": absorption.metadata["table_sha256"],
        "packets": args.packets,
        "chunk_size": args.chunk_size,
        "max_interactions": args.max_interactions,
        "seed": args.seed,
        "cloud_description": cloud_description,
        "source_model": args.source_model,
        "peak_band_fluxes": args.peak_band_fluxes,
        "baseline_band_fluxes": args.baseline_band_fluxes,
        "decay_time_days": args.decay_time_days if decay_mode else None,
        "decay_start_days": args.decay_start_days if decay_mode else None,
        "decay_duration_days": args.decay_duration_days if decay_mode else None,
        "source_time_bin_days": args.source_time_bin_days if decay_mode else None,
    }
    output_path = write_ideal_observer_npz(
        args.output,
        result,
        bin_geometry,
        source,
        cloud,
        physics,
        launch_geometry,
        run_metadata=run_metadata,
    )
    print(f"Saved: {output_path.resolve()}")
    write_ideal_observer_fits(
        fits_output,
        result,
        bin_geometry,
        source,
        cloud,
        physics,
        launch_geometry,
        run_metadata=run_metadata,
    )
    print(f"Saved: {fits_output.resolve()}")

    numerical_failures = int(status_count[0] + status_count[4:].sum())
    if numerical_failures:
        print(
            "WARNING: numerical-limit or invalid-input terminal states are "
            f"nonzero ({numerical_failures:,}); inspect the status counts."
        )
    if float(product_arrays["unbinned_weight_observer_fluence"]) > 0.0:
        print(
            "WARNING: some scored observer fluence lies outside the requested "
            "sky/energy/arrival-time bins."
        )


if __name__ == "__main__":
    main()
