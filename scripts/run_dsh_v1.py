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

from utils.absorption import load_photoelectric_absorption_table
from utils.clouds import (
    build_angular_distance_cloud,
    centers_to_edges,
    cloud_from_loaded_fits,
)
from utils.fits_output import write_ideal_observer_fits
from utils.newdust import (
    build_dust_physics_from_tables,
    load_newdust_scattering_table,
)
from utils.observer_binning import build_observer_bin_geometry
from utils.simulation import (
    TRANSPORT_STATUS_LABELS,
    run_tabulated_source_to_observer_chunked,
)
from utils.source import build_tabulated_band_source
from utils.source_launch import build_cloud_launch_geometry


DAY_S = 86_400.0


def build_synthetic_four_cloud_scene(source_distance_kpc=10.0):
    """Return a compact 500-arcsec four-cloud plus diffuse-H test scene."""

    x_arcsec = np.arange(-250.0, 250.0 + 10.0, 10.0)
    y_arcsec = np.arange(-250.0, 250.0 + 10.0, 10.0)
    z_kpc = np.arange(1.025, 8.975 + 0.025, 0.05)
    sky_x, sky_y = np.meshgrid(x_arcsec, y_arcsec, indexing="xy")
    delta_nh = np.zeros(
        (z_kpc.size, y_arcsec.size, x_arcsec.size), dtype=np.float64
    )

    # distance, x0, y0, sigma_x, sigma_y, sigma_z, peak column [cm^-2]
    cloud_parameters = (
        (2.0, -90.0, 65.0, 70.0, 45.0, 0.10, 8.0e21),
        (3.7, 75.0, -55.0, 55.0, 85.0, 0.14, 1.0e22),
        (5.6, -25.0, -20.0, 105.0, 60.0, 0.18, 1.2e22),
        (7.8, 55.0, 80.0, 80.0, 100.0, 0.22, 9.0e21),
    )
    for distance, x0, y0, sx, sy, sz, peak_column in cloud_parameters:
        projected_column = peak_column * np.exp(
            -0.5 * ((sky_x - x0) / sx) ** 2
            -0.5 * ((sky_y - y0) / sy) ** 2
        )
        radial_weight = np.exp(-0.5 * ((z_kpc - distance) / sz) ** 2)
        radial_weight /= radial_weight.sum()
        delta_nh += radial_weight[:, None, None] * projected_column[None, :, :]

    # A weak, spatially varying diffuse component with 2e21 cm^-2 mean total
    # column. It participates in the same dust scattering and absorption as
    # the molecular structures; V1 does not yet assign phase functions by gas
    # component.
    diffuse_column = 2.0e21 * (
        1.0 + 0.12 * sky_x / 250.0 - 0.08 * sky_y / 250.0
    )
    delta_nh += diffuse_column[None, :, :] / z_kpc.size

    return build_angular_distance_cloud(
        delta_nh,
        x_centers_arcsec=x_arcsec,
        y_centers_arcsec=y_arcsec,
        z_centers_kpc=z_kpc,
        source_distance_kpc=source_distance_kpc,
    )


def build_v1_test_source(energy_kev):
    """One-hour flare sampled at the three tabulated V1 energies."""

    return build_tabulated_band_source(
        time_edges_s=[0.0, 3_600.0],
        band_flux=np.asarray([[2.0e-2, 1.2e-2, 6.0e-3]]),
        effective_energy_kev=energy_kev,
    )


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
    parser.add_argument("--packets", type=int, default=4_096)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-interactions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--arrival-days", type=float, default=60.0)
    parser.add_argument("--time-bin-days", type=float, default=1.0)
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


def _arrival_edges(arrival_days, time_bin_days):
    if not np.isfinite(arrival_days) or arrival_days <= 0.0:
        raise ValueError("arrival_days must be finite and positive")
    if not np.isfinite(time_bin_days) or time_bin_days <= 0.0:
        raise ValueError("time_bin_days must be finite and positive")
    n_bins = int(np.ceil(arrival_days / time_bin_days))
    return np.linspace(0.0, arrival_days * DAY_S, n_bins + 1)


def _as_numpy_tree(named_tuple):
    return {
        name: np.asarray(getattr(named_tuple, name))
        for name in named_tuple._fields
    }


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

    scattering = load_newdust_scattering_table()
    absorption = load_photoelectric_absorption_table()
    physics = build_dust_physics_from_tables(scattering, absorption)

    if args.cloud_fits is None:
        cloud = build_synthetic_four_cloud_scene(args.source_distance_kpc)
        cloud_description = "built-in four-cloud synthetic scene"
    else:
        from utils.fits_cube import load_cube

        cloud = cloud_from_loaded_fits(
            load_cube(args.cloud_fits),
            source_distance_kpc=args.source_distance_kpc,
        )
        cloud_description = str(args.cloud_fits)

    source = build_v1_test_source(scattering.energy_kev)
    launch_geometry = build_cloud_launch_geometry(cloud)
    bin_geometry = build_observer_bin_geometry(
        sky_x_edges_arcsec=np.asarray(cloud.x_edges_arcsec),
        sky_y_edges_arcsec=np.asarray(cloud.y_edges_arcsec),
        energy_edges_kev=centers_to_edges(scattering.energy_kev),
        arrival_time_edges_s=_arrival_edges(
            args.arrival_days, args.time_bin_days
        ),
    )

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"Cloud: {cloud_description}")
    print(f"Cloud shape (z, y, x): {tuple(cloud.delta_nh_cm2.shape)}")
    print(f"Packets: {args.packets:,} in chunks of {args.chunk_size:,}")
    print("Input source: constant one-hour unabsorbed observer-equivalent flare")
    for energy, flux in zip(
        np.asarray(source.effective_energy_kev),
        np.asarray(source.band_flux)[0],
    ):
        print(f"  {float(energy):.1f} keV: {float(flux):.7g} ph cm^-2 s^-1")
    print(
        "  total: "
        f"{float(np.sum(np.asarray(source.band_flux)[0])):.7g} ph cm^-2 s^-1; "
        f"fluence={float(np.asarray(source.total_fluence)):.7g} ph cm^-2"
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

    product_arrays = _as_numpy_tree(result.products)
    diagnostic_arrays = _as_numpy_tree(result.diagnostics)
    status_count = diagnostic_arrays["transport_status_count"]
    print("Transport terminal states:")
    for label, count in zip(TRANSPORT_STATUS_LABELS, status_count):
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        **product_arrays,
        **diagnostic_arrays,
        sky_x_edges_arcsec=np.asarray(bin_geometry.sky_x_edges_arcsec),
        sky_y_edges_arcsec=np.asarray(bin_geometry.sky_y_edges_arcsec),
        energy_edges_kev=np.asarray(bin_geometry.energy_edges_kev),
        arrival_time_edges_s=np.asarray(bin_geometry.arrival_time_edges_s),
        sky_pixel_solid_angle_sr=np.asarray(
            bin_geometry.sky_pixel_solid_angle_sr
        ),
        transport_status_labels=np.asarray(TRANSPORT_STATUS_LABELS),
        source_energy_kev=np.asarray(source.effective_energy_kev),
        source_time_edges_s=np.asarray(source.time_edges_s),
        source_band_flux=np.asarray(source.band_flux),
        source_cell_fluence=np.asarray(source.cell_fluence),
        source_flat_cdf=np.asarray(source.flat_cdf),
        source_total_fluence=np.asarray(source.total_fluence),
        cloud_delta_nh_cm2=np.asarray(cloud.delta_nh_cm2),
        cloud_n_h_cm3=np.asarray(cloud.n_h_cm3),
        cloud_x_edges_arcsec=np.asarray(cloud.x_edges_arcsec),
        cloud_y_edges_arcsec=np.asarray(cloud.y_edges_arcsec),
        cloud_z_edges_kpc=np.asarray(cloud.z_edges_kpc),
        cloud_radial_bin_width_cm=np.asarray(cloud.radial_bin_width_cm),
        source_distance_kpc=np.asarray(cloud.source_distance_kpc),
        physics_energy_kev=np.asarray(physics.energy_kev),
        physics_scattering_cross_section_cm2_per_h=np.asarray(
            physics.scattering_cross_section_cm2_per_h
        ),
        physics_absorption_cross_section_cm2_per_h=np.asarray(
            physics.absorption_cross_section_cm2_per_h
        ),
        physics_scattering_angle_rad=np.asarray(
            physics.scattering_angle_rad
        ),
        physics_scattering_angle_cdf=np.asarray(
            physics.scattering_angle_cdf
        ),
        physics_differential_cross_section_cm2_per_sr_per_h=np.asarray(
            physics.differential_cross_section_cm2_per_sr_per_h
        ),
        launch_source_position_pc=np.asarray(
            launch_geometry.source_position_pc
        ),
        launch_source_distance_pc=np.asarray(
            launch_geometry.source_distance_pc
        ),
        launch_slope_x_bounds=np.asarray(launch_geometry.slope_x_bounds),
        launch_slope_y_bounds=np.asarray(launch_geometry.slope_y_bounds),
        launch_slope_area=np.asarray(launch_geometry.slope_area),
        launch_solid_angle_sr=np.asarray(
            launch_geometry.launch_solid_angle_sr
        ),
        output_schema_version=np.asarray(2),
        cloud_description=np.asarray(cloud_description),
        source_flux_convention=np.asarray(
            "unabsorbed observer-equivalent photon flux"
        ),
        random_seed=np.asarray(args.seed),
        requested_packet_count=np.asarray(args.packets),
        chunk_size=np.asarray(args.chunk_size),
        max_interactions=np.asarray(args.max_interactions),
    )
    np.savez_compressed(args.output, **payload)
    print(f"Saved: {args.output.resolve()}")
    write_ideal_observer_fits(
        fits_output,
        result,
        bin_geometry,
        source,
        cloud,
        physics,
        launch_geometry,
        run_metadata={
            "packets": args.packets,
            "chunk_size": args.chunk_size,
            "max_interactions": args.max_interactions,
            "seed": args.seed,
            "cloud_description": cloud_description,
        },
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
