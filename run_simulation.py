"""Run the DSH simulation described by sim_config.py and save the DSH image as a PNG."""

import os
import time
from pathlib import Path

import sim_config as cfg

JAX_PLATFORM_BY_DEVICE = {"cpu": "cpu", "gpu": "cuda"}
if cfg.DEVICE not in JAX_PLATFORM_BY_DEVICE:
    raise ValueError("DEVICE must be 'cpu' or 'gpu'")
# must be set before importing jax
os.environ["JAX_PLATFORMS"] = JAX_PLATFORM_BY_DEVICE[cfg.DEVICE]

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")  # no display on compute nodes

import matplotlib.pyplot as plt
import numpy as np
from jax import random

from dsh.examples import DAY_S, build_v1_decay_source, build_v1_test_source
from dsh.geometry.clouds import (
    KPC_TO_CM,
    build_angular_distance_cloud,
    centers_to_edges,
    cloud_from_loaded_fits,
)
from dsh.observer.binning import build_observer_bin_geometry
from dsh.physics.absorption import load_photoelectric_absorption_table
from dsh.physics.materials import load_2_10_material_tables
from dsh.physics.newdust import (
    build_dust_physics_from_tables,
    load_newdust_scattering_table,
)
from dsh.pipeline import (
    TRANSPORT_STATUS_LABELS,
    run_tabulated_source_to_observer_chunked,
)
from dsh.sources.launch import build_cloud_launch_geometry
from dsh.sources.models import build_tabulated_band_source


def build_disc_cloud():
    x_arcsec = np.arange(*cfg.GRID_X_ARCSEC)
    y_arcsec = np.arange(*cfg.GRID_Y_ARCSEC)
    z_kpc = np.arange(*cfg.GRID_Z_KPC)

    sky_x, sky_y = np.meshgrid(x_arcsec, y_arcsec, indexing="xy")
    r = np.sqrt(
        (sky_x - cfg.DISC_CENTER_X_ARCSEC) ** 2
        + (sky_y - cfg.DISC_CENTER_Y_ARCSEC) ** 2
    )
    radial_bin_width_cm = np.diff(centers_to_edges(z_kpc)) * KPC_TO_CM  # (n_z,)

    inside_disc = (r[None, :, :] <= cfg.DISC_RADIUS_ARCSEC) & (
        np.abs(z_kpc[:, None, None] - cfg.DISC_CENTER_Z_KPC)
        <= 0.5 * cfg.DISC_THICKNESS_KPC
    )
    n_h_grid = np.where(inside_disc, cfg.DISC_N_H_CM3, 0.0)  # (n_z, n_y, n_x)
    delta_nh = n_h_grid * radial_bin_width_cm[:, None, None]  # column increments

    return build_angular_distance_cloud(
        delta_nh,
        x_centers_arcsec=x_arcsec,
        y_centers_arcsec=y_arcsec,
        z_centers_kpc=z_kpc,
        source_distance_kpc=cfg.SOURCE_DISTANCE_KPC,
    )


def build_cloud():
    if cfg.CLOUD_FITS_PATH is None:
        return build_disc_cloud(), "uniform-density disc cloud"

    from dsh.io.cloud_fits import load_cube

    cloud = cloud_from_loaded_fits(
        load_cube(Path(cfg.CLOUD_FITS_PATH)),
        source_distance_kpc=cfg.SOURCE_DISTANCE_KPC,
    )
    return cloud, str(cfg.CLOUD_FITS_PATH)


def load_flux_file():
    """Return source time edges [s] and the flux columns of FLUX_FILE."""
    table = np.loadtxt(cfg.FLUX_FILE)
    mjd = table[:, 0]
    step_days = np.diff(mjd)
    # each row's flux holds from its MJD until the next row's MJD
    edges_days = np.append(mjd, mjd[-1] + step_days[-1]) - mjd[0]
    return edges_days * DAY_S, table[:, 1:]


def build_band_source(table_energy_kev):
    """Build a flux-file source whose energies are drawn inside ENERGY_EDGES_KEV."""
    if cfg.SOURCE_MODEL != "flux-file":
        raise ValueError("ENERGY_EDGES_KEV requires SOURCE_MODEL = 'flux-file'")
    edges = np.asarray(cfg.ENERGY_EDGES_KEV, dtype=np.float64)
    columns = np.asarray(cfg.FLUX_COLUMNS, dtype=int) - 1
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0.0):
        raise ValueError("ENERGY_EDGES_KEV must be increasing band edges")
    if columns.size != edges.size - 1:
        raise ValueError("FLUX_COLUMNS needs one column per energy band")
    if edges[0] < table_energy_kev[0] or edges[-1] > table_energy_kev[-1]:
        raise ValueError(
            f"ENERGY_EDGES_KEV must lie within the dust tables' "
            f"{table_energy_kev[0]:g}-{table_energy_kev[-1]:g} keV range"
        )
    time_edges_s, flux = load_flux_file()
    if columns.min() < 0 or columns.max() >= flux.shape[1]:
        raise ValueError(f"FLUX_COLUMNS must be between 1 and {flux.shape[1]}")

    # photon-weighted mean energy of each band under dN/dE ~ E^-PHOTON_INDEX
    def integral(power):
        if abs(power + 1.0) < 1e-10:
            return np.log(edges[1:] / edges[:-1])
        return (edges[1:] ** (power + 1.0) - edges[:-1] ** (power + 1.0)) / (power + 1.0)

    mean_energy = integral(1.0 - cfg.PHOTON_INDEX) / integral(-cfg.PHOTON_INDEX)
    source = build_tabulated_band_source(
        time_edges_s=time_edges_s,
        band_flux=flux[:, columns] * cfg.FLUX_SCALE,
        effective_energy_kev=mean_energy,
    )
    dtype = source.effective_energy_kev.dtype
    return source._replace(
        energy_edges_kev=jnp.asarray(edges, dtype=dtype),
        photon_index=jnp.asarray(cfg.PHOTON_INDEX, dtype=dtype),
    )


def build_source(energy_kev):
    if cfg.SOURCE_MODEL == "constant-flare":
        return build_v1_test_source(energy_kev, band_flux=cfg.PEAK_BAND_FLUXES)
    if cfg.SOURCE_MODEL == "exponential-decay":
        return build_v1_decay_source(
            energy_kev,
            peak_band_flux=cfg.PEAK_BAND_FLUXES,
            baseline_band_flux=cfg.BASELINE_BAND_FLUXES,
            decay_time_days=cfg.DECAY_TIME_DAYS,
            decay_start_days=cfg.DECAY_START_DAYS,
            decay_duration_days=cfg.DECAY_DURATION_DAYS,
            source_time_bin_days=cfg.SOURCE_TIME_BIN_DAYS,
        )
    if cfg.SOURCE_MODEL == "custom":
        return build_tabulated_band_source(
            time_edges_s=np.asarray(cfg.CUSTOM_TIME_EDGES_DAYS, dtype=np.float64)
            * DAY_S,
            band_flux=np.asarray(cfg.CUSTOM_BAND_FLUXES, dtype=np.float64),
            effective_energy_kev=energy_kev,
        )
    if cfg.SOURCE_MODEL == "flux-file":
        time_edges_s, flux = load_flux_file()
        return build_tabulated_band_source(
            time_edges_s=time_edges_s,
            band_flux=flux[:, :3] * cfg.FLUX_SCALE,
            effective_energy_kev=energy_kev,
        )
    raise ValueError(
        "SOURCE_MODEL must be 'constant-flare', 'exponential-decay', 'custom' "
        "or 'flux-file'"
    )


def build_arrival_edges_s(arrival_days, time_bin_days):
    n_bins = int(np.ceil(arrival_days / time_bin_days))
    return np.linspace(0.0, arrival_days * DAY_S, n_bins + 1)


def format_duration(seconds):
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def make_progress_reporter(log_path):
    """Return a callback that appends one line per chunk to log_path and stdout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    with log_path.open("w") as log:
        log.write(
            f"# DSH run started {time.strftime('%Y-%m-%d %H:%M:%S')}, job {job_id}, "
            f"{cfg.PACKETS:,} packets in chunks of {cfg.CHUNK_SIZE:,}\n"
        )

    chunks_done = 0

    def report_progress(completed, total):
        nonlocal chunks_done
        chunks_done += 1
        if chunks_done % cfg.PROGRESS_EVERY_CHUNKS and completed < total:
            return
        elapsed = time.time() - start
        fraction = completed / total
        eta = elapsed * (1.0 - fraction) / fraction if completed else float("nan")
        line = (
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {100.0 * fraction:6.2f}%  "
            f"{completed:,} / {total:,} packets  "
            f"elapsed {format_duration(elapsed)}  "
            f"ETA {format_duration(eta) if completed else '--'}"
        )
        print(line, flush=True)
        with log_path.open("a") as log:
            log.write(line + "\n")

    return report_progress


def print_diagnostics(result):
    status_count = np.asarray(result.diagnostics.transport_status_count)
    print("Transport terminal states:")
    for label, count in zip(TRANSPORT_STATUS_LABELS, status_count, strict=True):
        print(f"  {label}: {int(count):,}")
    print(
        "Analog interactions/scatterings: "
        f"{int(result.diagnostics.analog_interaction_count):,} / "
        f"{int(result.diagnostics.analog_scattering_count):,}"
    )
    print(
        "Scored/binned observer fluence [ph cm^-2]: "
        f"{float(result.diagnostics.scored_observer_fluence):.7g} / "
        f"{float(result.products.binned_weight_observer_fluence):.7g}"
    )
    products = result.products
    print(
        "Scored events valid/binned: "
        f"{int(products.valid_event_count):,} / {int(products.binned_event_count):,}"
    )
    print(
        "Unbinned events outside sky/energy/arrival time: "
        f"{int(products.outside_sky_event_count):,} / "
        f"{int(products.outside_energy_event_count):,} / "
        f"{int(products.outside_arrival_time_event_count):,}"
    )


def epoch_bins(bin_geometry):
    """Return (first bin, end bin, start day, stop day) for IMAGE_EPOCH_DAYS."""
    edges_days = np.asarray(bin_geometry.arrival_time_edges_s) / DAY_S
    if cfg.IMAGE_EPOCH_DAYS is None:
        return 0, edges_days.size - 1, edges_days[0], edges_days[-1]
    start, stop = cfg.IMAGE_EPOCH_DAYS
    # keep the arrival-time bins that lie entirely inside the requested interval
    first = int(np.searchsorted(edges_days, start - 1e-9, side="left"))
    end = int(np.searchsorted(edges_days, stop + 1e-9, side="right")) - 1
    if end <= first:
        raise ValueError(
            "IMAGE_EPOCH_DAYS must contain at least one whole arrival-time bin "
            f"(bins are {cfg.TIME_BIN_DAYS:g} d wide, from 0 to {edges_days[-1]:g} d)"
        )
    return first, end, edges_days[first], edges_days[end]


def epoch_label(start_day, stop_day):
    label = f"days {start_day:g}-{stop_day:g}"
    if cfg.SOURCE_MODEL == "flux-file":
        mjd0 = float(np.loadtxt(cfg.FLUX_FILE, usecols=0)[0])
        label += f" (MJD {mjd0 + start_day:g}-{mjd0 + stop_day:g})"
    return label


def project_image(cube, bin_geometry, bands):
    """Sum a (n_t, n_E, n_y, n_x) cube over the configured epoch, band and FOV.

    ``bands`` is (band center energies [keV], band labels, band widths [keV] or
    None). Also returns the summed band width [keV] (None for line energies).
    """
    first, end, _, _ = epoch_bins(bin_geometry)
    cube = np.asarray(cube)[first:end]
    centers_kev, labels, widths_kev = bands
    if cfg.IMAGE_ENERGY_KEV is None and len(labels) > 1:
        image = cube.sum(axis=(0, 1))
        band_label = "all energies"
        width_kev = None if widths_kev is None else float(np.sum(widths_kev))
    else:
        target = centers_kev[0] if cfg.IMAGE_ENERGY_KEV is None else cfg.IMAGE_ENERGY_KEV
        index = int(np.argmin(np.abs(centers_kev - target)))
        image = cube[:, index, :, :].sum(axis=0)
        band_label = labels[index]
        width_kev = None if widths_kev is None else float(widths_kev[index])

    x_edges = np.asarray(bin_geometry.sky_x_edges_arcsec)
    y_edges = np.asarray(bin_geometry.sky_y_edges_arcsec)
    if cfg.IMAGE_FOV_ARCSEC is not None:
        half_fov = 0.5 * cfg.IMAGE_FOV_ARCSEC
        x_lo = np.searchsorted(x_edges, -half_fov, side="left")
        x_hi = np.searchsorted(x_edges, half_fov, side="right") - 1
        y_lo = np.searchsorted(y_edges, -half_fov, side="left")
        y_hi = np.searchsorted(y_edges, half_fov, side="right") - 1
        image = image[y_lo:y_hi, x_lo:x_hi]
        x_edges = x_edges[x_lo : x_hi + 1]
        y_edges = y_edges[y_lo : y_hi + 1]
    return image, x_edges, y_edges, band_label, width_kev


def scaled(values, label, scale, log_values=None):
    """Return (image, colorbar label) for a colour scale; log masks empty pixels."""
    if scale == "linear":
        return values, label
    if scale != "log":
        raise ValueError("IMAGE_SCALES entries must be 'log' or 'linear'")
    if log_values is None:
        log_values = np.ma.log10(np.ma.masked_less_equal(values, 0))
    return log_values, f"log10 {label}"


def save_log_image(log_image, x_edges, y_edges, title, colorbar_label, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("black")  # masked (empty) pixels
    im = ax.imshow(
        log_image,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap=cmap,
        aspect="equal",
    )
    ax.set_xlabel("sky x [arcsec]")
    ax.set_ylabel("sky y [arcsec]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=colorbar_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=cfg.IMAGE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_image(result, bin_geometry, bands, output_path, scale):
    image, x_edges, y_edges, band_label, width_kev = project_image(
        result.products.total_fluence, bin_geometry, bands
    )
    if cfg.IMAGE_EPOCH_DAYS is None:
        plotted, label = scaled(
            image, "fluence [ph cm$^{-2}$]", scale, log_values=np.log10(image + 1e-30)
        )
        save_log_image(
            plotted,
            x_edges,
            y_edges,
            f"DSH image: {band_label}, {cfg.PACKETS:,} packets",
            label,
            output_path,
        )
        return

    # mean specific intensity over the epoch: fluence / (time * solid angle * dE)
    _, _, start_day, stop_day = epoch_bins(bin_geometry)
    pixel_arcsec2 = abs((x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0]))
    intensity = image / ((stop_day - start_day) * DAY_S * pixel_arcsec2)
    unit = "ph cm$^{-2}$ s$^{-1}$ arcsec$^{-2}$"
    if width_kev is not None:
        intensity = intensity / width_kev
        unit = "ph cm$^{-2}$ s$^{-1}$ keV$^{-1}$ arcsec$^{-2}$"
    plotted, label = scaled(intensity, f"$I_\\nu$ [{unit}]", scale)
    save_log_image(
        plotted,
        x_edges,
        y_edges,
        f"DSH intensity: {band_label}\n{epoch_label(start_day, stop_day)}",
        label,
        output_path,
    )
    return (
        f"Intensity: {epoch_label(start_day, stop_day)}, "
        f"peak {intensity.max():.4g}, mean {intensity.mean():.4g} {unit.replace('$', '')}"
    )


def save_count_image(result, bin_geometry, bands, output_path, scale):
    """Save the number of scored Monte Carlo events per pixel."""
    counts, x_edges, y_edges, band_label, _ = project_image(
        result.products.event_count, bin_geometry, bands
    )
    if cfg.IMAGE_EPOCH_DAYS is not None:
        _, _, start_day, stop_day = epoch_bins(bin_geometry)
        band_label += f", {epoch_label(start_day, stop_day)}"
    plotted, label = scaled(counts, "scored events per pixel", scale)
    save_log_image(
        plotted,
        x_edges,
        y_edges,
        f"DSH photon counts: {band_label}, {int(counts.sum()):,} events",
        label,
        output_path,
    )
    return counts


def report_device():
    try:
        device = jax.devices()[0]
    except RuntimeError as error:
        raise RuntimeError(
            f"DEVICE = {cfg.DEVICE!r} but JAX could not initialise that backend. "
            "For 'gpu', run through submit_simulation.sh so the job requests a GPU."
        ) from error
    print(f"JAX device: {device} ({device.device_kind})")


def main():
    run_start = time.time()
    report_device()
    if cfg.ENERGY_EDGES_KEV is None:
        scattering = load_newdust_scattering_table()
        absorption = load_photoelectric_absorption_table()
        physics = build_dust_physics_from_tables(scattering, absorption)
        source = build_source(scattering.energy_kev)
        energy_edges_kev = centers_to_edges(scattering.energy_kev)
        centers_kev = np.asarray(scattering.energy_kev)
        bands = (centers_kev, [f"{energy:.1f} keV" for energy in centers_kev], None)
    else:
        scattering, absorption, physics = load_2_10_material_tables()
        source = build_band_source(np.asarray(scattering.energy_kev))
        energy_edges_kev = np.asarray(cfg.ENERGY_EDGES_KEV, dtype=np.float64)
        bands = (
            np.asarray(source.effective_energy_kev),
            [
                f"{low:g}-{high:g} keV"
                for low, high in zip(energy_edges_kev[:-1], energy_edges_kev[1:])
            ],
            np.diff(energy_edges_kev),
        )
    print(f"Energy bands: {', '.join(bands[1])}")

    cloud, cloud_description = build_cloud()
    print(f"Cloud: {cloud_description}")
    print(f"Cloud shape (z, y, x): {tuple(cloud.delta_nh_cm2.shape)}")
    print(
        "Simulated source fluence: "
        f"{float(np.asarray(source.total_fluence)):.7g} ph cm^-2"
    )

    arrival_edges_s = build_arrival_edges_s(cfg.ARRIVAL_DAYS, cfg.TIME_BIN_DAYS)
    if float(np.asarray(source.time_edges_s)[-1]) >= arrival_edges_s[-1]:
        raise ValueError(
            "ARRIVAL_DAYS must extend beyond the final source-emission time "
            "to leave room for dust-scattering delays"
        )

    launch_geometry = build_cloud_launch_geometry(cloud)
    bin_geometry = build_observer_bin_geometry(
        sky_x_edges_arcsec=np.asarray(cloud.x_edges_arcsec),
        sky_y_edges_arcsec=np.asarray(cloud.y_edges_arcsec),
        energy_edges_kev=energy_edges_kev,
        arrival_time_edges_s=arrival_edges_s,
    )

    progress_log = Path(cfg.PROGRESS_LOG)
    report_progress = make_progress_reporter(progress_log)
    print(f"Progress log: {progress_log}")

    start = time.time()
    result = run_tabulated_source_to_observer_chunked(
        random.PRNGKey(cfg.SEED),
        source,
        launch_geometry,
        cloud,
        physics,
        bin_geometry,
        total_packets=cfg.PACKETS,
        chunk_size=cfg.CHUNK_SIZE,
        max_interactions=cfg.MAX_INTERACTIONS,
        progress_callback=report_progress,
    )
    simulation_s = time.time() - start
    print(f"Simulation wall time: {simulation_s:.1f} s")
    print_diagnostics(result)

    saved = []
    for scale in cfg.IMAGE_SCALES:
        # one file per colour scale, e.g. dsh_image_log.png and dsh_image_linear.png
        output_path = Path(cfg.OUTPUT_PNG)
        output_path = output_path.with_name(f"{output_path.stem}_{scale}.png")
        intensity_summary = save_image(result, bin_geometry, bands, output_path, scale)
        count_path = Path(cfg.OUTPUT_COUNT_PNG)
        count_path = count_path.with_name(f"{count_path.stem}_{scale}.png")
        counts = save_count_image(result, bin_geometry, bands, count_path, scale)
        print(f"Saved {output_path} and {count_path}")
        saved += [str(output_path), str(count_path)]
    count_npy_path = Path(cfg.OUTPUT_COUNT_NPY)
    count_npy_path.parent.mkdir(parents=True, exist_ok=True)
    # same matrix as the count image: rows are sky y, columns are sky x
    np.save(count_npy_path, np.asarray(counts, dtype=np.int64))
    print(f"Saved {count_npy_path}: shape {counts.shape} (y, x)")
    saved.append(str(count_npy_path))
    if intensity_summary is not None:
        print(intensity_summary)
    print(
        f"Image events: {int(counts.sum()):,}, "
        f"max {int(counts.max()):,} per pixel, "
        f"{int((counts == 0).sum()):,} of {counts.size:,} pixels empty"
    )
    total_s = time.time() - run_start
    print(f"Total run time: {total_s:.1f} s")
    with progress_log.open("a") as log:
        log.write(
            f"# Finished {time.strftime('%Y-%m-%d %H:%M:%S')}, saved {', '.join(saved)}\n"
            f"# Simulation runtime: {format_duration(simulation_s)} "
            f"({simulation_s:.1f} s) on {cfg.DEVICE}\n"
            f"# Total runtime incl. setup and image: {format_duration(total_s)} "
            f"({total_s:.1f} s)\n"
        )


if __name__ == "__main__":
    main()
