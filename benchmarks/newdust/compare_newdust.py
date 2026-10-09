"""Absolute NewDust/DSH thin-screen benchmark; run from a DSH repository.

No fitted amplitude, image renormalization, ARF, PSF, or Poisson realization.
Uses the public transport and peel-off functions and history-level MC moments.
The default comparison is first-scattering-order, time-integrated fluence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ARCSEC_TO_RAD = np.pi / (180 * 3600)
KPC_TO_CM = 3.0856775814913673e21


def api(module):
    """Support both the dsh baseline layout and the later core refactor."""
    parts = module.split(".")
    if parts[0] in {"geometry", "observer", "transport"} or module == "sources.launch":
        names = [f"dsh.core.{module}", f"dsh.{module}"]
    elif module == "physics.newdust":
        names = ["dsh.materials.scattering", "dsh.physics.newdust"]
    else:
        names = [f"dsh.{module}"]
    for name in names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name and not name.startswith(error.name + "."):
                raise
    raise ImportError(f"Cannot locate {module} in this DSH checkout")


def validate(config):
    if config.get("launch_proposal", "radial") not in {
        "radial",
        "nested_near",
        "nested_far",
    }:
        raise ValueError("launch_proposal must be radial, nested_near, or nested_far")
    positive = [
        "source_distance_kpc",
        "source_fluence_ph_cm2",
        "thickness_kpc",
        "half_width_arcsec",
        "pixel_arcsec",
        "annulus_width_arcsec",
        "reference_radial_step_arcsec",
        "maximum_total_tau",
        "maximum_aperture_relative_se",
        "maximum_numerical_budget_fraction",
    ]
    for key in positive:
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in [
        "packets_per_seed",
        "chunk_size",
        "max_interactions",
        "pixel_quadrature_order",
        "minimum_annulus_neff",
        "minimum_resolved_annuli",
    ]:
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_interactions"] < 2:
        raise ValueError("Use at least two interactions so cap hits remain diagnostic")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Provide distinct integer seeds")
    if any(not isinstance(s, int) or s < 0 for s in config["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    if not config["energies_kev"] or any(
        not np.isfinite(e) or e <= 0 for e in config["energies_kev"]
    ):
        raise ValueError("Provide positive finite energies")
    fractions = []
    for screen in config["screens"]:
        f, nh = screen["fraction_from_observer"], screen["nh_cm2"]
        if not np.isfinite(f) or not 0 < f < 1 or not np.isfinite(nh) or nh <= 0:
            raise ValueError("Screen requires 0 < fraction < 1 and NH > 0")
        fractions.append(f)
    if not fractions or len(set(fractions)) != len(fractions):
        raise ValueError("Specify at least one screen with distinct distances")
    centers = np.sort(fractions) * config["source_distance_kpc"]
    width = config["thickness_kpc"]
    if (
        centers[0] <= width / 2
        or centers[-1] + width / 2 >= config["source_distance_kpc"]
    ):
        raise ValueError("Slabs must lie strictly between observer and source")
    if np.any(np.diff(centers) <= width):
        raise ValueError("Slabs overlap")
    n = 2 * config["half_width_arcsec"] / config["pixel_arcsec"]
    if not np.isclose(n, round(n)):
        raise ValueError("Full field width must be divisible by pixel size")
    if config["minimum_resolved_annuli"] > int(
        config["half_width_arcsec"] / config["annulus_width_arcsec"]
    ):
        raise ValueError("Minimum resolved annuli exceeds available aperture bins")


def history_se(weight_sum, squared_sum, histories):
    """SE for weighted first-order sums, including zero-score histories."""
    if histories < 2:
        raise ValueError("At least two histories are needed")
    variance = (
        histories
        / (histories - 1)
        * np.maximum(
            np.asarray(squared_sum) - np.asarray(weight_sum) ** 2 / histories, 0
        )
    )
    return np.sqrt(variance)


def absolute_gate(
    measured, expected, standard_error, systematic_fraction, numeric_error=0
):
    """No amplitude adjustment: accept only an uncertainty-bounded difference."""
    return bool(
        abs(measured - expected)
        <= 3 * standard_error + systematic_fraction * expected + numeric_error
    )


def make_cloud(config):
    """Exact slab edges with explicit zero-density gaps; no midpoint smearing."""
    import jax.numpy as jnp

    cloud_api = api("geometry.clouds")
    width, distance = config["thickness_kpc"], config["source_distance_kpc"]
    slabs = sorted(config["screens"], key=lambda s: s["fraction_from_observer"])
    z = np.array(
        [
            edge
            for s in slabs
            for edge in [
                s["fraction_from_observer"] * distance - width / 2,
                s["fraction_from_observer"] * distance + width / 2,
            ]
        ]
    )
    columns = np.zeros((len(z) - 1, 2, 2))
    for i, s in enumerate(slabs):
        columns[2 * i] = s["nh_cm2"]
    widths = np.diff(z) * KPC_TO_CM
    sky_edges = [-config["half_width_arcsec"], 0, config["half_width_arcsec"]]
    arrays = [
        columns,
        columns / widths[:, None, None],
        sky_edges,
        sky_edges,
        z,
        widths,
        distance,
    ]
    cloud = cloud_api.AngularDistanceCloud(
        *(jnp.asarray(a, dtype=jnp.float32) for a in arrays)
    )
    np.testing.assert_allclose(
        np.asarray(cloud.n_h_cm3)
        * np.asarray(cloud.radial_bin_width_cm)[:, None, None],
        columns,
        rtol=2e-6,
        atol=0,
    )
    return cloud


def load_newdust():
    try:
        from xdust import grainpop, halos

        name = "xdust"
    except ImportError:
        from newdust import grainpop, halos

        name = "newdust"
    return grainpop, halos, name


def population(metadata, nh):
    grainpop, _, _ = load_newdust()
    if metadata.get("model") != "MRN power law + Rayleigh-Gans/Drude":
        raise ValueError(
            "This baseline benchmark requires the packaged MRN RG/Drude dust"
        )
    return grainpop.make_MRN_RGDrude(
        amin=metadata["grain_radius_min_micron"],
        amax=metadata["grain_radius_max_micron"],
        p=metadata["grain_power_law_index"],
        rho=metadata["grain_material_density_g_cm3"],
        md=nh
        * metadata["dust_mass_column_g_cm2"]
        / metadata["hydrogen_column_reference_cm2"],
        na=metadata["grain_radius_samples"],
        log=metadata["grain_radius_spacing"] == "log",
    )


def material_audit(table, energy_index):
    """Re-evaluate NewDust directly; never derive the reference from DSH rows."""
    import astropy.units as u

    metadata = table.metadata
    nh = metadata["hydrogen_column_reference_cm2"]
    pop = population(metadata, nh)
    energy = table.energy_kev[energy_index]
    pop.calculate_ext(np.array([energy]), theta=table.scattering_angle_rad * u.rad)
    direct = pop.int_diff.to_value("sr^-1")[0] / nh
    stored = table.differential_cross_section_cm2_per_sr_per_h[energy_index]
    active = direct > direct.max() * 1e-12
    discrepancy = float(np.max(np.abs(stored[active] / direct[active] - 1)))
    if discrepancy > 1e-6:
        raise ValueError(
            f"Dust material mismatch: {discrepancy:.3g}; stop before image comparison"
        )
    sigma = float(table.scattering_cross_section_cm2_per_h[energy_index])
    reported = float(np.asarray(pop.tau_sca).ravel()[0] / nh)
    return {
        "direct_differential_max_relative_error": discrepancy,
        "integrated_sigma_cm2_per_h": sigma,
        "newdust_reported_sigma_cm2_per_h": reported,
        "integrated_to_reported_sigma": sigma / reported,
    }


def reference_profiles(config, table, energy):
    import astropy.units as u

    _, halos, _ = load_newdust()
    max_radius = np.sqrt(2) * config["half_width_arcsec"] + config["pixel_arcsec"]
    grid = np.linspace(
        0,
        max_radius,
        int(np.ceil(max_radius / config["reference_radial_step_arcsec"])) + 1,
    )
    combined = np.zeros_like(grid)
    rows = []
    for screen in config["screens"]:
        pop = population(table.metadata, screen["nh_cm2"])
        halo = halos.ScreenGalHalo(np.array([energy]), grid * u.arcsec)
        halo.calculate(pop, x=1 - screen["fraction_from_observer"])
        profile = halo.norm_int.to_value("arcsec^-2")[0]
        combined += profile
        rows.append(profile)
    return grid, combined * config["source_fluence_ph_cm2"], np.array(rows)


def pixel_reference(config, radius_grid, profile, order):
    """Gauss-Legendre pixel integration with tangent-plane solid-angle Jacobian."""
    n = round(2 * config["half_width_arcsec"] / config["pixel_arcsec"])
    edges = np.linspace(
        -config["half_width_arcsec"], config["half_width_arcsec"], n + 1
    )
    centers = (edges[1:] + edges[:-1]) / 2
    x, y = np.meshgrid(centers, centers)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    image = np.zeros((n, n))
    pixel = config["pixel_arcsec"]
    for a, wa in zip(nodes, weights, strict=True):
        for b, wb in zip(nodes, weights, strict=True):
            xx, yy = x + a * pixel / 2, y + b * pixel / 2
            tx, ty = np.tan(xx * ARCSEC_TO_RAD), np.tan(yy * ARCSEC_TO_RAD)
            jac = (1 + tx**2) * (1 + ty**2) / (1 + tx**2 + ty**2) ** 1.5
            image += (
                wa
                * wb
                * np.interp(np.hypot(xx, yy), radius_grid, profile)
                * jac
                * pixel**2
                / 4
            )
    return edges, image


def radial_sums(image, squared_image, edges, config):
    centers = (edges[1:] + edges[:-1]) / 2
    x, y = np.meshgrid(centers, centers)
    radius = np.hypot(x, y)
    radial_edges = np.arange(
        0, config["half_width_arcsec"] + 1e-8, config["annulus_width_arcsec"]
    )
    # Include the final narrow annulus if the chosen width does not divide the aperture.
    if radial_edges[-1] < config["half_width_arcsec"]:
        radial_edges = np.append(radial_edges, config["half_width_arcsec"])
    ix = np.searchsorted(radial_edges, radius, side="right") - 1
    valid = (ix >= 0) & (ix < len(radial_edges) - 1)
    count = len(radial_edges) - 1
    sums = np.bincount(ix[valid], weights=image[valid], minlength=count)
    sq = np.bincount(ix[valid], weights=squared_image[valid], minlength=count)
    pixels = np.bincount(ix[valid], minlength=count)
    return radial_edges, sums, sq, pixels


def run_seed(config, cloud, physics, table, ei, edges, seed):
    import jax
    import jax.numpy as jnp
    from jax import random

    launches = api("sources.launch")
    source = api("sources.models")
    try:
        source = importlib.import_module("dsh.contracts")
    except ModuleNotFoundError as error:
        if error.name != "dsh.contracts":
            raise
    transport = api("transport.kernel")
    scoring = api("observer.scoring")
    proposal_kind = config.get("launch_proposal", "radial")
    if proposal_kind == "radial":
        radial_launch = api("sources.radial_launch")
        launch_geometry = radial_launch.build_radial_cloud_launch_geometry(
            cloud, physics
        )
        launch_jit = jax.jit(launches.sample_source_launches)
    else:
        outer = launches.build_cloud_launch_geometry(cloud)
        validation_launch = api("validation.launch")
        fraction = (min if proposal_kind == "nested_near" else max)(
            s["fraction_from_observer"] for s in config["screens"]
        )
        median = float(
            np.interp(0.5, table.scattering_angle_cdf[ei], table.scattering_angle_rad)
        )
        geometries = validation_launch.nested_screen_launch_geometries(
            outer,
            source_distance_kpc=config["source_distance_kpc"],
            fractional_distance=fraction,
            median_scattering_angle_rad=median,
        )
        launch_geometry = geometries
        launch_jit = jax.jit(validation_launch.sample_mixture_source_launches)
    transport_jit = jax.jit(
        transport.transport_photon_batch, static_argnames=("max_interactions",)
    )
    score_jit = jax.jit(scoring.score_peeloff_events)
    n = len(edges) - 1
    image = np.zeros((n, n))
    squares = np.zeros_like(image)
    all_image = np.zeros_like(image)
    counts = np.zeros((n, n), dtype=np.int64)
    statuses = np.zeros(len(transport.STATUS_NAMES), dtype=np.int64)
    packets = config["packets_per_seed"]
    complete = 0
    chunks = 0
    outside_weight = 0.0
    total_first = 0.0
    started = time.monotonic()
    while complete < packets:
        size = min(config["chunk_size"], packets - complete)
        pk = source.SourcePackets(
            jnp.full(size, table.energy_kev[ei], dtype=jnp.float32),
            jnp.zeros(size, dtype=jnp.float32),
            jnp.full(
                size, config["source_fluence_ph_cm2"] / packets, dtype=jnp.float32
            ),
            jnp.zeros(size, dtype=jnp.int32),
            jnp.zeros(size, dtype=jnp.int32),
        )
        lk, tk = random.split(random.fold_in(random.PRNGKey(seed), chunks))
        launched = launch_jit(lk, pk, launch_geometry)
        history = transport_jit(
            tk,
            launched.position_pc,
            launched.momentum_kev,
            cloud,
            physics,
            max_interactions=config["max_interactions"],
        )
        ev = score_jit(launched, history, cloud, physics)
        valid = np.asarray(ev.valid)
        order = np.asarray(ev.scattering_order)
        x, y = np.asarray(ev.sky_x_arcsec), np.asarray(ev.sky_y_arcsec)
        w = np.asarray(ev.weight_observer_fluence).astype(np.float64)
        first = valid & (order == 1)
        a, b, weights = x[first], y[first], w[first]
        ix = np.searchsorted(edges, a, side="right") - 1
        iy = np.searchsorted(edges, b, side="right") - 1
        keep = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        np.add.at(image, (iy[keep], ix[keep]), weights[keep])
        np.add.at(squares, (iy[keep], ix[keep]), weights[keep] ** 2)
        np.add.at(counts, (iy[keep], ix[keep]), 1)
        outside_weight += weights[~keep].sum()
        total_first += weights.sum()
        # Higher orders retained as a diagnostic, outside the NewDust first-order gate.
        all_image += np.histogram2d(
            y[valid], x[valid], bins=(edges, edges), weights=w[valid]
        )[0]
        statuses += np.bincount(np.asarray(history.status), minlength=len(statuses))
        complete += size
        chunks += 1
        if chunks == 1 or chunks % 20 == 0 or complete == packets:
            print(
                f"E={table.energy_kev[ei]:g} seed={seed}: {complete:,}/{packets:,}, "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )
    return {
        "image": image,
        "squares": squares,
        "all_orders_image": all_image,
        "counts": counts,
        "statuses": statuses,
        "outside_weight": outside_weight,
        "total_first_weight": total_first,
        "seconds": time.monotonic() - started,
        "proposal_kind": proposal_kind,
    }


def write_fits(path, image, config, energy, label):
    from astropy.io import fits

    header = fits.Header()
    header["BUNIT"] = "ph cm-2"
    header["CTYPE1"] = "SKY_X"
    header["CTYPE2"] = "SKY_Y"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = (image.shape[1] + 1) / 2
    header["CRPIX2"] = (image.shape[0] + 1) / 2
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = config["pixel_arcsec"]
    header["CDELT2"] = config["pixel_arcsec"]
    header["ENERGY"] = energy
    header["SRCFLU"] = config["source_fluence_ph_cm2"]
    header["DISTKPC"] = config["source_distance_kpc"]
    header["MODEL"] = label
    header["COMMENT"] = (
        "Time-integrated monochromatic halo fluence per pixel; no direct source."
    )
    fits.PrimaryHDU(image, header).writeto(path, overwrite=True)


def plot_result(path, reference, mc, uncertainty, edges, rows, energy, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    # Coarsen both displayed images by the same integer factor; saved FITS/NPZ
    # and radial statistics retain their original pixels. This is a sum, not smoothing.
    factor = 5 if reference.shape[0] % 5 == 0 else 1

    def display_image(values):
        n = values.shape[0] // factor
        return values.reshape(n, factor, n, factor).sum(axis=(1, 3))

    displayed_ref, displayed_mc = display_image(reference), display_image(mc)
    positive = displayed_ref[displayed_ref > 0]
    norm = LogNorm(vmin=np.quantile(positive, 0.02), vmax=positive.max())
    extent = [edges[0], edges[-1], edges[0], edges[-1]]
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(cmap(0))
    for ax, values, title in zip(
        axs[0],
        [displayed_ref, displayed_mc],
        ["NewDust first-order reference", "DSH first-order Monte Carlo"],
        strict=True,
    ):
        artist = ax.imshow(
            np.ma.masked_less_equal(values, 0),
            origin="lower",
            extent=extent,
            norm=norm,
            cmap=cmap,
        )
        ax.set(xlabel="Sky x (arcsec)", ylabel="Sky y (arcsec)", title=title)
        fig.colorbar(artist, ax=ax, label="Halo fluence (ph cm$^{-2}$ pixel$^{-1}$)")
    radius = np.array([r["radius_mid_arcsec"] for r in rows])
    ref = np.array([r["newdust_ph_cm2_arcsec2"] for r in rows])
    sim = np.array([r["dsh_ph_cm2_arcsec2"] for r in rows])
    se = np.array([r["dsh_se_ph_cm2_arcsec2"] for r in rows])
    axs[1, 0].plot(radius, ref, color="black", label="NewDust, absolute")
    axs[1, 0].errorbar(
        radius, sim, yerr=se, fmt=".", color="tab:blue", label="DSH ± 1 MC SE"
    )
    axs[1, 0].set(
        yscale="log",
        xlabel="Radius (arcsec)",
        ylabel="Surface fluence (ph cm$^{-2}$ arcsec$^{-2}$)",
    )
    axs[1, 0].legend()
    ratio = np.array([r["dsh_to_newdust"] for r in rows])
    rse = np.divide(se, ref, out=np.full_like(se, np.nan), where=ref > 0)
    axs[1, 1].errorbar(radius, ratio, yerr=rse, fmt=".", color="tab:blue")
    axs[1, 1].axhline(1, color="black", lw=1)
    axs[1, 1].set(xlabel="Radius (arcsec)", ylabel="DSH / NewDust (no fitted scale)")
    fig.suptitle(
        f"{energy:g} keV · {len(config['screens'])} thin screen(s) · D={config['source_distance_kpc']:g} kpc · display pixels {factor * config['pixel_arcsec']:g} arcsec"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare(config, table, physics, cloud, ei, output):
    energy = float(table.energy_kev[ei])
    sigma = table.scattering_cross_section_cm2_per_h[ei]
    tau = float(sigma * sum(s["nh_cm2"] for s in config["screens"]))
    if tau > config["maximum_total_tau"]:
        raise ValueError(f"total tau={tau:g} exceeds optically thin comparison limit")
    audit = material_audit(table, ei)
    grid, profile, components = reference_profiles(config, table, energy)
    order = config["pixel_quadrature_order"]
    edges, ref = pixel_reference(config, grid, profile, order)
    _, ref_check = pixel_reference(config, grid, profile, order * 2)
    fine_config = dict(
        config, reference_radial_step_arcsec=config["reference_radial_step_arcsec"] / 2
    )
    fine_grid, fine_profile, _ = reference_profiles(fine_config, table, energy)
    _, ref_fine = pixel_reference(config, fine_grid, fine_profile, order * 2)
    reference_error = np.abs(ref_fine - ref_check) + np.abs(ref_check - ref)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in config["seeds"]:
        run = run_seed(config, cloud, physics, table, ei, edges, seed)
        np.savez_compressed(
            output / f"seed_{seed}.npz",
            **run,
            sky_edges_arcsec=edges,
            histories=config["packets_per_seed"],
        )
        runs.append(run)
    count = len(runs)
    histories = config["packets_per_seed"] * count
    mc = sum(r["image"] for r in runs) / count
    sq = sum(r["squares"] for r in runs) / count**2
    all_orders = sum(r["all_orders_image"] for r in runs) / count
    r_edges, sim_r, sq_r, pixels = radial_sums(mc, sq, edges, config)
    _, ref_r, _, _ = radial_sums(ref, ref * 0, edges, config)
    _, num_r, _, _ = radial_sums(reference_error, reference_error * 0, edges, config)
    errors = history_se(sim_r, sq_r, histories)
    neff = np.divide(sim_r**2, sq_r, out=np.zeros_like(sim_r), where=sq_r > 0)
    fmin = min(
        min(s["fraction_from_observer"], 1 - s["fraction_from_observer"])
        for s in config["screens"]
    )
    # Deliberately conservative limits; refinement studies are still required.
    geometry_budget = (
        8 * (np.sqrt(2) * config["half_width_arcsec"] * ARCSEC_TO_RAD / fmin) ** 2
    )
    thickness_budget = (
        4 * config["thickness_kpc"] / (config["source_distance_kpc"] * fmin)
    )
    systematic = tau + geometry_budget + thickness_budget
    resolved = (neff >= config["minimum_annulus_neff"]) & (ref_r > 0)
    numeric_relative = np.divide(
        num_r, ref_r, out=np.zeros_like(num_r), where=ref_r > 0
    )
    numeric_ok = bool(
        np.max(numeric_relative) <= config["maximum_numerical_budget_fraction"]
    )
    radial_ok = bool(
        np.all(
            np.abs(sim_r[resolved] - ref_r[resolved])
            <= 4 * errors[resolved] + systematic * ref_r[resolved] + num_r[resolved]
        )
    )
    aperture_mc = float(sim_r.sum())
    aperture_ref = float(ref_r.sum())
    aperture_se = float(history_se(sim_r.sum(), sq_r.sum(), histories))
    rel_se = aperture_se / aperture_mc if aperture_mc > 0 else float("inf")
    absolute_ok = absolute_gate(
        aperture_mc, aperture_ref, aperture_se, systematic, num_r.sum()
    )
    seed_summaries = []
    for seed, run in zip(config["seeds"], runs, strict=True):
        _, rs, rq, _ = radial_sums(run["image"], run["squares"], edges, config)
        seed_summaries.append(
            {
                "seed": seed,
                "aperture_fluence": float(rs.sum()),
                "aperture_se": float(
                    history_se(rs.sum(), rq.sum(), config["packets_per_seed"])
                ),
            }
        )
    from scipy.stats import chi2

    seed_p = None
    if len(seed_summaries) >= 3 and all(s["aperture_se"] > 0 for s in seed_summaries):
        seed_values = np.array([s["aperture_fluence"] for s in seed_summaries])
        seed_errors = np.array([s["aperture_se"] for s in seed_summaries])
        consistency_mean = np.average(seed_values, weights=1 / seed_errors**2)
        seed_p = float(
            chi2.sf(
                np.sum(((seed_values - consistency_mean) / seed_errors) ** 2),
                len(seed_values) - 1,
            )
        )
    statuses = sum(r["statuses"] for r in runs)
    transport = api("transport.kernel")
    bad_status = int(
        statuses[
            [
                transport.ACTIVE,
                transport.MAX_INTERACTIONS,
                transport.INVALID_ENERGY,
                transport.INVALID_STATE,
            ]
        ].sum()
    )
    coverage_ok = all(
        r["outside_weight"] <= 1e-6 * max(r["total_first_weight"], 1e-300) for r in runs
    )
    enough = bool(
        rel_se <= config["maximum_aperture_relative_se"]
        and resolved.sum() >= config["minimum_resolved_annuli"]
        and count >= 3
    )
    gates = {
        "absolute_aperture": bool(absolute_ok),
        "resolved_annuli": radial_ok,
        "sampling_sufficient": enough,
        "reference_numerics": numeric_ok,
        "transport_complete": bad_status == 0,
        "first_order_image_coverage": coverage_ok,
        "seed_consistency": seed_p is not None and seed_p >= 0.001,
    }
    from azimuth import analyze_azimuth

    azimuth = analyze_azimuth(ref, mc, sq, edges, histories, output)
    if config.get("require_azimuth", True):
        gates["azimuthal_sampling_sufficient"] = azimuth["sampling_sufficient"]
        gates["azimuthal_harmonics"] = azimuth["resolved_harmonics_passed"]
    summary = {
        "energy_kev": energy,
        "total_scattering_tau": tau,
        "material_audit": audit,
        "aperture_newdust_ph_cm2": aperture_ref,
        "aperture_dsh_ph_cm2": aperture_mc,
        "absolute_ratio": aperture_mc / aperture_ref,
        "aperture_mc_se": aperture_se,
        "aperture_relative_mc_se": rel_se,
        "resolved_annuli": int(resolved.sum()),
        "systematic_comparison_budget_fraction": systematic,
        "budget_components": {
            "optically_thin_vs_extinguished": tau,
            "conservative_small_angle": geometry_budget,
            "conservative_finite_thickness": thickness_budget,
        },
        "max_reference_numeric_relative_error": float(np.max(numeric_relative)),
        "histories": histories,
        "seed_summaries": seed_summaries,
        "launch_proposal_kind": str(runs[0]["proposal_kind"])
        if "proposal_kind" in runs[0]
        else "legacy_rectangular_mixture",
        "seed_consistency_p": seed_p,
        "terminal_status_counts": {
            transport.STATUS_NAMES[i]: int(v) for i, v in enumerate(statuses)
        },
        "all_orders_to_first_order_image_ratio": float(all_orders.sum() / mc.sum())
        if mc.sum()
        else None,
        "azimuthal_analysis": azimuth,
        "gates": gates,
        "all_passed": all(gates.values()),
        "scope": "First-scatter order only; low total tau; no absorption; time-integrated; finite aperture.",
    }
    rows = []
    for i in range(len(sim_r)):
        area = pixels[i] * config["pixel_arcsec"] ** 2
        rows.append(
            {
                "radius_lo_arcsec": r_edges[i],
                "radius_hi_arcsec": r_edges[i + 1],
                "radius_mid_arcsec": 0.5 * (r_edges[i] + r_edges[i + 1]),
                "pixel_area_arcsec2": area,
                "newdust_ph_cm2_arcsec2": ref_r[i] / area if area else 0,
                "dsh_ph_cm2_arcsec2": sim_r[i] / area if area else 0,
                "dsh_se_ph_cm2_arcsec2": errors[i] / area if area else 0,
                "dsh_to_newdust": sim_r[i] / ref_r[i] if ref_r[i] else float("nan"),
                "neff": neff[i],
                "used_for_gate": bool(resolved[i]),
            }
        )
    with (output / "radial_profiles.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    se_image = history_se(mc, sq, histories)
    np.savez_compressed(
        output / "comparison.npz",
        newdust_fluence=ref,
        dsh_first_fluence=mc,
        dsh_first_se=se_image,
        dsh_all_orders_fluence=all_orders,
        event_counts=sum(r["counts"] for r in runs),
        sky_edges_arcsec=edges,
        reference_radius_arcsec=grid,
        reference_surface_fluence=profile,
        reference_screen_profiles_per_arcsec2=components,
    )
    write_fits(output / "newdust_halo.fits", ref, config, energy, "NewDust first-order")
    write_fits(output / "dsh_halo.fits", mc, config, energy, "DSH first-order")
    write_fits(
        output / "dsh_halo_se.fits", se_image, config, energy, "DSH MC standard error"
    )
    plot_result(
        output / "comparison.png", ref, mc, se_image, edges, rows, energy, config
    )
    (output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("benchmark_config.json")
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("newdust_validation"))
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--packets",
        type=int,
        help="Override packets per seed (pilot runs remain sampling-gated)",
    )
    parser.add_argument(
        "--seed", type=int, help="One-seed pilot; cannot pass the final gate"
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Nine single-screen tests: x=.25,.5,.75 and E=3.3,4.9,6.9",
    )
    parser.add_argument(
        "--multiple",
        action="store_true",
        help="Three screens at x=.25,.5,.75; conserve total column",
    )
    args = parser.parse_args()
    if args.matrix and args.multiple:
        parser.error("--matrix and --multiple select different stages")
    os.environ["JAX_PLATFORMS"] = args.device
    sys.path.insert(0, str(args.repo.resolve()))
    config = json.loads(args.config.read_text())
    if args.packets is not None:
        config["packets_per_seed"] = args.packets
    if args.seed is not None:
        config["seeds"] = [args.seed]
    validate(config)
    materials = api("physics.newdust")
    table = materials.load_newdust_scattering_table()
    physics = materials.build_dust_physics_from_newdust(
        table, np.zeros_like(table.energy_kev)
    )
    configs = [config]
    total_nh = sum(s["nh_cm2"] for s in config["screens"])
    if args.matrix:
        configs = [
            dict(
                config,
                energies_kev=[3.3, 4.9, 6.9],
                screens=[{"fraction_from_observer": f, "nh_cm2": total_nh}],
            )
            for f in [0.25, 0.5, 0.75]
        ]
    if args.multiple:
        configs = [
            dict(
                config,
                energies_kev=[3.3, 4.9, 6.9],
                screens=[
                    {"fraction_from_observer": f, "nh_cm2": total_nh / 3}
                    for f in [0.25, 0.5, 0.75]
                ],
            )
        ]
    args.output.mkdir(parents=True, exist_ok=True)
    _, _, package = load_newdust()
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "source-checkout"
    git = subprocess.run(
        ["git", "-C", str(args.repo.resolve()), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    code_hashes = {}
    for name in [
        "geometry.clouds",
        "transport.kernel",
        "observer.scoring",
        "physics.newdust",
        "sources.launch",
        "sources.radial_launch",
    ]:
        module = api(name)
        code_hashes[name] = hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest()
    provenance = {
        "newdust_import": package,
        "newdust_version": version,
        "dsh_reviewed_commit": config["dsh_revision_reviewed"],
        "dsh_actual_git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "dsh_code_sha256": code_hashes,
        "validation_launch_rule": config.get("launch_proposal", "radial"),
        "validation_launch_code_sha256": hashlib.sha256(
            (
                Path(api("sources.radial_launch").__file__)
                if config.get("launch_proposal", "radial") == "radial"
                else Path(api("validation.launch").__file__)
            ).read_bytes()
        ).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "material_metadata": dict(table.metadata),
        "configurations": configs,
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    summaries = []
    for setting in configs:
        validate(setting)
        cloud = make_cloud(setting)
        label = (
            "multi"
            if len(setting["screens"]) > 1
            else f"x{setting['screens'][0]['fraction_from_observer']:g}"
        )
        for energy in setting["energies_kev"]:
            indexes = np.flatnonzero(
                np.isclose(table.energy_kev, energy, rtol=0, atol=1e-10)
            )
            if len(indexes) != 1:
                raise ValueError(
                    "Use exact table-node energies for the baseline comparison"
                )
            summaries.append(
                compare(
                    setting,
                    table,
                    physics,
                    cloud,
                    int(indexes[0]),
                    args.output / f"{label}_E{energy:g}",
                )
            )
    (args.output / "suite_report.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    return 0 if all(s["all_passed"] for s in summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
