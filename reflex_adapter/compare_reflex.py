"""Matched absolute 4.9-keV RefleX/DSH comparison using production transport.

Run from the repository root. Every moment is grouped by photon history,
including zero contributions. The physical halo is compared with raw RefleX
camera counts through a predetermined source/camera normalization, with no
fitted amplitude. The angular aperture response is sampled at the event level.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY))
ARCSEC = np.pi / 648000.0
PC_CM = 3.0856775814913673e18


def make_cloud(config):
    import jax.numpy as jnp
    from dsh.geometry.clouds import AngularDistanceCloud

    center = config["cloud_center_kpc"]
    dz = config["cloud_thickness_pc"] / 1000
    extent = np.arctan(config["cloud_radius_pc"] / (center * 1000)) / ARCSEC
    edges = np.linspace(-extent, extent, config["cloud_sky_cells"] + 1)
    midpoint = (edges[1:] + edges[:-1]) / 2
    xx, yy = np.meshgrid(midpoint, midpoint)
    occupied = np.hypot(np.tan(xx * ARCSEC), np.tan(yy * ARCSEC)) * center * 1000 <= config["cloud_radius_pc"]
    z = np.array([center - dz / 2, center + dz / 2])
    width_cm = np.diff(z) * 1000 * PC_CM
    columns = occupied[None] * config["nh_cm2"]
    return AngularDistanceCloud(*(jnp.asarray(a, dtype=jnp.float32) for a in [
        columns, columns / width_cm[:, None, None], edges, edges, z,
        width_cm, config["source_distance_kpc"]]))


def image_edges(config):
    half, pixel = config["image_half_width_arcsec"], config["pixel_arcsec"]
    return np.linspace(-half, half, round(2 * half / pixel) + 1)


def radial_pixel_map(config):
    edges = image_edges(config)
    centers = (edges[1:] + edges[:-1]) / 2
    x, y = np.meshgrid(centers, centers)
    return np.searchsorted(config["annulus_edges_arcsec"], np.hypot(x, y), side="right") - 1


def run_seed(config, physics, cloud, seed, output):
    import jax
    import jax.numpy as jnp
    from jax import random
    from dsh.observer.scoring import score_peeloff_events
    from dsh.sources.launch import sample_source_launches
    from dsh.sources.models import SourcePackets
    from dsh.sources.radial_launch import build_radial_cloud_launch_geometry
    from dsh.transport.kernel import transport_photon_batch

    geometry = build_radial_cloud_launch_geometry(cloud, physics)
    chunk = config["chunk_size"]
    packets = config["packets_per_seed"]
    guard = config["max_interactions"]

    @jax.jit
    def batch(key):
        launch_key, transport_key = random.split(key)
        pk = SourcePackets(jnp.full(chunk, config["energy_kev"]), jnp.zeros(chunk),
                           jnp.full(chunk, config["source_fluence_ph_cm2"] / packets),
                           jnp.zeros(chunk, dtype=jnp.int32), jnp.zeros(chunk, dtype=jnp.int32))
        launched = sample_source_launches(launch_key, pk, geometry)
        history = transport_photon_batch(transport_key, launched.position_pc,
            launched.momentum_kev, cloud, physics, max_interactions=guard)
        ev = score_peeloff_events(launched, history, cloud, physics)
        return (ev.valid, ev.scattering_order, ev.sky_x_arcsec, ev.sky_y_arcsec,
                ev.weight_observer_fluence, history.status,
                jnp.linalg.norm(history.interactions.position_pc, axis=-1),
                ev.arrival_time_s, history.n_scatter)

    n = len(image_edges(config)) - 1
    nr = len(config["annulus_edges_arcsec"]) - 1
    image = np.zeros((3, n, n))
    image_q = np.zeros((n, n))
    radial = np.zeros(nr)
    radial_q = np.zeros((nr, nr))
    order_sum = np.zeros(3)
    order_q = np.zeros((3, 3))
    statuses = np.zeros(7, dtype=np.int64)
    radial_map = radial_pixel_map(config)
    rng = np.random.default_rng(seed + 10000000)
    all_weight = 0.0
    outside_weight = 0.0
    start = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    completed = 0
    # Static chunk shape; padding histories are removed before all accounting.
    for chunk_number in range((packets + chunk - 1) // chunk):
        size = min(chunk, packets - completed)
        values = batch(random.fold_in(random.PRNGKey(seed), chunk_number))
        valid, order, x, y, w, status, radius, arrival, scatter_count = [np.asarray(a)[:size] for a in values]
        hist, slot = np.nonzero(valid)
        weights = w[hist, slot].astype(np.float64)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or not np.all(np.isfinite(arrival[hist, slot])):
            raise ValueError("invalid observer score")
        all_weight += weights.sum()
        # A camera's finite circular aperture perturbs the inferred direction
        # by aperture_radius / event-to-camera distance. Use each event radius.
        angle = rng.uniform(0, 2 * np.pi, len(hist))
        offset = np.sqrt(rng.uniform(size=len(hist))) * config["camera_aperture_radius_pc"] / radius[hist, slot] / ARCSEC
        xx = x[hist, slot] + offset * np.cos(angle)
        yy = y[hist, slot] + offset * np.sin(angle)
        edges = image_edges(config)
        ix = np.searchsorted(edges, xx, side="right") - 1
        iy = np.searchsorted(edges, yy, side="right") - 1
        keep = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        outside_weight += weights[~keep].sum()
        hist, slot, ix, iy, weights = hist[keep], slot[keep], ix[keep], iy[keep], weights[keep]
        group = np.minimum(order[hist, slot] - 1, 2)
        np.add.at(image, (group, iy, ix), weights)
        # Combine repeated scores in the same pixel/history before squaring.
        combined_key, inv = np.unique(hist * n * n + iy * n + ix, return_inverse=True)
        combined = np.bincount(inv, weights=weights)
        np.add.at(image_q.ravel(), combined_key % (n * n), combined**2)
        per_history = np.zeros((size, nr))
        np.add.at(per_history, (hist, radial_map[iy, ix]), weights)
        radial += per_history.sum(axis=0)
        radial_q += per_history.T @ per_history
        per_order = np.zeros((size, 3))
        np.add.at(per_order, (hist, group), weights)
        order_sum += per_order.sum(axis=0)
        order_q += per_order.T @ per_order
        statuses += np.bincount(status.astype(int), minlength=7)
        completed += size
        if chunk_number == 0 or chunk_number % 25 == 0 or completed == packets:
            print(f"seed={seed}: {completed:,}/{packets:,}, {time.monotonic()-start:.1f}s, halo={radial.sum():.7g}", flush=True)
    result = dict(image=image, image_q=image_q, radial=radial, radial_q=radial_q,
                  order_sum=order_sum, order_q=order_q, statuses=statuses,
                  packets=packets, seed=seed, elapsed_seconds=time.monotonic()-start,
                  all_weight=all_weight, outside_weight=outside_weight)
    np.savez_compressed(output / f"seed_{seed}.npz", **result)
    return result


def covariance(sums, cross_moments, histories):
    return histories / (histories - 1) * (cross_moments - np.outer(sums, sums) / histories)


def normalize_camera_counts(config):
    side = config["camera_grid_side"]
    coords = (np.arange(side) - (side - 1) / 2) * config["camera_spacing_pc"]
    x, y = np.meshgrid(coords, coords)
    distance = np.sqrt((config["source_distance_kpc"] * 1000)**2 + x*x + y*y)
    aperture = config["camera_aperture_radius_pc"]
    root = np.sqrt(distance**2 + aperture**2)
    solid_angle = 2 * np.pi * aperture**2 / (root * (root + distance))
    cone = 2 * np.pi * (1 - np.cos(np.deg2rad(config["source_cone_half_angle_deg"])))
    return config["reflex_photons"] * solid_angle.sum() / cone


def compare(config, runs, physics, output):
    from astropy.io import fits
    from scipy.stats import chi2

    count = len(runs)
    histories = sum(int(r["packets"]) for r in runs)
    if len(set(int(r["packets"]) for r in runs)) != 1:
        raise ValueError("combine equal-size independent runs")
    image = sum(r["image"] for r in runs) / count
    radial = sum(r["radial"] for r in runs) / count
    radial_q = sum(r["radial_q"] for r in runs) / count**2
    cov = covariance(radial, radial_q, histories)
    order_sum = sum(r["order_sum"] for r in runs) / count
    order_q = sum(r["order_q"] for r in runs) / count**2
    order_cov = covariance(order_sum, order_q, histories)
    reference = fits.getdata(REPOSITORY / config["reflex_dust_image"]).astype(float)
    all_reference = fits.getdata(REPOSITORY / config["reflex_all_image"]).astype(float)
    direct = all_reference - reference
    if reference.shape != image.shape[1:] or np.any(reference < 0) or np.any(direct < 0):
        raise ValueError("reference image mismatch")
    if not np.all(reference == np.rint(reference)) or not np.all(direct == np.rint(direct)):
        raise ValueError("this comparison requires raw, unweighted RefleX photon-count images")
    center_mask = np.zeros_like(reference, dtype=bool)
    center = len(reference) // 2
    center_mask[center-1:center+1, center-1:center+1] = True
    if np.any(direct[~center_mask] != 0):
        raise ValueError("direct image does not match the calibrated four-pixel source-center convention")
    # RefleX's Y axis decreases. Radial statistics are reflection invariant.
    reference = reference[::-1].copy()
    radial_map = radial_pixel_map(config)
    observed = np.bincount(radial_map.ravel(), weights=reference.ravel(), minlength=len(radial))
    normalization = normalize_camera_counts(config) / config["source_fluence_ph_cm2"]
    expected = radial * normalization
    model_cov = cov * normalization**2
    # Disjoint camera counts arise from a fixed number of analog histories.
    reference_cov = np.diag(expected) - np.outer(expected, expected) / config["reflex_photons"]
    total_cov = model_cov + reference_cov
    residual = observed - expected
    statistic = float(residual @ np.linalg.solve(total_cov, residual))
    global_p = float(chi2.sf(statistic, len(radial)))
    aperture_variance = float(total_cov.sum())
    aperture_z = float(residual.sum() / np.sqrt(aperture_variance))
    annulus_z = residual / np.sqrt(np.diag(total_cov))
    model_rel_se = float(np.sqrt(cov.sum()) / radial.sum())
    ei = int(np.argmin(abs(np.asarray(physics.energy_kev) - config["energy_kev"])))
    extinction = float(np.asarray(physics.scattering_cross_section_cm2_per_h)[ei] + np.asarray(physics.absorption_cross_section_cm2_per_h)[ei])
    predicted_direct = normalize_camera_counts(config) * np.exp(-config["nh_cm2"] * extinction)
    direct_z = float((direct.sum() - predicted_direct) / np.sqrt(predicted_direct * (1 - predicted_direct / config["reflex_photons"])))
    seed_fluxes = np.array([r["radial"].sum() for r in runs])
    seed_variances = np.array([covariance(r["radial"], r["radial_q"], int(r["packets"])).sum() for r in runs])
    weighted_mean = np.average(seed_fluxes, weights=1 / seed_variances)
    seed_p = float(chi2.sf(np.sum((seed_fluxes-weighted_mean)**2 / seed_variances), count-1)) if count > 1 else None
    statuses = sum(r["statuses"] for r in runs)
    allowance = config["geometry_response_fraction_allowance"]
    gates = {
        "absolute_aperture": bool(abs(residual.sum()) <= config["aperture_sigma_limit"] * np.sqrt(aperture_variance) + allowance * expected.sum()),
        "radial_annuli": bool(np.all(abs(residual) <= config["annulus_sigma_limit"] * np.sqrt(np.diag(total_cov)) + allowance * expected)),
        "global_radial_profile": global_p >= config["minimum_global_pvalue"],
        "direct_transmission": abs(direct_z) <= config["aperture_sigma_limit"],
        "sampling_sufficient": bool(model_rel_se <= config["maximum_aperture_relative_mc_se"] and count >= 3 and observed.min() >= config["minimum_reflex_counts_per_annulus"]),
        "seed_consistency": seed_p is not None and seed_p >= config["minimum_global_pvalue"],
        "transport_complete": int(statuses[[0, 4, 5, 6]].sum()) == 0,
        "order_closure": bool(np.allclose(image.sum(axis=0).sum(), radial.sum(), rtol=1e-11)),
    }
    summary = dict(all_passed=all(gates.values()), gates=gates, histories=histories,
        reference_halo_counts=float(reference.sum()), predicted_halo_counts=float(expected.sum()),
        reference_direct_counts=float(direct.sum()), predicted_direct_counts=float(predicted_direct),
        absolute_aperture_z=aperture_z, direct_z=direct_z, global_radial_chi2=statistic,
        radial_degrees_of_freedom=len(radial), global_radial_pvalue=global_p,
        maximum_annulus_absolute_z=float(np.max(abs(annulus_z))), aperture_relative_mc_se=model_rel_se,
        seed_consistency_pvalue=seed_p, source_camera_normalization=float(normalization),
        order_fluences=order_sum.tolist(), order_covariance=order_cov.tolist(),
        multiple_to_first_ratio=float(order_sum[1:].sum()/order_sum[0]),
        transport_status_counts=statuses.tolist(), annulus_observed_counts=observed.tolist(),
        annulus_predicted_counts=expected.tolist(), annulus_z=annulus_z.tolist(),
        geometry_response_fraction_allowance=allowance, fitted_parameters=0,
        passes_without_geometry_response_allowance=bool(abs(aperture_z) <= config["aperture_sigma_limit"]
            and np.max(abs(annulus_z)) <= config["annulus_sigma_limit"]),
        config=config)
    (output / "comparison_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(output / "comparison_arrays.npz", image_by_order=image,
        reference_counts=reference, expected_counts=image.sum(axis=0)*normalization,
        radial_fluence=radial, radial_covariance=cov, observed_radial_counts=observed,
        expected_radial_counts=expected, radial_count_covariance=total_cov,
        sky_edges_arcsec=image_edges(config), annulus_edges_arcsec=config["annulus_edges_arcsec"])
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "order_covariance")}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("comparison_config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not args.reuse and args.output.exists() and list(args.output.glob("seed_*.npz")):
        raise FileExistsError("output contains saved seeds; use --reuse or a new output")
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.output / "frozen_config.json"
    if args.reuse and frozen.exists() and json.loads(frozen.read_text()) != config:
        raise ValueError("saved configuration differs; use a new output directory")
    frozen.write_text(json.dumps(config, indent=2) + "\n")
    import jax
    import astropy
    import scipy
    paths = ["dsh/geometry/rays.py", "dsh/geometry/clouds.py", "dsh/sources/launch.py",
             "dsh/sources/radial_launch.py", "dsh/transport/kernel.py", "dsh/observer/scoring.py",
             "dsh/physics/dust.py", "reflex_adapter/reflex_materials.py",
             "reflex_adapter/materials/reflex_scattering_v1.npz",
             "reflex_adapter/materials/reflex_absorption_v1.npz",
             config["reflex_dust_image"], config["reflex_all_image"]]
    hashes = {}
    for name in paths:
        raw = (REPOSITORY / name).read_bytes()
        if name.endswith(".py"):
            raw = raw.replace(b"\r\n", b"\n")
        hashes[name] = hashlib.sha256(raw).hexdigest()
    fingerprint = hashlib.sha256((json.dumps(config, sort_keys=True) + json.dumps(hashes, sort_keys=True)
                                 + inspect.getsource(make_cloud) + inspect.getsource(run_seed)).encode()).hexdigest()
    provenance_path = args.output / "provenance.json"
    saved_provenance = json.loads(provenance_path.read_text()) if args.reuse and provenance_path.exists() else None
    if saved_provenance and saved_provenance["collection_fingerprint"] != fingerprint:
        raise ValueError("saved seed collection used different physics, inputs or sampling code")
    provenance = dict(collection_fingerprint=fingerprint, sha256=hashes,
        repository_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
        jax=jax.__version__, numpy=np.__version__, astropy=astropy.__version__, scipy=scipy.__version__,
        backend=jax.default_backend())
    if saved_provenance and any(not (args.output / f"seed_{seed}.npz").exists() for seed in config["seeds"]):
        if any(saved_provenance[k] != provenance[k] for k in ("jax", "numpy", "backend")):
            raise ValueError("collect missing seeds with the original JAX/NumPy/backend, or use a new output directory")
    if saved_provenance is None:
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    from reflex_adapter.reflex_materials import load_reflex_materials
    scattering, absorption, physics = load_reflex_materials()
    cloud = make_cloud(config)
    runs = []
    for seed in config["seeds"]:
        path = args.output / f"seed_{seed}.npz"
        if args.reuse and path.exists():
            with np.load(path, allow_pickle=False) as data:
                runs.append(dict(data))
        else:
            runs.append(run_seed(config, physics, cloud, seed, args.output))
    compare(config, runs, physics, args.output)


if __name__ == "__main__":
    main()
