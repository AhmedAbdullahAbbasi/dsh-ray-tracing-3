"""Compare the independent WD01/D03 material with the frozen RefleX images.

Geometry, camera response and absolute normalization come from the accepted
RefleX benchmark. Only the material and independent random seeds change.
This is a comparison of physical models, not a matched-physics acceptance test.
PNG/PDF plots, native-pixel FITS, CSV and history-grouped moments are automatic.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsh.io.atomic import atomic_savez_compressed  # noqa: E402
from reflex_adapter.compare_reflex import (  # noqa: E402
    ARCSEC,
    covariance,
    image_edges,
    make_cloud,
    normalize_camera_counts,
    radial_pixel_map,
    run_seed,
)

DEFAULT_MATERIAL = ROOT / "dsh/data/materials/wd01_wilm_c50_si90_v1"
DEFAULT_SEEDS = [28611, 28612, 28613]


def combined_moments(runs):
    """Average equal-size independent runs, retaining zero-history covariance."""
    if not runs or len({int(r["packets"]) for r in runs}) != 1:
        raise ValueError("require equal-size independent runs")
    if len({int(r["seed"]) for r in runs}) != len(runs):
        raise ValueError("duplicate seeds are not independent")
    count = len(runs)
    histories = sum(int(r["packets"]) for r in runs)
    if histories < 2:
        raise ValueError("at least two histories are required")
    result = {"histories": histories}
    for name in ("image", "radial", "order_sum"):
        result[name] = sum(r[name] for r in runs) / count
    for name in ("radial", "order"):
        sums = result["order_sum" if name == "order" else name]
        raw = sum(r[f"{name}_q"] for r in runs) / count**2
        result[f"{name}_covariance"] = covariance(sums, raw, histories)
    raw_image = sum(r["image_q"] for r in runs) / count**2
    total_image = result["image"].sum(axis=0)
    result["image_variance"] = (
        histories / (histories - 1) * (raw_image - total_image**2 / histories)
    )
    return result


def independent_ratio(model, reference, model_variance, reference_variance):
    """Ratio and delta-method SE for two independent estimates."""
    model, reference = np.asarray(model), np.asarray(reference)
    if np.any(reference <= 0):
        raise ValueError("ratio requires positive reference counts")
    variance = model_variance / reference**2 + (
        model**2 * reference_variance / reference**4
    )
    return model / reference, np.sqrt(np.maximum(variance, 0))


def material_diagnostics(physics, config):
    from dsh.validation.absorbed_observer import host_material

    lo, hi, fraction, scattering, absorption = host_material(
        physics, config["energy_kev"]
    )
    cdf = np.asarray(physics.scattering_angle_cdf)
    cdf = (1 - fraction) * cdf[lo] + fraction * cdf[hi]
    quantiles = np.interp(
        [0.1, 0.5, 0.9], cdf, np.asarray(physics.scattering_angle_rad) / ARCSEC
    )
    return {
        "scattering_cm2_per_h": scattering,
        "absorption_cm2_per_h": absorption,
        "central_direct_transmission": float(
            np.exp(-config["nh_cm2"] * (scattering + absorption))
        ),
        "physical_angle_quantiles_arcsec": quantiles.tolist(),
    }


def summarize(config, runs, output, material_info):
    from astropy.io import fits
    from scipy.stats import chi2

    moments = combined_moments(runs)
    image, radial = moments["image"], moments["radial"]
    reference = fits.getdata(ROOT / config["reflex_dust_image"]).astype(float)
    all_reference = fits.getdata(ROOT / config["reflex_all_image"]).astype(float)
    direct = all_reference - reference
    if (
        reference.shape != image.shape[1:]
        or np.any(reference < 0)
        or np.any(direct < 0)
        or not np.all(reference == np.rint(reference))
        or not np.all(direct == np.rint(direct))
    ):
        raise ValueError("reference requires raw unweighted camera counts")
    center_mask = np.zeros_like(reference, dtype=bool)
    center = len(reference) // 2
    center_mask[center - 1 : center + 1, center - 1 : center + 1] = True
    if np.any(direct[~center_mask]):
        raise ValueError("reference direct-source center convention changed")
    reference = reference[::-1].copy()  # RefleX Y decreases; DSH Y increases.
    radial_map = radial_pixel_map(config)
    if radial_map.min() < 0 or radial_map.max() >= len(radial):
        raise ValueError("annuli must cover every image pixel center")
    observed = np.bincount(radial_map.ravel(), reference.ravel(), len(radial))
    normalization = normalize_camera_counts(config) / config["source_fluence_ph_cm2"]
    expected = radial * normalization
    model_cov = moments["radial_covariance"] * normalization**2
    # Plug-in multinomial count covariance for the observed RefleX population.
    # No assumption that the different WD01 material is the null truth.
    reference_cov = (
        np.diag(observed) - np.outer(observed, observed) / config["reflex_photons"]
    )
    total_cov = model_cov + reference_cov
    difference = expected - observed
    statistic = float(difference @ np.linalg.solve(total_cov, difference))
    z = difference / np.sqrt(np.diag(total_cov))
    ratio, ratio_se = independent_ratio(
        expected, observed, np.diag(model_cov), np.diag(reference_cov)
    )
    seed_fluxes = np.array([r["radial"].sum() for r in runs])
    seed_variances = np.array(
        [covariance(r["radial"], r["radial_q"], int(r["packets"])).sum() for r in runs]
    )
    weighted_mean = np.average(seed_fluxes, weights=1 / seed_variances)
    seed_p = (
        float(
            chi2.sf(
                np.sum((seed_fluxes - weighted_mean) ** 2 / seed_variances),
                len(runs) - 1,
            )
        )
        if len(runs) > 1
        else None
    )
    statuses = sum(r["statuses"] for r in runs)
    relative_mc_se = float(np.sqrt(moments["radial_covariance"].sum()) / radial.sum())
    gates = {
        "transport_complete": int(statuses[[0, 4, 5, 6]].sum()) == 0,
        "order_closure": bool(np.isclose(image.sum(), radial.sum(), rtol=1e-11)),
        "sampling_sufficient": bool(
            relative_mc_se <= config["maximum_aperture_relative_mc_se"]
            and len(runs) >= 3
            and observed.min() >= config["minimum_reflex_counts_per_annulus"]
        ),
        "seed_consistency": seed_p is not None
        and seed_p >= config["minimum_global_pvalue"],
    }
    predicted_direct = (
        normalize_camera_counts(config)
        * material_info["wd01"]["central_direct_transmission"]
    )
    direct_variance = direct.sum() * (1 - direct.sum() / config["reflex_photons"])
    order_sum = moments["order_sum"]
    summary = {
        "comparison_kind": "different physical materials, fixed absolute normalization",
        "numerical_quality_passed": all(gates.values()),
        "numerical_quality_gates": gates,
        "histories": moments["histories"],
        "reference_halo_counts": float(observed.sum()),
        "wd01_halo_counts": float(expected.sum()),
        "halo_ratio": float(expected.sum() / observed.sum()),
        "halo_ratio_se": float(
            independent_ratio(
                expected.sum(), observed.sum(), model_cov.sum(), reference_cov.sum()
            )[1]
        ),
        "halo_difference_z": float(difference.sum() / np.sqrt(total_cov.sum())),
        "global_radial_chi2": statistic,
        "radial_degrees_of_freedom": len(radial),
        "global_radial_pvalue": float(chi2.sf(statistic, len(radial))),
        "maximum_annulus_absolute_z": float(abs(z).max()),
        "aperture_relative_mc_se": relative_mc_se,
        "seed_consistency_pvalue": seed_p,
        "source_camera_normalization": float(normalization),
        "fitted_parameters": 0,
        "reference_direct_counts": float(direct.sum()),
        "wd01_direct_counts": float(predicted_direct),
        "direct_difference_z": float(
            (predicted_direct - direct.sum()) / np.sqrt(direct_variance)
        ),
        "multiple_to_first_ratio": float(order_sum[1:].sum() / order_sum[0]),
        "order_fluences": order_sum.tolist(),
        "order_covariance": moments["order_covariance"].tolist(),
        "transport_status_counts": statuses.tolist(),
        "annulus_reference_counts": observed.tolist(),
        "annulus_wd01_counts": expected.tolist(),
        "annulus_ratio": ratio.tolist(),
        "annulus_ratio_se": ratio_se.tolist(),
        "annulus_difference_z": z.tolist(),
        "statistical_scope": (
            "Approximate Gaussian radial diagnostic with observed RefleX multinomial "
            "covariance plus independent history-grouped DSH covariance. No geometry "
            "allowance added. This is not a same-material transport acceptance test."
        ),
        "material": material_info,
        "config": config,
    }
    (output / "comparison_results.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    atomic_savez_compressed(
        output / "comparison_arrays.npz",
        image_by_order=image,
        image_mc_variance=moments["image_variance"],
        reference_counts=reference,
        expected_counts=image.sum(axis=0) * normalization,
        radial_fluence=radial,
        radial_covariance=moments["radial_covariance"],
        observed_radial_counts=observed,
        expected_radial_counts=expected,
        model_radial_count_covariance=model_cov,
        reference_radial_count_covariance=reference_cov,
        radial_count_covariance=total_cov,
        sky_edges_arcsec=image_edges(config),
        annulus_edges_arcsec=config["annulus_edges_arcsec"],
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--material", type=Path, default=DEFAULT_MATERIAL)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    # Fail on missing scientific/plotting dependencies before expensive transport.
    import astropy
    import matplotlib
    import scipy

    from dsh.observer.response_transport import response_provenance
    from dsh.physics.wd01_ism import load_wd01_ism
    from dsh.sources.radial_launch import build_radial_cloud_launch_geometry
    from dsh.validation.absorbed_observer import host_phase
    from reflex_adapter.reflex_materials import load_reflex_materials
    from scripts.plot_wd01_reflex import plot_comparison

    matplotlib.use("Agg")
    config = json.loads((ROOT / "reflex_adapter/comparison_config.json").read_text())
    config["seeds"] = DEFAULT_SEEDS
    definition, _, _, _, physics = load_wd01_ism(args.material)
    _, _, reflex_physics = load_reflex_materials()
    material_info = {
        "directory_name": args.material.name,
        "definition_sha256": definition.sha256,
        "recipe": definition.as_dict(),
        "audit": definition.audit(),
        "scope": (
            "Experimental WD01 shapes renormalized to the declared Wilms budget; "
            "not the nominal published WD01 extinction normalization. "
            "D03 graphite/MgFeSiO4 spheres, Mie totals and hybrid Mie/ADT phase; "
            "residual neutral atomic gas plus H2 plus Mie grain absorption once."
        ),
        "wd01": material_diagnostics(physics, config),
        "reflex": material_diagnostics(reflex_physics, config),
    }
    cloud = make_cloud(config)
    geometry = build_radial_cloud_launch_geometry(cloud, physics)
    provenance = response_provenance(cloud, physics, geometry)
    paths = [
        Path(__file__),
        ROOT / "reflex_adapter/compare_reflex.py",
        *(args.material.glob("*.json")),
        *(args.material.glob("*.npz")),
        ROOT / config["reflex_dust_image"],
        ROOT / config["reflex_all_image"],
    ]
    hashes = {
        str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")
            if p.suffix in (".py", ".json")
            else p.read_bytes()
        ).hexdigest()
        for p in paths
    }
    provenance.update(
        input_sha256=hashes,
        config=config,
        material_definition_sha256=definition.sha256,
        collection_source_sha256=hashlib.sha256(
            (inspect.getsource(make_cloud) + inspect.getsource(run_seed)).encode()
        ).hexdigest(),
    )
    fingerprint = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode()
    ).hexdigest()
    provenance.update(
        collection_fingerprint=fingerprint,
        repository_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        astropy_version=astropy.__version__,
        scipy_version=scipy.__version__,
        matplotlib_version=matplotlib.__version__,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    saved_path = args.output / "provenance.json"
    if args.reuse:
        if not saved_path.exists():
            raise ValueError("--reuse requires saved collection provenance")
        saved = json.loads(saved_path.read_text())
        if saved["collection_fingerprint"] != fingerprint:
            raise ValueError("saved inputs/runtime differ; use a new output directory")
    elif saved_path.exists() or list(args.output.glob("seed_*.npz")):
        raise FileExistsError("use --reuse or a fresh output directory")
    else:
        saved_path.write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output / "material_diagnostics.json").write_text(
        json.dumps(material_info, indent=2) + "\n"
    )
    print(json.dumps({k: material_info[k] for k in ("wd01", "reflex")}, indent=2))
    runs = []
    for seed in config["seeds"]:
        path = args.output / f"seed_{seed}.npz"
        if args.reuse and path.exists():
            with np.load(path, allow_pickle=False) as data:
                result = dict(data)
            if (
                int(result["seed"]) != seed
                or int(result["packets"]) != config["packets_per_seed"]
            ):
                raise ValueError("saved seed metadata mismatch")
        else:
            # The immutable benchmark collector writes in place. Isolate its
            # working file, then atomically publish the complete seed archive.
            with tempfile.TemporaryDirectory(dir=args.output) as temporary:
                result = run_seed(config, physics, cloud, seed, Path(temporary))
                atomic_savez_compressed(path, **result)
        runs.append(result)
    summary = summarize(config, runs, args.output, material_info)
    angles = np.geomspace(0.5, 2000, 512)
    atomic_savez_compressed(
        args.output / "phase_comparison.npz",
        physical_angle_arcsec=angles,
        wd01_phase_per_sr=host_phase(physics, config["energy_kev"], angles * ARCSEC),
        reflex_phase_per_sr=host_phase(
            reflex_physics, config["energy_kev"], angles * ARCSEC
        ),
    )
    plot_comparison(args.output)
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("config", "material", "order_covariance")
            },
            indent=2,
        )
    )
    if not summary["numerical_quality_passed"]:
        raise SystemExit("Numerical quality gates failed; inspect the saved report")


if __name__ == "__main__":
    main()
