"""Compare available complete materials against the fixed 4.9 keV RefleX image.

Reuses verified independent RefleX-adapter/WD01-hybrid histories by default;
collects fresh MRN/RG-Drude and WD01 full-Mie histories. All cases share the
accepted geometry, camera response, column and predetermined normalization.
Full Mie is composed from the checked five-energy dust reference: this run
uses its exact 4.9 keV node, not its unvalidated broadband interpolation.
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
from scripts.compare_wd01_reflex import (  # noqa: E402
    combined_moments,
    independent_ratio,
    material_diagnostics,
)

CASES = {
    "reflex_adapter": {
        "label": "DSH: RefleX-matched material",
        "short_label": "RefleX-matched adapter",
        "color": "#227c9d",
        "seeds": [8261, 8262, 8263],
        "cached": "benchmarks/materials/absorption_edges_2026-10-16/reflex_fresh",
        "scattering": "RefleX exported dust opacity and analytic angular law",
        "absorption": "RefleX PHOTO + DUST*(1-ALBEDO), already depletion-scaled",
        "abundances": "Exported Wilms; H2 nuclei fraction 0.2; iron solid fraction 0.691",
    },
    "mrn_rg": {
        "label": "DSH: MRN/RG–Drude + TBabs",
        "short_label": "MRN/RG–Drude + TBabs",
        "color": "#7851a9",
        "seeds": [38611, 38612, 38613],
        "scattering": "Historical MRN Gaussian RG/Drude, radii 0.005–0.3 micron",
        "absorption": "Frozen XSPEC TBabs, abund wilm, xsect vern; full effective absorption",
        "abundances": "Historical effective dust normalization; dust mass/H 2.32475e-26 g",
    },
    "wd01_hybrid": {
        "label": "DSH: WD01/D03 Mie–ADT",
        "short_label": "WD01/D03 Mie–ADT",
        "color": "#c64a26",
        "seeds": [28611, 28612, 28613],
        "cached": "benchmarks/materials/wd01_reflex_2026-10-16",
        "scattering": "WD01 shapes, D03 optics; Mie totals, hybrid Mie/ADT phase",
        "absorption": "Residual neutral TBnew/Verner + H2 + Mie grain absorption",
        "abundances": "Wilms budget; 50% C and 90% Si in solids; MgFeSiO4 stoichiometry; H2=0.2",
    },
    "wd01_mie": {
        "label": "DSH: WD01/D03 full Mie",
        "short_label": "WD01/D03 full Mie",
        "color": "#238b45",
        "seeds": [48611, 48612, 48613],
        "scattering": "Same WD01/D03 C50/Si90 recipe; full-Mie angular reference at 4.9 keV",
        "absorption": "Same residual neutral TBnew/Verner + H2 + Mie grain absorption",
        "abundances": "Same Wilms C50/Si90 inventory as hybrid; sparse dust energy grid",
    },
}


def hash_file(path):
    raw = Path(path).read_bytes()
    if Path(path).suffix in (".py", ".json", ".md", ".csv", ".log"):
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def geometry_equal(first, second):
    """Seeds may differ; every other sampling/experiment setting must agree."""
    return {k: v for k, v in first.items() if k != "seeds"} == {
        k: v for k, v in second.items() if k != "seeds"
    }


def load_case_material(key, output):
    from dsh.physics.absorption import load_photoelectric_absorption_table
    from dsh.physics.material_definitions import load_wilm_abundances
    from dsh.physics.materials import load_2_10_material_tables
    from dsh.physics.newdust import (
        build_dust_physics_from_tables,
        load_newdust_scattering_table,
    )
    from dsh.physics.wd01_ism import (
        build_wd01_ism,
        load_wd01_ism,
        wilm_budget_population,
        write_wd01_ism,
    )
    from reflex_adapter.reflex_materials import load_reflex_materials

    definition = None
    details = {}
    if key == "reflex_adapter":
        _, _, physics = load_reflex_materials()
    elif key == "mrn_rg":
        _, _, physics = load_2_10_material_tables()
        legacy = build_dust_physics_from_tables(
            load_newdust_scattering_table(), load_photoelectric_absorption_table()
        )
        # The 2–10 keV bank extends the same historical physical prescription.
        old = material_diagnostics(legacy, {"energy_kev": 4.9, "nh_cm2": 1e22})
        new = material_diagnostics(physics, {"energy_kev": 4.9, "nh_cm2": 1e22})
        for name in ("scattering_cm2_per_h", "absorption_cm2_per_h"):
            if not np.isclose(old[name], new[name], rtol=1e-7, atol=0):
                raise ValueError(
                    "historical and extended MRN materials differ at 4.9 keV"
                )
        details["legacy_v1_4p9_cross_sections_match_rtol"] = 1e-7
    elif key == "wd01_hybrid":
        definition, _, _, _, physics = load_wd01_ism(
            ROOT / "dsh/data/materials/wd01_wilm_c50_si90_v1"
        )
    elif key == "wd01_mie":
        directory = output / "full_mie_material"
        if not directory.exists():
            source = ROOT / "benchmarks/materials/wd01_2026-10-16/full_mie_reference"
            with np.load(source / "wd01_scattering_v1.npz", allow_pickle=False) as data:
                source_energy = data["energy_kev"]
            if 4.9 not in source_energy:
                raise ValueError("full-Mie source must contain exact 4.9 keV node")
            material = build_wd01_ism(
                source,
                wilm_budget_population(
                    carbon_gas_fraction=0.5, silicon_gas_fraction=0.1
                ),
                load_wilm_abundances(),
                molecular_hydrogen_fraction=0.2,
            )
            write_wd01_ism(
                directory,
                material,
                recipe={
                    "recipe": "Same Wilms C50/Si90 budget as hybrid; full Mie dust reference",
                    "source": str(source.relative_to(ROOT)),
                    "validated_comparison_energy_kev": 4.9,
                    "scope": "Exact source energy node; sparse dust grid is not broadband edge validation",
                },
            )
        definition, _, _, _, physics = load_wd01_ism(directory)
        details["comparison_scope"] = (
            "4.9 keV exact source node; full Mie checked dust reference"
        )
    else:
        raise ValueError(f"unknown material: {key}")
    if definition is not None:
        details.update(definition_sha256=definition.sha256, audit=definition.audit())
    return physics, details


def read_runs(directory, config):
    runs = []
    for seed in config["seeds"]:
        with np.load(directory / f"seed_{seed}.npz", allow_pickle=False) as data:
            run = dict(data)
        if (
            int(run["seed"]) != seed
            or int(run["packets"]) != config["packets_per_seed"]
        ):
            raise ValueError("saved seed metadata mismatch")
        runs.append(run)
    return runs


def verified_cache(key, config, actual_provenance):
    """Verify actual collection inputs/runtime, not just a previous pass flag."""
    directory = ROOT / CASES[key]["cached"]
    saved = json.loads((directory / "provenance.json").read_text())
    saved_config = saved.get("config")
    if saved_config is None:
        saved_config = json.loads((directory / "frozen_config.json").read_text())
    if (
        not geometry_equal(config, saved_config)
        or config["seeds"] != saved_config["seeds"]
    ):
        raise ValueError("cached geometry, sampling or seeds differ")
    hashes = saved.get("input_sha256", saved.get("sha256", {}))
    if not hashes:
        raise ValueError("cached input hashes missing")
    for name, expected in hashes.items():
        if hash_file(ROOT / name) != expected:
            raise ValueError(f"cached input differs: {name}")
    if "material_sha256" in saved:
        for name in (
            "cloud_sha256",
            "material_sha256",
            "launch_sha256",
            "runtime_sha256",
        ):
            if actual_provenance[name] != saved[name]:
                raise ValueError(f"cached {name} differs")
    else:
        fingerprint = hashlib.sha256(
            (
                json.dumps(saved_config, sort_keys=True)
                + json.dumps(hashes, sort_keys=True)
                + inspect.getsource(make_cloud)
                + inspect.getsource(run_seed)
            ).encode()
        ).hexdigest()
        if fingerprint != saved["collection_fingerprint"]:
            raise ValueError("cached collector differs")
    return read_runs(directory, config), {
        "kind": "verified existing independent histories",
        "source_directory": str(directory.relative_to(ROOT)),
        "source_provenance_sha256": hash_file(directory / "provenance.json"),
        "seed_sha256": {
            f"seed_{s}.npz": hash_file(directory / f"seed_{s}.npz")
            for s in config["seeds"]
        },
    }


def analyze(config, runs, reference):
    from scipy.stats import chi2

    moments = combined_moments(runs)
    nr = len(moments["radial"])
    radial_map = radial_pixel_map(config)
    if (
        reference.shape != moments["image"].shape[1:]
        or radial_map.min() < 0
        or radial_map.max() >= nr
    ):
        raise ValueError("reference/grid/annuli mismatch")
    observed = np.bincount(radial_map.ravel(), reference.ravel(), nr)
    scale = normalize_camera_counts(config) / config["source_fluence_ph_cm2"]
    predicted = moments["radial"] * scale
    model_cov = moments["radial_covariance"] * scale**2
    reference_cov = (
        np.diag(observed) - np.outer(observed, observed) / config["reflex_photons"]
    )
    total_cov = model_cov + reference_cov
    difference = predicted - observed
    statistic = float(difference @ np.linalg.solve(total_cov, difference))
    ratio, ratio_se = independent_ratio(
        predicted, observed, np.diag(model_cov), np.diag(reference_cov)
    )
    fluxes = np.array([r["radial"].sum() for r in runs])
    variances = np.array(
        [covariance(r["radial"], r["radial_q"], int(r["packets"])).sum() for r in runs]
    )
    mean = np.average(fluxes, weights=1 / variances)
    seed_p = float(chi2.sf(np.sum((fluxes - mean) ** 2 / variances), len(runs) - 1))
    statuses = sum(r["statuses"] for r in runs)
    rel_se = float(np.sqrt(model_cov.sum()) / predicted.sum())
    gates = {
        "transport_complete": int(statuses[[0, 4, 5, 6]].sum()) == 0,
        "sampling_sufficient": bool(
            rel_se <= config["maximum_aperture_relative_mc_se"]
            and len(runs) >= 3
            and observed.min() >= config["minimum_reflex_counts_per_annulus"]
        ),
        "seed_consistency": seed_p >= config["minimum_global_pvalue"],
        "order_closure": bool(
            np.isclose(moments["image"].sum(), moments["radial"].sum(), rtol=1e-11)
        ),
    }
    total_ratio, total_ratio_se = independent_ratio(
        predicted.sum(), observed.sum(), model_cov.sum(), reference_cov.sum()
    )
    summary = {
        "histories": moments["histories"],
        "halo_counts": float(predicted.sum()),
        "reference_halo_counts": float(observed.sum()),
        "halo_ratio": float(total_ratio),
        "halo_ratio_combined_se": float(total_ratio_se),
        "halo_difference_z": float(difference.sum() / np.sqrt(total_cov.sum())),
        "radial_chi2": statistic,
        "radial_dof": nr,
        "radial_pvalue": float(chi2.sf(statistic, nr)),
        "maximum_annulus_absolute_z": float(
            np.max(abs(difference) / np.sqrt(np.diag(total_cov)))
        ),
        "aperture_relative_mc_se": rel_se,
        "seed_consistency_pvalue": seed_p,
        "multiple_to_first_ratio": float(
            moments["order_sum"][1:].sum() / moments["order_sum"][0]
        ),
        "numerical_quality_passed": all(gates.values()),
        "numerical_quality_gates": gates,
        "transport_status_counts": statuses.tolist(),
        "annulus_halo_counts": predicted.tolist(),
        "annulus_ratio": ratio.tolist(),
        "annulus_ratio_se": ratio_se.tolist(),
        "fitted_parameters": 0,
    }
    arrays = {
        "image_by_order": moments["image"],
        "image_mc_variance": moments["image_variance"],
        "predicted_counts": moments["image"].sum(axis=0) * scale,
        "radial_counts": predicted,
        "radial_model_covariance": model_cov,
        "radial_reference_covariance": reference_cov,
        "radial_total_covariance": total_cov,
        "reference_radial_counts": observed,
        "radial_ratio": ratio,
        "radial_ratio_se": ratio_se,
    }
    return summary, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="collect all four models without using existing benchmark seeds",
    )
    args = parser.parse_args()
    import astropy
    import jax
    import matplotlib
    import scipy
    from astropy.io import fits

    from dsh.observer.response_transport import response_provenance
    from dsh.sources.radial_launch import build_radial_cloud_launch_geometry
    from dsh.validation.absorbed_observer import host_phase
    from scripts.plot_material_models import plot_comparison

    matplotlib.use("Agg")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "reflex_adapter/comparison_config.json").read_text())
    config.pop("seeds")
    reference = (
        fits.getdata(ROOT / config["reflex_dust_image"]).astype(float)[::-1].copy()
    )
    if np.any(reference < 0) or not np.all(reference == np.rint(reference)):
        raise ValueError("RefleX reference must contain raw unweighted counts")
    angles = np.geomspace(0.5, 2000, 1024)
    phase_arrays = {"physical_angle_arcsec": angles}
    all_summary = {}
    provenance = {}
    all_arrays = {
        "reference_counts": reference,
        "sky_edges_arcsec": image_edges(config),
        "annulus_edges_arcsec": np.asarray(config["annulus_edges_arcsec"]),
    }
    for key, case in CASES.items():
        print(f"\nMODEL {key}: {case['label']}", flush=True)
        case_config = dict(config, seeds=case["seeds"])
        physics, details = load_case_material(key, args.output)
        cloud = make_cloud(case_config)
        launch = build_radial_cloud_launch_geometry(cloud, physics)
        current = response_provenance(cloud, physics, launch)
        current.update(
            config=case_config,
            collector_sha256=hash_file(ROOT / "reflex_adapter/compare_reflex.py"),
            runner_sha256=hash_file(Path(__file__)),
            reference_sha256=hash_file(ROOT / config["reflex_dust_image"]),
            uses_external_cache=bool("cached" in case and not args.fresh),
        )
        fingerprint = hashlib.sha256(
            json.dumps(current, sort_keys=True).encode()
        ).hexdigest()
        directory = args.output / key
        directory.mkdir(exist_ok=True)
        saved_path = directory / "provenance.json"
        if saved_path.exists():
            if not args.reuse:
                raise FileExistsError("saved model exists; use --reuse or a new output")
            saved = json.loads(saved_path.read_text())
            if saved["collection_fingerprint"] != fingerprint:
                raise ValueError(f"saved {key} material/runtime/config differs")
        if "cached" in case and not args.fresh:
            runs, origin = verified_cache(key, case_config, current)
        else:
            runs = []
            # Publish initial provenance to allow a clearly identified partial
            # run; completed seeds are atomic and metadata checked on resume.
            origin = {
                "kind": "fresh independent histories",
                "source_directory": str(directory),
            }
            initial = dict(current, collection_fingerprint=fingerprint, origin=origin)
            saved_path.write_text(json.dumps(initial, indent=2) + "\n")
            for seed in case["seeds"]:
                path = directory / f"seed_{seed}.npz"
                if args.reuse and path.exists():
                    with np.load(path, allow_pickle=False) as data:
                        run = dict(data)
                    if (
                        int(run["seed"]) != seed
                        or int(run["packets"]) != case_config["packets_per_seed"]
                    ):
                        raise ValueError("saved seed metadata mismatch")
                else:
                    with tempfile.TemporaryDirectory(dir=directory) as temporary:
                        run = run_seed(
                            case_config, physics, cloud, seed, Path(temporary)
                        )
                        atomic_savez_compressed(path, **run)
                runs.append(run)
        current.update(collection_fingerprint=fingerprint, origin=origin)
        saved_path.write_text(json.dumps(current, indent=2) + "\n")
        result, arrays = analyze(case_config, runs, reference)
        result.update(
            model=case,
            diagnostics=material_diagnostics(physics, case_config),
            material=details,
        )
        atomic_savez_compressed(directory / "comparison_arrays.npz", **arrays)
        (directory / "comparison_results.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        for name, value in arrays.items():
            all_arrays[f"{key}__{name}"] = value
        phase_arrays[key] = host_phase(physics, config["energy_kev"], angles * ARCSEC)
        all_summary[key] = result
        provenance[key] = current
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "halo_counts",
                        "halo_ratio",
                        "radial_pvalue",
                        "numerical_quality_passed",
                    )
                }
            ),
            flush=True,
        )
    for first in CASES:
        for second in CASES:
            if first != second and set(CASES[first]["seeds"]) & set(
                CASES[second]["seeds"]
            ):
                raise ValueError("model seeds must be independent")
    hybrid = all_arrays["wd01_hybrid__radial_counts"]
    mie = all_arrays["wd01_mie__radial_counts"]
    pair_cov = (
        all_arrays["wd01_hybrid__radial_model_covariance"]
        + all_arrays["wd01_mie__radial_model_covariance"]
    )
    from scipy.stats import chi2

    pair_chi2 = float((mie - hybrid) @ np.linalg.solve(pair_cov, mie - hybrid))
    hybrid_phase, mie_phase = phase_arrays["wd01_hybrid"], phase_arrays["wd01_mie"]
    relevant = mie_phase > mie_phase.max() * 1e-5
    summary = {
        "config": config,
        "models": all_summary,
        "reference": {
            "label": "RefleX analog simulation",
            "halo_counts": float(reference.sum()),
            "launched_photons": config["reflex_photons"],
        },
        "numerical_quality_passed": all(
            r["numerical_quality_passed"] for r in all_summary.values()
        ),
        "independent_model_histories": sum(
            r["histories"] for r in all_summary.values()
        ),
        "full_mie_vs_hybrid": {
            "radial_chi2": pair_chi2,
            "radial_dof": len(mie),
            "radial_pvalue": float(chi2.sf(pair_chi2, len(mie))),
            "halo_ratio": float(mie.sum() / hybrid.sum()),
            "maximum_phase_relative_difference_where_phase_above_1e_minus5_peak": float(
                np.max(abs(hybrid_phase[relevant] / mie_phase[relevant] - 1))
            ),
        },
        "scope": "4.9 keV fixed geometry and absolute normalization; different materials are not expected to match exactly. Ratio curves share the same RefleX denominator and are correlated. Model-pair statistics use only independent model MC covariance.",
        "excluded": {
            "nominal_wd01_plus_wilms": "Dust-only nominal WD01 bank overdraws Wilms abundances; a complete consistent alternative abundance recipe has not been selected",
            "zda04": "not yet implemented",
        },
        "versions": {
            "jax": jax.__version__,
            "numpy": np.__version__,
            "astropy": astropy.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    (args.output / "comparison_results.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    atomic_savez_compressed(args.output / "comparison_arrays.npz", **all_arrays)
    atomic_savez_compressed(args.output / "phase_comparison.npz", **phase_arrays)
    plot_comparison(args.output)
    print(
        json.dumps(
            {
                "numerical_quality_passed": summary["numerical_quality_passed"],
                "full_mie_vs_hybrid": summary["full_mie_vs_hybrid"],
            },
            indent=2,
        )
    )
    if not summary["numerical_quality_passed"]:
        raise SystemExit("A numerical quality gate failed; inspect saved model results")


if __name__ == "__main__":
    main()
