"""Fixed-budget paired Compton halo validation against the supplied RefleX pair.

Use --pilot for a numerical pilot and packet planning. The scientific run has
three fresh fixed seed budgets; prefix convergence points reuse histories.
No changes are made to the accepted transport/scoring or frozen benchmarks.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from dsh.io.atomic import atomic_savez_compressed
from dsh.validation.compton_halo import (
    CHANNELS,
    HaloCollector,
    add_moments,
    file_sha,
    load_checkpoint,
    mean_covariance,
    merge_seed_estimates,
    ratio_and_error,
    save_checkpoint,
    verify_reference,
)


def analyze(config, states, images, sigma_compton, *, bootstrap=10000):
    from scipy.stats import chi2

    from reflex_adapter.compare_reflex import normalize_camera_counts, radial_pixel_map

    mean, covariance = merge_seed_estimates(states)
    nr = len(config["annulus_edges_arcsec"]) - 1
    profiles = mean.reshape(len(CHANNELS), nr)
    scale = normalize_camera_counts(config) / config["source_fluence_ph_cm2"]
    radial_map = radial_pixel_map(config)
    condition_results, arrays = (
        {},
        {"joint_radial_mean": profiles, "joint_radial_mean_covariance": covariance},
    )
    n = int(states[0]["histories"])
    blocks = {name: slice(i * nr, (i + 1) * nr) for i, name in enumerate(CHANNELS)}
    for condition, channel in (("off", "off_dust"), ("on", "on_dust_selected")):
        block = blocks[channel]
        predicted = mean[block] * scale
        mc = covariance[block, block] * scale**2
        observed = np.bincount(
            radial_map.ravel(), images[condition + "_dust"].ravel(), minlength=nr
        )
        reference_cov = (
            np.diag(predicted)
            - np.outer(predicted, predicted) / config["reflex_photons"]
        )
        combined = mc + reference_cov
        residual = predicted - observed
        statistic = float(residual @ np.linalg.solve(combined, residual))
        flux = np.array([state["sum"][block].sum() / n for state in states])
        variances = np.array(
            [
                mean_covariance(state["sum"], state["cross"], n)[block, block].sum()
                for state in states
            ]
        )
        average = np.average(flux, weights=1 / variances)
        consistency = float(
            chi2.sf(np.sum((flux - average) ** 2 / variances), len(states) - 1)
        )
        # Predictive calibration: exact fixed-N camera multinomial plus the
        # independent finite-MC normal approximation; no fitted amplitude.
        eigenvalues, vectors = np.linalg.eigh((mc + mc.T) / 2)
        root = vectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
        rng = np.random.default_rng(112699 if condition == "off" else 112700)
        probability = np.r_[
            predicted / config["reflex_photons"],
            1 - predicted.sum() / config["reflex_photons"],
        ]
        counts = rng.multinomial(config["reflex_photons"], probability, size=bootstrap)[
            :, :nr
        ]
        mc_error = rng.standard_normal((bootstrap, nr)) @ root.T
        differences = counts - predicted - mc_error
        null = np.einsum(
            "ij,ji->i", differences, np.linalg.solve(combined, differences.T)
        )
        p = (1 + int(np.sum(null >= statistic))) / (bootstrap + 1)
        relse = float(np.sqrt(max(mc.sum(), 0)) / predicted.sum())
        annulus_precision = np.sqrt(np.maximum(np.diag(mc), 0)) / np.sqrt(
            np.maximum(observed, 1)
        )
        result = {
            "predicted_camera_counts": float(predicted.sum()),
            "reference_camera_counts": int(observed.sum()),
            "model_to_reference": float(predicted.sum() / observed.sum()),
            "model_relative_mc_se": relse,
            "aperture_residual_z": float(residual.sum() / np.sqrt(combined.sum())),
            "radial_chi2": statistic,
            "radial_dof": nr,
            "radial_asymptotic_pvalue": float(chi2.sf(statistic, nr)),
            "radial_predictive_pvalue": p,
            "predictive_replicates": bootstrap,
            "maximum_absolute_annulus_z": float(
                np.max(abs(residual) / np.sqrt(np.diag(combined)))
            ),
            "maximum_annulus_mc_to_reference_se": float(annulus_precision.max()),
            "independent_seed_consistency_pvalue": consistency,
            "fitted_parameters": 0,
        }
        condition_results[condition] = result
        arrays.update(
            {
                condition + "__radial_counts": predicted,
                condition + "__radial_mc_covariance": mc,
                condition + "__radial_reference_covariance": reference_cov,
                condition + "__reference_radial_counts": observed,
            }
        )
    off, on = blocks["off_dust"], blocks["on_dust_selected"]
    ratio, se = ratio_and_error(
        mean[on].sum(),
        mean[off].sum(),
        covariance[on, on].sum(),
        covariance[off, off].sum(),
        covariance[on, off].sum(),
    )
    dust_only = blocks["on_dust_only"]
    survival, survival_se = ratio_and_error(
        mean[dust_only].sum(),
        mean[off].sum(),
        covariance[dust_only, dust_only].sum(),
        covariance[off, off].sum(),
        covariance[dust_only, off].sum(),
    )
    per_annulus = [
        ratio_and_error(
            mean[on][i],
            mean[off][i],
            covariance[on, on][i, i],
            covariance[off, off][i, i],
            covariance[on, off][i, i],
        )
        for i in range(nr)
    ]
    arrays["on_off_radial_ratio"] = np.array([v[0] for v in per_annulus])
    arrays["on_off_radial_ratio_mc_se"] = np.array([v[1] for v in per_annulus])
    expected_survival = float(np.exp(-config["nh_cm2"] * sigma_compton))
    combined_state = states[0]
    for state in states[1:]:
        combined_state = add_moments(combined_state, state)
    statuses = combined_state["statuses"]
    numerical = bool(
        np.all(statuses[:, [0, 4, 5, 6, 7, 8]] == 0)
        and np.all(statuses.sum(axis=1) == n * len(states))
        and int(combined_state["unsupported_observer_count"]) == 0
        and float(combined_state["max_energy_error_kev"]) < 2e-5
        and int(combined_state["analog_compton_count"][0]) == 0
    )
    closure = bool(
        np.allclose(profiles[4], profiles[1] + profiles[3], rtol=1e-12, atol=1e-25)
    )
    sampling = bool(
        len(states) >= 3
        and all(
            value["model_relative_mc_se"] <= config["maximum_aperture_relative_mc_se"]
            and value["maximum_annulus_mc_to_reference_se"]
            <= config["maximum_annulus_mc_to_reference_se"]
            for value in condition_results.values()
        )
        and se <= config["maximum_delta_ratio_mc_se"]
    )
    seed_ok = all(
        value["independent_seed_consistency_pvalue"] >= config["minimum_global_pvalue"]
        for value in condition_results.values()
    )
    agreement = all(
        value["radial_predictive_pvalue"] >= config["minimum_global_pvalue"]
        and abs(value["aperture_residual_z"]) <= config["aperture_sigma_limit"]
        and value["maximum_absolute_annulus_z"] <= config["annulus_sigma_limit"]
        for value in condition_results.values()
    )
    survival_ok = abs(survival - expected_survival) <= 5 * survival_se + 1e-5
    components = {
        name: {
            "expected_camera_counts": float(profiles[i].sum() * scale),
            "mc_standard_error_counts": float(
                np.sqrt(max(covariance[blocks[name], blocks[name]].sum(), 0)) * scale
            ),
        }
        for i, name in enumerate(CHANNELS)
    }
    report = {
        "numerical_complete": numerical,
        "component_closure": closure,
        "sampling_sufficient": sampling,
        "seed_consistency": bool(seed_ok),
        "no_compton_survival_control": bool(survival_ok),
        "within_declared_reflex_thresholds": bool(agreement),
        "passes_declared_gates": bool(
            numerical and closure and sampling and seed_ok and survival_ok and agreement
        ),
        "histories_per_seed": n,
        "seed_count": len(states),
        "paired_histories": n * len(states),
        "conditions": condition_results,
        "on_off_dust_selected_ratio": ratio,
        "on_off_dust_selected_ratio_mc_se": se,
        "dust_only_survival_ratio": survival,
        "dust_only_survival_ratio_mc_se": survival_se,
        "near_forward_no_compton_survival_ratio": expected_survival,
        "physical_compton_cross_section_cm2_per_h": sigma_compton,
        "components": components,
        "transport_status_counts": statuses.tolist(),
        "analog_compton_counts": combined_state["analog_compton_count"].tolist(),
        "maximum_energy_closure_error_kev": float(
            combined_state["max_energy_error_kev"]
        ),
        "unsupported_observer_events": int(
            combined_state["unsupported_observer_count"]
        ),
        "reflex_on_off_ratio_observed": float(
            images["on_dust"].sum() / images["off_dust"].sum()
        ),
        "reflex_on_off_covariance_available": False,
        "reflex_non_dust_outside_core_95_upper_expected_counts": float(
            config["reflex_photons"]
            * (-np.expm1(np.log(0.05) / config["reflex_photons"]))
        ),
        "normalization_counts_per_ph_cm2": float(scale),
    }
    for name in (
        "image",
        "image_square",
        "energy_sum",
        "energy_square",
        "time_sum",
        "time_square",
    ):
        total = combined_state[name]
        count = n * len(states)
        if name.endswith("_square"):
            source_name = (
                name.replace("_square", "_sum") if name != "image_square" else "image"
            )
            sums = combined_state[source_name]
            arrays[name.replace("_square", "_mc_variance")] = (
                (total - sums * sums / count) / (count * (count - 1)) * scale**2
            )
        else:
            arrays[name] = total / count * scale
    return report, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/compton/halo_validation.json")
    )
    parser.add_argument(
        "--output-directory", type=Path, default=Path("outputs/compton-halo-validation")
    )
    parser.add_argument("--packets-per-seed", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--seed-only", type=int)
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float32"
    )
    args = parser.parse_args(argv)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "1" if args.precision == "float64" else "0"
    import jax

    jax.config.update("jax_enable_x64", args.precision == "float64")
    from jax import random

    from dsh.observer.response import array_fingerprint
    from dsh.observer.response_transport import response_provenance
    from dsh.transport.inelastic import interaction_opacities

    repo = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.packets_per_seed is not None:
        config["packets_per_seed"] = args.packets_per_seed
    elif args.pilot:
        config["packets_per_seed"] = 65536
    if args.pilot:
        config["seeds"] = config.get("pilot_seeds", config["seeds"])
    expected = {
        "source_distance_kpc": 10.0,
        "cloud_center_kpc": 5.0,
        "cloud_thickness_pc": 30.0,
        "cloud_radius_pc": 30.0,
        "cloud_sky_cells": 16,
        "nh_cm2": 1e22,
        "energy_kev": 4.9,
        "image_half_width_arcsec": 150.0,
        "pixel_arcsec": 5.0,
        "camera_aperture_radius_pc": 0.1,
        "reflex_photons": 100000000,
        "source_cone_half_angle_deg": 1.0,
        "camera_grid_side": 100,
        "camera_spacing_pc": 0.2,
        "source_fluence_ph_cm2": 1.0,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("configuration does not match the fixed RefleX reference")
    if len(config["seeds"]) < 3 or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("need at least three distinct seeds")
    budget, chunk = config["packets_per_seed"], config["chunk_size"]
    if budget < 2 or chunk < 2 or budget % chunk:
        raise ValueError("packet budget must be a positive multiple of chunk size")
    root = args.output_directory.resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError("output exists; choose a fresh directory or --resume")
    root.mkdir(parents=True, exist_ok=True)
    manifest, images = verify_reference(repo, repo / config["reference_manifest"])
    collector = HaloCollector(config, repo)
    _, sd, sc, sa = interaction_opacities(
        collector.physics, collector.compton, np.asarray(config["energy_kev"])
    )
    sigma_compton = float(sc)
    provenance = response_provenance(
        collector.cloud, collector.physics, collector.launch, collector.compton
    )
    provenance.update(
        collector_sha256=file_sha(repo / "dsh/validation/compton_halo.py"),
        reference_sha256=file_sha(repo / config["reference_manifest"]),
        grain_electrons=config["grain_electrons"],
        energy_cutoff_kev=config["energy_cutoff_kev"],
        precision=args.precision,
        config={
            key: value for key, value in config.items() if key != "packets_per_seed"
        },
    )
    # Arrays are per-history and independent of the final budget, so the same
    # exact sequence can be extended on resume. All collection inputs are pinned.
    fingerprint = array_fingerprint(
        {"provenance": np.asarray(json.dumps(provenance, sort_keys=True))}
    )
    metadata_path = root / "provenance.json"
    if (
        metadata_path.exists()
        and json.loads(metadata_path.read_text())["collection_fingerprint"]
        != fingerprint
    ):
        raise ValueError("cached collection inputs/runtime/config differ")
    if not metadata_path.exists():
        metadata_path.write_text(
            json.dumps(
                {
                    "collection_fingerprint": fingerprint,
                    "collection": provenance,
                    "material": collector.material,
                    "reference": manifest,
                    "python": platform.python_version(),
                    "jax_enable_x64": bool(jax.config.x64_enabled),
                    "requested_packets_per_seed": budget,
                },
                indent=2,
            )
            + "\n"
        )
    if args.prepare_only:
        print("Collection prepared: " + str(root), flush=True)
        return 0
    if args.seed_only is not None and args.seed_only not in config["seeds"]:
        raise ValueError("requested seed is not in the configured collection")
    checkpoints = sorted(
        set(
            [
                value
                for value in (
                    65536,
                    131072,
                    262144,
                    524288,
                    1048576,
                    2097152,
                    4194304,
                    8388608,
                )
                if value <= budget
            ]
            + [budget]
        )
    )
    started = time.monotonic()
    states = []
    for seed in config["seeds"] if args.seed_only is None else [args.seed_only]:
        seed_started = time.monotonic()
        seed_dir = root / ("seed_" + str(seed))
        seed_dir.mkdir(exist_ok=True)
        path = seed_dir / "latest.npz"
        state = collector.empty()
        if path.exists():
            state = load_checkpoint(path, seed, fingerprint)
        completed = int(state["histories"])
        initial_completed = completed
        if completed > budget:
            raise ValueError("cached history count exceeds requested budget")
        for start in range(completed, budget, chunk):
            index = start // chunk
            key = random.fold_in(random.PRNGKey(seed), index)
            current = collector.collect(
                key,
                chunk,
                int(np.random.SeedSequence([seed, index, 712998]).generate_state(1)[0]),
            )
            state = add_moments(state, current)
            completed = int(state["histories"])
            if completed in checkpoints:
                save_checkpoint(
                    seed_dir / f"n_{completed}.npz", state, seed, fingerprint
                )
            if completed % 16384 == 0 or completed == budget:
                save_checkpoint(path, state, seed, fingerprint)
                print(
                    f"seed={seed}: {completed:,}/{budget:,} paired histories; elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
        if completed > initial_completed:
            timing_path = seed_dir / "collection_timing.json"
            timing = json.loads(timing_path.read_text()) if timing_path.exists() else []
            timing.append(
                {
                    "start_histories": initial_completed,
                    "stop_histories": completed,
                    "elapsed_seconds": time.monotonic() - seed_started,
                }
            )
            timing_path.write_text(json.dumps(timing, indent=2) + "\n")
        states.append(state)
    if args.seed_only is not None:
        state = states[0]
        return (
            0
            if np.all(state["statuses"][:, [0, 4, 5, 6, 7, 8]] == 0)
            and int(state["unsupported_observer_count"]) == 0
            and float(state["max_energy_error_kev"]) < 2e-5
            else 3
        )
    result, arrays = analyze(config, states, images, sigma_compton)
    result["external_equivalence_claimed"] = False
    result["radial_tension_at_5_percent"] = {
        condition: value["radial_predictive_pvalue"] < 0.05
        for condition, value in result["conditions"].items()
    }
    result["elapsed_seconds_scope"] = (
        "Current invocation wall time; excludes histories collected before a resume."
    )
    result["collection_timings"] = {
        str(seed): json.loads(
            (root / f"seed_{seed}/collection_timing.json").read_text()
        )
        for seed in config["seeds"]
        if (root / f"seed_{seed}/collection_timing.json").exists()
    }
    convergence = []
    for level in checkpoints:
        saved = []
        for seed in config["seeds"]:
            saved.append(
                load_checkpoint(
                    root / f"seed_{seed}/n_{level}.npz", seed, fingerprint, level
                )
            )
        stage, _ = analyze(config, saved, images, sigma_compton, bootstrap=2000)
        convergence.append(
            {
                key: stage[key]
                for key in (
                    "histories_per_seed",
                    "on_off_dust_selected_ratio",
                    "on_off_dust_selected_ratio_mc_se",
                    "sampling_sufficient",
                )
            }
            | {
                "off_relative_mc_se": stage["conditions"]["off"][
                    "model_relative_mc_se"
                ],
                "on_relative_mc_se": stage["conditions"]["on"]["model_relative_mc_se"],
            }
        )
    result.update(
        config=config,
        precision=args.precision,
        pilot=args.pilot,
        elapsed_seconds=time.monotonic() - started,
        convergence=convergence,
        source_config_sha256=file_sha(config_path),
        material=collector.material,
        reference_material_caveat="WD01 C50/Si90 plus chosen cold prescription; RefleX uses a different inventory, effective opacity scale and angular law. No exact material-equivalence claim.",
        opacity={
            "dust_cm2_per_h": float(sd),
            "absorption_cm2_per_h": float(sa),
            "compton_cm2_per_h": sigma_compton,
            "reflex_compton_cm2_per_h": manifest["opacity_compton_4p9_cm2_per_h"],
        },
        off_control="Positive 1e-35 cm2/H negligible-opacity control on the inelastic six-stream kernel (nominal tau=1e-13); no realized Compton events required. Same elastic distribution to this declared error, not the legacy random stream.",
        uncertainty="Photon-grouped joint on/off covariance including zero scorers. Three independent seeds. RefleX on/off covariance unavailable. Predictive calibration is condition-wise, not a joint test of the RefleX pair.",
        convergence_note="Prefix points reuse histories and are correlated; production budget is fixed, not selected by a significance stopping rule.",
    )
    camera_scale = result["normalization_counts_per_ph_cm2"]
    direct = {}
    for condition, sigma in (("off", float(sd + sa)), ("on", float(sd + sa + sc))):
        proxy = float(
            (images[condition + "_all"] - images[condition + "_dust"])[
                29:31, 29:31
            ].sum()
        )
        expected = float(camera_scale * np.exp(-config["nh_cm2"] * sigma))
        direct[condition] = {
            "reference_direct_dominated_core_proxy_counts": proxy,
            "nominal_straight_path_prediction": expected,
            "nominal_residual_over_reference_marginal_se": float(
                (expected - proxy)
                / np.sqrt(proxy * (1 - proxy / config["reflex_photons"]))
            ),
        }
    result["direct_beam_diagnostic"] = {
        "conditions": direct,
        "predicted_on_off_ratio": float(np.exp(-config["nh_cm2"] * sigma_compton)),
        "observed_core_proxy_on_off_ratio": direct["on"][
            "reference_direct_dominated_core_proxy_counts"
        ]
        / direct["off"]["reference_direct_dominated_core_proxy_counts"],
        "scope": "Nominal straight column and common camera normalization. The on core proxy can include unresolved Compton-only photons; no RefleX paired-difference significance is inferred.",
    }
    arrays.update(dict(images.items()))
    arrays.update(
        energy_edges_kev=collector.energy_edges,
        time_edges_s=collector.time_edges,
        annulus_edges_arcsec=np.asarray(config["annulus_edges_arcsec"]),
        sky_edges_arcsec=collector.edges,
    )
    atomic_savez_compressed(root / "comparison_arrays.npz", **arrays)
    (root / "report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    from scripts.plot_compton_halo import write_report

    write_report(root, result, arrays)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "numerical_complete",
                    "sampling_sufficient",
                    "within_declared_reflex_thresholds",
                    "passes_declared_gates",
                    "on_off_dust_selected_ratio",
                    "on_off_dust_selected_ratio_mc_se",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    print("Report: " + str(root / "REPORT.md"), flush=True)
    return (
        0
        if (
            result["numerical_complete"]
            and result["component_closure"]
            and (args.pilot or result["passes_declared_gates"])
        )
        else 3
    )


if __name__ == "__main__":
    raise SystemExit(main())
