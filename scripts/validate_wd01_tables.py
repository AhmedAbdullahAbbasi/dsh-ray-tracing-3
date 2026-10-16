"""Compare WD01 production tables with a finer, independently integrated Mie run.

This is a numerical producer check, not an external RefleX halo comparison.
The optional transport check deliberately uses grain absorption alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dsh.physics.newdust import build_dust_physics_from_tables
from dsh.physics.wd01_tables import load_wd01_dust_tables

REPOSITORY = Path(__file__).resolve().parent.parent
ARCSEC = np.pi / 648000

# Chosen before examining the generated reference comparison. These gates
# measure numerical agreement, not uncertainty in the physical grain model.
LIMITS = {
    "total_cross_section_relative": 1e-3,
    "cdf_absolute": 5e-4,
    "angular_bin_relative": 0.01,
    "minimum_reference_bin_probability": 1e-4,
    "raw_angular_closure_relative": 3e-3,
}
ANGULAR_EDGES_ARCSEC = [0, 15, 30, 60, 120, 240, 480, 960]


def check_reference_totals(reference_directory):
    """Refine size integration again without recomputing a costly phase grid."""
    from dsh.physics.grain_populations import WD01Population
    from dsh.physics.optical_constants import load_draine_optical_constants
    from dsh.physics.sphere_scattering import mie_sphere

    reference, absorption = load_wd01_dust_tables(reference_directory)
    info = reference.metadata["population"]
    population = WD01Population(
        **{
            key: info[key]
            for key in (
                "normalization",
                "carbon_scale",
                "silicate_scale",
                "maximum_radius_um",
            )
        }
    )
    order = 2 * reference.metadata["quadrature_order_per_log_radius_interval"]
    scattering_totals, absorption_totals = [], []
    for energy in reference.energy_kev:
        sca, absorb = 0.0, 0.0
        for component, orientations in (
            (
                "carbonaceous",
                (("graphite_parallel", 1 / 3), ("graphite_perpendicular", 2 / 3)),
            ),
            ("silicate", (("silicate", 1.0),)),
        ):
            radius, count = population.quadrature(component, order)
            for name, orientation_weight in orientations:
                index = load_draine_optical_constants(name).index_at_energy(energy)
                for a, number in zip(radius, count, strict=True):
                    result = mie_sphere(a, energy, index, [0.0])
                    sca += number * orientation_weight * result.scattering_cm2
                    absorb += number * orientation_weight * result.absorption_cm2
        scattering_totals.append(sca)
        absorption_totals.append(absorb)
    sca_error = reference.scattering_cross_section_cm2_per_h / scattering_totals - 1
    abs_error = absorption.absorption_cross_section_cm2_per_h / absorption_totals - 1
    limit = 1e-4
    return {
        "all_passed": bool(
            np.max(abs(sca_error)) <= limit and np.max(abs(abs_error)) <= limit
        ),
        "scope": "Independent higher-order size integration of exact Mie totals; phase grid is not recomputed",
        "refined_quadrature_order": order,
        "relative_error_limit": limit,
        "energy_kev": reference.energy_kev.tolist(),
        "reference_scattering_relative_error": sca_error.tolist(),
        "reference_grain_absorption_relative_error": abs_error.tolist(),
    }


def _table_fingerprint(directory):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(directory).glob("*.npz"))
    }


def compare_tables(table_directory, reference_directory):
    scattering, absorption = load_wd01_dust_tables(table_directory)
    reference, reference_abs = load_wd01_dust_tables(reference_directory)
    if reference.metadata["scattering_method"] != "mie":
        raise ValueError("reference must use full Mie")
    for key in ("population_sha256", "optical_source_sha256"):
        if scattering.metadata[key] != reference.metadata[key]:
            raise ValueError(f"reference has different {key}")
    if (
        reference.metadata["quadrature_order_per_log_radius_interval"]
        <= scattering.metadata["quadrature_order_per_log_radius_interval"]
        or reference.scattering_angle_rad.size <= scattering.scattering_angle_rad.size
    ):
        raise ValueError("reference must refine both size and angle quadrature")
    edges = np.asarray(ANGULAR_EDGES_ARCSEC) * ARCSEC
    rows = []
    for i, energy in enumerate(reference.energy_kev):
        indices = np.flatnonzero(scattering.energy_kev == energy)
        if indices.size != 1:
            raise ValueError(f"production table lacks exact reference node {energy}")
        j = int(indices[0])
        # Compare on the union: both runtime CDFs are piecewise linear in angle.
        angles = np.union1d(
            scattering.scattering_angle_rad, reference.scattering_angle_rad
        )
        prod_cdf = np.interp(
            angles, scattering.scattering_angle_rad, scattering.scattering_angle_cdf[j]
        )
        ref_cdf = np.interp(
            angles, reference.scattering_angle_rad, reference.scattering_angle_cdf[i]
        )
        prod_bins = np.diff(
            np.interp(
                edges,
                scattering.scattering_angle_rad,
                scattering.scattering_angle_cdf[j],
            )
        )
        ref_bins = np.diff(
            np.interp(
                edges, reference.scattering_angle_rad, reference.scattering_angle_cdf[i]
            )
        )
        significant = ref_bins >= LIMITS["minimum_reference_bin_probability"]
        row = {
            "energy_kev": float(energy),
            "scattering_relative_error": float(
                scattering.scattering_cross_section_cm2_per_h[j]
                / reference.scattering_cross_section_cm2_per_h[i]
                - 1
            ),
            "grain_absorption_relative_error": float(
                absorption.absorption_cross_section_cm2_per_h[j]
                / reference_abs.absorption_cross_section_cm2_per_h[i]
                - 1
            ),
            "maximum_cdf_absolute_error": float(np.max(abs(prod_cdf - ref_cdf))),
            "maximum_significant_bin_relative_error": float(
                np.max(abs(prod_bins[significant] / ref_bins[significant] - 1))
            ),
            "production_bin_probabilities": prod_bins.tolist(),
            "reference_bin_probabilities": ref_bins.tolist(),
        }
        row["passed"] = bool(
            abs(row["scattering_relative_error"])
            <= LIMITS["total_cross_section_relative"]
            and abs(row["grain_absorption_relative_error"])
            <= LIMITS["total_cross_section_relative"]
            and row["maximum_cdf_absolute_error"] <= LIMITS["cdf_absolute"]
            and row["maximum_significant_bin_relative_error"]
            <= LIMITS["angular_bin_relative"]
        )
        rows.append(row)
    closure = float(scattering.metadata["maximum_raw_angular_closure_relative_error"])
    return {
        "all_passed": all(row["passed"] for row in rows)
        and closure <= LIMITS["raw_angular_closure_relative"],
        "scope": "Numerical size/angle integration and forward ADT check against finer full Mie; no independent external halo validation",
        "limits": LIMITS,
        "angular_bin_edges_arcsec": ANGULAR_EDGES_ARCSEC,
        "angle_convention": "intrinsic deflection, not observer halo radius",
        "production_shape": [
            scattering.energy_kev.size,
            scattering.scattering_angle_rad.size,
        ],
        "reference_shape": [
            reference.energy_kev.size,
            reference.scattering_angle_rad.size,
        ],
        "production_quadrature_order": scattering.metadata[
            "quadrature_order_per_log_radius_interval"
        ],
        "reference_quadrature_order": reference.metadata[
            "quadrature_order_per_log_radius_interval"
        ],
        "maximum_raw_angular_closure_relative_error": closure,
        "production_npz_sha256": _table_fingerprint(table_directory),
        "reference_npz_sha256": _table_fingerprint(reference_directory),
        "energies": rows,
    }


def transport_smoke(table_directory, output):
    from reflex_adapter.compare_reflex import make_cloud, run_seed

    scattering, absorption = load_wd01_dust_tables(table_directory)
    physics = build_dust_physics_from_tables(scattering, absorption)
    config = json.loads(
        (REPOSITORY / "reflex_adapter/comparison_config.json").read_text()
    )
    config.update(packets_per_seed=8192, chunk_size=8192, seeds=[29017])
    run = run_seed(config, physics, make_cloud(config), 29017, output)
    gates = {
        "all_histories_terminated": int(run["statuses"].sum())
        == config["packets_per_seed"],
        "no_active_capped_or_invalid_histories": int(
            run["statuses"][[0, 4, 5, 6]].sum()
        )
        == 0,
        "positive_finite_observer_fluence": bool(
            np.isfinite(run["all_weight"]) and run["all_weight"] > 0
        ),
        "order_image_radial_closure": bool(
            np.allclose(run["image"].sum(), run["order_sum"].sum(), rtol=1e-11)
            and np.allclose(run["image"].sum(), run["radial"].sum(), rtol=1e-11)
        ),
        "aperture_accounting_closure": bool(
            np.allclose(
                run["image"].sum() + run["outside_weight"],
                run["all_weight"],
                rtol=1e-11,
            )
        ),
    }
    return {
        "all_passed": all(gates.values()),
        "gates": gates,
        "scope": "Production transport/peeloff integration smoke, grain absorption only; not a complete ISM model or RefleX agreement test",
        "configuration": config,
        "transport_status_counts": run["statuses"].tolist(),
        "order_fluences": run["order_sum"].tolist(),
        "all_observer_fluence": float(run["all_weight"]),
        "outside_image_fluence": float(run["outside_weight"]),
    }


def make_figure(table_directory, reference_directory, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scattering, absorption = load_wd01_dust_tables(table_directory)
    reference, _ = load_wd01_dust_tables(reference_directory)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    axes[0].loglog(
        scattering.energy_kev,
        scattering.scattering_cross_section_cm2_per_h,
        label="Dust scattering",
    )
    axes[0].loglog(
        absorption.energy_kev,
        absorption.absorption_cross_section_cm2_per_h,
        label="Grain absorption",
    )
    axes[0].set(
        xlabel="Photon energy (keV)",
        ylabel=r"Cross section (cm$^2$ per H)",
        title="WD01 size law + D03 optical constants",
    )
    axes[0].legend(frameon=False)
    for i, e in enumerate(reference.energy_kev):
        j = int(np.flatnonzero(scattering.energy_kev == e)[0])
        angles = reference.scattering_angle_rad
        error = (
            np.interp(
                angles,
                scattering.scattering_angle_rad,
                scattering.scattering_angle_cdf[j],
            )
            - reference.scattering_angle_cdf[i]
        )
        axes[1].semilogx(angles[1:] / ARCSEC, error[1:], label=f"{e:g} keV")
    axes[1].axhline(LIMITS["cdf_absolute"], color="gray", linestyle=":")
    axes[1].axhline(-LIMITS["cdf_absolute"], color="gray", linestyle=":")
    axes[1].set(
        xlim=(1, 1e4),
        xlabel="Intrinsic deflection (arcsec)",
        ylabel="CDF difference",
        title="Production hybrid minus finer full Mie",
    )
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.savefig(output / "wd01_validation.png", dpi=180)
    fig.savefig(output / "wd01_validation.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables", type=Path, default=REPOSITORY / "dsh/data/materials/wd01_d03_v1"
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport-smoke", action="store_true")
    parser.add_argument("--refine-reference-totals", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = compare_tables(args.tables, args.reference)
    if args.refine_reference_totals:
        report["reference_total_convergence"] = check_reference_totals(args.reference)
        report["all_passed"] &= report["reference_total_convergence"]["all_passed"]
    if args.transport_smoke:
        report["transport_smoke"] = transport_smoke(
            args.tables, args.output / "transport_smoke"
        )
        report["all_passed"] &= report["transport_smoke"]["all_passed"]
    if args.plot:
        make_figure(args.tables, args.reference, args.output)
    (args.output / "validation_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["all_passed"]:
        raise SystemExit("WD01 validation failed")


if __name__ == "__main__":
    main()
