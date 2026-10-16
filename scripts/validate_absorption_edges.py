"""Validate a complete sphere material against author gas routines and grain solves."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jax
import numpy as np

from dsh.physics.energy_edges import integrate_spectrum
from dsh.physics.gas_absorption import (
    ELEMENTS,
    TBNEW_ELEMENTS,
    TBnewGasBackend,
    load_atomic_basis,
)
from dsh.physics.wd01_ism import load_wd01_ism, wilm_budget_population
from dsh.physics.wd01_tables import calculate_wd01_tables, default_wd01_angle_grid
from dsh.transport.kernel import _physics_at_energy
from dsh.validation.absorbed_observer import host_material


def _grain_row(energy):
    pop = wilm_budget_population(carbon_gas_fraction=0.5, silicon_gas_fraction=0.1)
    arrays = calculate_wd01_tables(
        pop,
        [energy],
        default_wd01_angle_grid(1024, 128),
        method="mie_adt",
        quadrature_order=24,
    )
    return {
        key: arrays[key]
        for key in (
            "energy_kev",
            "scattering_angle_rad",
            "scattering_cross_section_cm2_per_h",
            "grain_absorption_cross_section_cm2_per_h",
            "scattering_angle_cdf",
        )
    }


def validate(directory, backend, *, direct_grain_check=False):
    definition, scattering, components, total, physics = load_wd01_ism(directory)
    basis = load_atomic_basis()
    rng = np.random.default_rng(617203)
    edges = np.asarray(basis.metadata["edge_energies_kev"])
    points = np.unique(
        np.concatenate(
            [
                np.linspace(2, 10, 2049),
                np.exp(rng.uniform(np.log(2), np.log(10), 1024)),
                *(
                    edges * (1 + offset)
                    for offset in (-1e-4, -5e-6, -1e-6, 0, 1e-6, 5e-6, 1e-4)
                ),
            ]
        )
    )
    direct, direct_h2 = backend.evaluate(points)
    portable, portable_h2 = basis.evaluate(points)
    basis_error = float(np.max(abs(portable / direct - 1)))
    h2_error = float(np.max(abs(portable_h2 / direct_h2 - 1)))
    audit = definition.audit()
    residual = np.array(
        [audit["elements"].get(e, {}).get("gas_atoms_per_h", 0) for e in ELEMENTS]
    )
    atomic_count = residual.copy()
    atomic_count[0] = audit["atomic_hydrogen_atoms_per_h"]
    mask = np.array(TBNEW_ELEMENTS) - 1
    core_errors = []
    for energy in (2, 3.3, 4.9, 6.9, 7.12399, 7.12401, 10):
        atoms, molecule = backend.evaluate([energy])
        sigma = float(
            atomic_count[mask] @ atoms[mask, 0]
            + audit["molecular_hydrogen_molecules_per_h"] * molecule[0]
        )
        for nh in (1e20, 1e22, 3e23):
            native = backend.core_transmission(
                energy, nh, residual, definition.molecular_hydrogen_fraction
            )
            core_errors.append(abs(native / np.exp(-nh * sigma) - 1))

    @jax.jit
    def runtime_cross_sections(energy):
        return jax.vmap(
            lambda e: (
                _physics_at_energy(physics, e)[1],
                _physics_at_energy(physics, e)[2],
            )
        )(energy)

    def grain_absorption(e):
        return np.exp(
            np.interp(
                np.log(e),
                np.log(scattering.energy_kev),
                np.log(components[2].absorption_cross_section_cm2_per_h),
            )
        )

    panels = np.unique(np.r_[scattering.energy_kev, edges])
    folded = []
    for bounds in ([2, 4], [4, 6], [6, 10]):
        for nh in (1e21, 1e22, 1e23):

            def production(e, nh=nh):
                sca, absorbed = runtime_cross_sections(e)
                return e**-1.7 * np.exp(
                    -nh
                    * (np.asarray(sca, dtype=float) + np.asarray(absorbed, dtype=float))
                )

            def reference(e, nh=nh):
                atoms, molecule = backend.evaluate(e)
                absorbed = (
                    atomic_count @ atoms
                    + audit["molecular_hydrogen_molecules_per_h"] * molecule
                    + grain_absorption(e)
                )
                sca = np.exp(
                    np.interp(
                        np.log(e),
                        np.log(scattering.energy_kev),
                        np.log(scattering.scattering_cross_section_cm2_per_h),
                    )
                )
                return e**-1.7 * np.exp(-nh * (sca + absorbed))

            prod = integrate_spectrum(
                production, bounds, breakpoints_kev=panels, order=24
            )
            ref = integrate_spectrum(
                reference, bounds, breakpoints_kev=panels, order=32
            )
            ref_fine = integrate_spectrum(
                reference, bounds, breakpoints_kev=panels, order=48
            )
            folded.append(
                {
                    "band_kev": bounds,
                    "nh_cm2": nh,
                    "production": prod,
                    "author_gas_reference": ref_fine,
                    "relative_error": prod / ref_fine - 1,
                    "reference_quadrature_relative_difference": ref / ref_fine - 1,
                }
            )
    inverse = []
    for nh in (1e20, 1e22, 1e24):
        sigma = total.absorption_cross_section_cm2_per_h
        inverse.append(
            float(np.max(abs(-np.log(np.exp(-nh * sigma)) / nh / sigma - 1)))
        )
    result = {
        "material_definition_sha256": definition.sha256,
        "scope": "Neutral 2--10 keV gas/H2 plus declared WD01-shape Mie grains; standalone author routines, not an XSPEC-process validation",
        "gas_basis_check": {
            "energies": len(points),
            "maximum_atomic_relative_error": basis_error,
            "maximum_h2_relative_error": h2_error,
            "relative_error_limit": 5e-4,
        },
        "upstream_full_gas_summation_check": {
            "energies": 7,
            "columns_cm2": [1e20, 1e22, 3e23],
            "maximum_transmission_relative_error": float(max(core_errors)),
            "relative_error_limit": 1e-10,
            "scope": "18 TBnew atoms + H2, residual element columns, rho=0; extended atoms validated through PHFIT2 separately",
        },
        "folded_direct_beam_check": {
            "spectrum": "E^-1.7",
            "relative_error_limit": 1e-3,
            "reference_quadrature_limit": 1e-5,
            "rows": folded,
            "scope": "Actual JAX total extinction versus direct author gas + interpolated grain absorption/scattering; independent grain checks reported separately",
        },
        "beer_lambert_maximum_coefficient_recovery_relative_error": max(inverse),
        "source_precision_note": "PHFIT2 uses float32 energies internally; sub-ulp threshold locations are not claimed. Runtime thresholds follow stored JAX precision.",
    }
    if direct_grain_check:
        rows = []
        test_energies = [2.05, 4.95, 7.1125, 7.124]
        with ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for energy, ref in zip(
                test_energies, pool.map(_grain_row, test_energies), strict=True
            ):
                low, high, f, sca, _ = host_material(physics, energy)
                cdf = (1 - f) * scattering.scattering_angle_cdf[
                    low
                ] + f * scattering.scattering_angle_cdf[high]
                grain = grain_absorption(energy)
                rows.append(
                    {
                        "energy_kev": energy,
                        "scattering_relative_error": float(
                            sca / ref["scattering_cross_section_cm2_per_h"][0] - 1
                        ),
                        "grain_absorption_relative_error": float(
                            grain / ref["grain_absorption_cross_section_cm2_per_h"][0]
                            - 1
                        ),
                        "maximum_cdf_absolute_error": float(
                            np.max(abs(cdf - ref["scattering_angle_cdf"][0]))
                        ),
                    }
                )
        result["direct_grain_interpolation_check"] = {
            "scope": "Direct order-24 Mie totals / forward-ADT phase, between original energy nodes, same optical constants and shape recipe",
            "total_relative_error_limit": 1e-3,
            "cdf_absolute_error_limit": 5e-4,
            "rows": rows,
        }
    gates = {
        "gas_basis": max(basis_error, h2_error) <= 5e-4,
        "native_tbnew_summation": max(core_errors) <= 1e-10,
        "folded_extinction": all(abs(r["relative_error"]) <= 1e-3 for r in folded),
        "folded_reference_convergence": all(
            abs(r["reference_quadrature_relative_difference"]) <= 1e-5 for r in folded
        ),
        "beer_lambert": max(inverse) <= 1e-11,
    }
    if direct_grain_check:
        gates["grain_energy_interpolation"] = all(
            abs(r["scattering_relative_error"]) <= 1e-3
            and abs(r["grain_absorption_relative_error"]) <= 1e-3
            and r["maximum_cdf_absolute_error"] <= 5e-4
            for r in result["direct_grain_interpolation_check"]["rows"]
        )
    gates = {key: bool(value) for key, value in gates.items()}
    result.update(gates=gates, all_passed=all(gates.values()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--material-directory",
        type=Path,
        default=Path("dsh/data/materials/wd01_wilm_c50_si90_v1"),
    )
    p.add_argument("--backend", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--direct-grain-check", action="store_true")
    p.add_argument("--transport-control", action="store_true")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    result = validate(
        a.material_directory,
        TBnewGasBackend(a.backend),
        direct_grain_check=a.direct_grain_check,
    )
    if a.transport_control:
        result["transport_control"] = transport_control(a.material_directory)
        result["all_passed"] &= result["transport_control"]["all_passed"]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.plot:
        plot_material(a.material_directory, a.output.parent)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise SystemExit("absorption/edge validation failed")


def transport_control(directory):
    """First-flight competing risks using actual new tables and production MC."""
    import jax.numpy as jnp
    from jax import random

    from dsh.transport.kernel import transport_photon_batch
    from reflex_adapter.compare_reflex import make_cloud

    _, _, _, _, physics = load_wd01_ism(directory)
    config = json.loads(Path("reflex_adapter/comparison_config.json").read_text())
    config["nh_cm2"] = 3e23
    cloud = make_cloud(config)
    chunk, histories = 8192, 32768
    position = jnp.broadcast_to(jnp.array([10000.0, 0.0, 0.0]), (chunk, 3))

    @jax.jit
    def batch(key, energy):
        momentum = jnp.broadcast_to(jnp.array([energy, -energy, 0.0, 0.0]), (chunk, 4))
        result = transport_photon_batch(
            key, position, momentum, cloud, physics, max_interactions=16
        )
        return (
            result.interactions.interaction_type[:, 0],
            result.status,
            result.n_scatter,
        )

    rows = []
    for index, energy in enumerate((6.9, 7.1239, 7.1241)):
        first = np.zeros(3, dtype=int)
        statuses = np.zeros(7, dtype=int)
        multiple = 0
        for step in range(histories // chunk):
            types, status, nscatter = map(
                np.asarray, batch(random.PRNGKey(41203 + index * 100 + step), energy)
            )
            first += np.bincount(types, minlength=3)
            statuses += np.bincount(status, minlength=7)
            multiple += int(np.count_nonzero(nscatter >= 2))
        _, _, _, sca, absorbed = host_material(physics, energy)
        tau = config["nh_cm2"] * (sca + absorbed)
        event = -np.expm1(-tau)
        probability = np.array(
            [
                np.exp(-tau),
                event * sca / (sca + absorbed),
                event * absorbed / (sca + absorbed),
            ]
        )
        z = (first - histories * probability) / np.sqrt(
            histories * probability * (1 - probability)
        )
        passed = bool(
            np.max(abs(z)) <= 5
            and statuses[[0, 4, 5, 6]].sum() == 0
            and statuses.sum() == histories
            and multiple > 0
        )
        rows.append(
            {
                "energy_kev": energy,
                "histories": histories,
                "first_interaction_counts_none_scatter_absorb": first.tolist(),
                "analytic_first_probabilities": probability.tolist(),
                "first_probability_z": z.tolist(),
                "multiple_scattering_histories": multiple,
                "terminal_status_counts": statuses.tolist(),
                "passed": passed,
            }
        )
    return {
        "all_passed": all(r["passed"] for r in rows),
        "nh_cm2": config["nh_cm2"],
        "maximum_first_event_absolute_z": 5,
        "scope": "Exact central first-flight survival/competing risks; later multiple scatterings present, with termination checked but no independent new-model multiple-order halo prediction",
        "rows": rows,
    }


def plot_material(directory, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, scattering, components, total, _ = load_wd01_ism(directory)
    e = scattering.energy_kev
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    labels = ("Atomic gas", "H2", "Grains")
    colors = ("#cc7722", "#999999", "#2867a5")
    for table, label, color in zip(components, labels, colors, strict=True):
        axes[0].loglog(
            e, table.absorption_cross_section_cm2_per_h, label=label, color=color
        )
        axes[1].plot(
            e,
            table.absorption_cross_section_cm2_per_h / 1e-24,
            label=label,
            color=color,
        )
    axes[0].loglog(
        e,
        total.absorption_cross_section_cm2_per_h,
        color="black",
        label="Total absorption",
        lw=1.5,
    )
    axes[0].loglog(
        e,
        scattering.scattering_cross_section_cm2_per_h,
        "--",
        color="#228855",
        label="Scattering",
    )
    axes[0].set(
        xlabel="Energy (keV)",
        ylabel=r"Cross section (cm$^2$ per H)",
        title="Experimental WD01 shapes / Wilms budget",
    )
    axes[1].plot(
        e,
        total.absorption_cross_section_cm2_per_h / 1e-24,
        color="black",
        label="Total absorption",
        lw=1.5,
    )
    axes[1].axvline(7.124, color="gray", linestyle=":", lw=1)
    axes[1].set(
        xlim=(7.08, 7.22),
        ylim=(0, 2.3),
        xlabel="Energy (keV)",
        ylabel=r"Absorption ($10^{-24}$ cm$^2$ per H)",
        title="Native grain structure and gas K threshold",
    )
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(Path(output) / "absorption_components.png", dpi=180)
    fig.savefig(Path(output) / "absorption_components.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
