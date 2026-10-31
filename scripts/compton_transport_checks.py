"""Independent analog/peel-off control and matched cold-material core example.

This calls the explicit inelastic APIs, not the unchanged TOML/response runner.
Run with ``python -m scripts.compton_transport_checks --output-directory ...``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from dsh.geometry.clouds import build_angular_distance_cloud
from dsh.observer.binning import build_observer_bin_geometry
from dsh.observer.inelastic import bin_inelastic_events, score_inelastic_peeloff
from dsh.physics.cold_compton import build_cold_compton_table
from dsh.physics.compton import (
    build_free_electron_compton_table,
    to_compton_physics,
    write_compton_material_table,
)
from dsh.physics.dust import build_dust_physics_table
from dsh.physics.wd01_ism import load_wd01_ism
from dsh.sources.launch import (
    LaunchedSourcePackets,
    build_cloud_launch_geometry,
    sample_source_launches,
)
from dsh.sources.models import SourcePackets
from dsh.transport.inelastic import INELASTIC_STATUS_NAMES, transport_inelastic_batch
from dsh.transport.kernel import REACHED_OBSERVER_PLANE, STATUS_NAMES


def _scene(column, angular_spacing):
    return build_angular_distance_cloud(
        np.full((2, 3, 3), column / 2),
        x_centers_arcsec=[-angular_spacing, 0, angular_spacing],
        y_centers_arcsec=[-angular_spacing, 0, angular_spacing],
        z_centers_kpc=[4.0, 6.0],
        source_distance_kpc=10.0,
    )


def analog_peeloff_control(n_packets=262144, chunk_size=4096):
    """Single-scatter pencil beam: finite observer disk versus central density.

    Compare analog hits/area with peel-off fluence density at disk center.
    Units are pc^-2 per injected pencil-beam photon. The 500 pc disk is
    small compared with the 3--7 kpc scatter distances. Allow a declared 2%
    finite-aperture approximation plus five PAIRED Monte Carlo standard
    errors. This is a transport/estimator control, not a halo source model.
    """
    if n_packets < chunk_size or n_packets % chunk_size:
        raise ValueError("n_packets must be a positive multiple of chunk_size")
    scene = _scene(5e23, 60000.0)
    theta = np.linspace(0, np.pi, 33)
    cdf = np.broadcast_to((1 - np.cos(theta)) / 2, (2, theta.size))
    dust = build_dust_physics_table([1, 10], [0, 0], [0, 0], theta, cdf)
    compton = to_compton_physics(
        build_free_electron_compton_table(
            [1, 2, 3.3, 4.9, 6.9, 10],
            electrons_per_h=1,
            material_definition_sha256="b" * 64,
            cosine_nodes=513,
        )
    )
    n, energy, radius = chunk_size, 4.9, 500.0
    packets = LaunchedSourcePackets(
        jnp.broadcast_to(jnp.array([10000.0, 0, 0]), (n, 3)),
        jnp.broadcast_to(jnp.array([energy, -energy, 0, 0]), (n, 4)),
        jnp.ones(n),
        jnp.full(n, 1 / (4 * np.pi)),
        jnp.zeros(n),
        jnp.full(n, 1 / (n_packets * 10000.0**2)),
        jnp.zeros(n, dtype=jnp.int32),
        jnp.zeros(n, dtype=jnp.int32),
    )

    @jax.jit
    def simulate(key):
        result = transport_inelastic_batch(
            key,
            packets.position_pc,
            packets.momentum_kev,
            scene,
            dust,
            compton,
            energy_cutoff_kev=1.0,
            max_interactions=8,
        )
        scored = score_inelastic_peeloff(packets, result, scene, dust, compton)
        hit = (
            (result.status == REACHED_OBSERVER_PLANE)
            & (result.n_compton == 1)
            & (jnp.sum(result.position_pc[:, 1:] ** 2, axis=-1) <= radius**2)
        )
        peel = jnp.sum(
            jnp.where(
                scored.events.valid & (scored.events.scattering_order == 1),
                scored.events.weight_observer_fluence,
                0,
            ),
            axis=1,
        )
        return hit, peel

    hits, analog, peeloff, difference_squared = 0, 0.0, 0.0, 0.0
    for index in range(n_packets // chunk_size):
        hit, peel = jax.device_get(
            simulate(random.fold_in(random.PRNGKey(20261101), index))
        )
        a = hit.astype(np.float64) / (n_packets * np.pi * radius**2)
        p = peel.astype(np.float64)
        hits += int(hit.sum())
        analog += a.sum()
        peeloff += p.sum()
        difference_squared += np.sum((a - p) ** 2)
    difference = analog - peeloff
    standard_error = np.sqrt(
        n_packets
        / (n_packets - 1)
        * max(difference_squared - difference**2 / n_packets, 0.0)
    )
    tolerance = 5 * standard_error + 0.02 * peeloff
    passed = hits >= 100 and np.isfinite(peeloff) and abs(difference) < tolerance
    report = {
        "status": "passed" if passed else "failed",
        "histories": n_packets,
        "seed": 20261101,
        "analog_single_scatter_hits": hits,
        "disk_radius_pc": radius,
        "units": "pc^-2 per injected pencil-beam photon",
        "analog_fluence_density": float(analog),
        "central_peeloff_fluence_density": float(peeloff),
        "relative_difference": float(difference / peeloff),
        "paired_standard_error": float(standard_error),
        "paired_standard_errors_from_zero": float(difference / standard_error),
        "finite_aperture_relative_allowance": 0.02,
        "absolute_acceptance_tolerance": float(tolerance),
        "scope": "Single scatter free-electron normalization; not a cold-material or RefleX comparison.",
    }
    jax.clear_caches()
    return report


def matched_material_example(destination, n_packets=4096, chunk_size=256):
    """Use one declared elemental budget for WD01, absorption and Compton."""
    if n_packets < chunk_size or n_packets % chunk_size:
        raise ValueError("n_packets must be a positive multiple of chunk_size")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    material_directory = (
        Path(__file__).resolve().parents[1] / "dsh/data/materials/wd01_wilm_c50_si90_v1"
    )
    definition, scattering, _, absorption, dust = load_wd01_ism(material_directory)
    energy_grid = np.unique(np.r_[np.geomspace(2, 10, 129), 3.3, 4.9, 6.9])
    built = build_cold_compton_table(
        definition, energy_grid, grain_electrons="atomic_incoherent"
    )
    for metadata in (scattering.metadata, absorption.metadata, built.table.metadata):
        if metadata["material_definition_sha256"] != definition.sha256:
            raise ValueError("dust, absorption and Compton compositions differ")
    write_compton_material_table(destination / "cold_compton.npz", built.table)
    compton = to_compton_physics(built.table)
    scene = _scene(5e23, 500.0)
    geometry = build_cloud_launch_geometry(scene)
    bins = build_observer_bin_geometry(
        [-1000, 0, 1000],
        [-1000, 0, 1000],
        [2, 3.25, 3.35, 4.8, 5.0, 6.8, 7.0, 10],
        [0, 1e5, 1e7, 1e9, 1e11, 1e14],
    )

    @jax.jit
    def simulate(key, source):
        launch_key, transport_key = random.split(key)
        launched = sample_source_launches(launch_key, source, geometry)
        transported = transport_inelastic_batch(
            transport_key,
            launched.position_pc,
            launched.momentum_kev,
            scene,
            dust,
            compton,
            energy_cutoff_kev=2.0,
            max_interactions=24,
        )
        scored = score_inelastic_peeloff(launched, transported, scene, dust, compton)
        return transported, scored, bin_inelastic_events(scored, bins)

    packet_arrays, event_arrays, cubes = {}, {}, {}
    statuses = {}
    max_energy_error, unsupported, unsupported_phase = 0.0, 0, 0
    counts = np.zeros(4, dtype=np.int64)
    max_scatter = 0
    for start in range(0, n_packets, chunk_size):
        # Equal numbers at the three accepted anchors, with total injected
        # observer-equivalent fluence 1 ph/cm^2. This is a core API example.
        energy = jnp.asarray(
            np.array([3.3, 4.9, 6.9])[(start + np.arange(chunk_size)) % 3]
        )
        source = SourcePackets(
            energy,
            jnp.zeros(chunk_size),
            jnp.full(chunk_size, 1 / n_packets),
            jnp.zeros(chunk_size, dtype=jnp.int32),
            jnp.asarray((start + np.arange(chunk_size)) % 3, dtype=jnp.int32),
        )
        result, scored, products = jax.device_get(
            simulate(random.fold_in(random.PRNGKey(2049), start // chunk_size), source)
        )
        for value, count in zip(
            *np.unique(result.status, return_counts=True), strict=True
        ):
            statuses[int(value)] = statuses.get(int(value), 0) + int(count)
        closure = result.deposited_energy_kev + result.momentum_kev[:, 0] - energy
        max_energy_error = max(max_energy_error, float(np.max(np.abs(closure))))
        max_scatter = max(max_scatter, int(result.n_scatter.max()))
        unsupported += int(scored.unsupported_escape.sum())
        unsupported_phase += int(scored.unsupported_phase.sum())
        counts += np.bincount(scored.component[scored.events.valid], minlength=4)
        for field in result._fields:
            if field != "interactions":
                packet_arrays.setdefault(field, []).append(getattr(result, field))
        selected = scored.events.valid
        for field in scored.events._fields:
            event_arrays.setdefault(field, []).append(
                getattr(scored.events, field)[selected]
            )
        for field in scored._fields[1:]:
            event_arrays.setdefault(field, []).append(getattr(scored, field)[selected])
        packet_ids = np.broadcast_to(
            np.arange(start, start + chunk_size)[:, None], selected.shape
        )
        event_arrays.setdefault("packet_index", []).append(packet_ids[selected])
        for component in products._fields:
            for field in getattr(products, component)._fields:
                name = component + "__" + field
                value = np.asarray(getattr(getattr(products, component), field))
                cubes[name] = cubes.get(name, np.zeros_like(value)) + value
    arrays = {
        "packet__" + name: np.concatenate(values)
        for name, values in packet_arrays.items()
    }
    arrays.update(
        {
            "event__" + name: np.concatenate(values)
            for name, values in event_arrays.items()
        }
    )
    arrays.update(cubes)
    arrays.update(
        {"axis__" + field: np.asarray(getattr(bins, field)) for field in bins._fields}
    )
    np.savez_compressed(destination / "core_example.npz", **arrays)
    total = cubes["total__total_fluence"]
    components = sum(
        cubes[name + "__total_fluence"]
        for name in ("dust_only", "compton_only", "mixed")
    )
    component_error = float(np.max(np.abs(total - components)))
    positive_components = all(
        float(cubes[name + "__total_fluence"].sum()) > 0
        for name in ("dust_only", "compton_only", "mixed")
    )
    complete = set(statuses) <= {1, 2, 3}  # observer plane, outer sphere, absorbed
    passed = (
        complete
        and unsupported == 0
        and unsupported_phase == 0
        and max_energy_error < 2e-5
        and np.isfinite(total).all()
        and positive_components
        and np.allclose(total, components, rtol=5e-6, atol=1e-20)
        and float(cubes["total__unbinned_weight_observer_fluence"]) == 0
    )
    names = STATUS_NAMES | INELASTIC_STATUS_NAMES
    report = {
        "status": "passed" if passed else "failed",
        "histories": n_packets,
        "seed": 2049,
        "injected_observer_equivalent_fluence_ph_cm2": 1,
        "source_anchor_energy_kev": [3.3, 4.9, 6.9],
        "material_definition_sha256": definition.sha256,
        "grain_electrons": "atomic_incoherent",
        "common_table_energy_support_kev": [2, 10],
        "numerical_continuation_cutoff_kev": 2,
        "maximum_interactions": 24,
        "maximum_realized_scattering_order": max_scatter,
        "terminal_status_counts": {
            str(key): {"name": names[key], "count": value}
            for key, value in sorted(statuses.items())
        },
        "maximum_per_history_energy_closure_error_kev": max_energy_error,
        "unsupported_virtual_escape_count": unsupported,
        "unsupported_virtual_phase_count": unsupported_phase,
        "valid_event_counts": dict(
            zip(
                ("none", "dust_only", "compton_only", "mixed"),
                counts.tolist(),
                strict=True,
            )
        ),
        "component_fluence_ph_cm2": {
            name: float(cubes[name + "__total_fluence"].sum())
            for name in products._fields
        },
        "maximum_component_cube_closure_error_ph_cm2": component_error,
        "unbinned_fluence_ph_cm2": float(
            cubes["total__unbinned_weight_observer_fluence"]
        ),
        "scope": "Matched declared WD01/cold budget core example. No TOML, response, FITS, or RefleX equivalence claim.",
        "material_metadata": built.table.metadata,
    }
    jax.clear_caches()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/compton-transport-checks"),
    )
    parser.add_argument("--check", choices=("all", "analog", "material"), default="all")
    args = parser.parse_args(argv)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    report = {}
    if args.check in ("all", "analog"):
        print("Checking analog versus peel-off normalization ...", flush=True)
        report["analog"] = analog_peeloff_control()
    if args.check in ("all", "material"):
        print("Running matched cold-material example ...", flush=True)
        report["material"] = matched_material_example(
            args.output_directory / "material"
        )
    (args.output_directory / "checks.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({name: value["status"] for name, value in report.items()}),
        flush=True,
    )
    return 0 if all(value["status"] == "passed" for value in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
