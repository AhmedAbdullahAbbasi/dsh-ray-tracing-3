"""Build a numerical plan without transport, plotting or output publication."""

from dataclasses import dataclass, field

import numpy as np

from .config.load import (
    input_paths,
    validate_destinations,
    validate_relationships,
)
from .examples import DAY_S
from .inputs.common import check_backend, load_input
from .inputs.material import load_material
from .inputs.scene import load_scene
from .inputs.source import load_source
from .io.provenance import hash_files, runtime_identity, verify_inputs
from .observer.binning import build_observer_bin_geometry
from .sources.launch import build_cloud_launch_geometry
from .sources.radial_launch import build_radial_cloud_launch_geometry


@dataclass(frozen=True)
class RunPlan:
    config: object
    source: object
    cloud: object
    physics: object
    launch: object
    bins: object
    input_hashes: dict
    metadata: dict
    summary: dict
    publication_paths: dict = field(default_factory=dict)
    extra_inputs: tuple = ()


def build_run_plan(config, *, publication_paths=None, extra_inputs=()):
    validate_relationships(config)
    validate_destinations(
        config, publication_paths=publication_paths, extra_inputs=extra_inputs
    )
    actual_backend = check_backend(config.execution.device)
    hashes = hash_files((*input_paths(config), *extra_inputs))
    if hashes[str(config.config_path)] != config.config_sha256:
        raise ValueError("configuration file changed after parsing")
    scattering, absorption, physics, material_metadata = load_input(
        "materials", load_material, config.materials
    )
    source, source_metadata = load_input(
        f"source.file ({config.source.file})" if config.source.file else "source",
        load_source,
        config.source,
    )
    cloud = load_input(
        f"scene.cloud_file ({config.scene.cloud_file})"
        if config.scene.cloud_file
        else "scene",
        load_scene,
        config.scene,
    )
    energies = np.asarray(
        source.energy_edges_kev
        if source.energy_edges_kev is not None
        else source.effective_energy_kev
    )
    material_energy = np.asarray(physics.energy_kev)
    if energies.min() < material_energy.min() or energies.max() > material_energy.max():
        raise ValueError(
            "source: energy support extends beyond selected material tables"
        )
    if source.energy_edges_kev is None and not np.all(
        np.isin(energies, material_energy)
    ):
        raise ValueError(
            "source.effective_energy_kev: representative energies must be selected material nodes"
        )
    if (
        config.observer.energy_edges_kev[0] > energies.min()
        or config.observer.energy_edges_kev[-1] < energies.max()
    ):
        raise ValueError(
            "observer.energy_edges_kev: initial configured interface requires full source energy coverage"
        )
    arrival = np.asarray(config.observer.arrival.edges_days) * DAY_S
    if arrival[-1] <= float(np.asarray(source.time_edges_s)[-1]):
        raise ValueError(
            "observer.arrival: must extend beyond the final source emission time"
        )
    bins = build_observer_bin_geometry(
        cloud.x_edges_arcsec,
        cloud.y_edges_arcsec,
        config.observer.energy_edges_kev,
        arrival,
    )
    for name in (
        "energy_edges_kev",
        "arrival_time_edges_s",
        "sky_x_edges_arcsec",
        "sky_y_edges_arcsec",
    ):
        if not np.all(np.diff(np.asarray(getattr(bins, name))) > 0):
            raise ValueError(f"observer.{name}: edges collapse at JAX precision")
    if "fits" in config.output.formats:
        from .io.fits_output import _import_fits

        _import_fits()
    if config.plots.enabled:
        import matplotlib  # noqa: F401
    from .plots import plot_selection

    plot_selection(
        config.plots, bins, source, declared_days=config.observer.arrival.edges_days
    )
    launch = (
        build_radial_cloud_launch_geometry(cloud, physics)
        if config.execution.launch_sampler == "radial"
        else build_cloud_launch_geometry(cloud)
    )
    shape = (
        len(arrival) - 1,
        len(config.observer.energy_edges_kev) - 1,
        len(cloud.y_edges_arcsec) - 1,
        len(cloud.x_edges_arcsec) - 1,
    )
    metadata = {
        **runtime_identity(),
        **source_metadata,
        "material_tables": config.materials.kind,
        "material_identity": material_metadata,
        "scattering_table_sha256": scattering.metadata["table_sha256"],
        "absorption_table_sha256": absorption.metadata["table_sha256"],
        "material_definition_sha256": material_metadata.get(
            "definition_sha256", "unspecified"
        ),
        "cloud_description": str(config.scene.cloud_file or config.scene.kind),
        "packets": config.execution.packets,
        "chunk_size": config.execution.chunk_size,
        "max_interactions": config.execution.max_interactions,
        "seed": config.execution.seed,
        "launch_sampler": config.execution.launch_sampler,
        "source_dtype": str(source.band_flux.dtype),
        "physics_dtype": str(physics.energy_kev.dtype),
    }
    summary = {
        "run": config.name,
        "source_kind": config.source.kind,
        "source_fluence_ph_cm2": float(source.total_fluence),
        "source_emission_days": (
            np.asarray(source.time_edges_s)[[0, -1]] / DAY_S
        ).tolist(),
        "source_energy_support_kev": [float(energies.min()), float(energies.max())],
        "source_epoch": metadata.get("source_epoch", "relative day zero"),
        "source_time_scale": metadata.get("source_time_scale", "relative seconds"),
        "material_kind": config.materials.kind,
        "material_identity": {
            "material_id": material_metadata.get("definition", {}).get(
                "material_id", "legacy effective pair"
            ),
            "definition_sha256": material_metadata.get(
                "definition_sha256", "unspecified"
            ),
            "scattering_model": scattering.metadata.get(
                "model",
                scattering.metadata.get("scattering_method", "see input metadata"),
            ),
            "absorption_producer": absorption.metadata.get(
                "producer", "declared gas/H2/grain total"
            ),
            "energy_nodes": len(scattering.energy_kev),
            "description": material_metadata.get(
                "description",
                "Explicit pair; see resolved manifest for full table metadata.",
            ),
        },
        "cloud_shape_zyx": list(cloud.delta_nh_cm2.shape),
        "observer_shape_teyx": list(shape),
        "backend": actual_backend,
        "packets": config.execution.packets,
        "output_directory": str(config.output.directory),
        "product_array_lower_bound_bytes": int(np.prod(shape)) * 8 * 5,
        "memory_estimate_note": "Product arrays only; excludes cloud, phase tables, histories, compilation and transient buffers.",
    }
    verify_inputs(hashes)
    return RunPlan(
        config,
        source,
        cloud,
        physics,
        launch,
        bins,
        hashes,
        metadata,
        summary,
        publication_paths or {},
        tuple(extra_inputs),
    )
