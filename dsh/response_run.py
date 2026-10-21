"""Configured orchestration around the established physical halo response API."""

from __future__ import annotations

import json
from dataclasses import dataclass

import jax
import numpy as np

from .config.load import input_paths, validate_destinations
from .inputs.common import check_backend, load_input
from .inputs.material import load_material
from .inputs.scene import load_scene
from .inputs.source import load_source
from .io.atomic import atomic_savez_compressed
from .io.provenance import hash_files, runtime_identity, verify_inputs
from .io.publication import publish_outputs, staged_run, write_run_records
from .observer.response import (
    fold_spectral_response,
    load_spectral_response,
    write_spectral_response,
)
from .observer.response_transport import collect_spectral_response
from .sources.launch import build_cloud_launch_geometry
from .sources.radial_launch import build_radial_cloud_launch_geometry


class IncompleteResponse(ValueError):
    """A physical response cannot be published as a complete product."""


@dataclass(frozen=True)
class ResponsePlan:
    config: object
    source: object
    input_hashes: dict
    metadata: dict
    summary: dict
    archive: object = None
    cloud: object = None
    physics: object = None
    launch: object = None


def build_response_plan(config):
    validate_destinations(config)
    check_backend(config.execution.device)
    hashes = hash_files(input_paths(config))
    if hashes[str(config.config_path)] != config.config_sha256:
        raise ValueError("configuration file changed after parsing")
    source, source_metadata = load_input("source", load_source, config.source)
    if source.energy_edges_kev is None or source.photon_index is None:
        raise ValueError("source: response workflows require a continuous spectrum")
    metadata = {
        **runtime_identity(),
        **source_metadata,
        "workflow": config.workflow,
        "config_sha256": config.config_sha256,
    }
    summary = {
        "workflow": config.workflow,
        "name": config.name,
        "backend": jax.default_backend(),
        "output_directory": str(config.output.directory),
    }
    if config.workflow == "response_collect":
        if source.band_flux.shape[0] != 1 or np.any(np.asarray(source.band_flux) <= 0):
            raise ValueError(
                "source: response proposal requires one interval and strictly positive flux in every band"
            )
        _, _, physics, material = load_input(
            "materials", load_material, config.materials
        )
        cloud = load_input("scene", load_scene, config.scene)
        bounds = np.asarray(source.energy_edges_kev)
        if bounds[0] < float(physics.energy_kev[0]) or bounds[-1] > float(
            physics.energy_kev[-1]
        ):
            raise ValueError("source: proposal energy support exceeds material tables")
        launch = (
            build_radial_cloud_launch_geometry(cloud, physics)
            if config.execution.launch_sampler == "radial"
            else build_cloud_launch_geometry(cloud)
        )
        metadata["material_identity"] = material
        summary.update(
            packets=config.execution.packets,
            proposal_energy_support_kev=[float(bounds[0]), float(bounds[-1])],
            archive_schema_version=1,
        )
        plan = ResponsePlan(
            config,
            source,
            hashes,
            metadata,
            summary,
            cloud=cloud,
            physics=physics,
            launch=launch,
        )
    else:
        archive = load_input(
            "response.file", load_spectral_response, config.response.file
        )
        bounds = np.asarray(source.energy_edges_kev)
        if (
            bounds[0] < archive.energy_support_kev[0]
            or bounds[-1] > archive.energy_support_kev[1]
        ):
            raise ValueError("source: energy support extends outside response proposal")
        settings = config.response
        shape = (
            len(settings.arrival_time_edges_days) - 1,
            len(settings.energy_edges_kev) - 1,
            len(settings.radial_edges_arcsec) - 1,
        )
        workspace = (
            int(np.prod(shape))
            * min(archive.n_histories, settings.history_block_size)
            * 8
        )
        summary.update(
            n_histories=archive.n_histories,
            response_samples=len(archive.energy_kev),
            shape_time_energy_radius=list(shape),
            group_workspace_bytes=workspace,
            covariance=settings.covariance,
            archive_provenance=archive.provenance,
        )
        metadata["response_provenance"] = archive.provenance
        plan = ResponsePlan(config, source, hashes, metadata, summary, archive=archive)
    verify_inputs(hashes)
    return plan


def execute_response_plan(plan, progress=None):
    verify_inputs(plan.input_hashes)
    config = plan.config
    if config.workflow == "response_collect":
        settings = config.execution
        response = collect_spectral_response(
            plan.source,
            plan.cloud,
            plan.physics,
            plan.launch,
            n_packets=settings.packets,
            chunk_size=settings.chunk_size,
            max_interactions=settings.max_interactions,
            seed=settings.seed,
            progress=progress,
        )
        response.provenance["configured_run"] = plan.metadata
        response.provenance["input_sha256"] = plan.input_hashes
        return response
    settings = config.response
    result = fold_spectral_response(
        plan.archive,
        plan.source,
        np.asarray(settings.arrival_time_edges_days) * 86400,
        settings.energy_edges_kev,
        settings.radial_edges_arcsec,
        scattering_orders=settings.scattering_orders,
        covariance=settings.covariance,
        history_block_size=settings.history_block_size,
    )

    values = (result.fluence, result.standard_error, result.covariance)
    if any(value is not None and not np.all(np.isfinite(value)) for value in values):
        raise IncompleteResponse(
            "response_fold: nonfinite folded values or uncertainty"
        )
    return result


def publish_response_run(plan):
    config = plan.config
    directory = config.output.directory
    with staged_run(config, plan.input_hashes) as stage:
        complete = True
        message = None
        log_path = stage / "run.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write(json.dumps(plan.summary, allow_nan=False) + "\n")

            def progress(n, total):
                text = f"Response histories: {n}/{total}"
                print(text, flush=True)
                log.write(text + "\n")
                log.flush()

            try:
                result = execute_response_plan(plan, progress)
            except ValueError as error:
                # The unchanged collector exposes this exact quality-gate error.
                # Other ValueErrors remain configuration/runtime errors (exit 2).
                if (
                    not isinstance(error, IncompleteResponse)
                    and str(error)
                    != "response requires completed, uncapped, valid histories"
                ):
                    raise
                complete = False
                message = str(error)
        paths = [log_path]
        if complete:
            product = stage / (config.name + ".npz")
            if config.workflow == "response_collect":
                write_spectral_response(product, result)
            else:
                settings = config.response
                metadata = {
                    "schema_version": 1,
                    "fluence_units": "ph cm^-2",
                    "axes": ["arrival_time", "energy", "radius"],
                    "uncertainty": "photon-grouped finite-response Monte Carlo covariance; C-order flattening",
                    "n_histories": result.n_histories,
                    **plan.metadata,
                }
                payload = {
                    "fluence": result.fluence,
                    "standard_error": result.standard_error,
                    "arrival_time_edges_s": np.asarray(settings.arrival_time_edges_days)
                    * 86400,
                    "energy_edges_kev": settings.energy_edges_kev,
                    "radial_edges_arcsec": settings.radial_edges_arcsec,
                    "metadata_json": np.asarray(
                        json.dumps(metadata, sort_keys=True, allow_nan=False)
                    ),
                }
                if result.covariance is not None:
                    payload["covariance"] = result.covariance
                atomic_savez_compressed(product, **payload)
            paths.append(product)
        report = {
            "report_version": 1,
            "workflow": config.workflow,
            "status": "complete" if complete else "incomplete",
            "numerical_complete": complete,
            "error": message,
            "config_sha256": config.config_sha256,
        }
        paths = write_run_records(
            stage,
            config,
            plan.input_hashes,
            plan.metadata,
            report,
            paths,
            log_path=log_path,
        )
        publish_outputs(stage, config, plan.input_hashes, paths, complete=complete)
    print(
        f"Response {'complete' if complete else 'incomplete'}; products: {directory}",
        flush=True,
    )
    return 0 if complete else 3
