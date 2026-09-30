"""Compile validated source, material, and scene inputs into numerical plans."""

from __future__ import annotations

import numpy as np

from dsh.contracts import RunPlan

from .config.schema import ResolvedRun
from .core.geometry.clouds import cloud_from_loaded_fits
from .core.launch import build_cloud_launch_geometry
from .core.observer.binning import build_observer_bin_geometry
from .io.cloud_fits import load_cube
from .materials import load_material_inputs
from .scenes.examples import DAY_S, build_synthetic_four_cloud_scene
from .sources.cells import build_source_cells
from .sources.format import load_source_fits


def build_run(config: ResolvedRun):
    """Validate files and produce source, material, cloud and observer arrays."""

    source_file = load_source_fits(config.source_fits)
    cells = build_source_cells(source_file, components=config.components)
    material = load_material_inputs(
        config.scattering, config.absorption, grid_path=config.material_grid
    )
    low = np.asarray(cells.energy_low_kev)
    high = np.asarray(cells.energy_high_kev)
    material_energy = material.scattering.energy_kev
    if np.min(low) < material_energy[0] or np.max(high) > material_energy[-1]:
        raise ValueError("selected source energies exceed material support")
    if config.scene_kind == "example_four_cloud":
        cloud = build_synthetic_four_cloud_scene(config.source_distance_kpc)
    else:
        cloud = cloud_from_loaded_fits(
            load_cube(config.cloud_fits), source_distance_kpc=config.source_distance_kpc
        )
    arrival_edges_s = np.asarray(config.time_edges_days) * DAY_S
    if float(np.asarray(cells.time_edges_s)[-1]) >= arrival_edges_s[-1]:
        raise ValueError("observer time range must extend beyond source emission")
    if float(np.asarray(cells.time_edges_s)[0]) < arrival_edges_s[0]:
        raise ValueError("observer time range must include source start")
    launch = build_cloud_launch_geometry(cloud)
    bins = build_observer_bin_geometry(
        np.asarray(cloud.x_edges_arcsec),
        np.asarray(cloud.y_edges_arcsec),
        np.asarray(config.energy_edges_kev),
        arrival_edges_s,
    )
    return source_file, cells, material, cloud, launch, bins


def plan_from_inputs(
    config: ResolvedRun, cells, material, cloud, launch, bins
) -> RunPlan:
    """Keep host provenance separate from the unchanged numerical runner inputs."""
    return RunPlan(
        source=cells,
        cloud=cloud,
        material=material.physics,
        launch=launch,
        observer=bins,
        packets=config.packets,
        chunk_size=config.chunk_size,
        max_interactions=config.max_interactions,
        seed=config.seed,
    )


def build_run_plan(config: ResolvedRun) -> RunPlan:
    """Public numerical-plan interface; build_run retains its six-item API."""
    _, cells, material, cloud, launch, bins = build_run(config)
    return plan_from_inputs(config, cells, material, cloud, launch, bins)
