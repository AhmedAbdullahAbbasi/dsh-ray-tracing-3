"""Write the complete, reproducible ideal-observer NPZ payload."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..geometry.clouds import AngularDistanceCloud
from ..observer.binning import ObserverBinGeometry
from ..physics.dust import DustPhysicsTable
from ..pipeline import TRANSPORT_STATUS_LABELS, IdealObserverSimulationResult
from ..sources.cells import SourceCells
from ..sources.launch import SourceLaunchGeometry
from ..sources.models import TabulatedBandSource


def _named_tuple_arrays(value) -> dict[str, np.ndarray]:
    return {field: np.asarray(getattr(value, field)) for field in value._fields}


def _optional_float(metadata: Mapping[str, object], name: str) -> np.ndarray:
    value = metadata.get(name)
    return np.asarray(np.nan if value is None else value)


def _source_arrays(source: TabulatedBandSource | SourceCells) -> dict[str, np.ndarray]:
    if isinstance(source, SourceCells):
        arrays = {
            "source_time_edges_s": np.asarray(source.time_edges_s),
            "source_total_fluence": np.asarray(source.total_fluence),
            "source_cell_time_start_s": np.asarray(source.start_s),
            "source_cell_time_stop_s": np.asarray(source.stop_s),
            "source_cell_energy_low_kev": np.asarray(source.energy_low_kev),
            "source_cell_energy_high_kev": np.asarray(source.energy_high_kev),
            "source_cell_photon_index": np.asarray(source.photon_index),
            "source_cell_kind": np.asarray(source.kind),
            "source_cell_time_index": np.asarray(source.time_index),
            "source_cell_spectral_bin_index": np.asarray(source.spectral_bin_index),
            "source_cell_fluence": np.asarray(source.cell_fluence),
            "source_flat_cdf": np.asarray(source.flat_cdf),
        }
        return arrays
    return {
        "source_energy_kev": np.asarray(source.effective_energy_kev),
        "source_energy_edges_kev": (
            np.asarray(source.energy_edges_kev)
            if source.energy_edges_kev is not None
            else np.asarray([], dtype=np.float64)
        ),
        "source_photon_index": (
            np.asarray(source.photon_index)
            if source.photon_index is not None
            else np.asarray(np.nan)
        ),
        "source_time_edges_s": np.asarray(source.time_edges_s),
        "source_band_flux": np.asarray(source.band_flux),
        "source_cell_fluence": np.asarray(source.cell_fluence),
        "source_flat_cdf": np.asarray(source.flat_cdf),
        "source_total_fluence": np.asarray(source.total_fluence),
    }


def write_ideal_observer_npz(
    path: str | Path,
    result: IdealObserverSimulationResult,
    bin_geometry: ObserverBinGeometry,
    source: TabulatedBandSource | SourceCells,
    cloud: AngularDistanceCloud,
    physics: DustPhysicsTable,
    launch_geometry: SourceLaunchGeometry,
    *,
    run_metadata: Mapping[str, object],
) -> Path:
    """Write observer products, inputs, physics tables, and run metadata."""

    metadata = dict(run_metadata)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        **_named_tuple_arrays(result.products),
        **_named_tuple_arrays(result.diagnostics),
        sky_x_edges_arcsec=np.asarray(bin_geometry.sky_x_edges_arcsec),
        sky_y_edges_arcsec=np.asarray(bin_geometry.sky_y_edges_arcsec),
        energy_edges_kev=np.asarray(bin_geometry.energy_edges_kev),
        arrival_time_edges_s=np.asarray(bin_geometry.arrival_time_edges_s),
        sky_pixel_solid_angle_sr=np.asarray(bin_geometry.sky_pixel_solid_angle_sr),
        transport_status_labels=np.asarray(TRANSPORT_STATUS_LABELS),
        **_source_arrays(source),
        cloud_delta_nh_cm2=np.asarray(cloud.delta_nh_cm2),
        cloud_n_h_cm3=np.asarray(cloud.n_h_cm3),
        cloud_x_edges_arcsec=np.asarray(cloud.x_edges_arcsec),
        cloud_y_edges_arcsec=np.asarray(cloud.y_edges_arcsec),
        cloud_z_edges_kpc=np.asarray(cloud.z_edges_kpc),
        cloud_radial_bin_width_cm=np.asarray(cloud.radial_bin_width_cm),
        source_distance_kpc=np.asarray(cloud.source_distance_kpc),
        physics_energy_kev=np.asarray(physics.energy_kev),
        physics_scattering_cross_section_cm2_per_h=np.asarray(
            physics.scattering_cross_section_cm2_per_h
        ),
        physics_absorption_cross_section_cm2_per_h=np.asarray(
            physics.absorption_cross_section_cm2_per_h
        ),
        physics_scattering_angle_rad=np.asarray(physics.scattering_angle_rad),
        physics_scattering_angle_cdf=np.asarray(physics.scattering_angle_cdf),
        physics_differential_cross_section_cm2_per_sr_per_h=np.asarray(
            physics.differential_cross_section_cm2_per_sr_per_h
        ),
        launch_source_position_pc=np.asarray(launch_geometry.source_position_pc),
        launch_source_distance_pc=np.asarray(launch_geometry.source_distance_pc),
        launch_slope_x_bounds=np.asarray(launch_geometry.slope_x_bounds),
        launch_slope_y_bounds=np.asarray(launch_geometry.slope_y_bounds),
        launch_slope_area=np.asarray(launch_geometry.slope_area),
        launch_solid_angle_sr=np.asarray(launch_geometry.launch_solid_angle_sr),
        output_schema_version=np.asarray(7 if isinstance(source, SourceCells) else 6),
        simulation_git_head=np.asarray(
            metadata.get("simulation_git_head", "unavailable")
        ),
        simulation_git_dirty=np.asarray(metadata.get("simulation_git_dirty", True)),
        simulation_python=np.asarray(metadata.get("simulation_python", "unavailable")),
        simulation_numpy=np.asarray(metadata.get("simulation_numpy", "unavailable")),
        simulation_jax=np.asarray(metadata.get("simulation_jax", "unavailable")),
        uncertainty_convention=np.asarray(
            "photon-grouped raw moments; order groups 1,2,>=3; zero-score histories included"
        ),
        cloud_description=np.asarray(metadata["cloud_description"]),
        source_flux_convention=np.asarray("unabsorbed observer-equivalent photon flux"),
        source_model=np.asarray(metadata["source_model"]),
        source_spectrum=np.asarray(metadata.get("source_spectrum", "representative")),
        material_tables=np.asarray(metadata.get("material_tables", "unspecified")),
        scattering_table_sha256=np.asarray(
            metadata.get("scattering_table_sha256", "unspecified")
        ),
        absorption_table_sha256=np.asarray(
            metadata.get("absorption_table_sha256", "unspecified")
        ),
        source_peak_band_flux=np.asarray(metadata.get("peak_band_fluxes", [])),
        source_baseline_band_flux=np.asarray(metadata.get("baseline_band_fluxes", [])),
        source_decay_time_days=_optional_float(metadata, "decay_time_days"),
        source_decay_start_days=_optional_float(metadata, "decay_start_days"),
        source_decay_duration_days=_optional_float(metadata, "decay_duration_days"),
        source_time_bin_days=_optional_float(metadata, "source_time_bin_days"),
        random_seed=np.asarray(metadata["seed"]),
        requested_packet_count=np.asarray(metadata["packets"]),
        chunk_size=np.asarray(metadata["chunk_size"]),
        max_interactions=np.asarray(metadata["max_interactions"]),
    )
    if isinstance(source, SourceCells):
        payload.update(
            source_fits_sha256=np.asarray(metadata["source_fits_sha256"]),
            cloud_fits_sha256=np.asarray(metadata.get("cloud_fits_sha256") or ""),
            source_components=np.asarray(metadata["source_components"]),
            source_mjdref=_optional_float(metadata, "source_mjdref"),
            source_timesys=np.asarray(metadata.get("source_timesys") or "RELATIVE"),
            resolved_config=np.asarray(metadata["resolved_config"]),
            resolved_config_sha256=np.asarray(metadata["resolved_config_sha256"]),
            original_config=np.asarray(metadata["original_config"]),
            original_config_sha256=np.asarray(metadata["original_config_sha256"]),
        )
    np.savez_compressed(output_path, **payload)
    return output_path
