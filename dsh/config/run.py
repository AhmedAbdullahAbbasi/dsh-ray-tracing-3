"""Resolve a TOML run into the existing source-to-observer pipeline."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
from jax import random

from ..examples import DAY_S, build_synthetic_four_cloud_scene
from ..geometry.clouds import cloud_from_loaded_fits
from ..io.cloud_fits import load_cube
from ..io.fits_output import write_ideal_observer_fits
from ..io.npz_output import write_ideal_observer_npz
from ..materials import load_material_inputs
from ..observer.binning import build_observer_bin_geometry
from ..pipeline import run_source_cells_to_observer_chunked
from ..sources.cells import build_source_cells
from ..sources.launch import build_cloud_launch_geometry
from ..sources.source_fits import load_source_fits

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


@dataclass(frozen=True)
class ResolvedRun:
    """Host-only configuration, paths and provenance; never passed to JIT."""

    name: str
    packets: int
    chunk_size: int
    max_interactions: int
    seed: int
    fail_on_cap: bool
    source_distance_kpc: float
    scene_kind: str
    cloud_fits: Path | None
    source_fits: Path
    components: str
    scattering: Path
    absorption: Path
    material_grid: Path | None
    time_edges_days: tuple[float, ...]
    energy_edges_kev: tuple[float, ...]
    output_npz: Path
    output_fits: Path
    config_path: Path
    config_text: str
    config_sha256: str


def _integer(section, name: str, *, min_value: int = 0) -> int:
    value = section[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < min_value:
        raise ValueError(f"{name} must be an integer >= {min_value}")
    return value


def _edges(values, name: str) -> tuple[float, ...]:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size < 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional edge list")
    if not np.all(np.diff(result) > 0.0):
        raise ValueError(f"{name} must increase strictly")
    return tuple(float(value) for value in result)


def load_run_config(path: str | Path) -> ResolvedRun:
    """Load strict input paths relative to the configuration file."""

    config_path = Path(path).resolve()
    raw = config_path.read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    if data.get("format_version") != 1:
        raise ValueError("run configuration requires format_version = 1")
    required = {"run", "scene", "source", "materials", "observer", "output"}
    if set(data) != required | {"format_version"}:
        raise ValueError("run configuration has missing or unknown sections")

    def local(value):
        candidate = Path(value)
        return (
            candidate if candidate.is_absolute() else config_path.parent / candidate
        ).resolve()

    run, scene, source, materials, observer, output = (
        data[name]
        for name in ("run", "scene", "source", "materials", "observer", "output")
    )
    allowed_keys = {
        "run": {
            "name",
            "packets",
            "chunk_size",
            "max_interactions",
            "seed",
            "fail_on_cap",
        },
        "scene": {"kind", "path", "source_distance_kpc"},
        "source": {"file", "components"},
        "materials": {"scattering", "absorption", "grid"},
        "observer": {"time_edges_days", "energy_edges_kev"},
        "output": {"npz", "fits"},
    }
    for section_name in allowed_keys:
        section = data[section_name]
        if not isinstance(section, dict) or set(section) - allowed_keys[section_name]:
            raise ValueError(f"unknown keys or invalid section: {section_name}")
    name = run["name"]
    if not isinstance(name, str) or not name:
        raise ValueError("run.name must be nonempty")
    scene_kind = scene["kind"]
    if scene_kind not in {"example_four_cloud", "fits"}:
        raise ValueError("scene.kind must be example_four_cloud or fits")
    if scene_kind == "fits" and "path" not in scene:
        raise ValueError("FITS scenes require scene.path")
    components = source.get("components", "both")
    if components not in {"both", "continuum", "lines"}:
        raise ValueError("source.components must be both, continuum or lines")
    distance = float(scene["source_distance_kpc"])
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("scene.source_distance_kpc must be finite and positive")
    time_edges = _edges(observer["time_edges_days"], "observer.time_edges_days")
    energy_edges = _edges(observer["energy_edges_kev"], "observer.energy_edges_kev")
    if energy_edges[0] <= 0.0:
        raise ValueError("observer energies must be positive")
    output_npz = local(output["npz"])
    if output_npz.suffix.lower() != ".npz":
        raise ValueError("output.npz must end in .npz")
    output_fits = local(output.get("fits", str(output_npz.with_suffix(".fits"))))
    if output_fits.suffix.lower() != ".fits":
        raise ValueError("output.fits must end in .fits")
    input_paths = {
        local(source["file"]),
        local(materials["scattering"]),
        local(materials["absorption"]),
    }
    if "grid" in materials:
        input_paths.add(local(materials["grid"]))
    if scene_kind == "fits":
        input_paths.add(local(scene["path"]))
    if output_npz in input_paths or output_fits in input_paths:
        raise ValueError("output paths must not overwrite input files")
    fail_on_cap = run.get("fail_on_cap", True)
    if not isinstance(fail_on_cap, bool):
        raise ValueError("run.fail_on_cap must be boolean")
    return ResolvedRun(
        name=name,
        packets=_integer(run, "packets", min_value=1),
        chunk_size=_integer(run, "chunk_size", min_value=1),
        max_interactions=_integer(run, "max_interactions", min_value=1),
        seed=_integer(run, "seed"),
        fail_on_cap=fail_on_cap,
        source_distance_kpc=distance,
        scene_kind=scene_kind,
        cloud_fits=local(scene["path"]) if scene_kind == "fits" else None,
        source_fits=local(source["file"]),
        components=components,
        scattering=local(materials["scattering"]),
        absorption=local(materials["absorption"]),
        material_grid=local(materials["grid"]) if "grid" in materials else None,
        time_edges_days=time_edges,
        energy_edges_kev=energy_edges,
        output_npz=output_npz,
        output_fits=output_fits,
        config_path=config_path,
        config_text=raw.decode("utf-8"),
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


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


def _git_value(*args) -> str | None:
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_manifest(config: ResolvedRun, cloud_sha256: str | None) -> str:
    """Make input paths and all numerical settings explicit in the archive."""

    values = {
        "format_version": 1,
        "run": {
            "name": config.name,
            "packets": config.packets,
            "chunk_size": config.chunk_size,
            "max_interactions": config.max_interactions,
            "seed": config.seed,
            "fail_on_cap": config.fail_on_cap,
        },
        "scene": {
            "kind": config.scene_kind,
            "path": str(config.cloud_fits) if config.cloud_fits else None,
            "cloud_fits_sha256": cloud_sha256,
            "source_distance_kpc": config.source_distance_kpc,
        },
        "source": {"file": str(config.source_fits), "components": config.components},
        "materials": {
            "scattering": str(config.scattering),
            "absorption": str(config.absorption),
            "grid": str(config.material_grid) if config.material_grid else None,
        },
        "observer": {
            "time_edges_days": config.time_edges_days,
            "energy_edges_kev": config.energy_edges_kev,
            "sky_edges": "native cloud edges",
        },
        "output": {"npz": str(config.output_npz), "fits": str(config.output_fits)},
        "original_config_path": str(config.config_path),
        "original_config_sha256": config.config_sha256,
    }
    return json.dumps(values, indent=2, sort_keys=True) + "\n"


def run_configured_simulation(config: ResolvedRun, *, progress_callback=None):
    """Execute and write schema-7 outputs plus a numerical run report."""

    cloud_sha256 = _file_sha256(config.cloud_fits) if config.cloud_fits else None
    source_file, cells, material, cloud, launch, bins = build_run(config)
    if config.cloud_fits and _file_sha256(config.cloud_fits) != cloud_sha256:
        raise ValueError("cloud FITS changed while the input scene was being loaded")
    result = run_source_cells_to_observer_chunked(
        random.PRNGKey(config.seed),
        cells,
        launch,
        cloud,
        material.physics,
        bins,
        total_packets=config.packets,
        chunk_size=config.chunk_size,
        max_interactions=config.max_interactions,
        progress_callback=progress_callback,
    )
    status = np.asarray(result.diagnostics.transport_status_count, dtype=np.int64)
    resolved_text = _resolved_manifest(config, cloud_sha256)
    resolved_sha256 = hashlib.sha256(resolved_text.encode("utf-8")).hexdigest()
    metadata = {
        "simulation_git_head": _git_value("rev-parse", "HEAD") or "unavailable",
        "simulation_git_dirty": bool(_git_value("status", "--porcelain")),
        "simulation_python": platform.python_version(),
        "simulation_numpy": np.__version__,
        "simulation_jax": jax.__version__,
        "material_tables": "external-components",
        "source_model": "source-fits",
        "source_spectrum": "file-cells",
        "source_components": config.components,
        "source_fits_sha256": source_file.file_sha256,
        "source_mjdref": source_file.mjdref,
        "source_timesys": source_file.timesys,
        "cloud_fits_sha256": cloud_sha256,
        "scattering_table_sha256": material.scattering_sha256,
        "absorption_table_sha256": material.absorption_sha256,
        "packets": config.packets,
        "chunk_size": config.chunk_size,
        "max_interactions": config.max_interactions,
        "seed": config.seed,
        "cloud_description": str(config.cloud_fits)
        if config.cloud_fits
        else "built-in synthetic four-cloud scene",
        "resolved_config": resolved_text,
        "resolved_config_sha256": resolved_sha256,
        "original_config": config.config_text,
        "original_config_sha256": config.config_sha256,
    }
    config.output_npz.parent.mkdir(parents=True, exist_ok=True)
    config.output_npz.with_suffix(".resolved_config.json").write_text(
        resolved_text, encoding="utf-8"
    )
    write_ideal_observer_npz(
        config.output_npz,
        result,
        bins,
        cells,
        cloud,
        material.physics,
        launch,
        run_metadata=metadata,
    )
    write_ideal_observer_fits(
        config.output_fits,
        result,
        bins,
        cells,
        cloud,
        material.physics,
        launch,
        run_metadata=metadata,
    )
    report = {
        "name": config.name,
        "source_sha256": source_file.file_sha256,
        "cloud_sha256": cloud_sha256,
        "scattering_sha256": material.scattering_sha256,
        "absorption_sha256": material.absorption_sha256,
        "config_sha256": resolved_sha256,
        "original_config_sha256": config.config_sha256,
        "status_counts": status.tolist(),
        "cap_passed": bool(status[4] == 0),
        "numerical_passed": bool(status[0] == 0 and np.sum(status[4:]) == 0),
        "validated_science_product": False,
    }
    report_path = config.output_npz.with_suffix(".run_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if config.fail_on_cap and not report["numerical_passed"]:
        raise RuntimeError(f"numerical transport states are nonzero; see {report_path}")
    return report
