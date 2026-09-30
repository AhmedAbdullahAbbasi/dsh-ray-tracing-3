"""Parse TOML, validate scalar settings, and resolve paths once."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .schema import ResolvedRun

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


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
