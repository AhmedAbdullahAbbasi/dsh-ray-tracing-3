"""Strict TOML validation with standard-library-only imports on Python 3.11+."""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import fields
from difflib import get_close_matches
from pathlib import Path

from .schema import (
    SOURCE_FIELDS,
    ArrivalConfig,
    ExecutionConfig,
    LoggingConfig,
    MaterialConfig,
    ObserverConfig,
    OutputConfig,
    PlotConfig,
    QualityConfig,
    ResolvedRun,
    SceneConfig,
    SourceConfig,
)

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


class ConfigError(ValueError):
    """An invalid setting, identified by its configuration field."""


def _table(value, name, allowed, required=()):
    if not isinstance(value, dict):
        raise ConfigError(f"{name}: expected a TOML table")
    unknown = set(value) - set(allowed)
    missing = set(required) - set(value)
    if unknown:
        hints = []
        for key in sorted(unknown):
            matches = get_close_matches(key, sorted(allowed), n=1, cutoff=0.75)
            if matches:
                hints.append(f"{key!r}: did you mean {matches[0]!r}?")
        detail = "; " + "; ".join(hints) if hints else ""
        raise ConfigError(
            f"{name}: unknown or irrelevant fields {sorted(unknown)}{detail}"
        )
    if missing:
        raise ConfigError(f"{name}: missing fields {sorted(missing)}")
    return value


def _number(value, name, *, minimum=None, strictly=False):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ConfigError(f"{name}: expected a finite number")
    if minimum is not None and (value <= minimum if strictly else value < minimum):
        raise ConfigError(f"{name}: must be {'>' if strictly else '>='} {minimum}")
    return float(value)


def _integer(value, name, minimum=1, maximum=None):
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ConfigError(
            f"{name}: expected an integer in [{minimum}, {maximum or 'inf'}]"
        )
    return value


def _boolean(value, name):
    if type(value) is not bool:
        raise ConfigError(f"{name}: expected true or false")
    return value


def _choice(value, name, choices):
    if not isinstance(value, str) or value not in choices:
        raise ConfigError(f"{name}: expected one of {tuple(choices)}")
    return value


def _vector(value, name, *, minimum=None, edges=False):
    if not isinstance(value, list) or len(value) < (2 if edges else 1):
        raise ConfigError(f"{name}: expected a nonempty numeric array")
    result = tuple(_number(x, name, minimum=minimum) for x in value)
    if edges and any(b <= a for a, b in zip(result[:-1], result[1:], strict=True)):
        raise ConfigError(f"{name}: edges must increase strictly")
    return result


def _path(value, name, base, *, directory=False, output=False):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name}: expected a nonempty path")
    if "$" in value or re.search(r"%[^%]+%", value):
        raise ConfigError(f"{name}: environment-variable expansion is unsupported")
    if os.name != "nt" and (
        re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("\\\\")
    ):
        raise ConfigError(f"{name}: Windows absolute paths require a Windows host")
    try:
        candidate = Path(value).expanduser()
        candidate = (
            (base / candidate).resolve()
            if not candidate.is_absolute()
            else candidate.resolve()
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise ConfigError(f"{name}: invalid path: {error}") from error
    if not output and not (candidate.is_dir() if directory else candidate.is_file()):
        raise ConfigError(
            f"{name}: {'directory' if directory else 'file'} not found: {candidate}"
        )
    return candidate


def input_paths(config):
    """Exact files loaded by the selected adapters, including sidecars."""
    paths = [config.config_path]
    paths += [p for p in (config.scene.cloud_file, config.source.file) if p is not None]
    material = config.materials
    if material.kind == "declared_bundle":
        paths.append(material.directory / "material_definition.json")
        for stem in (
            "scattering",
            "atomic_gas",
            "molecular_hydrogen",
            "grains",
            "total_absorption",
        ):
            paths += [
                material.directory / (stem + suffix) for suffix in (".npz", ".json")
            ]
        # These describe the producer recipe rather than runtime arrays, but
        # preserve them when present as part of the selected bundle identity.
        paths += [
            material.directory / name
            for name in ("recipe.json", "abundance_audit.json")
            if (material.directory / name).is_file()
        ]
    else:
        for path in (material.scattering_file, material.absorption_file):
            paths += [path, path.with_suffix(".json")]
        if material.grid_file:
            paths.append(material.grid_file)
    return tuple(dict.fromkeys(p.resolve() for p in paths))


def output_paths(config):
    directory = config.output.directory
    prefix = "diagnostic_" if config.quality.allow_incomplete else ""
    paths = [
        directory / (prefix + config.name + "." + kind)
        for kind in config.output.formats
    ]
    paths += [
        directory / name
        for name in (
            "original.toml",
            "resolved.json",
            "report.json",
            "complete.json",
            config.logging.file,
        )
    ]
    if config.plots.enabled:
        paths += [
            directory / (prefix + "intensity_" + scale + ".png")
            for scale in config.plots.scales
        ]
        if config.plots.save_count_plots:
            paths += [
                directory / (prefix + "scored_events_" + scale + ".png")
                for scale in config.plots.scales
            ]
    if config.plots.save_scored_event_map:
        paths.append(directory / (prefix + "scored_mc_events.npy"))
    return tuple(paths)


def validate_destinations(config, *, publication_paths=None, extra_inputs=()):
    inputs = input_paths(config) + tuple(Path(p).resolve() for p in extra_inputs)
    declared = output_paths(config)
    mapping = publication_paths or {}
    if set(mapping) & {
        "original.toml",
        "resolved.json",
        "report.json",
        "complete.json",
    }:
        raise ConfigError("output: provenance files must stay in the run directory")
    if set(mapping) - {p.name for p in declared}:
        raise ConfigError("output: publication mapping contains an unknown product")
    outputs = [Path(mapping.get(p.name, p)).resolve() for p in declared]
    resolved = [path.resolve() for path in outputs]
    if len(set(resolved)) != len(resolved):
        raise ConfigError("output: derived output paths collide")
    for output in outputs:
        for parent in output.parents:
            if parent.exists():
                if not parent.is_dir():
                    field = (
                        "output.directory"
                        if output.parent == config.output.directory
                        else "output.publication_paths"
                    )
                    raise ConfigError(f"{field}: parent is not a directory: {parent}")
                break
        for source in inputs:
            if output.resolve() == source or (
                output.exists() and source.exists() and os.path.samefile(output, source)
            ):
                raise ConfigError(f"output: would overwrite input {source}")
        if output.exists() and (not config.output.overwrite or not output.is_file()):
            raise ConfigError(
                f"output: destination exists: {output}; choose another output.directory "
                "or explicitly set output.overwrite = true"
            )
    if config.output.directory.exists() and not config.output.directory.is_dir():
        raise ConfigError("output.directory: must be a directory")
    for parent in config.output.directory.parents:
        if parent.exists():
            if not parent.is_dir():
                raise ConfigError(
                    f"output.directory: parent is not a directory: {parent}"
                )
            break


def _source(data, base):
    kind = _choice(
        data.get("kind"),
        "source.kind",
        (
            "spectral_history",
            "flux_table",
            "representative",
            "band_powerlaw",
            "powerlaw",
            "exponential_decay",
        ),
    )
    common = {"kind"}
    mode_fields = SOURCE_FIELDS
    selected = mode_fields[kind]
    if kind == "flux_table":
        spectral = _choice(
            data.get("spectral_mode"),
            "source.spectral_mode",
            ("band_powerlaw", "representative"),
        )
        selected = selected - (
            {"effective_energy_kev"}
            if spectral == "band_powerlaw"
            else {"energy_edges_kev", "photon_index"}
        )
    required = selected - {"photon_index", "baseline_band_flux", "flux_scale"}
    _table(data, f"source (kind={kind})", common | selected, common | required)
    values = {"kind": kind}
    for name, value in data.items():
        field_name = "source." + name
        if name == "kind":
            continue
        if name == "file":
            values[name] = _path(value, field_name, base)
        elif name in {
            "time_edges_days",
            "energy_edges_kev",
            "effective_energy_kev",
            "photon_flux",
            "peak_band_flux",
            "baseline_band_flux",
        }:
            values[name] = _vector(
                value,
                field_name,
                minimum=0,
                edges=name in {"time_edges_days", "energy_edges_kev"},
            )
            if "energy" in name and min(values[name]) <= 0:
                raise ConfigError(f"{field_name}: energies must be positive")
        elif name == "band_flux":
            if not isinstance(value, list) or not value:
                raise ConfigError(f"{field_name}: expected an array of rows")
            values[name] = tuple(_vector(row, field_name, minimum=0) for row in value)
        elif name == "time_column":
            values[name] = _integer(value, field_name)
        elif name == "flux_columns":
            if not isinstance(value, list) or not value:
                raise ConfigError(f"{field_name}: expected column numbers")
            values[name] = tuple(_integer(x, field_name) for x in value)
        elif name in {"photon_index", "epoch_mjd"}:
            values[name] = _number(value, field_name)
        elif name in {
            "decay_time_days",
            "last_interval_days",
            "max_interval_days",
            "flux_scale",
        }:
            values[name] = _number(value, field_name, minimum=0, strictly=True)
        else:
            choices = {
                "column_numbering": ("one_based_full_table",),
                "time_format": ("mjd",),
                "time_scale": ("UTC", "TT", "TDB", "TAI", "unspecified"),
                "interval_policy": ("hold_until_next",),
                "missing_data": ("error",),
                "flux_convention": ("unabsorbed observer-equivalent",),
                "flux_unit": ("ph cm^-2 s^-1",),
                "spectral_mode": ("band_powerlaw", "representative"),
            }
            values[name] = _choice(value, field_name, choices[name])
    source = SourceConfig(**values)
    nband = (
        len(source.energy_edges_kev) - 1
        if source.energy_edges_kev
        else len(source.effective_energy_kev)
    )
    ntime = len(source.time_edges_days) - 1
    if source.band_flux and (
        len(source.band_flux) != ntime
        or any(len(row) != nband for row in source.band_flux)
    ):
        raise ConfigError(
            "source.band_flux: expected (number of time intervals, number of bands)"
        )
    if source.photon_flux and len(source.photon_flux) != ntime:
        raise ConfigError("source.photon_flux: expected one flux per time interval")
    if kind == "exponential_decay" and (
        len(source.peak_band_flux) != nband
        or (source.baseline_band_flux and len(source.baseline_band_flux) != nband)
    ):
        raise ConfigError("source: decay flux vectors must match effective energies")
    if kind == "flux_table":
        columns = (source.time_column, *source.flux_columns)
        if len(set(columns)) != len(columns) or len(source.flux_columns) != nband:
            raise ConfigError(
                "source.flux_columns: distinct from time_column, with one column per band"
            )
        if source.last_interval_days > source.max_interval_days:
            raise ConfigError(
                "source.last_interval_days: must be <= source.max_interval_days"
            )
    if source.band_flux and not any(x > 0 for row in source.band_flux for x in row):
        raise ConfigError("source.band_flux: at least one cell must have positive flux")
    if source.photon_flux and not any(x > 0 for x in source.photon_flux):
        raise ConfigError(
            "source.photon_flux: at least one interval must have positive flux"
        )
    if kind == "exponential_decay":
        baseline = source.baseline_band_flux or (0.0,) * nband
        if any(a < b for a, b in zip(source.peak_band_flux, baseline, strict=True)):
            raise ConfigError(
                "source.peak_band_flux: must be >= source.baseline_band_flux"
            )
        if not any(x > 0 for x in source.peak_band_flux):
            raise ConfigError(
                "source.peak_band_flux: at least one band must have positive flux"
            )
    return source


def validate_relationships(config):
    """Reject host-only conflicts before importing a numerical backend."""
    source, observer = config.source, config.observer
    energies = source.energy_edges_kev or source.effective_energy_kev
    if energies and (
        observer.energy_edges_kev[0] > min(energies)
        or observer.energy_edges_kev[-1] < max(energies)
    ):
        raise ConfigError(
            "observer.energy_edges_kev: must cover the full source energy support"
        )
    if (
        source.time_edges_days
        and observer.arrival.edges_days[-1] <= source.time_edges_days[-1]
    ):
        raise ConfigError(
            "observer.arrival: must extend beyond the final source emission time"
        )
    if config.plots.epoch_days:
        indices = []
        for edge in config.plots.epoch_days:
            matches = [
                i
                for i, x in enumerate(observer.arrival.edges_days)
                if math.isclose(edge, x, rel_tol=0, abs_tol=1e-9)
            ]
            if len(matches) != 1:
                raise ConfigError(
                    "plots.epoch_days: each bound must align unambiguously with one observer bin edge"
                )
            indices.append(matches[0])
        if indices[1] <= indices[0]:
            raise ConfigError(
                "plots.epoch_days: must select at least one complete observer bin"
            )
    if (config.plots.energy_selection == "nearest") != (
        config.plots.energy_kev is not None
    ):
        raise ConfigError(
            "plots.energy_kev: required only for energy_selection=nearest"
        )
    if (config.plots.crop == "centered") != (config.plots.fov_arcsec is not None):
        raise ConfigError("plots.fov_arcsec: required only for crop=centered")


def load_run_config(path: str | Path) -> ResolvedRun:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = config_path.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"{config_path}: {error}") from error
    allowed = {
        "config_version",
        "run",
        "scene",
        "source",
        "materials",
        "execution",
        "observer",
        "quality",
        "output",
        "plots",
        "logging",
        "slurm",
    }
    _table(
        data,
        "config",
        allowed,
        {"config_version", "run", "scene", "source", "materials", "observer", "output"},
    )
    if type(data["config_version"]) is not int or data["config_version"] != 1:
        raise ConfigError("config_version: only integer version 1 is supported")
    run = _table(data["run"], "run", {"name", "workflow"}, {"name"})
    name = run["name"]
    if not isinstance(name, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name
    ):
        raise ConfigError("run.name: use 1–80 letters, numbers, underscores or hyphens")
    workflow = _choice(run.get("workflow", "simulate"), "run.workflow", ("simulate",))
    base = config_path.parent
    scene = _table(
        data["scene"],
        "scene",
        {f.name for f in fields(SceneConfig)},
        {"kind", "source_distance_kpc"},
    )
    kind = _choice(
        scene["kind"], "scene.kind", ("fits", "example_four_cloud", "uniform_disc")
    )
    disc_fields = {f.name for f in fields(SceneConfig)} - {
        "kind",
        "cloud_file",
        "source_distance_kpc",
        "coordinates",
    }
    disc_values = {}
    if kind == "uniform_disc":
        _table(
            scene,
            "scene",
            {"kind", "source_distance_kpc", "coordinates"} | disc_fields,
            {"kind", "source_distance_kpc"} | disc_fields,
        )
        for key in disc_fields:
            if key.startswith("grid_"):
                values = _vector(scene[key], "scene." + key)
                if len(values) != 3 or values[2] <= 0 or values[1] <= values[0]:
                    raise ConfigError(
                        f"scene.{key}: expected [start, exclusive_stop, positive_step]"
                    )
                count = (values[1] - values[0]) / values[2]
                if not math.isfinite(count) or count <= 1 or count > 1_000_000:
                    raise ConfigError(f"scene.{key}: requires 2 to one million centers")
                disc_values[key] = values
            else:
                disc_values[key] = _number(
                    scene[key],
                    "scene." + key,
                    minimum=0
                    if key
                    in {"disc_radius_arcsec", "disc_thickness_kpc", "disc_n_h_cm3"}
                    else None,
                    strictly=key in {"disc_radius_arcsec", "disc_thickness_kpc"},
                )
    elif set(scene) & disc_fields:
        raise ConfigError("scene: disc settings require kind=uniform_disc")
    if (kind == "fits") != ("cloud_file" in scene):
        raise ConfigError("scene.cloud_file: required only for scene.kind=fits")
    scene_config = SceneConfig(
        kind,
        _number(
            scene["source_distance_kpc"],
            "scene.source_distance_kpc",
            minimum=0,
            strictly=True,
        ),
        _path(scene["cloud_file"], "scene.cloud_file", base)
        if kind == "fits"
        else None,
        _choice(
            scene.get("coordinates", "source_centered_offsets"),
            "scene.coordinates",
            ("source_centered_offsets",),
        ),
        **disc_values,
    )
    if not isinstance(data["source"], dict):
        raise ConfigError("source: expected a TOML table")
    source = _source(data["source"], base)
    material = _table(
        data["materials"],
        "materials",
        {f.name for f in fields(MaterialConfig)},
        {"kind"},
    )
    kind = _choice(
        material["kind"], "materials.kind", ("table_pair", "declared_bundle")
    )
    selected = (
        {"directory"}
        if kind == "declared_bundle"
        else {"scattering_file", "absorption_file", "grid_file", "absorption_enabled"}
    )
    _table(
        material,
        "materials",
        selected | {"kind"},
        (selected - {"grid_file", "absorption_enabled"}) | {"kind"},
    )
    material_config = MaterialConfig(
        kind,
        **{
            key: _boolean(value, "materials.absorption_enabled")
            if key == "absorption_enabled"
            else _path(value, "materials." + key, base, directory=key == "directory")
            for key, value in material.items()
            if key != "kind"
        },
    )
    execution = _table(
        data.get("execution", {}),
        "execution",
        {f.name for f in fields(ExecutionConfig)},
    )
    execution_values = {}
    for key, value in execution.items():
        if key == "device":
            execution_values[key] = _choice(value, "execution.device", ("cpu", "gpu"))
        elif key == "launch_sampler":
            execution_values[key] = _choice(
                value, "execution.launch_sampler", ("radial", "rectangular")
            )
        else:
            execution_values[key] = _integer(
                value,
                "execution." + key,
                minimum=0 if key == "seed" else 1,
                maximum=2**32 - 1 if key == "seed" else None,
            )
    observer = _table(
        data["observer"],
        "observer",
        {"sky_grid", "energy_edges_kev", "arrival"},
        {"energy_edges_kev", "arrival"},
    )
    arrival = _table(
        observer["arrival"],
        "observer.arrival",
        {"kind", "edges_days", "start_days", "stop_days", "step_days", "last_bin"},
        {"kind"},
    )
    arrival_kind = _choice(
        arrival["kind"], "observer.arrival.kind", ("explicit", "uniform")
    )
    if arrival_kind == "explicit":
        _table(
            arrival, "observer.arrival", {"kind", "edges_days"}, {"kind", "edges_days"}
        )
        arrival_config = ArrivalConfig(
            arrival_kind,
            _vector(
                arrival["edges_days"],
                "observer.arrival.edges_days",
                minimum=0,
                edges=True,
            ),
        )
    else:
        _table(
            arrival,
            "observer.arrival",
            {"kind", "start_days", "stop_days", "step_days", "last_bin"},
            {"kind", "start_days", "stop_days", "step_days"},
        )
        start = _number(arrival["start_days"], "observer.arrival.start_days", minimum=0)
        stop = _number(
            arrival["stop_days"],
            "observer.arrival.stop_days",
            minimum=start,
            strictly=True,
        )
        step = _number(
            arrival["step_days"], "observer.arrival.step_days", minimum=0, strictly=True
        )
        last = _choice(
            arrival.get("last_bin", "shorten"),
            "observer.arrival.last_bin",
            ("shorten",),
        )
        ratio = (stop - start) / step
        if not math.isfinite(ratio) or ratio > 1_000_000:
            raise ConfigError(
                "observer.arrival: more than one million bins; use a larger step"
            )
        count = math.ceil(ratio)
        interior = tuple(start + i * step for i in range(count))
        if len(interior) > 1 and math.isclose(
            interior[-1], stop, rel_tol=0, abs_tol=8 * math.ulp(stop)
        ):
            interior = interior[:-1]
        edges = interior + (stop,)
        arrival_config = ArrivalConfig(arrival_kind, edges, start, stop, step, last)
    if arrival_config.edges_days[0] != 0:
        raise ConfigError(
            "observer.arrival: initial configured interface requires bins beginning at zero"
        )
    energies = _vector(
        observer["energy_edges_kev"], "observer.energy_edges_kev", minimum=0, edges=True
    )
    if energies[0] <= 0:
        raise ConfigError("observer.energy_edges_kev: must be positive")
    observer_config = ObserverConfig(
        energies,
        arrival_config,
        _choice(observer.get("sky_grid", "cloud"), "observer.sky_grid", ("cloud",)),
    )
    quality = _table(
        data.get("quality", {}), "quality", {f.name for f in fields(QualityConfig)}
    )
    quality_config = QualityConfig(
        **{key: _boolean(value, "quality." + key) for key, value in quality.items()}
    )
    output = _table(
        data["output"], "output", {"directory", "formats", "overwrite"}, {"directory"}
    )
    formats = output.get("formats", ["npz", "fits"])
    if (
        not isinstance(formats, list)
        or not formats
        or len({str(x) for x in formats}) != len(formats)
    ):
        raise ConfigError("output.formats: expected a nonempty list without duplicates")
    formats = tuple(_choice(x, "output.formats", ("npz", "fits")) for x in formats)
    output_config = OutputConfig(
        _path(output["directory"], "output.directory", base, output=True),
        formats,
        _boolean(output.get("overwrite", False), "output.overwrite"),
    )
    plots = _table(data.get("plots", {}), "plots", {f.name for f in fields(PlotConfig)})
    plot_values = {}
    for key, value in plots.items():
        if key in {
            "enabled",
            "save_scored_event_map",
            "save_count_plots",
            "density_per_kev",
        }:
            plot_values[key] = _boolean(value, "plots." + key)
        elif key == "dpi":
            plot_values[key] = _integer(value, "plots.dpi", maximum=1200)
        elif key == "epoch_days":
            plot_values[key] = _vector(value, "plots.epoch_days", minimum=0, edges=True)
            if len(plot_values[key]) != 2:
                raise ConfigError("plots.epoch_days: expected exactly [start, stop]")
        elif key == "scales":
            if (
                not isinstance(value, list)
                or not value
                or len({str(x) for x in value}) != len(value)
            ):
                raise ConfigError("plots.scales: expected a nonempty unique list")
            plot_values[key] = tuple(
                _choice(x, "plots.scales", ("log", "linear")) for x in value
            )
        elif key in {"energy_kev", "fov_arcsec"}:
            plot_values[key] = _number(value, "plots." + key, minimum=0, strictly=True)
        else:
            choices = {
                "energy_selection": ("all", "nearest"),
                "crop": ("full", "centered"),
                "quantity": ("intensity", "fluence"),
            }
            plot_values[key] = _choice(value, "plots." + key, choices[key])
    plots_config = PlotConfig(**plot_values)
    logging = _table(
        data.get("logging", {}), "logging", {"progress_every_chunks", "file"}
    )
    log_name = logging.get("file", "run.log")
    if not isinstance(log_name, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+\.log", log_name
    ):
        raise ConfigError("logging.file: expected a simple filename ending in .log")
    logging_config = LoggingConfig(
        _integer(
            logging.get("progress_every_chunks", 5), "logging.progress_every_chunks"
        ),
        log_name,
    )
    slurm_config = None
    if "slurm" in data:
        from ..cluster import parse_slurm

        slurm_config = parse_slurm(
            data["slurm"], base, execution_values.get("device", "cpu")
        )
    config = ResolvedRun(
        config_path,
        hashlib.sha256(raw).hexdigest(),
        raw,
        name,
        scene_config,
        source,
        material_config,
        observer_config,
        output_config,
        ExecutionConfig(**execution_values),
        quality_config,
        plots_config,
        logging_config,
        workflow=workflow,
        slurm=slurm_config,
    )
    validate_relationships(config)
    for path in input_paths(config):
        if not path.is_file():
            raise ConfigError(f"inputs: required input or sidecar not found: {path}")
    validate_destinations(config)
    return config
