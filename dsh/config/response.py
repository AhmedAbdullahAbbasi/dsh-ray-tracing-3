"""Strict host-only configuration for collecting and folding physical responses."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .load import (
    ConfigError,
    _boolean,
    _integer,
    _path,
    _source,
    _table,
    _vector,
    input_paths,
    parse_execution,
    parse_material,
    parse_scene,
    validate_destinations,
)
from .schema import (
    ExecutionConfig,
    LoggingConfig,
    OutputConfig,
    PlotConfig,
    QualityConfig,
)


@dataclass(frozen=True)
class ResponseSettings:
    file: Path | None = None
    covariance: bool = True
    history_block_size: int = 2048
    scattering_orders: tuple[int, ...] | None = None
    arrival_time_edges_days: tuple[float, ...] = ()
    energy_edges_kev: tuple[float, ...] = ()
    radial_edges_arcsec: tuple[float, ...] = ()


@dataclass(frozen=True)
class ResponseRun:
    config_path: Path
    config_sha256: str
    config_bytes: bytes = field(repr=False)
    name: str
    workflow: str
    source: object
    output: OutputConfig
    execution: ExecutionConfig
    response: ResponseSettings
    scene: object = None
    materials: object = None
    slurm: object = None
    config_version: int = 1
    quality: QualityConfig = field(default_factory=QualityConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self):
        def convert(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        data = asdict(self)
        del data["config_bytes"]
        return convert(data)


def load_response_config(data, path, raw):
    collect = data["run"]["workflow"] == "response_collect"
    common = {"config_version", "run", "source", "output", "execution", "slurm"}
    selected = {"scene", "materials"} if collect else {"response", "observer"}
    _table(
        data,
        "config",
        common | selected,
        {"config_version", "run", "source", "output"} | selected,
    )
    if type(data["config_version"]) is not int or data["config_version"] != 1:
        raise ConfigError("config_version: only integer version 1 is supported")
    run = _table(data["run"], "run", {"name", "workflow"}, {"name", "workflow"})
    name = run["name"]
    if not isinstance(name, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name
    ):
        raise ConfigError("run.name: use 1–80 letters, numbers, underscores or hyphens")
    base = path.parent
    if not isinstance(data["source"], dict):
        raise ConfigError("source: expected a TOML table")
    source = _source(data["source"], base)
    if source.kind not in {
        "spectral_history",
        "powerlaw",
        "band_powerlaw",
        "flux_table",
    } or (source.kind == "flux_table" and source.spectral_mode != "band_powerlaw"):
        raise ConfigError("source: response workflows require a continuous spectrum")
    execution_data = parse_execution(data, base)
    if not collect and set(data.get("execution", {})) - {"device"}:
        raise ConfigError(
            "execution: folding accepts only device; it launches no photons"
        )
    execution = ExecutionConfig(**execution_data)
    if not collect and execution.device != "cpu":
        raise ConfigError("execution.device: response_fold uses CPU NumPy; select cpu")
    if collect and (execution.packets < 2 or execution.chunk_size < 2):
        raise ConfigError(
            "execution: response collection needs at least two packets and chunk_size >= 2"
        )
    output = _table(data["output"], "output", {"directory", "overwrite"}, {"directory"})
    output = OutputConfig(
        _path(output["directory"], "output.directory", base, output=True),
        ("npz",),
        _boolean(output.get("overwrite", False), "output.overwrite"),
    )
    settings = ResponseSettings()
    if not collect:
        response = _table(
            data["response"],
            "response",
            {"file", "covariance", "history_block_size", "scattering_orders"},
            {"file"},
        )
        orders = response.get("scattering_orders")
        if orders is not None:
            if not isinstance(orders, list) or not orders:
                raise ConfigError("response.scattering_orders: expected nonempty list")
            orders = tuple(_integer(v, "response.scattering_orders") for v in orders)
            if len(set(orders)) != len(orders):
                raise ConfigError("response.scattering_orders: duplicate orders")
        observer = _table(
            data["observer"],
            "observer",
            {"arrival_time_edges_days", "energy_edges_kev", "radial_edges_arcsec"},
            {"arrival_time_edges_days", "energy_edges_kev", "radial_edges_arcsec"},
        )
        axes = {
            key: _vector(value, "observer." + key, minimum=0, edges=True)
            for key, value in observer.items()
        }
        if axes["energy_edges_kev"][0] <= 0:
            raise ConfigError("observer.energy_edges_kev: must be positive")
        covariance = _boolean(response.get("covariance", True), "response.covariance")
        bins = 1
        for axis in axes.values():
            bins *= len(axis) - 1
        if covariance and bins > 2048:
            raise ConfigError(
                "response.covariance: dense covariance limited to 2048 bins; set false"
            )
        settings = ResponseSettings(
            file=_path(response["file"], "response.file", base),
            covariance=covariance,
            history_block_size=_integer(
                response.get("history_block_size", 2048), "response.history_block_size"
            ),
            scattering_orders=orders,
            **axes,
        )
    slurm = None
    if "slurm" in data:
        from ..cluster import parse_slurm

        slurm = parse_slurm(data["slurm"], base, execution.device)
    config = ResponseRun(
        path,
        hashlib.sha256(raw).hexdigest(),
        raw,
        name,
        run["workflow"],
        source,
        output,
        execution,
        settings,
        parse_scene(data, base) if collect else None,
        parse_material(data, base) if collect else None,
        slurm,
    )
    for file in input_paths(config):
        if not file.is_file():
            raise ConfigError(f"inputs: required input or sidecar not found: {file}")
    validate_destinations(config)
    return config
