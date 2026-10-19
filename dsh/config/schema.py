"""Immutable settings, separate from numerical arrays and JIT arguments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

SOURCE_FIELDS = {
    "spectral_history": {"file"},
    "representative": {"time_edges_days", "effective_energy_kev", "band_flux"},
    "band_powerlaw": {
        "time_edges_days",
        "energy_edges_kev",
        "band_flux",
        "photon_index",
    },
    "powerlaw": {
        "time_edges_days",
        "energy_edges_kev",
        "photon_flux",
        "photon_index",
    },
    "exponential_decay": {
        "time_edges_days",
        "effective_energy_kev",
        "peak_band_flux",
        "baseline_band_flux",
        "decay_time_days",
    },
    "flux_table": {
        "file",
        "time_column",
        "flux_columns",
        "column_numbering",
        "time_format",
        "time_scale",
        "epoch_mjd",
        "interval_policy",
        "last_interval_days",
        "max_interval_days",
        "missing_data",
        "flux_convention",
        "flux_unit",
        "flux_scale",
        "spectral_mode",
        "energy_edges_kev",
        "photon_index",
    },
}


@dataclass(frozen=True)
class SceneConfig:
    kind: str
    source_distance_kpc: float
    cloud_file: Path | None = None
    coordinates: str = "source_centered_offsets"


@dataclass(frozen=True)
class SourceConfig:
    kind: str
    file: Path | None = None
    time_edges_days: tuple[float, ...] = ()
    energy_edges_kev: tuple[float, ...] = ()
    effective_energy_kev: tuple[float, ...] = ()
    band_flux: tuple[tuple[float, ...], ...] = ()
    photon_flux: tuple[float, ...] = ()
    photon_index: float = 1.7
    peak_band_flux: tuple[float, ...] = ()
    baseline_band_flux: tuple[float, ...] = ()
    decay_time_days: float = 25.0
    time_column: int = 1
    flux_columns: tuple[int, ...] = ()
    column_numbering: str = "one_based_full_table"
    time_format: str = "mjd"
    time_scale: str = ""
    epoch_mjd: float | None = None
    interval_policy: str = "hold_until_next"
    last_interval_days: float | None = None
    max_interval_days: float | None = None
    missing_data: str = "error"
    flux_convention: str = "unabsorbed observer-equivalent"
    flux_unit: str = "ph cm^-2 s^-1"
    flux_scale: float = 1.0
    spectral_mode: str = "band_powerlaw"


@dataclass(frozen=True)
class MaterialConfig:
    kind: str
    directory: Path | None = None
    scattering_file: Path | None = None
    absorption_file: Path | None = None
    grid_file: Path | None = None


@dataclass(frozen=True)
class ExecutionConfig:
    packets: int = 4096
    chunk_size: int = 256
    max_interactions: int = 32
    seed: int = 2026
    device: str = "cpu"
    launch_sampler: str = "radial"


@dataclass(frozen=True)
class ArrivalConfig:
    kind: str
    edges_days: tuple[float, ...]
    start_days: float | None = None
    stop_days: float | None = None
    step_days: float | None = None
    last_bin: str = "shorten"


@dataclass(frozen=True)
class ObserverConfig:
    energy_edges_kev: tuple[float, ...]
    arrival: ArrivalConfig
    sky_grid: str = "cloud"


@dataclass(frozen=True)
class QualityConfig:
    allow_incomplete: bool = False
    save_failure_diagnostics: bool = True


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    formats: tuple[str, ...] = ("npz", "fits")
    overwrite: bool = False


@dataclass(frozen=True)
class PlotConfig:
    enabled: bool = False
    scales: tuple[str, ...] = ("log", "linear")
    energy_selection: str = "all"
    epoch_days: tuple[float, ...] = ()
    crop: str = "full"
    save_scored_event_map: bool = False
    dpi: int = 200


@dataclass(frozen=True)
class LoggingConfig:
    progress_every_chunks: int = 5
    file: str = "run.log"


@dataclass(frozen=True)
class ResolvedRun:
    config_path: Path
    config_sha256: str
    config_bytes: bytes = field(repr=False)
    name: str
    scene: SceneConfig
    source: SourceConfig
    materials: MaterialConfig
    observer: ObserverConfig
    output: OutputConfig
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    config_version: int = 1
    workflow: str = "simulate"

    def to_dict(self):
        """JSON-ready effective settings, including expanded defaults."""

        def convert(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(item) for item in value]
            return value

        data = asdict(self)
        del data["config_bytes"]
        active_source = {"kind"} | SOURCE_FIELDS[self.source.kind]
        data["source"] = {
            key: value for key, value in data["source"].items() if key in active_source
        }
        return convert(data)
