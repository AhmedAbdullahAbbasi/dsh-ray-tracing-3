"""Immutable host-side run configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
