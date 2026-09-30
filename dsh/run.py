"""Execute numerical run plans and write reproducible configured products."""

from __future__ import annotations

import hashlib
import json
import platform

import jax
import numpy as np
from jax import random

from dsh.contracts import RunPlan

from .build import build_run, plan_from_inputs
from .config.schema import ResolvedRun
from .core.pipeline import run_source_cells_to_observer_chunked
from .io.fits_output import write_ideal_observer_fits
from .io.npz_output import write_ideal_observer_npz
from .io.provenance import _file_sha256, _git_value


def run_plan(plan: RunPlan, *, progress_callback=None):
    """Run arrays only; no input files or formats enter the numerical pipeline."""
    return run_source_cells_to_observer_chunked(
        random.PRNGKey(plan.seed),
        plan.source,
        plan.launch,
        plan.cloud,
        plan.material,
        plan.observer,
        total_packets=plan.packets,
        chunk_size=plan.chunk_size,
        max_interactions=plan.max_interactions,
        progress_callback=progress_callback,
    )


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
    plan = plan_from_inputs(config, cells, material, cloud, launch, bins)
    result = run_plan(plan, progress_callback=progress_callback)
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
