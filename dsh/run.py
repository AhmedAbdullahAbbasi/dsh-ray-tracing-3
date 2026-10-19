"""Execute an existing plan and publish a complete, provenance-bearing run."""

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from jax import random

from .config.load import validate_destinations
from .io.fits_output import write_ideal_observer_fits
from .io.npz_output import write_ideal_observer_npz
from .io.provenance import hash_files, verify_inputs
from .pipeline import run_tabulated_source_to_observer_chunked
from .plots import save_plots
from .quality import evaluate_quality


def execute_plan(plan, *, progress_callback=None):
    """Numerical API: return products, with no file creation or plotting."""
    verify_inputs(plan.input_hashes)
    execution = plan.config.execution
    return run_tabulated_source_to_observer_chunked(
        random.PRNGKey(execution.seed),
        plan.source,
        plan.launch,
        plan.cloud,
        plan.physics,
        plan.bins,
        total_packets=execution.packets,
        chunk_size=execution.chunk_size,
        max_interactions=execution.max_interactions,
        progress_callback=progress_callback,
    )


def _json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _publish(source, destination, overwrite):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if overwrite:
            os.replace(source, destination)
        else:
            os.link(source, destination)
            source.unlink()
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        with tempfile.TemporaryDirectory(
            prefix=".dsh-publish-", dir=destination.parent
        ) as temporary:
            local = Path(temporary) / destination.name
            shutil.copyfile(source, local)
            _publish(local, destination, overwrite)
        source.unlink()


def publish_run(plan):
    config = plan.config
    mapping = plan.publication_paths

    def validate():
        validate_destinations(
            config, publication_paths=mapping, extra_inputs=plan.extra_inputs
        )

    validate()
    verify_inputs(plan.input_hashes)
    directory = config.output.directory
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "complete.json"
    if config.output.overwrite:
        marker.unlink(missing_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=".dsh-run-", dir=directory) as temporary:
        stage = Path(temporary)
        log_path = stage / config.logging.file
        with log_path.open("w", encoding="utf-8") as log:
            log.write(json.dumps(plan.summary, allow_nan=False) + "\n")

            def progress(completed, total):
                chunks = (
                    completed + config.execution.chunk_size - 1
                ) // config.execution.chunk_size
                if (
                    completed == total
                    or chunks % config.logging.progress_every_chunks == 0
                ):
                    message = f"Completed packets: {completed:,} / {total:,}"
                    print(message, flush=True)
                    log.write(message + "\n")
                    log.flush()

            result = execute_plan(plan, progress_callback=progress)
        quality = evaluate_quality(result, config.execution.packets)
        complete = quality["numerical_complete"]
        diagnostic = config.quality.allow_incomplete
        report = {
            "report_version": 1,
            "run": config.name,
            "status": "complete" if complete else "incomplete",
            "diagnostic_only": diagnostic,
            **quality,
            "elapsed_seconds": time.monotonic() - started,
            "config_sha256": config.config_sha256,
            "output_inventory": [],
        }
        paths = []
        if complete or diagnostic:
            prefix = "diagnostic_" if diagnostic else ""
            metadata = {
                **plan.metadata,
                "peak_band_fluxes": np.asarray(plan.source.band_flux)[0],
                "baseline_band_fluxes": config.source.baseline_band_flux
                or np.zeros(plan.source.band_flux.shape[1]),
                "config_sha256": config.config_sha256,
                "numerical_complete": complete,
                "diagnostic_only": diagnostic,
            }
            for kind in config.output.formats:
                path = stage / (prefix + config.name + "." + kind)
                writer = (
                    write_ideal_observer_npz
                    if kind == "npz"
                    else write_ideal_observer_fits
                )
                writer(
                    path,
                    result,
                    plan.bins,
                    plan.source,
                    plan.cloud,
                    plan.physics,
                    plan.launch,
                    run_metadata=metadata,
                )
                paths.append(path)
            paths += save_plots(plan, result, stage, prefix)
        if complete or diagnostic or config.quality.save_failure_diagnostics:
            original = stage / "original.toml"
            original.write_bytes(config.config_bytes)
            resolved = stage / "resolved.json"
            _json(
                resolved,
                {
                    "manifest_version": 1,
                    "configuration": config.to_dict(),
                    "input_sha256": plan.input_hashes,
                    "runtime": plan.metadata,
                    "publication_paths": {
                        key: str(value) for key, value in mapping.items()
                    },
                    "input_hash_policy": "SHA256 of raw bytes, checked before and after loading and before publication; grid recipe checks separately normalize CRLF.",
                },
            )
            paths += [original, resolved, log_path]
            report["output_inventory"] = [path.name for path in paths] + ["report.json"]
            report_path = stage / "report.json"
            _json(report_path, report)
            paths.append(report_path)
        verify_inputs(plan.input_hashes)
        validate()
        # Marker is always last. An interrupted group publication has no
        # production success marker, even if some individual files exist.
        inventory = {
            path.name: digest
            for name, digest in hash_files(paths).items()
            for path in [Path(name)]
        }
        for path in paths:
            _publish(
                path,
                Path(mapping.get(path.name, directory / path.name)),
                config.output.overwrite,
            )
        if complete and not diagnostic:
            completed = stage / "complete.json"
            _json(
                completed,
                {
                    "completion_manifest_version": 1,
                    "run": config.name,
                    "config_sha256": config.config_sha256,
                    "output_sha256": inventory,
                    "output_paths": {
                        name: str(mapping[name])
                        for name in inventory
                        if name in mapping
                    },
                },
            )
            _publish(completed, marker, config.output.overwrite)
    print(f"Run {report['status']}; products: {directory}", flush=True)
    return 0 if complete else 3
