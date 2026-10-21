"""Execute an existing plan and publish a complete, provenance-bearing run."""

from __future__ import annotations

import json
import time

import numpy as np
from jax import random

from .io.fits_output import write_ideal_observer_fits
from .io.npz_output import write_ideal_observer_npz
from .io.provenance import verify_inputs
from .io.publication import publish_outputs, staged_run, write_run_records
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


def publish_run(plan):
    config = plan.config
    mapping = plan.publication_paths

    directory = config.output.directory
    with staged_run(
        config, plan.input_hashes, mapping=mapping, extra_inputs=plan.extra_inputs
    ) as stage:
        started = time.monotonic()
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
            paths = write_run_records(
                stage,
                config,
                plan.input_hashes,
                plan.metadata,
                report,
                paths,
                log_path=log_path,
                resolved_extra={
                    "publication_paths": {
                        key: str(value) for key, value in mapping.items()
                    },
                    "input_hash_policy": "SHA256 of raw bytes, checked before and after loading and before publication; grid recipe checks separately normalize CRLF.",
                },
            )
        publish_outputs(
            stage,
            config,
            plan.input_hashes,
            paths,
            complete=complete and not diagnostic,
            mapping=mapping,
            extra_inputs=plan.extra_inputs,
        )
    print(f"Run {report['status']}; products: {directory}", flush=True)
    return 0 if complete else 3
