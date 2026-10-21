"""Measure configured execution without exporting or altering production products."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import pstats
import statistics
import sys
import time
from dataclasses import fields, is_dataclass
from pathlib import Path

from .command import bootstrap
from .config import load_run_config
from .io.provenance import verify_inputs


def result_fingerprint(value):
    """Hash scientific arrays, dimensions and diagnostic values, excluding provenance."""
    import numpy as np

    digest = hashlib.sha256()

    def visit(item):
        if is_dataclass(item):
            for field in fields(item):
                if field.name != "provenance":
                    digest.update(field.name.encode())
                    visit(getattr(item, field.name))
        elif isinstance(item, (list, tuple)):
            for member in item:
                visit(member)
        elif item is None:
            digest.update(b"None")
        else:
            array = np.asarray(item)
            if array.dtype.hasobject:
                raise ValueError("profile fingerprint requires numerical results")
            digest.update(json.dumps([array.dtype.str, list(array.shape)]).encode())
            digest.update(np.ascontiguousarray(array).tobytes())

    visit(value)
    return digest.hexdigest()


def synchronize(value):
    """Wait for asynchronous JAX arrays before stopping an execution timer."""
    if is_dataclass(value):
        for field in fields(value):
            if field.name != "provenance":
                synchronize(getattr(value, field.name))
    elif isinstance(value, (list, tuple)):
        for member in value:
            synchronize(member)
    elif hasattr(value, "block_until_ready"):
        value.block_until_ready()


def profile_config(config_path, output_directory, *, repeats=3):
    if type(repeats) is not int or not 1 <= repeats <= 20:
        raise ValueError("repeats must be an integer from 1 to 20")
    output_directory = Path(output_directory).resolve()
    if output_directory.exists():
        raise ValueError(
            "profile output_directory must be new; choose a fresh directory"
        )
    started = time.perf_counter()
    config = load_run_config(config_path)
    if (
        output_directory == config.output.directory
        or config.output.directory in output_directory.parents
    ):
        raise ValueError(
            "profile output_directory must be outside the production output directory"
        )
    parse_seconds = time.perf_counter() - started
    bootstrap(config.execution.device)
    started = time.perf_counter()
    if config.workflow == "simulate":
        from .build import build_run_plan as build
        from .quality import evaluate_quality
        from .run import execute_plan as execute
    else:
        from .response_run import build_response_plan as build
        from .response_run import execute_response_plan as execute
    imports_seconds = time.perf_counter() - started
    output_directory.mkdir(parents=True)
    report = {
        "profile_version": 1,
        "status": "failed",
        "workflow": config.workflow,
        "config_sha256": config.config_sha256,
        "repeats": repeats,
        "production_products_published": False,
        "parse_seconds": parse_seconds,
        "numerical_import_seconds": imports_seconds,
        "timing_scope": "execution includes input SHA256 rechecks; excludes result fingerprinting, exports and plotting; first call includes compilation as required; repeat calls may also compile",
        "performance_improvement_claimed": False,
    }
    try:
        started = time.perf_counter()
        plan = build(config)
        report["build_seconds"] = time.perf_counter() - started
        report["runtime"] = plan.metadata
        report["input_sha256"] = plan.input_hashes
        report["plan_summary"] = plan.summary

        def run():
            result = execute(plan)
            synchronize(result)
            if (
                config.workflow == "simulate"
                and not evaluate_quality(result, config.execution.packets)[
                    "numerical_complete"
                ]
            ):
                raise ValueError(
                    "profile run has incomplete transport histories; increase max_interactions"
                )
            return result

        started = time.perf_counter()
        result = run()
        report["first_execution_seconds"] = time.perf_counter() - started
        baseline = result_fingerprint(result)
        del result
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            result = run()
            timings.append(time.perf_counter() - started)
            if result_fingerprint(result) != baseline:
                raise ValueError("seeded result changed between profile repetitions")
            del result
        report["repeat_execution_seconds"] = timings
        report["repeat_median_seconds"] = statistics.median(timings)
        report["scientific_result_sha256"] = baseline
        report["seeded_repeats_exact"] = True
        # Separate instrumented pass, so its overhead does not contaminate timings.
        profiler = cProfile.Profile()
        result = profiler.runcall(run)
        if result_fingerprint(result) != baseline:
            raise ValueError("instrumented execution changed scientific results")
        profiler.dump_stats(str(output_directory / "calls.pstats"))
        text = io.StringIO()
        pstats.Stats(profiler, stream=text).strip_dirs().sort_stats(
            "cumulative"
        ).print_stats(30)
        (output_directory / "calls.txt").write_text(text.getvalue(), encoding="utf-8")
        verify_inputs(plan.input_hashes)
        report["status"] = "passed"
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        report["error"] = str(error)
    (output_directory / "profile.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        report = profile_config(
            args.config, args.output_directory, repeats=args.repeats
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": str(args.output_directory.resolve() / "profile.json"),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0 if report["status"] == "passed" else 2
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"dsh profile: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
