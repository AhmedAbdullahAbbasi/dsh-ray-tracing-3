"""Configuration-first entry point: choose the backend before numerical imports."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config import ConfigError, load_run_config


def bootstrap(device):
    platform = "cpu" if device == "cpu" else "cuda"
    if "jax" in sys.modules:
        import jax

        actual = jax.default_backend()
        expected = "cpu" if device == "cpu" else "gpu"
        if actual != expected:
            raise ConfigError(
                f"execution.device: JAX already uses {actual}; restart before selecting {device}"
            )
    else:
        os.environ["JAX_PLATFORMS"] = platform
        os.environ["JAX_PLATFORM_NAME"] = device


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check or run a TOML-configured simulation or physical halo response."
    )
    parser.add_argument(
        "command", choices=("check", "run", "slurm-plan", "slurm-submit")
    )
    parser.add_argument(
        "config", help="TOML run file; relative input paths use its directory"
    )
    args = parser.parse_args(argv)
    try:
        config = load_run_config(args.config)
        if args.command.startswith("slurm-"):
            from .cluster import job_plan, submit_job

            plan = job_plan(config)
            if args.command == "slurm-plan":
                print(json.dumps(plan, indent=2), flush=True)
                return 0
            return submit_job(config, plan)
        bootstrap(config.execution.device)
        if config.workflow in {"response_collect", "response_fold"}:
            from .response_run import build_response_plan, publish_response_run

            plan = build_response_plan(config)
            print(json.dumps(plan.summary, indent=2, allow_nan=False), flush=True)
            return 0 if args.command == "check" else publish_response_run(plan)
        from .build import build_run_plan

        plan = build_run_plan(config)
        print(json.dumps(plan.summary, indent=2, allow_nan=False), flush=True)
        if args.command == "check":
            return 0
        from .run import publish_run

        return publish_run(plan)
    except (ConfigError, ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"dsh: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
