"""Compatibility entry point for sim_config.py or an explicit TOML run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", nargs="?", help="optional TOML file; otherwise use sim_config.py"
    )
    parser.add_argument(
        "--legacy-config", type=Path, help="explicit historical Python settings file"
    )
    parser.add_argument(
        "--check", action="store_true", help="validate without transport or outputs"
    )
    args = parser.parse_args(argv)
    if args.config and args.legacy_config:
        parser.error("choose a TOML file or --legacy-config, not both")
    if args.config:
        from dsh.command import main as configured_main

        return configured_main(["check" if args.check else "run", args.config])
    try:
        # Importing this module does not execute sim_config or initialize JAX.
        # Executing the historical entry point explicitly opts into Python settings.
        if args.legacy_config:
            import runpy
            from types import SimpleNamespace

            settings = SimpleNamespace(
                **runpy.run_path(str(args.legacy_config.resolve()))
            )
        else:
            import sim_config as settings
        from dsh.command import bootstrap

        bootstrap(settings.DEVICE)
        from dsh.compat import run_root

        return run_root(settings, check=args.check)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"run_simulation: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
