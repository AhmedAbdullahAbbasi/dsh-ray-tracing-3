"""Input-file CLI; the existing dsh-v1 entry remains a compatibility runner."""

from __future__ import annotations

import argparse
import json

from .build import build_run
from .config.load import load_run_config
from .products.audit import audit_configured_run
from .run import run_configured_simulation


def main(argv=None):
    parser = argparse.ArgumentParser(description="DSH ideal-observer file-input runner")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run", "audit"):
        commands.add_parser(name).add_argument("config")
    snapshot = commands.add_parser(
        "snapshot", help="extract audited line-source snapshots"
    )
    snapshot.add_argument("config")
    snapshot.add_argument("--output-dir", type=str)
    snapshot.add_argument("--days", type=int, nargs="+", default=[3, 6, 9])
    snapshot.add_argument("--exposure-days", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "snapshot":
        from pathlib import Path

        from .products.snapshots import extract_snapshots

        report = extract_snapshots(
            Path(args.config),
            Path(args.output_dir) if args.output_dir else None,
            days=tuple(args.days),
            exposure_days=args.exposure_days,
        )
        for item in report["snapshots"]:
            print(
                f"{item['file']}: {item['scored_event_count']:,} scored events, "
                f"{item['total_fluence_ph_cm2']:.7g} ph cm^-2"
            )
        return
    config = load_run_config(args.config)
    if args.command == "audit":
        report = audit_configured_run(config)
        destination = config.output_npz.with_suffix(".audit_report.json")
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Saved: {destination}; product integrity passed: {report['all_passed']}")
        if not report["all_passed"]:
            raise SystemExit(1)
        return
    source, cells, material, cloud, launch, bins = build_run(config)
    print(f"Run: {config.name}; source: {config.source_fits}")
    print(f"Selected source fluence: {float(cells.total_fluence):.8g} ph cm^-2")
    print(
        f"Material SHA-256: {material.scattering_sha256} / {material.absorption_sha256}"
    )
    print(f"Cloud (z,y,x): {tuple(cloud.delta_nh_cm2.shape)}")
    print(f"Launch solid angle: {float(launch.launch_solid_angle_sr):.8g} sr")
    cube_shape = (
        len(bins.arrival_time_edges_s) - 1,
        len(bins.energy_edges_kev) - 1,
        len(bins.sky_y_edges_arcsec) - 1,
        len(bins.sky_x_edges_arcsec) - 1,
    )
    print(f"Observer cube (t,E,y,x): {cube_shape}")
    if args.command == "run":
        # The runner rechecks inputs immediately before transport, preserving
        # the same boundary for API and command-line callers.
        report = run_configured_simulation(config)
        print(f"Saved: {config.output_npz} and {config.output_fits}")
        print(f"Numerical status passed: {report['numerical_passed']}")


if __name__ == "__main__":
    main()
