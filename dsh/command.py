"""Input-file CLI; the existing dsh-v1 entry remains a compatibility runner."""

from __future__ import annotations

import argparse

from .config import build_run, load_run_config, run_configured_simulation


def main():
    parser = argparse.ArgumentParser(description="DSH ideal-observer file-input runner")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run"):
        commands.add_parser(name).add_argument("config")
    args = parser.parse_args()
    config = load_run_config(args.config)
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
