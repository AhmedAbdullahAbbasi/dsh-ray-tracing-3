"""Controlled 5.35 keV FITS-input comparison with the independent Stage 9F reference.

The scientific comparison uses Stage 9F's declared finite launch cone. The
separate configured run exercises the automatic full-cloud launch cone and its
saved products; its audit is an integrity check, not a science acceptance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from dsh.config import (
    audit_configured_run,
    build_run,
    load_run_config,
    run_configured_simulation,
)
from dsh.materials import packaged_material_paths
from dsh.physics.materials import load_2_10_material_tables
from dsh.sources.source_fits import SourceFluxFile, write_source_fits
from dsh.validation.absorbed_observer import host_material
from dsh.validation.heterogeneous_observer import (
    LAUNCH_BOUNDS,
    RADIAL_EDGES_KPC,
    SKY_EDGES,
    scene_columns,
)
from scripts.run_heterogeneous_observer_validation import run_case


def write_cloud(path: Path, columns: np.ndarray) -> None:
    """Preserve the reference radial cells and quadrant edges in native FITS WCS."""
    primary = fits.PrimaryHDU(columns)
    primary.header["BUNIT"] = "cm-2"
    for axis, kind, unit, start, increment in (
        (1, "XOFFSET", "arcsec", -900.0, 1800.0),
        (2, "YOFFSET", "arcsec", -900.0, 1800.0),
        (3, "DISTANCE", "kpc", 2.5, 1.0),
    ):
        primary.header[f"CTYPE{axis}"] = kind
        primary.header[f"CUNIT{axis}"] = unit
        primary.header[f"CRPIX{axis}"] = 1.0
        primary.header[f"CRVAL{axis}"] = start
        primary.header[f"CDELT{axis}"] = increment
    primary.writeto(path, overwrite=True, checksum=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--packets", type=int, default=100_000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[912, 319, 141])
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--configured-packets", type=int, default=8192)
    args = parser.parse_args()
    if (
        args.packets < 2
        or args.chunk_size < 1
        or args.configured_packets < 1
        or len(args.seeds) < 3
        or len(set(args.seeds)) != len(args.seeds)
    ):
        parser.error(
            "positive packet counts and at least three distinct seeds are required"
        )

    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    scattering, absorption, grid = packaged_material_paths("2-10")
    # Obtain the reference column normalization from the pinned scattering table.
    _, _, physics = load_2_10_material_tables()
    energy = 5.35
    tau = 1.5
    sigma_scattering = host_material(physics, energy)[3]
    columns = scene_columns(tau / sigma_scattering)
    cloud_path = directory / "heterogeneous_cloud.fits"
    write_cloud(cloud_path, columns)
    source_path = write_source_fits(
        directory / "one_second_line.fits",
        SourceFluxFile(
            time_edges_s=np.array([0.0, 1.0]),
            continuum_energy_edges_kev=np.array([]),
            continuum_flux=np.empty((1, 0)),
            continuum_shape=(),
            continuum_photon_index=np.empty((1, 0)),
            line_energy_kev=np.array([energy]),
            line_flux=np.array([[1.0]]),
            line_labels=("5.35_keV",),
        ),
    )
    config_path = directory / "line_heterogeneous.toml"
    config_path.write_text(
        f'''format_version = 1
[run]
name = "controlled_heterogeneous_line"
packets = {args.configured_packets}
chunk_size = {args.chunk_size}
max_interactions = 32
seed = {args.seeds[0]}
[scene]
kind = "fits"
path = "heterogeneous_cloud.fits"
source_distance_kpc = 10.0
[source]
file = "one_second_line.fits"
components = "lines"
[materials]
scattering = "{scattering.as_posix()}"
absorption = "{absorption.as_posix()}"
grid = "{grid.as_posix()}"
[observer]
time_edges_days = [0.0, 365.0]
energy_edges_kev = [5.25, 5.45]
[output]
npz = "configured.npz"
fits = "configured.fits"
''',
        encoding="utf-8",
    )
    config = load_run_config(config_path)
    source, cells, material, cloud, full_launch, _ = build_run(config)
    np.testing.assert_allclose(np.asarray(cloud.delta_nh_cm2), columns, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(cloud.z_edges_kpc), RADIAL_EDGES_KPC)
    np.testing.assert_allclose(np.asarray(cloud.x_edges_arcsec), SKY_EDGES)
    np.testing.assert_allclose(np.asarray(cloud.y_edges_arcsec), SKY_EDGES)
    np.testing.assert_allclose(
        np.asarray(material.physics.scattering_cross_section_cm2_per_h),
        np.asarray(physics.scattering_cross_section_cm2_per_h),
    )
    if source.line_fluence != 1.0 or float(cells.total_fluence) != 1.0:
        raise ValueError("file-loaded line fluence differs from the reference")

    print("Running configured full-cloud input/output and archive audit...", flush=True)
    configured = run_configured_simulation(config)
    audit = audit_configured_run(config)
    print(
        "Comparing file-loaded inputs with independent Stage 9F references...",
        flush=True,
    )
    case = run_case(
        material.physics,
        energy=energy,
        target_tau=tau,
        packets=args.packets,
        chunk_size=args.chunk_size,
        seeds=args.seeds,
        max_interactions=32,
        sigma_limit=5.0,
        minimum_effective_histories=30,
        maximum_relative_error=0.10,
        input_cloud=cloud,
        source_cells=cells,
    )
    result = {
        "validation": "file_input_heterogeneous_5.35_keV",
        "config_path": str(config_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "cloud_sha256": hashlib.sha256(cloud_path.read_bytes()).hexdigest(),
        "scattering_sha256": material.scattering_sha256,
        "absorption_sha256": material.absorption_sha256,
        "file_input_observer": {
            "scope": "automatic full-cloud launch cone; output and integrity only",
            "slope_x_bounds": np.asarray(full_launch.slope_x_bounds).tolist(),
            "slope_y_bounds": np.asarray(full_launch.slope_y_bounds).tolist(),
            "run_report": configured,
            "audit_checks": audit["checks"],
            "audit_passed": audit["all_passed"],
        },
        "independent_science_reference": {
            "scope": (
                "file-loaded source/cloud/material with Stage 9F finite launch cone"
            ),
            "packets_per_seed": args.packets,
            "seeds": args.seeds,
            "source_emission_interval_s": [0.0, 1.0],
            "finite_launch_slope_bounds": LAUNCH_BOUNDS,
            "case": case,
        },
        "all_passed": bool(
            configured["numerical_passed"]
            and audit["all_passed"]
            and case["all_passed"]
        ),
    }
    output = directory / "validation_report.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output}; all_passed={result['all_passed']}", flush=True)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
