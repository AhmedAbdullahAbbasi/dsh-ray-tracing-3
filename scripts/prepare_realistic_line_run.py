"""Write a line-only source FITS and pilot/full configurations for a cloud FITS.

The default scene matches the earlier 500x500 arcsec, 10 kpc four-cloud test
cube. Run ``dsh check`` on the generated TOML before launching either job.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from dsh.materials import packaged_material_paths
from dsh.sources.source_fits import SourceFluxFile, write_source_fits


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def prepare_run(
    cloud_fits: Path,
    output_dir: Path,
    *,
    energy_kev: float = 5.35,
    photon_flux: float = 0.038,
    flare_duration_s: float = 3600.0,
    source_distance_kpc: float = 10.5,
    pilot_packets: int = 8192,
    packets: int = 2_500_000,
) -> tuple[Path, Path]:
    """Build reproducible inputs; ``dsh check`` performs the full cube audit."""
    cloud = Path(cloud_fits).resolve(strict=True)
    if not cloud.is_file():
        raise ValueError("cloud_fits must be a file")
    for value, name in (
        (energy_kev, "energy_kev"),
        (photon_flux, "photon_flux"),
        (flare_duration_s, "flare_duration_s"),
        (source_distance_kpc, "source_distance_kpc"),
    ):
        _positive(value, name)
    if not 2.0 <= energy_kev <= 10.0:
        raise ValueError("line energy must lie within the pinned 2-10 keV tables")
    if flare_duration_s >= 60 * 86400:
        raise ValueError("flare must end before the final arrival bin")
    if any(
        isinstance(n, bool) or not isinstance(n, int) or n < 1
        for n in (pilot_packets, packets)
    ):
        raise ValueError("pilot and full packet counts must be positive integers")
    output = Path(output_dir).resolve()
    if output == cloud:
        raise ValueError("output directory cannot be the cloud FITS")
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / name).exists()
        for name in ("monochromatic_flare.fits", "pilot.toml", "full.toml")
    ):
        raise FileExistsError("run inputs already exist; choose a new output directory")
    scattering, absorption, grid = packaged_material_paths("2-10")
    source_path = write_source_fits(
        output / "monochromatic_flare.fits",
        SourceFluxFile(
            time_edges_s=np.array([0.0, flare_duration_s]),
            continuum_energy_edges_kev=np.array([]),
            continuum_flux=np.empty((1, 0)),
            continuum_shape=(),
            continuum_photon_index=np.empty((1, 0)),
            line_energy_kev=np.array([energy_kev]),
            line_flux=np.array([[photon_flux]]),
            line_labels=("monochromatic_flare",),
        ),
    )

    def quoted(path: Path) -> str:
        # Forward slashes make Windows drive paths valid TOML basic strings.
        return json.dumps(path.as_posix())

    def config_text(name: str, n: int, chunk: int) -> str:
        return f'''format_version = 1
[run]
name = "{name}"
packets = {n}
chunk_size = {chunk}
max_interactions = 32
seed = 2026
fail_on_cap = true

[scene]
kind = "fits"
path = {quoted(cloud)}
source_distance_kpc = {source_distance_kpc}

[source]
file = {quoted(source_path)}
components = "lines"

[materials]
scattering = {quoted(scattering)}
absorption = {quoted(absorption)}
grid = {quoted(grid)}

[observer]
time_edges_days = [0, 3, 4, 6, 7, 9, 10, 60]
energy_edges_kev = [{energy_kev - 0.1:.12g}, {energy_kev + 0.1:.12g}]

[output]
npz = "{name}.npz"
fits = "{name}.fits"
'''

    pilot_path = output / "pilot.toml"
    full_path = output / "full.toml"
    pilot_path.write_text(config_text("pilot", pilot_packets, 256), encoding="utf-8")
    full_path.write_text(config_text("full", packets, 512), encoding="utf-8")
    return pilot_path, full_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud-fits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--energy-kev", type=float, default=5.35)
    parser.add_argument("--photon-flux", type=float, default=0.038)
    parser.add_argument("--flare-duration-s", type=float, default=3600.0)
    parser.add_argument("--source-distance-kpc", type=float, default=10.5)
    parser.add_argument("--pilot-packets", type=int, default=8192)
    parser.add_argument("--packets", type=int, default=2_500_000)
    args = parser.parse_args()
    pilot, full = prepare_run(
        args.cloud_fits,
        args.output_dir,
        energy_kev=args.energy_kev,
        photon_flux=args.photon_flux,
        flare_duration_s=args.flare_duration_s,
        source_distance_kpc=args.source_distance_kpc,
        pilot_packets=args.pilot_packets,
        packets=args.packets,
    )
    print(f"Source FITS and configurations saved in {pilot.parent}")
    for path in (pilot, full):
        print(f"python -m dsh.command check {quoted_cli(path)}")
    print(f"Pilot: python -m dsh.command run {quoted_cli(pilot)}")
    print(f"Full:  python -m dsh.command run {quoted_cli(full)}")


def quoted_cli(path: Path) -> str:
    return f'"{path}"'


if __name__ == "__main__":
    main()
