"""Generate the compact JAX-ready v1 NewDust scattering table.

This is an offline preprocessing script.  NewDust/xdust, Astropy, and SciPy
are not runtime dependencies of the photon transport kernel.

The explicit 100-point *linear* grain-radius grid is important because newer
xdust releases changed their default size grid. Historical 4U 1630-47 screen
files were used only to identify and cross-check this configuration; no
observer-space screen kernel is stored in the generated transport table.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np

ENERGY_KEV = np.array([3.3, 4.9, 6.9], dtype=np.float64)
NH_REFERENCE_CM2 = 1.0e22
DUST_MASS_COLUMN_G_CM2 = 2.32475e-4
N_GRAIN_RADII = 100
N_POSITIVE_ANGLES = 8192
MIN_ANGLE_ARCSEC = 1.0e-3
ARCSEC_PER_RADIAN = 180.0 * 3600.0 / np.pi


def _git_source_revision(module_file: Path):
    for directory in module_file.resolve().parents:
        if not (directory / ".git").exists():
            continue
        try:
            description = subprocess.run(
                ["git", "-C", str(directory), "describe", "--tags", "--always"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            revision = subprocess.run(
                ["git", "-C", str(directory), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            return description, revision
        except (OSError, subprocess.CalledProcessError):
            break
    return None, None


def _load_newdust_api():
    try:
        import xdust
        from xdust import grainpop

        distribution = "xdust"
        module_file = Path(xdust.__file__)
    except ImportError:
        # Historical releases used the import name ``newdust``.
        import newdust
        from newdust import grainpop

        distribution = "newdust"
        module_file = Path(newdust.__file__)
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    source_description, source_revision = _git_source_revision(module_file)
    if version is None:
        version = source_description or "source-checkout"
    return grainpop, distribution, version, source_revision


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(output: Path) -> None:
    try:
        import astropy.units as u
    except ImportError as error:
        raise SystemExit("generation requires astropy") from error

    grainpop, distribution, version, source_revision = _load_newdust_api()

    theta = np.concatenate(
        [
            np.array([0.0]),
            np.geomspace(
                MIN_ANGLE_ARCSEC / ARCSEC_PER_RADIAN,
                np.pi,
                N_POSITIVE_ANGLES,
            ),
        ]
    )
    population = grainpop.make_MRN_RGDrude(
        amin=0.005,
        amax=0.3,
        p=3.5,
        rho=3.0,
        md=DUST_MASS_COLUMN_G_CM2,
        na=N_GRAIN_RADII,
        log=False,
    )
    population.calculate_ext(ENERGY_KEV, theta=theta * u.radian)
    differential = population.int_diff.to("sr^-1").value / NH_REFERENCE_CM2

    # Repeat the transport module's trapezoidal solid-angle integration here
    # so generating data does not itself require JAX.
    integrand = 2.0 * np.pi * np.sin(theta)[None, :] * differential
    increments = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * np.diff(theta)
    cumulative = np.concatenate(
        [np.zeros((ENERGY_KEV.size, 1)), np.cumsum(increments, axis=1)],
        axis=1,
    )
    sigma = cumulative[:, -1]
    cdf = cumulative / sigma[:, None]
    cdf[:, 0] = 0.0
    cdf[:, -1] = 1.0

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        energy_kev=ENERGY_KEV,
        scattering_angle_rad=theta,
        differential_cross_section_cm2_per_sr_per_h=differential,
        scattering_cross_section_cm2_per_h=sigma,
        scattering_angle_cdf=cdf,
    )

    reported_tau_sca = np.asarray(population.tau_sca, dtype=float)
    metadata = {
        "schema_version": 1,
        "quantity": "intrinsic_differential_cross_section",
        "differential_cross_section_unit": "cm2 sr-1 H-1",
        "integrated_cross_section_unit": "cm2 H-1",
        "energy_unit": "keV",
        "angle_unit": "rad",
        "energy_kev": ENERGY_KEV.tolist(),
        "producer": "NewDust/xdust",
        "producer_repository": "https://github.com/eblur/xdust",
        "producer_import": distribution,
        "producer_version": version,
        "producer_source_revision": source_revision,
        "model": "MRN power law + Rayleigh-Gans/Drude",
        "grain_radius_min_micron": 0.005,
        "grain_radius_max_micron": 0.3,
        "grain_power_law_index": 3.5,
        "grain_material_density_g_cm3": 3.0,
        "grain_radius_samples": N_GRAIN_RADII,
        "grain_radius_spacing": "linear",
        "dust_mass_column_g_cm2": DUST_MASS_COLUMN_G_CM2,
        "hydrogen_column_reference_cm2": NH_REFERENCE_CM2,
        "normalization": "NewDust d(tau)/dOmega divided by NH_reference",
        "scattering_cross_section_definition": (
            "solid-angle integral of the tabulated differential cross-section"
        ),
        "normalization_choice": (
            "use the integral of dSigma/dOmega so interaction opacity and the "
            "sampled phase function close exactly"
        ),
        "newdust_reported_tau_sca": reported_tau_sca.tolist(),
        "newdust_reported_scattering_cross_section_cm2_per_h": (
            reported_tau_sca / NH_REFERENCE_CM2
        ).tolist(),
        "integrated_to_newdust_reported_cross_section_ratio": (
            sigma / (reported_tau_sca / NH_REFERENCE_CM2)
        ).tolist(),
        "absorption_included": False,
        "positive_angle_samples": N_POSITIVE_ANGLES,
        "minimum_positive_angle_arcsec": MIN_ANGLE_ARCSEC,
        "maximum_angle_rad": float(np.pi),
        "historical_validation": {
            "role": "provenance only; not an input to photon transport",
            "configuration_origin": (
                "numerical regression against supplied 4U 1630-47 screen files"
            ),
            "screen_kernel_convention": (
                "K(theta_obs,f)=NH*dSigma/dOmega(theta_obs/(1-f))/(1-f)^2"
            ),
            "reference_grid": {
                "matrix_shape": [1500, 1000],
                "observed_angle_arcsec": [1.0, 1500.0, 1.0],
                "fractional_distance_from_observer": [0.001, 1.0, 0.001],
                "energy_kev": [3.3, 4.9, 6.9],
            },
        },
        "table_sha256": _sha256(output),
    }
    with output.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(f"wrote {output}")
    print(f"wrote {output.with_suffix('.json')}")
    print("sigma_sca [cm^2/H] =", sigma)


def main() -> None:
    default_output = (
        Path(__file__).resolve().parent.parent
        / "dsh"
        / "data"
        / "newdust"
        / "mrn_rg_drude_v1.npz"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
    generate(args.output.resolve())


if __name__ == "__main__":
    main()
