"""Generate intrinsic MRN/RG-Drude scattering on a shared 2–10 keV axis."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dsh.physics.material_grid import load_material_grid
from dsh.physics.rg_drude import RGDrudeMRN, default_angle_grid, gaussian_rg_drude_table


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(output: Path, energy_grid: Path) -> None:
    energy = load_material_grid(energy_grid)
    angle = default_angle_grid()
    model = RGDrudeMRN()
    differential, sigma, cdf = gaussian_rg_drude_table(energy, angle, model=model)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        energy_kev=energy,
        scattering_angle_rad=angle,
        differential_cross_section_cm2_per_sr_per_h=differential,
        scattering_cross_section_cm2_per_h=sigma,
        scattering_angle_cdf=cdf,
    )
    metadata = {
        "schema_version": 1,
        "quantity": "intrinsic_differential_cross_section",
        "differential_cross_section_unit": "cm2 sr-1 H-1",
        "integrated_cross_section_unit": "cm2 H-1",
        "energy_unit": "keV",
        "angle_unit": "rad",
        "energy_kev": energy.tolist(),
        "producer": "DSH analytic Gaussian Rayleigh-Gans/Drude MRN calculator",
        "model_parameters": asdict(model),
        "dust_mass_column_at_nh_1e22_g_cm2": model.dust_mass_per_h_g * 1e22,
        "normalization": "grain mass per H; physical solid-angle integral of differential",
        "model_limitation": (
            "Rayleigh-Gans/Drude approximation; configured for 2–10 keV, "
            "not a general Mie calculation"
        ),
        "scattering_cross_section_definition": (
            "solid-angle integral of the tabulated differential cross-section"
        ),
        "positive_angle_samples": angle.size - 1,
        "minimum_positive_angle_arcsec": float(angle[1] * 180.0 * 3600.0 / np.pi),
        "maximum_angle_rad": float(np.pi),
        "absorption_included": False,
        "shared_energy_grid_sha256": _sha256(energy_grid),
        "table_sha256": _sha256(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {output} and {output.with_suffix('.json')} ({energy.size} energies)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy-grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output.resolve(), args.energy_grid.resolve())


if __name__ == "__main__":
    main()
