#!/usr/bin/env python3
"""Extract a source-independent absorption table from XSPEC ``tbabs``.

This is an offline provenance script.  XSPEC/HEASoft is not a runtime
dependency of the JAX transport code.  For each requested energy it evaluates
``tbabs*powerlaw`` and ``powerlaw`` through the same very narrow dummy response
and obtains the monochromatic material coefficient from their ratio:

    sigma_abs(E) = -log(F_abs/F_unabs) / NH.

The power law cancels in the narrow-bin ratio; it is only an additive source
component required to evaluate XSPEC's multiplicative model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np


ENERGY_KEV = np.array([3.3, 4.9, 6.9], dtype=np.float64)
REFERENCE_NH22 = 10.0
VALIDATION_NH22 = 1.0
BIN_HALF_WIDTH_KEV = 5.0e-6
PHOTON_INDEX = 2.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_one_energy(
    xspec_executable: str,
    energy_kev: float,
    nh22: float,
    half_width_kev: float,
):
    low = energy_kev - half_width_kev
    high = energy_kev + half_width_kev
    if low <= 0.0:
        raise ValueError("dummy-response lower energy must be positive")

    commands = "\n".join(
        [
            "abund wilm",
            "xsect vern",
            "xset TBABSVERSION 2",
            f"dummyrsp {low:.12g} {high:.12g} 1 lin",
            "model tbabs*powerlaw",
            f"{nh22:.12g}",
            f"{PHOTON_INDEX:.12g}",
            "1.0",
            "tclout modval 1",
            'puts "DSH_ABSORBED $xspec_tclout"',
            "newpar 1 0.0",
            "tclout modval 1",
            'puts "DSH_UNABSORBED $xspec_tclout"',
            "exit",
            "",
        ]
    )
    completed = subprocess.run(
        [xspec_executable],
        input=commands,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + "\n" + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"XSPEC failed for E={energy_kev:g} keV:\n{output}"
        )

    absorbed_match = re.search(
        r"DSH_ABSORBED\s+([0-9.eE+-]+)", output
    )
    unabsorbed_match = re.search(
        r"DSH_UNABSORBED\s+([0-9.eE+-]+)", output
    )
    version_match = re.search(r"XSPEC version:\s*([^\s]+)", output)
    if absorbed_match is None or unabsorbed_match is None:
        raise RuntimeError(
            "could not parse model values from XSPEC output; "
            "check that the executable supports tclout modval"
        )

    absorbed = float(absorbed_match.group(1))
    unabsorbed = float(unabsorbed_match.group(1))
    if not 0.0 < absorbed < unabsorbed:
        raise RuntimeError("XSPEC returned an invalid absorbed/unabsorbed ratio")
    transmission = absorbed / unabsorbed
    sigma = -np.log(transmission) / (nh22 * 1.0e22)
    return sigma, transmission, version_match.group(1) if version_match else None


def generate(output: Path, xspec_executable: str) -> None:
    resolved_xspec = shutil.which(xspec_executable)
    if resolved_xspec is None:
        candidate = Path(xspec_executable)
        if not candidate.is_file():
            raise SystemExit(
                "generation requires an initialized XSPEC/HEASoft environment"
            )
        resolved_xspec = str(candidate.resolve())

    reference_sigma = []
    reference_transmission = []
    validation_sigma = []
    producer_versions = set()
    for energy in ENERGY_KEV:
        sigma, transmission, version = _extract_one_energy(
            resolved_xspec,
            float(energy),
            REFERENCE_NH22,
            BIN_HALF_WIDTH_KEV,
        )
        check_sigma, _, check_version = _extract_one_energy(
            resolved_xspec,
            float(energy),
            VALIDATION_NH22,
            BIN_HALF_WIDTH_KEV,
        )
        reference_sigma.append(sigma)
        reference_transmission.append(transmission)
        validation_sigma.append(check_sigma)
        producer_versions.update(v for v in (version, check_version) if v)

    sigma = np.asarray(reference_sigma, dtype=np.float64)
    validation_sigma = np.asarray(validation_sigma, dtype=np.float64)
    if not np.allclose(sigma, validation_sigma, rtol=5.0e-7, atol=0.0):
        raise RuntimeError(
            "inferred TBabs cross-section changes with reference column"
        )
    if len(producer_versions) > 1:
        raise RuntimeError("XSPEC version changed during table generation")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        energy_kev=ENERGY_KEV,
        absorption_cross_section_cm2_per_h=sigma,
    )

    producer_version = next(iter(producer_versions), "unknown")
    metadata = {
        "schema_version": 1,
        "quantity": "photoelectric_absorption_cross_section",
        "cross_section_unit": "cm2 H-1",
        "energy_unit": "keV",
        "energy_kev": ENERGY_KEV.tolist(),
        "producer": "XSPEC tbabs",
        "producer_version": producer_version,
        "producer_url": (
            "https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/"
            "XSmodelTbabs.html"
        ),
        "tbabs_version": 2,
        "abundance_command": "wilm",
        "abundance_reference": "Wilms, Allen & McCray (2000)",
        "atomic_cross_section_baseline": "Verner",
        "xspec_cross_section_note": (
            "tbabs ignores the global xsect setting and uses Verner cross-sections "
            "as its internal baseline"
        ),
        "gas_phase_included": True,
        "molecular_hydrogen_included": True,
        "grain_phase_included": True,
        "grain_depletion_and_self_shielding_included": True,
        "source_spectrum_dependent": False,
        "extraction_method": (
            "-log((tbabs*powerlaw)/powerlaw)/NH in a one-bin dummy response"
        ),
        "dummy_powerlaw_photon_index": PHOTON_INDEX,
        "dummy_response_half_width_kev": BIN_HALF_WIDTH_KEV,
        "reference_hydrogen_column_nh22": REFERENCE_NH22,
        "reference_transmission": reference_transmission,
        "validation_hydrogen_column_nh22": VALIDATION_NH22,
        "validation_cross_section_cm2_per_h": validation_sigma.tolist(),
        "validation_max_relative_difference": float(
            np.max(np.abs(validation_sigma / sigma - 1.0))
        ),
        "intended_runtime_use": (
            "intrinsic material coefficient; transmission is exp(-NH*sigma_abs)"
        ),
        "broad_band_warning": (
            "do not replace with a band-integrated transmission table, which "
            "depends on source spectrum and column density"
        ),
        "table_sha256": _sha256(output),
    }
    with output.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(f"wrote {output}")
    print(f"wrote {output.with_suffix('.json')}")
    print("sigma_abs [cm^2/H] =", sigma)


def main() -> None:
    default_output = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "absorption"
        / "tbabs_wilm_vern_v1.npz"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--xspec",
        default="xspec",
        help="XSPEC executable in an initialized HEASoft environment",
    )
    args = parser.parse_args()
    generate(args.output.resolve(), args.xspec)


if __name__ == "__main__":
    main()
