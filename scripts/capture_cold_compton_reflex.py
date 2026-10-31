"""Re-run the inspected RefleX executable's cold opacity and angular controls.

This diagnostic is optional. Normal acceptance uses frozen reference data and
does not require RefleX. The supplied executable is not redistributed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.cold_compton_reference import REFERENCE, assess_reference, evidence_sha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    executable = args.executable.resolve()
    destination = args.output_directory.resolve()
    manifest = json.loads((REFERENCE / "reference.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    if digest != manifest["executable_sha256"]:
        raise ValueError(
            "This capture recipe is verified only for the recorded RefleX executable hash"
        )
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("capture output directory must be empty")
    from astropy.io import fits

    cases = [
        "wilm_h2_02_dust0691",
        "wilm_h2_02_dust0",
        "wilm_h2_0_dust0",
        "pure_h",
        "pure_h2",
        "h_he",
    ]
    rows = []
    energies = np.unique(np.r_[np.linspace(2, 10, 81), 3.3, 4.9, 6.9])
    native_hashes = {}
    for name in [*cases, "angular_h"]:
        parameter = name + ".par"
        (destination / parameter).write_text(
            (REFERENCE / parameter).read_text(encoding="utf-8"), encoding="utf-8"
        )
        print("Running RefleX " + name + " ...", flush=True)
        with (destination / (name + ".log")).open("w", encoding="utf-8") as stream:
            subprocess.run(
                [str(executable), parameter],
                cwd=destination,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=300,
            )
        fits_path = destination / (name + ".fits")
        native_hashes[fits_path.name] = hashlib.sha256(
            fits_path.read_bytes()
        ).hexdigest()
        with fits.open(fits_path) as hdus:
            data = hdus[1].data
            if name == "angular_h":
                np.savez_compressed(
                    destination / "reflex_angular_h.npz",
                    cosine=np.array(data["DIR_Z"], dtype=float),
                    energy_ev=np.array(data["E"], dtype=float),
                    compton_order=np.array(data["COMPTON"], dtype=int),
                )
                manifest["angular_escaping_selected_photons"] = len(data)
            else:
                if hdus[1].header["STEP"] != 1 or len(data) != 20000:
                    raise ValueError("unsupported reference energy grid")
                for energy in energies:
                    index = int(np.floor(energy * 1000 + 1e-9)) if energy < 10 else 9999
                    raw = float(data["COMPTON"][index])
                    rows.append(
                        {
                            "case": name,
                            "requested_energy_kev": float(energy),
                            "dump_row": index,
                            "tabulated_energy_kev": (index + 1) / 1000,
                            "dump_raw": raw,
                            "effective_cross_section_cm2_per_h": raw * 1e-18,
                            "hubbell_scale_removed_cm2_per_h": raw
                            * 1e-18
                            / manifest["legacy_scale_relative_to_barn_to_mb"],
                        }
                    )
    with (destination / "reflex_opacity_controls.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest["capture_date"] = datetime.now(timezone.utc).isoformat()
    manifest["source_sha256"] = native_hashes
    manifest["evidence_sha256"] = {
        name: evidence_sha(destination / name) for name in manifest["evidence_sha256"]
    }
    (destination / "reference.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    assessment = assess_reference(destination)
    (destination / "assessment.json").write_text(
        json.dumps(assessment, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(assessment, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
