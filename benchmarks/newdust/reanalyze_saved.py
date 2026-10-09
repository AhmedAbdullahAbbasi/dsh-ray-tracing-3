"""Recompute radial profiles from saved first-order MC moments, without rerunning photons.

Example: python newdust_benchmark/reanalyze_saved.py --repo .
  --input validation_outputs/newdust_multiple --config multiple_config.json --output reanalyzed
Physical parameters and DSH module hashes must match the original provenance.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import compare_newdust as bench
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Use a separate output directory to preserve original results")
    os.environ["JAX_PLATFORMS"] = "cpu"
    sys.path.insert(0, str(args.repo.resolve()))
    config = json.loads(args.config.read_text())
    bench.validate(config)
    provenance = json.loads((args.input / "provenance.json").read_text())
    physical = [
        "source_distance_kpc",
        "source_fluence_ph_cm2",
        "screens",
        "thickness_kpc",
        "half_width_arcsec",
        "pixel_arcsec",
        "packets_per_seed",
        "seeds",
        "chunk_size",
        "max_interactions",
    ]
    matches = [
        c
        for c in provenance["configurations"]
        if all(c[k] == config[k] for k in physical)
        and all(e in c["energies_kev"] for e in config["energies_kev"])
    ]
    if not matches:
        raise ValueError(
            "Saved run does not match the requested physical configuration"
        )
    for name, expected in provenance["dsh_code_sha256"].items():
        actual = hashlib.sha256(Path(bench.api(name).__file__).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"DSH module differs from the saved run: {name}")
    materials = bench.api("physics.newdust")
    table = materials.load_newdust_scattering_table()
    if (
        table.metadata["table_sha256"]
        != provenance["material_metadata"]["table_sha256"]
    ):
        raise ValueError("Material table differs from the saved run")
    physics = materials.build_dust_physics_from_newdust(
        table, np.zeros_like(table.energy_kev)
    )
    label = (
        "multi"
        if len(config["screens"]) > 1
        else f"x{config['screens'][0]['fraction_from_observer']:g}"
    )

    def reuse(setting, cloud, material, scattering, index, edges, seed):
        case = args.input / f"{label}_E{scattering.energy_kev[index]:g}"
        with np.load(case / f"seed_{seed}.npz", allow_pickle=False) as archive:
            if int(archive["histories"]) != setting["packets_per_seed"]:
                raise ValueError("Saved history count differs")
            np.testing.assert_array_equal(archive["sky_edges_arcsec"], edges)
            return {
                k: archive[k].copy()
                for k in archive.files
                if k not in ["histories", "sky_edges_arcsec"]
            }

    bench.run_seed = reuse
    args.output.mkdir(parents=True, exist_ok=True)
    updated = dict(provenance)
    updated["original_configurations"] = provenance["configurations"]
    updated["configurations"] = [config]
    updated["analysis_note"] = (
        "Radial/reference reanalysis of saved photon moments; no new photon histories or fitted amplitude."
    )
    updated["reanalysis_script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    (args.output / "provenance.json").write_text(json.dumps(updated, indent=2) + "\n")
    (args.output / "benchmark_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    cloud = bench.make_cloud(config)
    reports = []
    for energy in config["energies_kev"]:
        indexes = np.flatnonzero(
            np.isclose(table.energy_kev, energy, rtol=0, atol=1e-10)
        )
        if len(indexes) != 1:
            raise ValueError("Energy must match a material node")
        reports.append(
            bench.compare(
                config,
                table,
                physics,
                cloud,
                int(indexes[0]),
                args.output / f"{label}_E{energy:g}",
            )
        )
    (args.output / "suite_report.json").write_text(json.dumps(reports, indent=2) + "\n")
    return 0 if all(r["all_passed"] for r in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
