"""Prepare or run an independent XSPEC TBabs-v2 residual-gas cross-check.

Uses tbvarabs (or external tbnew) with rho=0 and residual abundances expressed
relative to the explicitly selected wilm vector. Only the model's 18 atoms and
H2 are compared; additional PHFIT2 atoms are outside this XSPEC interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from dsh.physics.gas_absorption import ELEMENTS, TBNEW_ELEMENTS, load_atomic_basis
from dsh.physics.material_definitions import (
    load_material_definition,
    load_wilm_abundances,
)
from scripts.generate_tbabs_table import resolve_xspec


def gas_parameters(definition, nh_cm2):
    audit = definition.audit()
    gas = np.array(
        [audit["elements"].get(e, {}).get("gas_atoms_per_h", 0) for e in ELEMENTS]
    )
    gas_h = gas[0]
    if gas_h <= 0 or not np.isfinite(nh_cm2) or nh_cm2 <= 0:
        raise ValueError("XSPEC control needs positive gas hydrogen and column")
    wilm = load_wilm_abundances()
    parameters = [nh_cm2 * gas_h / 1e22]
    parameters += [
        gas[z - 1] / gas_h / wilm.atoms_per_h[ELEMENTS[z - 1]]
        for z in TBNEW_ELEMENTS[1:]
    ]
    parameters += [definition.molecular_hydrogen_fraction / gas_h, 0, 0.025, 0.25, 3.5]
    parameters += [1.0] * 18 + [0.0]
    assert len(parameters) == 42
    return parameters


def make_commands(definition, energy_kev, nh_cm2, *, model="tbvarabs"):
    if model not in ("tbvarabs", "tbnew"):
        raise ValueError("expected full 42-parameter TBabs-family model")
    if not 2 <= energy_kev <= 10:
        raise ValueError("XSPEC check supports 2--10 keV")
    parameters = gas_parameters(definition, nh_cm2)
    width = 5e-7
    return "\n".join(
        [
            "abund wilm",
            "xsect vern",
            "xset TBABSVERSION 2",
            f"dummyrsp {energy_kev - width:.12g} {energy_kev + width:.12g} 1 lin",
            f"model {model}*powerlaw",
            *(f"{value:.17g}" for value in parameters),
            "2",
            "1",
            "tclout modval 1",
            'puts "DSH_ABSORBED $xspec_tclout"',
            "newpar 1 0",
            "tclout modval 1",
            'puts "DSH_UNABSORBED $xspec_tclout"',
            "exit",
            "",
        ]
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--definition", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--xspec", default="xspec")
    p.add_argument("--model", choices=("tbvarabs", "tbnew"), default="tbvarabs")
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args()
    definition = load_material_definition(a.definition)
    a.output.mkdir(parents=True, exist_ok=True)
    basis = load_atomic_basis()
    audit = definition.audit()
    counts = np.array(
        [audit["elements"].get(e, {}).get("gas_atoms_per_h", 0) for e in ELEMENTS]
    )
    counts[0] = audit["atomic_hydrogen_atoms_per_h"]
    index = np.array(TBNEW_ELEMENTS) - 1
    jobs = []
    for energy in (3.3, 4.9, 6.9, 7.1239, 7.1241):
        atoms, molecule = basis.evaluate(energy)
        expected = float(
            counts[index] @ atoms[index]
            + audit["molecular_hydrogen_molecules_per_h"] * molecule
        )
        for nh in (1e22, 1e23):
            commands = make_commands(definition, energy, nh, model=a.model)
            name = f"gas_e{energy:g}_nh{nh:g}.xcm"
            (a.output / name).write_text(commands, encoding="utf-8")
            jobs.append(
                {
                    "energy_kev": energy,
                    "nh_cm2": nh,
                    "commands_file": name,
                    "commands_sha256": hashlib.sha256(commands.encode()).hexdigest(),
                    "expected_core_cross_section_cm2_per_h": expected,
                }
            )
    report = {
        "material_definition_sha256": definition.sha256,
        "model": a.model,
        "tbabs_version": 2,
        "scope": "Independent XSPEC-process comparison, 18 core atoms + H2 only; no grain absorption or extra PHFIT2 atoms",
        "validation_executed": not a.prepare_only,
        "relative_cross_section_tolerance": 1e-3,
        "jobs": jobs,
    }
    if not a.prepare_only:
        executable = resolve_xspec(a.xspec)
        for job in jobs:
            commands = (a.output / job["commands_file"]).read_text()
            run = subprocess.run(
                [executable],
                input=commands,
                text=True,
                capture_output=True,
                check=False,
            )
            output = run.stdout + "\n" + run.stderr
            (a.output / (job["commands_file"] + ".log")).write_text(output)
            absorbed = re.search(r"DSH_ABSORBED\s+([0-9.eE+-]+)", output)
            unabsorbed = re.search(r"DSH_UNABSORBED\s+([0-9.eE+-]+)", output)
            if run.returncode or absorbed is None or unabsorbed is None:
                raise RuntimeError(
                    "XSPEC cross-check failed; inspect saved process log"
                )
            ratio = float(absorbed[1]) / float(unabsorbed[1])
            if not 0 < ratio < 1:
                raise ValueError("invalid XSPEC transmission")
            sigma = float(-np.log(ratio) / job["nh_cm2"])
            job.update(
                xspec_cross_section_cm2_per_h=sigma,
                relative_error=sigma / job["expected_core_cross_section_cm2_per_h"] - 1,
            )
            version = re.search(r"XSPEC version:\s*([^\s]+)", output)
            job["xspec_version"] = version[1] if version else "unknown"
        report["all_passed"] = all(abs(j["relative_error"]) <= 1e-3 for j in jobs)
    else:
        report["all_passed"] = None
    (a.output / "xspec_crosscheck.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "Prepared ten XSPEC checks"
        if a.prepare_only
        else f"XSPEC checks passed: {report['all_passed']}"
    )
    if not a.prepare_only and not report["all_passed"]:
        raise SystemExit("XSPEC checks failed")


if __name__ == "__main__":
    main()
