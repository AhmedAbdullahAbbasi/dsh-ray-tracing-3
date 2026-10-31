"""Assess frozen independent RefleX controls without running its executable."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from dsh.physics.cold_compton import (
    SOURCE_PATH,
    build_cold_compton_table,
    cold_compton_components,
)
from dsh.physics.compton import ELECTRON_REST_ENERGY_KEV
from dsh.physics.material_definitions import (
    ElementalAbundances,
    MaterialDefinition,
    load_material_definition,
)

REFERENCE = Path(__file__).resolve().parents[1] / "benchmarks/compton/cold_material_v1"


def evidence_sha(path):
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix in (".par", ".log", ".csv"):
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _cdf(mu, differential):
    result = np.r_[0, np.cumsum(np.diff(mu) * (differential[1:] + differential[:-1]))]
    return result / result[-1]


def _ks(samples, mu, cdf):
    ordered = np.sort(samples)
    n = ordered.size
    probability = np.interp(ordered, mu, cdf)
    return float(
        max(
            np.max(np.arange(1, n + 1) / n - probability),
            np.max(probability - np.arange(n) / n),
        )
    )


def assess_reference(directory=REFERENCE):
    directory = Path(directory)
    manifest = json.loads((directory / "reference.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["evidence_sha256"].items():
        if evidence_sha(directory / relative) != expected:
            raise ValueError("cold Compton reference evidence changed: " + relative)
    with (directory / "reflex_opacity_controls.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row["case"] == "wilm_h2_02_dust0691"]
    energy = np.array([float(row["tabulated_energy_kev"]) for row in selected])
    observed = np.array(
        [float(row["hubbell_scale_removed_cm2_per_h"]) for row in selected]
    )
    effective = np.array(
        [float(row["effective_cross_section_cm2_per_h"]) for row in selected]
    )
    material = load_material_definition(
        SOURCE_PATH.parent / "reflex_reference_definition.json"
    )
    # Match the reference's coarser linear-in-E interpolation separately from
    # the physical continuous kernel. Never modify the DSH material totals.
    nodes = np.array([2, 3, 4, 5, 6, 8, 10.0])
    at_nodes = build_cold_compton_table(material, nodes, grain_electrons="omit")
    matched_interpolation = np.interp(
        energy, nodes, at_nodes.table.cross_section_cm2_per_h
    )
    direct = build_cold_compton_table(material, energy, grain_electrons="omit")
    residual = matched_interpolation / observed - 1
    direct_residual = direct.table.cross_section_cm2_per_h / observed - 1
    with np.load(directory / "reflex_angular_h.npz", allow_pickle=False) as archive:
        cosine = archive["cosine"]
        outgoing = archive["energy_ev"]
        order = archive["compton_order"]
    if (
        len(cosine) != manifest["angular_escaping_selected_photons"]
        or not np.all(order == 1)
        or not np.isfinite(cosine).all()
        or np.any(abs(cosine) > 1)
    ):
        raise ValueError("invalid frozen RefleX angular sample")
    mu = np.linspace(-1, 1, 65537)
    ratio = 1 / (1 + 4.9 / ELECTRON_REST_ENERGY_KEV * (1 - mu))
    kn = ratio**3 + ratio - ratio**2 * (1 - mu**2)
    # Isolated H removes composition dependence from the angular check.
    hydrogen = MaterialDefinition(
        material_id="isolated-hydrogen-control",
        abundances=ElementalAbundances(
            abundance_id="pure-H",
            atoms_per_h={"H": 1.0},
            reference="Pure hydrogen control",
        ),
        grains=(),
        molecular_hydrogen_fraction=0.0,
        scattering_method="none",
        reference="Independent isolated-hydrogen angular control",
    )
    bound = sum(
        cold_compton_components(hydrogen, 4.9, mu, grain_electrons="omit").values()
    )
    kn_ks = _ks(cosine, mu, _cdf(mu, kn))
    bound_ks = _ks(cosine, mu, _cdf(mu, bound))
    predicted_energy = 4900 / (1 + 4.9 / ELECTRON_REST_ENERGY_KEV * (1 - cosine))
    recoil_error = float(np.max(abs(outgoing - predicted_energy)))
    return {
        "reference_executable_sha256": manifest["executable_sha256"],
        "reference_evidence_files_verified": len(manifest["evidence_sha256"]),
        "composition": "wilm, molecular nuclei fraction=0.2, Fe dust fraction=0.691",
        "grain_electrons": "omit",
        "reflex_initialization_scale_relative_to_barn_to_mb": manifest[
            "legacy_scale_relative_to_barn_to_mb"
        ],
        "maximum_opacity_relative_difference_matching_native_energy_interpolation": float(
            np.max(abs(residual))
        ),
        "maximum_opacity_relative_difference_continuous_physics": float(
            np.max(abs(direct_residual))
        ),
        "maximum_raw_effective_opacity_relative_difference": float(
            np.max(abs(direct.table.cross_section_cm2_per_h / effective - 1))
        ),
        "native_interpolation_opacity_within_0p5_percent": bool(
            np.max(abs(residual)) < 0.005
        ),
        "reflex_photon_histories_launched": 2000000,
        "angular_selected_photons": int(cosine.size),
        "ks_distance_to_free_kn": kn_ks,
        "ks_distance_to_bound_h": bound_ks,
        "maximum_stationary_recoil_difference_ev": recoil_error,
        "reflex_angular_equivalence": False,
        "angular_discrepancy_demonstrated": bool(kn_ks < 0.03 and bound_ks > 0.08),
        "interpretation": "Opacity scale and energy interpolation are accounted for diagnostically; the physical table is never rescaled. Supplied RefleX angular sampler follows KN rather than the bound KN*S law. Full RefleX equivalence is not established.",
    }
