"""Chemical inventory, hydrogen convention, and declared table composition."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from dsh.physics.absorption import PhotoelectricAbsorptionTable
from dsh.physics.material_definitions import (
    ATOMIC_MASS_UNIT_G,
    ATOMIC_WEIGHTS_U,
    ElementalAbundances,
    GrainComponent,
    MaterialDefinition,
    load_material_definition,
    load_wilm_abundances,
)
from dsh.physics.materials import build_declared_material_tables
from dsh.physics.newdust import load_newdust_scattering_table


def example_definition():
    # Counts are chosen independently of any WD01 or RefleX dust prescription.
    carbon = GrainComponent(
        "graphite",
        {"C": 1},
        2.2,
        1.2e-4 * 12.011 * ATOMIC_MASS_UNIT_G,
        "illustrative-size-law",
        "illustrative-graphite-index",
        "accounting test",
    )
    olivine = GrainComponent(
        "olivine",
        {"Mg": 1, "Fe": 1, "Si": 1, "O": 4},
        3.5,
        1e-5 * (24.305 + 55.845 + 28.085 + 4 * 15.999) * ATOMIC_MASS_UNIT_G,
        "illustrative-size-law",
        "illustrative-olivine-index",
        "accounting test",
    )
    return MaterialDefinition(
        "illustrative-wilm-mixture",
        load_wilm_abundances(),
        (carbon, olivine),
        0.2,
        "synthetic-test",
        "accounting test",
    )


class MaterialDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.definition = example_definition()

    def test_wilm_vector_matches_independent_reflex_export(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "reflex_adapter/materials/reflex_scattering_v1.json"
        )
        header = json.loads(path.read_text())["source_dump_header"]
        actual = load_wilm_abundances().atoms_per_h
        self.assertEqual(len(actual), 30)
        for number, element in enumerate(ATOMIC_WEIGHTS_U, start=1):
            self.assertEqual(actual[element], header[f"ELEM{number}"])

    def test_stoichiometric_counts_and_residual_gas(self):
        rows = self.definition.audit()["elements"]
        for element, solid in {
            "C": 1.2e-4,
            "Mg": 1e-5,
            "Fe": 1e-5,
            "Si": 1e-5,
            "O": 4e-5,
        }.items():
            self.assertAlmostEqual(rows[element]["grain_atoms_per_h"] / solid, 1.0)
            self.assertAlmostEqual(
                rows[element]["gas_atoms_per_h"] + solid,
                rows[element]["total_atoms_per_h"],
                places=16,
            )
        self.assertEqual(rows["Ne"]["gas_fraction"], 1)
        self.assertIsNone(rows["Li"]["gas_fraction"])

    def test_mass_and_hydrogen_nuclei_closure(self):
        report = self.definition.audit()
        self.assertAlmostEqual(
            (report["dust_mass_per_h_g"] + report["gas_mass_per_h_g"])
            / report["total_mass_per_h_g"],
            1.0,
            places=14,
        )
        self.assertEqual(report["atomic_hydrogen_atoms_per_h"], 0.8)
        self.assertEqual(report["molecular_hydrogen_molecules_per_h"], 0.1)
        self.assertAlmostEqual(
            report["dust_to_gas_mass_ratio"],
            report["dust_mass_per_h_g"] / report["gas_mass_per_h_g"],
        )

    def test_silicon_overdraw_rejected_without_renormalizing(self):
        grains = list(self.definition.grains)
        grains[1] = replace(grains[1], mass_per_h_g=grains[1].mass_per_h_g * 2)
        invalid = replace(self.definition, grains=tuple(grains))
        with self.assertRaisesRegex(ValueError, "exceeds Si abundance"):
            invalid.audit()
        self.assertEqual(invalid.grains[1].mass_per_h_g, grains[1].mass_per_h_g)

    def test_elements_with_zero_budget_cannot_be_put_in_grains(self):
        solid = replace(self.definition.grains[0], stoichiometry={"Zn": 1})
        with self.assertRaisesRegex(ValueError, "exceeds Zn abundance"):
            replace(self.definition, grains=(solid,)).audit()

    def test_hydrogen_in_solids_cannot_also_be_molecular(self):
        hydrogen_solid = GrainComponent(
            "hydrogen-bearing-test",
            {"H": 1},
            1,
            0.1 * 1.008 * ATOMIC_MASS_UNIT_G,
            "illustration",
            "illustration",
            "hydrogen accounting test",
        )
        definition = replace(
            self.definition, grains=(hydrogen_solid,), molecular_hydrogen_fraction=0.8
        )
        report = definition.audit()
        self.assertAlmostEqual(report["atomic_hydrogen_atoms_per_h"], 0.1)
        self.assertAlmostEqual(report["grain_hydrogen_atoms_per_h"], 0.1)
        with self.assertRaisesRegex(ValueError, "grain hydrogen exceeds"):
            replace(definition, molecular_hydrogen_fraction=0.95).audit()

    def test_invalid_chemical_inputs_rejected(self):
        for changes in (
            {"density_g_cm3": 0},
            {"mass_per_h_g": -1},
            {"mass_per_h_g": float("nan")},
            {"stoichiometry": {"Si": -1}},
            {"stoichiometry": {"Xx": 1}},
            {"optical_constants_id": ""},
            {"size_parameters": {"p": float("inf")}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.definition.grains[0], **changes)
        for fraction in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                replace(self.definition, molecular_hydrogen_fraction=fraction)
        with self.assertRaises(ValueError):
            ElementalAbundances("wrong normalization", {"H": 2}, "test")

    def test_duplicate_population_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be unique"):
            replace(self.definition, grains=(self.definition.grains[0],) * 2)

    def test_finite_inputs_cannot_overflow_the_derived_inventory(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            replace(self.definition.grains[0], stoichiometry={"C": 1e308})
        giant = replace(self.definition.grains[0], mass_per_h_g=1e308)
        with self.assertRaisesRegex(ValueError, "finite"):
            replace(self.definition, grains=(giant,)).audit()

    def test_definitions_copy_and_freeze_input_mappings(self):
        data = {"H": 1, "C": 1e-4}
        abundance = ElementalAbundances("test", data, "test")
        data["C"] = 0
        self.assertEqual(abundance.atoms_per_h["C"], 1e-4)
        with self.assertRaises(TypeError):
            abundance.atoms_per_h["C"] = 0

    def test_round_trip_fingerprint_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "definition.json"
            path.write_text(json.dumps(self.definition.as_dict()), encoding="utf-8")
            loaded = load_material_definition(path)
            self.assertEqual(loaded.sha256, self.definition.sha256)
            self.assertEqual(
                replace(loaded, grains=tuple(reversed(loaded.grains))).sha256,
                loaded.sha256,
            )
            self.assertNotEqual(
                replace(loaded, molecular_hydrogen_fraction=0.3).sha256, loaded.sha256
            )
            data = loaded.as_dict()
            data["normalization"] = "per H2 molecule"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "normalization"):
                load_material_definition(path)


class DeclaredMaterialCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = load_newdust_scattering_table()

    def setUp(self):
        self.definition = example_definition()
        # These synthetic tags exercise the contract, not a physical WD01 model.
        metadata = dict(
            self.original.metadata,
            material_definition_sha256=self.definition.sha256,
            scattering_method=self.definition.scattering_method,
        )
        self.scattering = replace(self.original, metadata=metadata)

    def component(self, coverage, value=1e-24):
        return PhotoelectricAbsorptionTable(
            self.scattering.energy_kev.copy(),
            np.full(self.scattering.energy_kev.shape, value),
            {
                "material_definition_sha256": self.definition.sha256,
                "quantity": "photoelectric_absorption_cross_section",
                "cross_section_unit": "cm2 H-1",
                "absorption_coverage": coverage,
                "producer": "synthetic contract test",
            },
        )

    def test_disjoint_components_sum_into_actual_jax_physics(self):
        components = [
            self.component(["atomic_gas"], 2e-24),
            self.component(["molecular_hydrogen"], 1e-25),
            self.component(["grains"], 4e-24),
        ]
        scattering, absorption, physics = build_declared_material_tables(
            self.definition, self.scattering, components
        )
        self.assertIs(scattering, self.scattering)
        np.testing.assert_allclose(
            absorption.absorption_cross_section_cm2_per_h, 6.1e-24, rtol=1e-14
        )
        np.testing.assert_allclose(
            physics.absorption_cross_section_cm2_per_h, 6.1e-24, rtol=1e-6
        )
        np.testing.assert_array_equal(
            physics.scattering_cross_section_cm2_per_h,
            np.asarray(
                self.original.scattering_cross_section_cm2_per_h,
                dtype=physics.energy_kev.dtype,
            ),
        )

    def test_full_ism_plus_grain_absorption_is_rejected(self):
        full = self.component(["atomic_gas", "molecular_hydrogen", "grains"])
        with self.assertRaisesRegex(ValueError, "overlapping absorption"):
            build_declared_material_tables(
                self.definition, self.scattering, [full, self.component(["grains"])]
            )

    def test_full_ism_and_split_absorption_are_equivalent(self):
        _, full, _ = build_declared_material_tables(
            self.definition,
            self.scattering,
            [self.component(["atomic_gas", "molecular_hydrogen", "grains"], 3e-24)],
        )
        _, split, _ = build_declared_material_tables(
            self.definition,
            self.scattering,
            [
                self.component([part])
                for part in ("atomic_gas", "molecular_hydrogen", "grains")
            ],
        )
        np.testing.assert_allclose(
            full.absorption_cross_section_cm2_per_h,
            split.absorption_cross_section_cm2_per_h,
            rtol=1e-15,
        )

    def test_mismatched_definition_or_method_rejected(self):
        full = self.component(["atomic_gas", "molecular_hydrogen", "grains"])
        for changes in (
            {"material_definition_sha256": "wrong"},
            {"scattering_method": "mie"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_declared_material_tables(
                    self.definition,
                    replace(
                        self.scattering,
                        metadata=dict(self.scattering.metadata, **changes),
                    ),
                    [full],
                )
        invalid = replace(
            full, metadata=dict(full.metadata, material_definition_sha256="wrong")
        )
        with self.assertRaisesRegex(ValueError, "absorption material definition"):
            build_declared_material_tables(self.definition, self.scattering, [invalid])

    def test_absorption_coverage_must_be_explicit_and_complete(self):
        for coverage in ([], ["atomic_gas"], ["all"], ["grains", "grains"], "grains"):
            with self.subTest(coverage=coverage), self.assertRaises(ValueError):
                build_declared_material_tables(
                    self.definition, self.scattering, [self.component(coverage)]
                )

    def test_zero_component_is_explicitly_supported(self):
        components = [
            self.component(["atomic_gas", "grains"]),
            self.component(["molecular_hydrogen"], 0),
        ]
        _, _, physics = build_declared_material_tables(
            self.definition, self.scattering, components
        )
        np.testing.assert_allclose(
            physics.absorption_cross_section_cm2_per_h, 1e-24, rtol=1e-6
        )

    def test_bad_units_arrays_and_grids_rejected(self):
        full = self.component(["atomic_gas", "molecular_hydrogen", "grains"])
        cases = [
            replace(full, energy_kev=full.energy_kev + 0.1),
            replace(
                full, metadata=dict(full.metadata, cross_section_unit="cm2 per grain")
            ),
            replace(
                full, metadata=dict(full.metadata, shared_energy_grid_sha256="wrong")
            ),
            replace(full, absorption_cross_section_cm2_per_h=np.array([-1e-24] * 3)),
            replace(full, absorption_cross_section_cm2_per_h=np.array([np.nan] * 3)),
        ]
        for invalid in cases:
            with self.assertRaises(ValueError):
                build_declared_material_tables(
                    self.definition, self.scattering, [invalid]
                )


if __name__ == "__main__":
    unittest.main()
