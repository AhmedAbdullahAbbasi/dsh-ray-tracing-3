"""Gas inventory, one-sided transport/observer edges and spectral folding."""

from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dsh.observer.scoring import _cross_sections_at_energy, scattering_phase_pdf_per_sr
from dsh.physics.absorption import load_photoelectric_absorption_table
from dsh.physics.dust import (
    build_dust_physics_table,
    phase_cdf_from_differential_cross_section,
)
from dsh.physics.energy_edges import (
    bracketed_edge_grid,
    edge_intervals,
    integrate_spectrum,
)
from dsh.physics.gas_absorption import (
    DEFAULT_GAS_BASIS,
    load_atomic_basis,
    residual_gas_components,
)
from dsh.physics.grain_populations import WD01Population
from dsh.physics.material_definitions import (
    ElementalAbundances,
    MaterialDefinition,
    load_wilm_abundances,
)
from dsh.physics.materials import build_declared_material_tables
from dsh.physics.newdust import (
    build_dust_physics_from_tables,
    load_newdust_scattering_table,
)
from dsh.physics.wd01_ism import (
    build_wd01_ism,
    load_wd01_ism,
    wilm_budget_population,
    write_wd01_ism,
)
from dsh.transport.kernel import _physics_at_energy
from dsh.validation.absorbed_observer import host_material

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DUST = ROOT / "dsh/data/materials/wd01_d03_v1"
COMPLETE = ROOT / "dsh/data/materials/wd01_wilm_c50_si90_v1"


def toy_edge_physics():
    energy = np.array([2, 4 * (1 - 2e-6), 4 * (1 + 2e-6), 10])
    angle = np.linspace(0, np.pi, 129)
    law = np.array(
        [
            np.ones_like(angle),
            np.ones_like(angle),
            1 + 0.5 * np.cos(angle),
            1 + 0.5 * np.cos(angle),
        ]
    )
    norm, _ = phase_cdf_from_differential_cross_section(angle, law)
    law *= (2e-23 / norm)[:, None]
    sigma, cdf = phase_cdf_from_differential_cross_section(angle, law)
    return build_dust_physics_table(
        energy,
        sigma,
        [1e-23, 1e-23, 9e-23, 9e-23],
        angle,
        cdf,
        law,
        edge_energies_kev=[4],
    )


class EnergyEdgeTests(unittest.TestCase):
    def test_jump_uses_one_side_in_jitted_transport_and_observer(self):
        p = toy_edge_physics()
        transport = jax.jit(lambda e: _physics_at_energy(p, e))
        observer = jax.jit(lambda e: _cross_sections_at_energy(p, e))
        for energy, expected, row in (
            (3.999996, 1e-23, 1),
            (4.0, 9e-23, 2),
            (4.000004, 9e-23, 2),
        ):
            t, o = transport(energy), observer(energy)
            self.assertTrue(bool(t[0]) and bool(o[0]))
            self.assertAlmostEqual(float(t[2]) / expected, 1, delta=5e-6)
            self.assertAlmostEqual(float(o[2]) / expected, 1, delta=5e-6)
            np.testing.assert_array_equal(t[3], p.scattering_angle_cdf[row])
            pdf = float(scattering_phase_pdf_per_sr(p, energy, 0))
            expected_pdf = float(
                p.differential_cross_section_cm2_per_sr_per_h[row, 0]
                / p.scattering_cross_section_cm2_per_h[row]
            )
            self.assertAlmostEqual(pdf / expected_pdf, 1, delta=5e-6)
            self.assertAlmostEqual(
                host_material(p, energy)[-1] / expected, 1, delta=5e-6
            )

    def test_empty_edge_declaration_preserves_legacy_results(self):
        s = load_newdust_scattering_table()
        p = build_dust_physics_from_tables(s, load_photoelectric_absorption_table())
        q = p._replace(energy_edge_kev=jnp.zeros(len(p.energy_kev) - 1))
        for e in np.linspace(3.3, 6.9, 11):
            for a, b in zip(
                _physics_at_energy(p, e), _physics_at_energy(q, e), strict=True
            ):
                np.testing.assert_array_equal(a, b)

    def test_grid_requires_resolved_narrow_distinct_brackets(self):
        grid = bracketed_edge_grid([2, 4, 6, 10], [4, 7.124])
        self.assertEqual(np.count_nonzero(edge_intervals(grid, [4, 7.124])), 2)
        for energy, edges in (
            ([2, 4, 10], [4]),
            ([2, 3, 5, 10], [4]),
            ([2, 3.99999999, 4.00000001, 10], [4]),
            ([2, 4, 10], [1]),
            ([2, 4, 10], [4, 4]),
        ):
            with self.assertRaises(ValueError):
                edge_intervals(energy, edges)

    def test_spectral_panels_split_at_discontinuities(self):
        nh = 3e22

        def f(e):
            return e**-2 * np.exp(-nh * np.where(e < 4, 1e-23, 9e-23))

        exact = np.exp(-nh * 1e-23) * (1 / 2 - 1 / 4) + np.exp(-nh * 9e-23) * (
            1 / 4 - 1 / 10
        )
        actual = integrate_spectrum(f, [2, 10], breakpoints_kev=[4], order=24)
        self.assertAlmostEqual(actual / exact, 1, delta=1e-13)

    def test_spectral_invalid_inputs_fail(self):
        for bounds, order in (([0, 10], 16), ([4, 2], 16), ([2, 10], 1)):
            with self.assertRaises(ValueError):
                integrate_spectrum(lambda e: e, bounds, order=order)
        with self.assertRaises(ValueError):
            integrate_spectrum(lambda e: np.nan, [2, 10])

    @unittest.skipUnless(importlib.util.find_spec("astropy"), "requires Astropy")
    def test_fits_retains_edge_thresholds(self):
        from astropy.io import fits

        from dsh.io.fits_output import _physics_table_hdu

        p = toy_edge_physics()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edges.fits"
            _physics_table_hdu(fits, p).writeto(path)
            with fits.open(path) as h:
                self.assertEqual(h[1].header["EDGEINT"], "ONESIDED")
                np.testing.assert_array_equal(
                    h[1].data["EDGE_KEV"][:-1], p.energy_edge_kev
                )


class GasInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.basis = load_atomic_basis()
        cls.definition, cls.scattering, cls.components, cls.total, cls.physics = (
            load_wd01_ism(COMPLETE)
        )

    def test_hydrogen_analytic_cross_section_and_molecule_units(self):
        energy = self.basis.energy_kev
        atomic, h2 = self.basis.evaluate(energy)
        y = 2 * energy * 1000 / 13.599
        expected = 605 / (y**1.5 * (1 + np.sqrt(y)) ** 4) * 1e-18
        np.testing.assert_allclose(atomic[0], expected, rtol=2e-14)
        self.assertAlmostEqual(
            float(self.basis.evaluate(4.9)[1]) / 1.539756006465403e-25, 1, delta=2e-4
        )
        self.assertTrue(np.all(h2 > 2 * atomic[0]))

    def test_all_elements_and_native_edges_present(self):
        self.assertEqual(self.basis.atomic_cm2.shape[0], 30)
        self.assertEqual(len(self.basis.metadata["edge_energies_kev"]), 16)
        self.assertIn("P", self.basis.metadata["additional_elements_outside_tbnew"])
        self.assertIn("Mn", self.basis.metadata["additional_elements_outside_tbnew"])
        self.assertIn("Ti", self.basis.metadata["additional_elements_outside_tbnew"])
        self.assertFalse(self.basis.atomic_cm2.flags.writeable)

    def test_nonrepresentable_edge_uses_same_precision_in_host_and_jax(self):
        for energy in (7.12399, 7.124, 7.12401):
            expected = host_material(self.physics, energy)[-1]
            actual = float(_physics_at_energy(self.physics, energy)[2])
            self.assertAlmostEqual(actual / expected, 1, delta=5e-6)

    def test_xspec_gas_control_has_explicit_residual_abundances(self):
        from scripts.check_gas_with_xspec import gas_parameters, make_commands

        parameters = gas_parameters(self.definition, 1e23)
        self.assertEqual(len(parameters), 42)
        self.assertEqual(parameters[0], 10)
        self.assertEqual(parameters[18], 0.2)
        self.assertEqual(parameters[19], 0)
        self.assertEqual(parameters[23:41], [1.0] * 18)
        self.assertAlmostEqual(parameters[2], 0.5)
        self.assertAlmostEqual(parameters[9], 0.1)
        commands = make_commands(self.definition, 4.9, 1e23)
        self.assertIn("xset TBABSVERSION 2", commands)
        self.assertIn("model tbvarabs*powerlaw", commands)
        self.assertIn("DSH_UNABSORBED", commands)

    def test_xspec_gas_control_converts_total_h_to_residual_gas_h(self):
        from dsh.physics.material_definitions import (
            ATOMIC_MASS_UNIT_G,
            ATOMIC_WEIGHTS_U,
            GrainComponent,
        )
        from scripts.check_gas_with_xspec import gas_parameters

        solid = GrainComponent(
            "H-control",
            {"H": 1},
            1,
            0.2 * ATOMIC_WEIGHTS_U["H"] * ATOMIC_MASS_UNIT_G,
            "control",
            "control",
            "hydrogen accounting control",
        )
        definition = MaterialDefinition(
            "H-solid-control",
            load_wilm_abundances(),
            (solid,),
            0.2,
            "none",
            "hydrogen accounting control",
        )
        parameters = gas_parameters(definition, 1e23)
        self.assertAlmostEqual(parameters[0], 8)
        self.assertAlmostEqual(parameters[18], 0.25)
        self.assertAlmostEqual(parameters[1], 1 / 0.8)

    def test_basis_no_extrapolation_or_checksum_repair(self):
        for energy in (1.9, 10.1, np.nan):
            with self.assertRaises(ValueError):
                self.basis.evaluate(energy)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "basis.npz"
            shutil.copy2(DEFAULT_GAS_BASIS, path)
            shutil.copy2(
                DEFAULT_GAS_BASIS.with_suffix(".json"), path.with_suffix(".json")
            )
            path.write_bytes(path.read_bytes() + b"wrong")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_atomic_basis(path)
            shutil.copy2(DEFAULT_GAS_BASIS, path)
            metadata = json.loads(path.with_suffix(".json").read_text())
            metadata["atomic_cross_section_unit"] = "cm2 H-1"
            path.with_suffix(".json").write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                load_atomic_basis(path)

    def test_budget_targets_and_derived_other_depletions(self):
        a = self.definition.audit()
        self.assertAlmostEqual(a["elements"]["C"]["gas_fraction"], 0.5)
        self.assertAlmostEqual(a["elements"]["Si"]["gas_fraction"], 0.1)
        self.assertAlmostEqual(
            a["elements"]["Mg"]["gas_atoms_per_h"], 2.51e-5 - 1.674e-5
        )
        self.assertAlmostEqual(
            a["elements"]["Fe"]["gas_atoms_per_h"], 2.69e-5 - 1.674e-5
        )
        self.assertAlmostEqual(
            a["elements"]["O"]["gas_atoms_per_h"], 4.9e-4 - 4 * 1.674e-5
        )
        self.assertAlmostEqual(a["atomic_hydrogen_atoms_per_h"], 0.8)
        self.assertAlmostEqual(a["molecular_hydrogen_molecules_per_h"], 0.1)

    def test_atomic_and_h2_absorption_use_actual_residual_counts(self):
        e = self.scattering.energy_kev
        atomic, h2 = self.basis.evaluate(e)
        a = self.definition.audit()
        from dsh.physics.gas_absorption import ELEMENTS

        counts = np.array([a["elements"][z]["gas_atoms_per_h"] for z in ELEMENTS])
        counts[0] = 0.8
        np.testing.assert_allclose(
            self.components[0].absorption_cross_section_cm2_per_h,
            counts @ atomic,
            rtol=2e-14,
        )
        np.testing.assert_allclose(
            self.components[1].absorption_cross_section_cm2_per_h, 0.1 * h2, rtol=2e-14
        )
        np.testing.assert_array_equal(
            self.total.absorption_cross_section_cm2_per_h,
            sum(t.absorption_cross_section_cm2_per_h for t in self.components),
        )

    def test_molecular_fraction_moves_hydrogen_without_duplication(self):
        a = ElementalAbundances("pure-H", {"H": 1}, "analytic hydrogen control")
        for fraction in (0, 1):
            definition = MaterialDefinition(
                "hydrogen-control", a, (), fraction, "none", "control"
            )
            atom, molecule = residual_gas_components(definition, self.basis)
            np.testing.assert_allclose(
                atom.absorption_cross_section_cm2_per_h,
                (1 - fraction) * self.basis.atomic_cm2[0],
                atol=0,
            )
            np.testing.assert_allclose(
                molecule.absorption_cross_section_cm2_per_h,
                0.5 * fraction * self.basis.h2_cm2,
                atol=0,
            )

    def test_nominal_wd01_wilm_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds C"):
            build_wd01_ism(
                SOURCE_DUST,
                WD01Population(),
                load_wilm_abundances(),
                molecular_hydrogen_fraction=0.2,
            )

    def test_no_duplicate_grains_or_missing_absorption_phases(self):
        for components in (self.components + self.components[2:], self.components[:2]):
            with self.assertRaises(ValueError):
                build_declared_material_tables(
                    self.definition, self.scattering, components
                )

    def test_zero_h2_component_roundtrip_is_explicit(self):
        pop = wilm_budget_population(carbon_gas_fraction=0.5, silicon_gas_fraction=0.1)
        material = build_wd01_ism(
            SOURCE_DUST, pop, load_wilm_abundances(), molecular_hydrogen_fraction=0
        )
        with tempfile.TemporaryDirectory() as directory:
            restored = write_wd01_ism(
                directory, material, recipe={"purpose": "zero-H2 control"}
            )
            self.assertEqual(
                np.count_nonzero(restored[2][1].absorption_cross_section_cm2_per_h), 0
            )
            self.assertEqual(
                restored[2][1].metadata["absorption_coverage"], ["molecular_hydrogen"]
            )

    def test_complete_material_rejects_mislabeled_total_or_edge(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "material"
            shutil.copytree(COMPLETE, target)
            path = target / "total_absorption.json"
            info = json.loads(path.read_text())
            info["components"][0]["table_sha256"] = "wrong"
            path.write_text(json.dumps(info))
            with self.assertRaises(ValueError):
                load_wd01_ism(target)
            for key, value in (
                ("shared_energy_grid_sha256", "wrong"),
                ("edge_energies_kev", []),
                ("absorption_coverage", ["grains"]),
            ):
                info = json.loads((COMPLETE / "total_absorption.json").read_text())
                info[key] = value
                path.write_text(json.dumps(info))
                with self.assertRaises(ValueError):
                    load_wd01_ism(target)


if __name__ == "__main__":
    unittest.main()
