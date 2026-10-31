"""Physical tests of neutral scattering functions and per-H material assembly."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dsh.physics.cold_compton import (
    BOHR_RADIUS_ANGSTROM,
    HC_KEV_ANGSTROM,
    SOURCE_PATH,
    _monotone_cubic,
    build_cold_compton_table,
    cold_compton_components,
    cold_scattering_inventory,
    compton_transfer_inverse_angstrom,
    load_incoherent_functions,
)
from dsh.physics.compton import (
    THOMSON_CROSS_SECTION_CM2,
    evaluate_compton,
    load_compton_material_table,
    sample_compton_scattering,
    to_compton_physics,
    write_compton_material_table,
)
from dsh.physics.material_definitions import (
    ElementalAbundances,
    MaterialDefinition,
    load_material_definition,
)


def simple_material(helium=0.0, h2=0.0):
    return MaterialDefinition(
        material_id=f"test-H-He-{helium}-{h2}",
        abundances=ElementalAbundances(
            abundance_id="explicit-test",
            atoms_per_h={"H": 1.0, "He": helium},
            reference="Controlled synthetic composition",
        ),
        grains=(),
        molecular_hydrogen_fraction=h2,
        scattering_method="none",
        reference="Synthetic cold gas for validation",
    )


def hydrogen_quadrature(energy):
    # Scalar Thomson radius expression plus analytic 1s electron form factor;
    # independent of the provider's tabulation and its angular mesh.
    mu, weights = np.polynomial.legendre.leggauss(256)
    ratio = 1 / (1 + energy / 510.99895069 * (1 - mu))
    radius_squared = 3 * THOMSON_CROSS_SECTION_CM2 / (8 * np.pi)
    kn = radius_squared / 2 * ratio**2 * (ratio + 1 / ratio - 1 + mu**2)
    outgoing = energy * ratio
    x = np.sqrt(energy**2 + outgoing**2 - 2 * energy * outgoing * mu) / (
        2 * HC_KEV_ANGSTROM
    )
    form = (1 + (2 * np.pi * BOHR_RADIUS_ANGSTROM * x) ** 2) ** -2
    return 2 * np.pi * np.dot(weights, kn * (1 - form**2))


class IncoherentFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = load_incoherent_functions()

    def test_isolated_hydrogen_against_published_nodes(self):
        q = np.array([0.005, 0.01, 0.1, 0.4])
        published = np.array([0.0011047, 0.0044098, 0.34257, 0.98298])
        np.testing.assert_allclose(
            self.functions.evaluate("H", q), published, rtol=5e-5
        )

    def test_h2_published_nodes_are_per_nucleus(self):
        q = np.array([0.01, 0.1, 0.2, 0.4, 1.0])
        np.testing.assert_allclose(
            self.functions.evaluate("H2", q),
            [0.0036668, 0.27262, 0.67154, 0.96111, 0.99991],
            rtol=1e-12,
        )

    def test_corrected_helium_low_x_nodes(self):
        np.testing.assert_allclose(
            self.functions.evaluate("He", [0.005, 0.01, 0.015, 0.02]),
            [0.00083186, 0.0033274, 0.0074867, 0.013310],
            rtol=1e-12,
        )

    def test_forward_limit_and_quadratic_behavior(self):
        for species in ("H", "H2", "He", "O", "Fe"):
            self.assertEqual(float(self.functions.evaluate(species, 0)), 0)
            small = self.functions.evaluate(species, np.array([1e-7, 2e-7]))
            self.assertAlmostEqual(float(small[1] / small[0]), 4, delta=1e-8)

    def test_nonnegative_monotone_and_electron_limits(self):
        q = np.r_[0, np.geomspace(1e-10, 2, 1000)]
        for species, electrons in (("H2", 1), ("He", 2), ("C", 6), ("Fe", 26)):
            values = self.functions.evaluate(species, q)
            self.assertTrue(np.all((values >= 0) & (values <= electrons)))
            self.assertTrue(np.all(np.diff(values) >= 0))

    def test_invalid_species_and_support(self):
        for species, q in (("unknown", 0.1), ("Fe", 3), ("H2", -1), ("H", np.nan)):
            with self.assertRaises(ValueError):
                self.functions.evaluate(species, q)

    def test_source_integrity_and_windows_newlines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_bytes(SOURCE_PATH.read_bytes().replace(b"\n", b"\r\n"))
            self.assertEqual(
                load_incoherent_functions(path).sha256, self.functions.sha256
            )
            data = json.loads(path.read_text())
            data["h2_s_per_h_nucleus"][0] *= 2
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_incoherent_functions(path)

    def test_recoil_corrected_transfer_from_photon_vectors(self):
        mu = np.linspace(-1, 1, 100)
        for energy in [2, 4.9, 10]:
            outgoing = energy / (1 + energy / 510.99895069 * (1 - mu))
            vector_transfer = np.sqrt(
                (energy - outgoing) ** 2 + 2 * energy * outgoing * (1 - mu)
            ) / (2 * HC_KEV_ANGSTROM)
            np.testing.assert_allclose(
                compton_transfer_inverse_angstrom(energy, mu),
                vector_transfer,
                rtol=1e-14,
                atol=1e-16,
            )

    def test_monotone_hermite_against_independent_pchip_fixture(self):
        # Independently evaluated with scipy.interpolate.PchipInterpolator.
        value = _monotone_cubic(
            np.array([0.0, 1, 2, 4]),
            np.array([0.0, 1, 1, 3]),
            np.array([0.25, 0.75, 1.5, 2.5, 3.5]),
        )
        np.testing.assert_allclose(
            value, [0.3671875, 0.9140625, 1, 1.15625, 2.21875], rtol=1e-14
        )


class ColdMaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.material = load_material_definition(
            SOURCE_PATH.parent / "reflex_reference_definition.json"
        )
        cls.result = build_cold_compton_table(
            cls.material, [2, 3.3, 4.9, 6.9, 10], grain_electrons="omit"
        )

    def test_hydrogen_nuclei_and_molecule_inventory(self):
        targets, audit = cold_scattering_inventory(
            self.material, grain_electrons="omit"
        )
        self.assertAlmostEqual(targets["gas:H"], 0.8)
        self.assertAlmostEqual(targets["gas:H2"], 0.1)
        self.assertAlmostEqual(targets["gas:H"] + 2 * targets["gas:H2"], 1)
        self.assertAlmostEqual(audit["elements"]["Fe"]["gas_fraction"], 0.309)

    def test_grain_policy_is_required_and_validated(self):
        with self.assertRaises(TypeError):
            cold_scattering_inventory(self.material)
        with self.assertRaises(ValueError):
            cold_scattering_inventory(self.material, grain_electrons="automatic")

    def test_grain_inventory_is_counted_once(self):
        targets, audit = cold_scattering_inventory(
            self.material, grain_electrons="atomic_incoherent"
        )
        for species in ("C", "O", "Mg", "Si", "Fe"):
            self.assertAlmostEqual(
                targets["gas:" + species] + targets["solid:" + species],
                audit["elements"][species]["total_atoms_per_h"],
            )

    def test_h2_fraction_linearity_without_factor_two_error(self):
        mu = np.linspace(-1, 1, 101)
        values = []
        for fraction in [0, 0.2, 1]:
            parts = cold_compton_components(
                simple_material(h2=fraction), 4.9, mu, grain_electrons="omit"
            )
            values.append(sum(parts.values()))
        np.testing.assert_allclose(
            values[1], 0.8 * values[0] + 0.2 * values[2], rtol=1e-14
        )

    def test_helium_additivity_and_abundance_scaling(self):
        mu = np.linspace(-1, 1, 51)
        a = cold_compton_components(
            simple_material(0.1), 4.9, mu, grain_electrons="omit"
        )
        b = cold_compton_components(
            simple_material(0.2), 4.9, mu, grain_electrons="omit"
        )
        np.testing.assert_array_equal(a["gas:H"], b["gas:H"])
        np.testing.assert_allclose(2 * a["gas:He"], b["gas:He"], rtol=1e-14)

    def test_all_atomic_inventory_is_depletion_invariant(self):
        gas = replace(self.material, grains=())
        a = build_cold_compton_table(
            gas, [2, 4.9, 10], grain_electrons="atomic_incoherent"
        )
        b = build_cold_compton_table(
            self.material, [2, 4.9, 10], grain_electrons="atomic_incoherent"
        )
        np.testing.assert_allclose(
            a.table.cross_section_cm2_per_h, b.table.cross_section_cm2_per_h, rtol=3e-15
        )

    def test_component_integrals_sum_to_total(self):
        np.testing.assert_allclose(
            sum(self.result.component_cross_sections_cm2_per_h.values()),
            self.result.table.cross_section_cm2_per_h,
            rtol=4e-15,
        )

    def test_pure_h_against_independent_analytic_quadrature(self):
        result = build_cold_compton_table(
            simple_material(), [2, 3.3, 4.9, 6.9, 10], grain_electrons="omit"
        )
        expected = [hydrogen_quadrature(e) for e in result.table.energy_kev]
        np.testing.assert_allclose(
            result.table.cross_section_cm2_per_h, expected, rtol=2e-6
        )

    def test_hydrogen_published_total_cross_sections(self):
        # Hubbell Table V, barns/H at exact 2, 5, 10 keV nodes. Published
        # precision/1975 constants are distinct from our grid discretization.
        result = build_cold_compton_table(
            simple_material(), [2, 5, 10], grain_electrons="omit"
        )
        np.testing.assert_allclose(
            result.table.cross_section_cm2_per_h / 1e-24,
            [0.2478, 0.5187, 0.5993],
            rtol=7e-4,
        )

    def test_molecular_published_totals_per_h_nucleus(self):
        # Hubbell Table V is PER H IN H2, not per molecule. Independent
        # integrated reference tests both source interpolation and the ledger.
        result = build_cold_compton_table(
            simple_material(h2=1), [2, 4, 5, 10], grain_electrons="omit"
        )
        np.testing.assert_allclose(
            result.table.cross_section_cm2_per_h / 1e-24,
            [0.2024, 0.4206, 0.4809, 0.5855],
            rtol=4e-3,
        )

    def test_support_rejection(self):
        for energies in ([1.9, 3], [3, 10.1], [2, np.nan], [3, 2]):
            with self.assertRaises(ValueError):
                build_cold_compton_table(
                    self.material, energies, grain_electrons="omit"
                )
        with self.assertRaises(ValueError):
            cold_compton_components(self.material, 4.9, 1.1, grain_electrons="omit")

    def test_angular_grid_convergence(self):
        coarse = build_cold_compton_table(
            self.material, [2, 4.9, 10], grain_electrons="omit", cosine_nodes=1025
        )
        fine = build_cold_compton_table(
            self.material, [2, 4.9, 10], grain_electrons="omit", cosine_nodes=8193
        )
        np.testing.assert_allclose(
            coarse.table.cross_section_cm2_per_h,
            fine.table.cross_section_cm2_per_h,
            rtol=3e-5,
        )

    def test_phase_normalization_at_intermediate_energy(self):
        runtime = to_compton_physics(self.result.table)
        mu = np.linspace(-1, 1, 16385)
        value = jax.jit(evaluate_compton)(runtime, 4.2, mu)
        area = np.pi * np.sum(
            np.diff(mu)
            * (
                np.asarray(value.phase_pdf_per_sr)[1:]
                + np.asarray(value.phase_pdf_per_sr)[:-1]
            )
        )
        self.assertAlmostEqual(float(area), 1, delta=3e-6)
        self.assertEqual(float(evaluate_compton(runtime, 4.2, 1).phase_pdf_per_sr), 0)

    def test_dense_energy_grid_opacity_interpolation(self):
        energy = np.geomspace(2, 10, 129)
        midpoints = np.sqrt(energy[:-1] * energy[1:])
        built = build_cold_compton_table(self.material, energy, grain_electrons="omit")
        direct = build_cold_compton_table(
            self.material, midpoints, grain_electrons="omit"
        )
        evaluated = evaluate_compton(to_compton_physics(built.table), midpoints, 0.2)
        np.testing.assert_allclose(
            evaluated.cross_section_cm2_per_h,
            direct.table.cross_section_cm2_per_h,
            rtol=4e-5,
            atol=0,
        )

    def test_sampler_matches_bound_angular_cdf_and_recoil(self):
        runtime = to_compton_physics(self.result.table)
        keys = jax.random.split(jax.random.PRNGKey(8913), 40000)
        samples = jax.jit(
            jax.vmap(
                lambda key: sample_compton_scattering(key, runtime, jnp.asarray(4.9))
            )
        )(keys)
        mu = np.sort(np.asarray(samples.cosine))
        n = len(mu)
        cdf = np.interp(mu, self.result.table.cosine, self.result.table.cosine_cdf[2])
        ks = max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(n) / n))
        self.assertLess(ks, 0.013)
        self.assertTrue(np.asarray(samples.supported).all())
        np.testing.assert_allclose(
            np.asarray(samples.outgoing_energy_kev)
            + np.asarray(samples.recoil_energy_kev),
            4.9,
            rtol=2e-7,
        )

    def test_archive_preserves_physical_contract_and_material_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cold.npz"
            write_compton_material_table(path, self.result.table)
            loaded = load_compton_material_table(
                path, expected_material_definition_sha256=self.material.sha256
            )
            self.assertEqual(loaded.metadata["grain_electrons"], "omit")
            self.assertEqual(
                loaded.metadata["energy_redistribution"], "stationary_electron_compton"
            )
            np.testing.assert_array_equal(
                loaded.differential_cm2_per_sr_per_h,
                self.result.table.differential_cm2_per_sr_per_h,
            )
            with self.assertRaisesRegex(ValueError, "composition"):
                load_compton_material_table(
                    path, expected_material_definition_sha256="0" * 64
                )


if __name__ == "__main__":
    unittest.main()
