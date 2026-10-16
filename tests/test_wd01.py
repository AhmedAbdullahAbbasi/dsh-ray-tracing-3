"""Independent WD01 moments, sphere solutions, and actual table round trips."""

from __future__ import annotations

import importlib.util
import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dsh.physics.grain_populations import WD01Population
from dsh.physics.material_definitions import (
    ATOMIC_MASS_UNIT_G,
    ATOMIC_WEIGHTS_U,
    load_wilm_abundances,
)
from dsh.physics.optical_constants import (
    HC_KEV_UM,
    OPTICAL_DIRECTORY,
    default_wd01_energy_grid,
    load_draine_optical_constants,
)
from dsh.physics.sphere_scattering import (
    _shadow_transform,
    adt_forward_differential,
    adt_sphere_efficiencies,
    hybrid_sphere,
    mie_sphere,
)

HAS_PRODUCERS = all(
    importlib.util.find_spec(name) is not None for name in ("scipy", "miepython")
)


class WD01PopulationTests(unittest.TestCase):
    def test_export_cannot_mutate_population_law_or_inventory(self):
        population = WD01Population()
        fingerprint = population.sha256
        inventory = population.inventory()
        exported = population.as_dict()
        exported["parameters"]["silicate"] = (0, 0, 1, 1, 0)
        exported["densities_g_cm3"]["silicate"] = 0
        exported["d03_optical_material_densities_g_cm3"]["carbonaceous"] = 0
        self.assertEqual(population.sha256, fingerprint)
        self.assertEqual(population.inventory(), inventory)

    def test_moments_match_independent_published_table1(self):
        inventory = WD01Population().inventory()
        # Published Table 1 has rounded fit parameters AND rounded volumes.
        # Its column definitions use 2.07e-27 and 2.98e-27 cm3/H, respectively.
        self.assertAlmostEqual(
            inventory["carbonaceous"]["volume_cm3_per_h"] / 2.07e-27, 1.092, delta=0.005
        )
        self.assertAlmostEqual(
            inventory["silicate"]["volume_cm3_per_h"] / 2.98e-27, 1.322, delta=0.007
        )

    def test_lognormal_carbon_count_is_60ppm(self):
        population = WD01Population()
        radius, _ = population.quadrature("carbonaceous", 48)
        # Independent dense log-radius integration of just the log-normal term.
        radius = np.geomspace(0.00035, 2, 40001)
        count = population.dn_da_per_h_per_um(
            radius, "carbonaceous", lognormal_only=True
        )
        mass = np.trapezoid(count * 4 * np.pi / 3 * (radius * 1e-4) ** 3 * 2.24, radius)
        self.assertAlmostEqual(
            mass / (ATOMIC_WEIGHTS_U["C"] * ATOMIC_MASS_UNIT_G), 60e-6, delta=2e-11
        )

    def test_quadrature_and_radius_cutoff_converge(self):
        p = WD01Population()
        for name in ("carbonaceous", "silicate"):
            coarse = p.inventory(16)[name]["volume_cm3_per_h"]
            fine = p.inventory(48)[name]["volume_cm3_per_h"]
            extended = WD01Population(maximum_radius_um=3).inventory(48)[name][
                "volume_cm3_per_h"
            ]
            self.assertLess(abs(coarse / fine - 1), 1e-9)
            self.assertLess(abs(fine / extended - 1), 1e-11)

    def test_named_and_explicit_normalizations_are_linear(self):
        p = WD01Population()
        for name, scale in (("carbonaceous", 0.5), ("silicate", 0.35)):
            original = p.inventory()[name]["volume_cm3_per_h"]
            self.assertAlmostEqual(
                WD01Population("d03_0p93").inventory()[name]["volume_cm3_per_h"]
                / original,
                0.93,
            )
            actual = WD01Population(carbon_scale=0.5, silicate_scale=0.35).inventory()[
                name
            ]["volume_cm3_per_h"]
            self.assertAlmostEqual(actual / original, scale)

    def test_optical_density_and_abundance_budget_are_explicit(self):
        nominal = WD01Population()
        with self.assertRaisesRegex(ValueError, "exceeds .* abundance"):
            nominal.material_definition(load_wilm_abundances())
        example = WD01Population(carbon_scale=0.5, silicate_scale=0.35)
        definition = example.material_definition(load_wilm_abundances())
        report = definition.audit()
        self.assertGreater(report["elements"]["Si"]["gas_atoms_per_h"], 0)
        self.assertEqual(
            {g.component_id: g.density_g_cm3 for g in definition.grains},
            {"carbonaceous": 2.2, "silicate": 3.8},
        )
        self.assertNotEqual(
            nominal.inventory()["silicate"]["mass_g_per_h"],
            nominal.inventory(density_model="d03")["silicate"]["mass_g_per_h"],
        )

    def test_invalid_population_inputs_fail(self):
        for kwargs in (
            {"carbon_scale": 0},
            {"silicate_scale": -1},
            {"maximum_radius_um": float("nan")},
            {"normalization": "unknown"},
        ):
            with self.assertRaises(ValueError):
                WD01Population(**kwargs)
        with self.assertRaises(ValueError):
            WD01Population().dn_da_per_h_per_um([0], "silicate")


class OpticalConstantsTests(unittest.TestCase):
    def test_native_samples_and_passive_sign_are_preserved(self):
        for name in ("silicate", "graphite_parallel", "graphite_perpendicular"):
            table = load_draine_optical_constants(name)
            supported = (table.energy_kev >= 2) & (table.energy_kev <= 10)
            index = table.index_at_energy(table.energy_kev[supported])
            np.testing.assert_allclose(
                index.real - 1,
                table.real_index_minus_one[supported],
                rtol=1e-10,
                atol=1e-16,
            )
            np.testing.assert_allclose(
                index.imag, table.imaginary_index[supported], rtol=1e-12
            )
            self.assertTrue(np.all(index.imag > 0))

    def test_no_extrapolation_or_silent_checksum_repair(self):
        table = load_draine_optical_constants("silicate")
        for energy in (1.9, 10.1, np.nan):
            with self.assertRaises(ValueError):
                table.index_at_energy(energy)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "optics"
            shutil.copytree(OPTICAL_DIRECTORY, target)
            path = target / "silD03.dat"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_draine_optical_constants("silicate", target)

    def test_energy_axis_includes_native_edge_samples(self):
        grid = default_wd01_energy_grid()
        self.assertEqual((grid[0], grid[-1]), (2, 10))
        self.assertTrue(np.all(np.diff(grid) > 0))
        self.assertTrue(all(e in grid for e in (3.3, 4.9, 6.9)))
        table = load_draine_optical_constants("silicate")
        edge = table.energy_kev[(table.energy_kev > 7.1) & (table.energy_kev < 7.2)]
        self.assertGreater(edge.size, 20)
        self.assertTrue(np.all(np.isin(edge, grid)))


class WD01CommandTests(unittest.TestCase):
    def test_invalid_parallel_energy_grid_fails_before_generation(self):
        from scripts.generate_wd01_tables import main

        with tempfile.TemporaryDirectory() as directory:
            for energy in ([2], [3, 2], [2, 2], [1, 3], [2, 11], [2, "nan"]):
                arguments = [
                    "generate_wd01_tables",
                    "--output",
                    directory,
                    "--workers",
                    "2",
                    "--energies",
                    *map(str, energy),
                ]
                with (
                    patch("sys.argv", arguments),
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as result,
                ):
                    main()
                self.assertEqual(result.exception.code, 2)
            self.assertFalse(list(Path(directory).iterdir()))


@unittest.skipUnless(HAS_PRODUCERS, 'requires optional ".[materials]" dependencies')
class SphereScatteringTests(unittest.TestCase):
    def test_wiscombe_independent_published_amplitude_case14(self):
        # MIEV0 case 14, reproduced in miepython's normalization documentation.
        energy = HC_KEV_UM / (2 * np.pi)
        angle = np.deg2rad([0, 30, 60, 90])
        result = mie_sphere(1, energy, 1.5 + 1j, angle)
        k = 2 * np.pi / (HC_KEV_UM / energy * 1e-4)
        np.testing.assert_allclose(
            result.differential_cm2_per_sr * k * k,
            [0.377446, 0.313213, 0.192141, 0.120663],
            rtol=5e-6,
        )

    def test_rayleigh_limit_cross_sections_and_passive_absorption(self):
        x = 1e-3
        m = 1.5 + 0.1j
        result = mie_sphere(1, HC_KEV_UM * x / (2 * np.pi), m, [0, np.pi / 2, np.pi])
        polar = (m * m - 1) / (m * m + 2)
        area = np.pi * 1e-8
        self.assertAlmostEqual(
            result.scattering_cm2 / (area * 8 / 3 * x**4 * abs(polar) ** 2),
            1,
            delta=2e-6,
        )
        self.assertAlmostEqual(
            result.absorption_cm2 / (area * 4 * x * polar.imag), 1, delta=2e-6
        )
        self.assertAlmostEqual(
            result.extinction_cm2 / (result.scattering_cm2 + result.absorption_cm2), 1
        )

    def test_adt_totals_match_mie_in_supported_xray_regime(self):
        for energy in (2, 4.9, 10):
            m = load_draine_optical_constants("silicate").index_at_energy(energy)
            for radius in (0.03, 0.1, 0.3):
                exact = mie_sphere(radius, energy, m, [0])
                x = 2 * np.pi * radius * energy / HC_KEV_UM
                sca, absorbed, ext = adt_sphere_efficiencies(x, m)
                area = np.pi * (radius * 1e-4) ** 2
                self.assertLess(abs(area * sca / exact.scattering_cm2 - 1), 2e-3)
                self.assertLess(abs(area * absorbed / exact.absorption_cm2 - 1), 2e-3)
                self.assertAlmostEqual(ext, sca + absorbed)

    def test_adt_oscillatory_transform_matches_independent_adaptive_integral(self):
        from scipy.integrate import quad
        from scipy.special import j0

        q = np.array([0, 1, 10, 39.9, 40.1, 80, 200])
        for rho in (-0.4 + 0.1j, -7 + 2j):
            direct = np.array(
                [
                    quad(
                        lambda t, rho=rho, value=value: (
                            (
                                t
                                * -np.expm1(1j * rho * t)
                                * j0(value * np.sqrt(1 - t * t))
                            ).real
                        ),
                        0,
                        1,
                        epsabs=1e-13,
                        limit=500,
                    )[0]
                    + 1j
                    * quad(
                        lambda t, rho=rho, value=value: (
                            (
                                t
                                * -np.expm1(1j * rho * t)
                                * j0(value * np.sqrt(1 - t * t))
                            ).imag
                        ),
                        0,
                        1,
                        epsabs=1e-13,
                        limit=500,
                    )[0]
                    for value in q
                ]
            )
            np.testing.assert_allclose(
                _shadow_transform(rho, q), direct, rtol=1e-8, atol=1e-12
            )

    def test_hybrid_uses_mie_at_wide_angles_and_does_not_add_backward_peak(self):
        energy = 4.9
        m = load_draine_optical_constants("silicate").index_at_energy(energy)
        angle = np.r_[0, np.geomspace(1e-8, 0.1, 1024), [0.2, 1, np.pi]]
        exact = mie_sphere(0.1, energy, m, angle)
        hybrid = hybrid_sphere(0.1, energy, m, angle)
        self.assertEqual(hybrid.scattering_cm2, exact.scattering_cm2)
        np.testing.assert_array_equal(
            hybrid.differential_cm2_per_sr[-3:], exact.differential_cm2_per_sr[-3:]
        )
        integral = np.trapezoid(
            2
            * np.pi
            * np.sin(angle[:-3])
            * np.abs(
                hybrid.differential_cm2_per_sr[:-3] - exact.differential_cm2_per_sr[:-3]
            ),
            angle[:-3],
        )
        self.assertLess(integral / exact.scattering_cm2, 2e-3)
        with self.assertRaises(ValueError):
            adt_forward_differential(0.1, energy, m, [np.pi])

    def test_unsupported_sphere_inputs_fail(self):
        for radius, energy, m, angle in (
            (0, 4.9, 1.5, [0]),
            (1, 0, 1.5, [0]),
            (1, 4.9, 1.5 - 0.1j, [0]),
            (1, 4.9, 1.5, [4]),
            (100, 10, 1.5, [0]),
        ):
            with self.assertRaises(ValueError):
                mie_sphere(radius, energy, m, angle)
        with self.assertRaises(ValueError):
            adt_sphere_efficiencies(10, 1.001 + 0.001j)


@unittest.skipUnless(HAS_PRODUCERS, 'requires optional ".[materials]" dependencies')
class WD01TableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dsh.physics.wd01_tables import (
            calculate_wd01_tables,
            default_wd01_angle_grid,
        )

        cls.population = WD01Population()
        cls.arrays = calculate_wd01_tables(
            cls.population,
            [3.3, 6.9],
            default_wd01_angle_grid(256, 64),
            method="mie_adt",
            quadrature_order=8,
        )

    def test_actual_runtime_table_roundtrip_and_grain_only_scope(self):
        from dsh.physics.newdust import build_dust_physics_from_tables
        from dsh.physics.wd01_tables import write_wd01_tables

        with tempfile.TemporaryDirectory() as directory:
            s, a = write_wd01_tables(
                directory,
                self.arrays,
                self.population,
                method="mie_adt",
                quadrature_order=8,
            )
            physics = build_dust_physics_from_tables(s, a)
            self.assertEqual(a.metadata["absorption_coverage"], ["grains"])
            self.assertEqual(
                s.metadata["chemical_status"],
                "grain inventory only; no total abundance set selected",
            )
            self.assertNotIn("material_definition_sha256", s.metadata)
            np.testing.assert_allclose(
                s.scattering_cross_section_cm2_per_h,
                self.arrays["component_mie_scattering"].sum(axis=0),
                rtol=1e-12,
            )
            np.testing.assert_allclose(
                physics.absorption_cross_section_cm2_per_h,
                self.arrays["component_mie_absorption"].sum(axis=0),
                rtol=1e-6,
            )
            self.assertLess(s.metadata["maximum_omitted_phase_probability"], 1e-10)

    def test_writer_cannot_mislabel_population_or_method(self):
        from dsh.physics.wd01_tables import write_wd01_tables

        with tempfile.TemporaryDirectory() as directory:
            for population, method in (
                (WD01Population("d03_0p93"), "mie_adt"),
                (self.population, "mie"),
            ):
                with self.assertRaisesRegex(ValueError, "configuration"):
                    write_wd01_tables(
                        directory,
                        self.arrays,
                        population,
                        method=method,
                        quadrature_order=8,
                    )


if __name__ == "__main__":
    unittest.main()
