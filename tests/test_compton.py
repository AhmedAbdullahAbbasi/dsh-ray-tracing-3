"""Independent physical and numerical checks for the Compton foundation."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from dsh.physics.compton import (
    ELECTRON_REST_ENERGY_KEV,
    THOMSON_CROSS_SECTION_CM2,
    build_compton_material_table,
    build_free_electron_compton_table,
    compton_recoil_kinematics,
    evaluate_compton,
    klein_nishina_differential_cm2_per_sr,
    klein_nishina_total_cm2,
    load_compton_material_table,
    sample_compton_scattering,
    to_compton_physics,
    write_compton_material_table,
)

MATERIAL_SHA = "a" * 64


def angular_reference(energy, mu):
    """Independent scalar/NumPy expression via the classical electron radius."""
    k = 1 / (1 + energy / ELECTRON_REST_ENERGY_KEV * (1 - mu))
    radius_squared = THOMSON_CROSS_SECTION_CM2 * 3 / (8 * np.pi)
    return radius_squared / 2 * k * k * (k + 1 / k - (1 - mu * mu))


def quadrature_total(energy):
    mu, weights = np.polynomial.legendre.leggauss(256)
    return 2 * np.pi * np.sum(weights * angular_reference(energy, mu))


def free_table(energy=(0.1, 2.0, 3.3, 4.9, 6.9, 10.0, 100.0, 1000.0), electrons=1.2):
    return build_free_electron_compton_table(
        energy,
        electrons_per_h=electrons,
        material_definition_sha256=MATERIAL_SHA,
    )


class ComptonReferenceTests(unittest.TestCase):
    def test_thomson_limit_and_differential(self):
        mu = np.linspace(-1, 1, 101)
        value = np.asarray(klein_nishina_differential_cm2_per_sr(0.0, mu))
        expected = 3 * THOMSON_CROSS_SECTION_CM2 / (16 * np.pi) * (1 + mu**2)
        np.testing.assert_allclose(value, expected, rtol=3e-7, atol=0)
        self.assertAlmostEqual(
            float(klein_nishina_total_cm2(0.0)) / THOMSON_CROSS_SECTION_CM2,
            1.0,
            delta=2e-7,
        )

    def test_total_against_independent_full_sphere_quadrature(self):
        energies = np.array(
            [
                1e-7,
                0.1,
                2.0,
                3.3,
                4.9,
                6.9,
                10.0,
                63.8,
                63.9,
                64.0,
                100.0,
                511.0,
                1000.0,
            ]
        )
        expected = np.array([quadrature_total(e) for e in energies])
        actual = jax.jit(klein_nishina_total_cm2)(jnp.asarray(energies))
        tolerance = 2e-5 if not jax.config.x64_enabled else 2e-12
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=0)

    def test_float32_low_energy_accuracy_and_series_transition(self):
        energies = np.array([0.001, 2.0, 4.9, 10.0, 63.8, 63.9, 64.0], dtype=np.float32)
        expected = np.array([quadrature_total(float(e)) for e in energies])
        actual = jax.jit(klein_nishina_total_cm2)(jnp.asarray(energies))
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=0)

    def test_forward_and_backward_recoil_limits(self):
        e = np.array([0.1, 4.9, 10.0, 511.0, 1000.0])
        out, recoil = compton_recoil_kinematics(e, np.ones(e.shape))
        np.testing.assert_allclose(out, e, rtol=2e-7)
        np.testing.assert_array_equal(recoil, 0.0)
        out, recoil = compton_recoil_kinematics(e, -np.ones(e.shape))
        np.testing.assert_allclose(
            out, e / (1 + 2 * e / ELECTRON_REST_ENERGY_KEV), rtol=2e-7
        )
        np.testing.assert_allclose(np.asarray(out) + np.asarray(recoil), e, rtol=2e-7)

    def test_small_recoil_is_not_lost_to_energy_subtraction(self):
        energy, mu = jnp.float32(0.1), jnp.float32(1 - 1e-5)
        _, recoil = compton_recoil_kinematics(energy, mu)
        expected = float(energy) ** 2 / ELECTRON_REST_ENERGY_KEV * (1 - float(mu))
        self.assertGreater(float(recoil), 0)
        self.assertAlmostEqual(float(recoil) / expected, 1.0, delta=5e-7)

    def test_recoil_electron_mass_shell(self):
        e = np.array([2.0, 4.9, 10.0, 511.0])
        mu = np.array([-0.7, 0.3, 0.8, -0.5])
        outgoing, recoil = map(np.asarray, compton_recoil_kinematics(e, mu))
        # Incident photon along z; recoil spatial momentum is k-k'.
        p_squared = (outgoing * np.sqrt(1 - mu**2)) ** 2 + (e - outgoing * mu) ** 2
        np.testing.assert_allclose(
            p_squared,
            recoil * (recoil + 2 * ELECTRON_REST_ENERGY_KEV),
            rtol=3e-6,
            atol=0,
        )

    def test_invalid_analytic_inputs_are_explicit(self):
        values = klein_nishina_total_cm2([-1.0, np.nan, np.inf])
        self.assertTrue(np.isnan(values).all())
        for energy, mu in [(-1.0, 0.0), (1.0, 1.1), (1.0, np.nan), (np.inf, 0.0)]:
            self.assertTrue(np.isnan(klein_nishina_differential_cm2_per_sr(energy, mu)))
            self.assertTrue(np.isnan(compton_recoil_kinematics(energy, mu)[0]))


class ComptonTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = free_table()
        cls.physics = to_compton_physics(cls.table)

    def test_per_h_normalization_and_explicit_electron_inventory(self):
        other = free_table(electrons=0.6)
        np.testing.assert_allclose(
            other.cross_section_cm2_per_h,
            self.table.cross_section_cm2_per_h / 2,
            rtol=1e-14,
        )
        expected = 1.2 * np.array([quadrature_total(e) for e in self.table.energy_kev])
        np.testing.assert_allclose(
            self.table.cross_section_cm2_per_h, expected, rtol=2e-6, atol=0
        )

    def test_phase_and_cdf_have_consistent_integrals(self):
        phase = (
            self.table.differential_cm2_per_sr_per_h
            / self.table.cross_section_cm2_per_h[:, None]
        )
        integral = 2 * np.pi * np.trapezoid(phase, self.table.cosine, axis=1)
        np.testing.assert_allclose(integral, 1.0, rtol=1e-13)
        np.testing.assert_array_equal(self.table.cosine_cdf[:, 0], 0.0)
        np.testing.assert_array_equal(self.table.cosine_cdf[:, -1], 1.0)
        self.assertTrue((np.diff(self.table.cosine_cdf, axis=1) >= 0).all())
        self.assertFalse(self.table.energy_kev.flags.writeable)

    def test_evaluation_at_nodes_and_log_energy_mixture(self):
        compiled = jax.jit(evaluate_compton)
        mu = np.linspace(-1, 1, 4097)
        energy = math.sqrt(4.9 * 6.9)
        value = compiled(self.physics, energy, mu)
        p = (
            self.table.differential_cm2_per_sr_per_h
            / self.table.cross_section_cm2_per_h[:, None]
        )
        np.testing.assert_allclose(value.phase_pdf_per_sr, (p[3] + p[4]) / 2, rtol=2e-6)
        integral = 2 * np.pi * np.trapezoid(np.asarray(value.phase_pdf_per_sr), mu)
        self.assertAlmostEqual(integral, 1.0, delta=5e-7)
        target = math.sqrt(
            self.table.cross_section_cm2_per_h[3]
            * self.table.cross_section_cm2_per_h[4]
        )
        np.testing.assert_allclose(value.cross_section_cm2_per_h, target, rtol=5e-6)

    def test_support_bounds_are_not_silent_energy_clipping(self):
        for energy in [0.099, 1000.1, np.nan, np.inf, -1.0]:
            value = jax.jit(evaluate_compton)(self.physics, energy, 0.0)
            self.assertFalse(bool(value.supported))
            self.assertEqual(float(value.cross_section_cm2_per_h), 0.0)
            sample = jax.jit(sample_compton_scattering)(
                random.PRNGKey(1), self.physics, energy
            )
            self.assertFalse(bool(sample.supported))
            self.assertEqual(float(sample.outgoing_energy_kev), 0.0)
        for energy in [0.1, 1000.0]:
            self.assertTrue(bool(evaluate_compton(self.physics, energy, 0.0).supported))

    def test_malformed_angular_tables_and_inconsistent_totals_rejected(self):
        t = self.table
        for mu, diff in [
            (t.cosine[1:], t.differential_cm2_per_sr_per_h[:, 1:]),
            (t.cosine, np.zeros_like(t.differential_cm2_per_sr_per_h)),
            (t.cosine, -t.differential_cm2_per_sr_per_h),
            (t.cosine, np.full_like(t.differential_cm2_per_sr_per_h, np.nan)),
        ]:
            with self.assertRaises(ValueError):
                build_compton_material_table(
                    t.energy_kev, mu, diff, metadata=t.metadata
                )
        with self.assertRaisesRegex(ValueError, "disagrees"):
            build_compton_material_table(
                t.energy_kev,
                t.cosine,
                t.differential_cm2_per_sr_per_h,
                metadata=t.metadata,
                expected_cross_section_cm2_per_h=2 * t.cross_section_cm2_per_h,
            )

    def test_metadata_requires_composition_and_model_declarations(self):
        t = self.table
        for field, value in [
            ("model", "reflex_matched"),
            ("electron_state", "bound"),
            ("material_definition_sha256", "unknown"),
            ("normalization", "per electron"),
            ("electrons_per_h", 0.0),
            ("energy_redistribution", "unspecified"),
            ("polarization", "polarized"),
        ]:
            metadata = dict(t.metadata, **{field: value})
            with self.assertRaises(ValueError):
                build_compton_material_table(
                    t.energy_kev,
                    t.cosine,
                    t.differential_cm2_per_sr_per_h,
                    metadata=metadata,
                )

    def test_external_angular_table_is_explicitly_supported(self):
        t = self.table
        # Synthetic input exercises the external interface, NOT bound physics validation.
        metadata = dict(
            t.metadata,
            model="external_incoherent",
            electron_state="bound",
            producer="synthetic test only",
            reference="synthetic interface fixture",
        )
        external = build_compton_material_table(
            t.energy_kev, t.cosine, t.differential_cm2_per_sr_per_h, metadata=metadata
        )
        self.assertEqual(external.metadata["model"], "external_incoherent")

    def test_invalid_free_electron_inventory_and_coarse_grid_rejected(self):
        for value in [0.0, -1.0, np.inf, np.nan, True]:
            with self.assertRaises(ValueError):
                free_table(electrons=value)
        with self.assertRaisesRegex(ValueError, "too coarse"):
            build_free_electron_compton_table(
                [1.0, 1000.0],
                electrons_per_h=1.0,
                material_definition_sha256=MATERIAL_SHA,
                cosine_nodes=3,
            )


class ComptonSamplingTests(unittest.TestCase):
    def test_sampler_matches_independent_kn_cdf_and_recoil(self):
        table = to_compton_physics(free_table())
        keys = random.split(random.PRNGKey(84325), 24000)
        compiled = jax.jit(jax.vmap(sample_compton_scattering, in_axes=(0, None, None)))
        for energy in [4.9, 1000.0]:
            samples = compiled(keys, table, energy)
            self.assertTrue(np.asarray(samples.supported).all())
            mu = np.sort(np.asarray(samples.cosine))
            grid = np.linspace(-1, 1, 100001)
            density = angular_reference(energy, grid)
            cdf = np.concatenate(
                [[0.0], np.cumsum((density[1:] + density[:-1]) / 2 * np.diff(grid))]
            )
            cdf /= cdf[-1]
            expected = np.interp(mu, grid, cdf)
            n = len(mu)
            statistic = max(
                np.max(np.arange(1, n + 1) / n - expected),
                np.max(expected - np.arange(n) / n),
            )
            self.assertLess(statistic, 2.2 / math.sqrt(n))
            np.testing.assert_allclose(
                np.asarray(samples.outgoing_energy_kev) + samples.recoil_energy_kev,
                energy,
                rtol=3e-7,
            )
            phi = np.asarray(samples.azimuth_rad)
            self.assertTrue(((phi >= 0) & (phi < 2 * np.pi)).all())
            self.assertLess(abs(np.cos(phi).mean()), 0.025)
            self.assertLess(abs(np.sin(phi).mean()), 0.025)

    def test_inverse_sampler_is_consistent_on_coarse_linear_phase(self):
        original = free_table()
        mu = np.array([-1.0, 0.0, 1.0])
        # A triangular density, zero in the backward hemisphere. Exact CDF=mu^2
        # in the forward hemisphere; inverse MUST be sqrt(u), not u.
        differential = np.tile(np.array([0.0, 0.0, 1.0]), (2, 1))
        table = to_compton_physics(
            build_compton_material_table(
                [1.0, 10.0], mu, differential, metadata=original.metadata
            )
        )
        keys = random.split(random.PRNGKey(3719), 16000)
        result = jax.jit(jax.vmap(sample_compton_scattering, in_axes=(0, None, None)))(
            keys, table, 4.9
        )
        x = np.asarray(result.cosine)
        self.assertTrue((x >= 0).all())
        self.assertAlmostEqual(float(x.mean()), 2 / 3, delta=0.012)
        self.assertAlmostEqual(float((x * x).mean()), 0.5, delta=0.012)


class ComptonArchiveTests(unittest.TestCase):
    def test_roundtrip_and_wrong_material_rejection(self):
        table = free_table()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compton.npz"
            write_compton_material_table(path, table)
            loaded = load_compton_material_table(
                path, expected_material_definition_sha256=MATERIAL_SHA
            )
            np.testing.assert_array_equal(
                loaded.cross_section_cm2_per_h, table.cross_section_cm2_per_h
            )
            self.assertEqual(loaded.metadata, table.metadata)
            with self.assertRaisesRegex(ValueError, "composition"):
                load_compton_material_table(
                    path, expected_material_definition_sha256="b" * 64
                )

    def test_array_or_metadata_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compton.npz"
            for which in ["array", "metadata"]:
                write_compton_material_table(path, free_table())
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {key: np.array(archive[key]) for key in archive.files}
                if which == "array":
                    arrays["differential_cm2_per_sr_per_h"][0, 0] *= 2
                else:
                    metadata = json.loads(str(arrays["metadata_json"].item()))
                    metadata["material"]["electrons_per_h"] = 2.0
                    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
                np.savez_compressed(path, **arrays)
                with self.assertRaisesRegex(ValueError, "checksum"):
                    load_compton_material_table(
                        path, expected_material_definition_sha256=MATERIAL_SHA
                    )


if __name__ == "__main__":
    unittest.main()
