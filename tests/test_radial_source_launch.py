"""Independent distribution, support, and importance checks for circular launches."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from dsh.sources.launch import build_rectangular_launch_geometry, sample_source_launches
from dsh.sources.models import SourcePackets
from dsh.sources.radial_launch import (
    build_radial_launch_geometry,
    radial_launch_pdf_per_sr,
)


class TestRadialSourceLaunch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = build_radial_launch_geometry(
            build_rectangular_launch_geometry(11.5, [-0.01, 0.015], [-0.02, 0.008]),
            [3.3, 4.9],
            [[0.002, 0.006], [0.001, 0.003]],
            [1.0, 2.0],
        )
        n = 100000
        cls.packets = SourcePackets(
            jnp.full(n, 3.3),
            jnp.arange(n, dtype=jnp.float32),
            jnp.ones(n) / n,
            jnp.arange(n, dtype=jnp.int32) % 7,
            jnp.arange(n, dtype=jnp.int32) % 3,
        )
        cls.samples = jax.jit(sample_source_launches)(
            random.PRNGKey(912), cls.packets, cls.geometry
        )
        momentum = np.asarray(cls.samples.momentum_kev)
        cls.direction = momentum[:, 1:] / momentum[:, 0, None]
        cls.beta = np.arctan2(
            np.hypot(cls.direction[:, 1], cls.direction[:, 2]), -cls.direction[:, 0]
        )

    def test_sample_cdf_matches_analytic_mixture(self):
        for radius in [0.0007, 0.002, 0.006, 0.015, 0.05, 1.0]:
            scales = np.array([0.002, 0.006])
            expected = 0.95 * np.dot(
                [1 / 3, 2 / 3], radius**2 / (scales**2 + radius**2)
            )
            expected += 0.05 * (1 - 1 / np.sqrt(1 + radius**2))
            measured = np.mean(self.beta <= np.arctan(radius))
            self.assertLess(
                abs(measured - expected),
                5 * np.sqrt(expected * (1 - expected) / len(self.beta)) + 1e-4,
            )

    def test_pdf_normalization_with_independent_solid_angle_quadrature(self):
        beta = np.r_[0.0, np.geomspace(1e-8, np.pi / 2, 100000)]
        pdf = np.asarray(
            radial_launch_pdf_per_sr(
                jnp.asarray(np.cos(beta)),
                jnp.asarray(np.sin(beta) ** 2),
                jnp.array([0.002, 0.006]),
                self.geometry.component_probability,
                0.05,
            )
        )
        mass = np.trapezoid(pdf * 2 * np.pi * np.sin(beta), beta)
        self.assertAlmostEqual(float(mass), 1.0, delta=3e-6)

    def test_pdf_matches_slope_jacobian_and_has_no_box_edges(self):
        beta = np.array([0.0, 1e-7, 0.002, 0.01, 0.02, 0.8, 1.5])
        u = np.tan(beta)
        scales = np.array([0.002, 0.006])
        p_uv = np.sum(
            np.array([1 / 3, 2 / 3])
            * scales**2
            / (np.pi * (scales**2 + u[:, None] ** 2) ** 2),
            axis=1,
        )
        expected = 0.95 * p_uv * (1 + u * u) ** 1.5 + 0.05 / (2 * np.pi)
        actual = np.asarray(
            radial_launch_pdf_per_sr(
                jnp.asarray(np.cos(beta)),
                jnp.asarray(np.sin(beta) ** 2),
                jnp.asarray(scales),
                self.geometry.component_probability,
                0.05,
            )
        )
        np.testing.assert_allclose(actual, expected, rtol=5e-7)
        self.assertTrue(np.all(actual > 0))

    def test_importance_integrates_known_finite_cone(self):
        weights = np.asarray(self.samples.isotropic_importance, dtype=np.float64) * (
            self.beta < 0.015
        )
        expected = (1 - np.cos(0.015)) / 2
        error = weights.std(ddof=1) / np.sqrt(weights.size)
        self.assertLess(abs(weights.mean() - expected), 5 * error)

    def test_azimuth_is_uniform_including_fourfold_mode(self):
        phi = np.arctan2(self.direction[:, 2], self.direction[:, 1])
        bound = 5 / np.sqrt(2 * len(phi))
        for mode in [1, 2, 4]:
            self.assertLess(abs(np.cos(mode * phi).mean()), bound)
            self.assertLess(abs(np.sin(mode * phi).mean()), bound)

    def test_null_momenta_metadata_and_defensive_bound(self):
        np.testing.assert_allclose(np.linalg.norm(self.direction, axis=1), 1, rtol=3e-7)
        self.assertTrue(np.all(self.direction[:, 0] <= 0))
        q = np.asarray(self.samples.launch_pdf_per_sr)
        importance = np.asarray(self.samples.isotropic_importance)
        self.assertTrue(np.all(np.isfinite(q)) and np.all(q > 0))
        self.assertLessEqual(importance.max(), 10.00001)
        np.testing.assert_allclose(importance, 1 / (4 * np.pi * q), rtol=3e-7)
        for field in [
            "emission_time_s",
            "weight_observer_fluence",
            "time_index",
            "spectral_bin_index",
        ]:
            np.testing.assert_array_equal(
                getattr(self.samples, field), getattr(self.packets, field)
            )

    def test_continuous_energy_uses_normalized_interpolated_proposal(self):
        energy = np.sqrt(3.3 * 4.9)
        packets = self.packets._replace(energy_kev=jnp.full(len(self.beta), energy))
        samples = jax.jit(sample_source_launches)(
            random.PRNGKey(319), packets, self.geometry
        )
        p = np.asarray(samples.momentum_kev)
        beta = np.arctan2(np.hypot(p[:, 2], p[:, 3]), -p[:, 1])
        radius = 0.003
        scales = np.sqrt(np.array([0.002, 0.006]) * np.array([0.001, 0.003]))
        expected = 0.95 * np.dot(
            [1 / 3, 2 / 3], radius**2 / (scales**2 + radius**2)
        ) + 0.05 * (1 - 1 / np.sqrt(1 + radius**2))
        self.assertLess(
            abs(np.mean(beta <= np.arctan(radius)) - expected),
            5 * np.sqrt(expected * (1 - expected) / len(beta)),
        )

    def test_invalid_proposal_rejected(self):
        coverage = build_rectangular_launch_geometry(10.0, [-0.01, 0.01], [-0.01, 0.01])
        for scales, weights, defense in [
            ([[0.0], [0.01]], [1.0], 0.05),
            ([[0.01], [0.02]], [-1.0], 0.05),
            ([[0.01], [0.02]], [1.0], 0.0),
        ]:
            with self.assertRaises(ValueError):
                build_radial_launch_geometry(
                    coverage,
                    [3.0, 5.0],
                    scales,
                    weights,
                    hemisphere_probability=defense,
                )


if __name__ == "__main__":
    unittest.main()
