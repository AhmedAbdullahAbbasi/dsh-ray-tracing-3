"""Independent 9F reference checks against production material evaluation."""

import unittest
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from dsh.observer.scoring import scattering_phase_pdf_per_sr
from dsh.physics.materials import load_2_10_material_tables
from dsh.validation.absorbed_observer import (
    DAY_S,
    first_order_quadrature,
    first_order_radial_quadrature,
    host_material,
    host_phase,
    score_scattering_only_histories,
)


class TestAbsorbedObserverReference(unittest.TestCase):
    def test_reference_respects_production_sky_window(self):
        physics = load_2_10_material_tables()[2]
        row = SimpleNamespace(
            position_pc=np.array([[[4500.0, 0.2, 0.0]]]),
            incoming_momentum_kev=np.array([[[5.35, -5.35, 0.0, 0.0]]]),
            valid=np.array([[True]]),
            interaction_type=np.array([[1]]),
            scattering_order=np.array([[1]]),
        )
        history = SimpleNamespace(interactions=row, status=np.array([1]))
        launched = SimpleNamespace(
            launch_pdf_per_sr=np.array([1.0]), weight_observer_fluence=np.array([1.0])
        )
        args = (launched, history, physics, 5.35, 1.0e23, np.array([0.0, 30 * DAY_S]))
        narrow = score_scattering_only_histories(*args, [-1.0, 1.0], [-1800.0, 1800.0])
        wide = score_scattering_only_histories(
            *args, [-100.0, 100.0], [-1800.0, 1800.0]
        )
        self.assertEqual(narrow["sum"].sum(), 0.0)
        self.assertGreater(wide["sum"].sum(), 0.0)

    def test_host_off_node_phase_and_absorbed_first_order(self):
        physics = load_2_10_material_tables()[2]
        e = 4.0747680326
        for angle in (1e-8, 1e-5, 1e-4, 1e-3):
            host = float(host_phase(physics, e, angle))
            device = float(
                scattering_phase_pdf_per_sr(physics, jnp.float32(e), jnp.float32(angle))
            )
            with self.subTest(angle=angle):
                self.assertGreater(host, 0)
                self.assertLess(abs(host - device) / host, 2e-5)

        _, _, _, sigma, _ = host_material(physics, e)
        column = 1.5 / sigma
        bounds = ((-0.0015, 0.0015), (-0.0015, 0.0015))
        edges = np.array([0, 0.5, 2, 30]) * DAY_S
        with_abs = first_order_quadrature(physics, e, column, bounds, edges, 24, 16)
        scatter_only = physics._replace(
            absorption_cross_section_cm2_per_h=jnp.zeros_like(
                physics.absorption_cross_section_cm2_per_h
            )
        )
        without_abs = first_order_quadrature(
            scatter_only, e, column, bounds, edges, 24, 16
        )
        self.assertGreater(with_abs.sum(), 0)
        self.assertLess(with_abs.sum(), without_abs.sum())
        self.assertTrue(np.all(with_abs >= 0))

    def test_radial_time_bin_reference_crosschecks_cartesian_integral(self):
        physics = load_2_10_material_tables()[2]
        bounds = ((-0.003, 0.003), (-0.003, 0.003))
        edges = np.array([0, 0.5, 2, 6, 30]) * DAY_S
        for energy in (4.0747680326, 5.35):
            column = 1.5 / host_material(physics, energy)[3]
            low = first_order_radial_quadrature(
                physics, energy, column, bounds, edges, n_radius=24, n_depth=24
            )
            high = first_order_radial_quadrature(
                physics, energy, column, bounds, edges, n_radius=64, n_depth=56
            )
            cartesian = first_order_quadrature(
                physics, energy, column, bounds, edges, n_slope=128, n_depth=48
            )
            with self.subTest(energy=energy):
                np.testing.assert_allclose(low, high, rtol=0.002, atol=0)
                np.testing.assert_allclose(high.sum(), cartesian.sum(), rtol=1e-4)
                np.testing.assert_allclose(
                    high[:2].sum(), cartesian[:2].sum(), rtol=1e-3
                )

    def test_radial_reference_requires_centered_launch_cone(self):
        physics = load_2_10_material_tables()[2]
        with self.assertRaisesRegex(ValueError, "centered rectangular cone"):
            first_order_radial_quadrature(
                physics, 5.35, 1e23, ((0.0, 0.003), (-0.003, 0.003)), [0, DAY_S]
            )


if __name__ == "__main__":
    unittest.main()
