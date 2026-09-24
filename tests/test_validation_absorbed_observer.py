"""Independent 9F reference checks against production material evaluation."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

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
from scripts.run_absorbed_annulus_validation import _annular_photon_weights
from scripts.run_absorbed_observer_validation import _compare_first_order_time_bins
from scripts.run_spectral_observer_validation import (
    _allocate_band_packets,
    _edge_breaks,
    _powerlaw_integral,
    spectral_first_order_quadrature,
)


class TestAbsorbedObserverReference(unittest.TestCase):
    def test_annular_first_order_reference_partitions_the_shell(self):
        physics = load_2_10_material_tables()[2]
        energy = 5.35
        _, _, _, sigma, _ = host_material(physics, energy)
        args = (
            physics,
            energy,
            1.5 / sigma,
            ((-0.003, 0.003), (-0.003, 0.003)),
            [0.0, 30 * DAY_S],
        )
        annuli = ((0.0, 45.0), (45.0, 90.0), (90.0, 1800.0))
        whole = first_order_radial_quadrature(*args, n_radius=64, n_depth=56)
        pieces = [
            first_order_radial_quadrature(
                *args, n_radius=64, n_depth=56, annulus_arcsec=annulus
            )
            for annulus in annuli
        ]
        self.assertTrue(all(piece[0] > 0 for piece in pieces))
        np.testing.assert_allclose(sum(pieces), whole, rtol=0.001)

    def test_annular_event_moments_group_by_launched_photon(self):
        events = SimpleNamespace(
            valid=np.array([[True, True], [True, False]]),
            scattering_order=np.array([[1, 2], [1, 0]]),
            sky_x_arcsec=np.array([[20.0, 80.0], [60.0, 0.0]]),
            sky_y_arcsec=np.zeros((2, 2)),
            arrival_time_s=np.array([[DAY_S, DAY_S], [DAY_S, 0.0]]),
            weight_observer_fluence=np.array([[2.0, 7.0], [3.0, 0.0]]),
        )
        weights = _annular_photon_weights(
            events, (0.0, 45.0, 90.0), (0.0, 2 * DAY_S)
        )
        np.testing.assert_array_equal(weights, [[2.0, 0.0], [0.0, 3.0]])

    def test_stratified_spectrum_preserves_band_probabilities(self):
        probabilities = np.array([0.57, 0.23, 0.20])
        allocation = _allocate_band_packets(10, probabilities, [2, 3, 5])
        np.testing.assert_array_equal(allocation, [2, 3, 5])
        np.testing.assert_allclose(
            [np.full(n, p / n).sum() for n, p in zip(allocation, probabilities)],
            probabilities,
        )
        np.testing.assert_array_equal(
            _allocate_band_packets(10, probabilities), [6, 2, 2]
        )
        for requested in ([2, 3], [1, 4, 5], [2, 3, 4]):
            with self.subTest(requested=requested):
                with self.assertRaises(ValueError):
                    _allocate_band_packets(10, probabilities, requested)

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

    def test_independent_scorer_accepts_per_photon_continuous_energies(self):
        physics = load_2_10_material_tables()[2]
        rows = SimpleNamespace(
            position_pc=np.array([[[4500.0, 0.2, 0.0]]] * 2),
            incoming_momentum_kev=np.array(
                [[[2.6, -2.6, 0.0, 0.0]], [[7.3, -7.3, 0.0, 0.0]]]
            ),
            valid=np.ones((2, 1), dtype=bool),
            interaction_type=np.ones((2, 1), dtype=int),
            scattering_order=np.ones((2, 1), dtype=int),
        )
        launched = SimpleNamespace(
            launch_pdf_per_sr=np.ones(2), weight_observer_fluence=np.ones(2) / 2
        )
        history = SimpleNamespace(interactions=rows, status=np.ones(2, dtype=int))
        args = (physics, 1.0e23, [0.0, 30 * DAY_S], [-1800, 1800], [-1800, 1800])
        mixed = score_scattering_only_histories(
            launched, history, args[0], [2.6, 7.3], *args[1:]
        )
        singles = []
        for index, energy in enumerate((2.6, 7.3)):
            one_launch = SimpleNamespace(
                launch_pdf_per_sr=launched.launch_pdf_per_sr[index : index + 1],
                weight_observer_fluence=launched.weight_observer_fluence[
                    index : index + 1
                ],
            )
            one_history = SimpleNamespace(
                interactions=SimpleNamespace(
                    **{
                        key: getattr(rows, key)[index : index + 1]
                        for key in (
                            "position_pc",
                            "incoming_momentum_kev",
                            "valid",
                            "interaction_type",
                            "scattering_order",
                        )
                    }
                ),
                status=np.ones(1, dtype=int),
            )
            singles.append(
                score_scattering_only_histories(
                    one_launch, one_history, physics, energy, *args[1:]
                )
            )
        np.testing.assert_allclose(mixed["sum"], sum(x["sum"] for x in singles))
        np.testing.assert_allclose(mixed["cross"], sum(x["cross"] for x in singles))

    def test_spectral_quadrature_weights_each_band_and_splits_edges(self):
        physics = SimpleNamespace(energy_kev=np.array([2, 2.47, 2.4701, 4, 6, 10]))
        np.testing.assert_allclose(_edge_breaks(physics, 2, 4), [2, 2.47, 2.4701, 4])
        self.assertAlmostEqual(_powerlaw_integral(2, 10, 0.0), 8.0)
        with patch(
            "scripts.run_spectral_observer_validation.first_order_radial_quadrature",
            side_effect=lambda _, energy, *__args, **__kwargs: np.full(
                4, 1.0 + energy
            ),
        ):
            result = spectral_first_order_quadrature(
                physics, 1.0e23, 0.0, n_energy=2
            )
        for i, (low, high) in enumerate(((2, 4), (4, 6), (6, 10))):
            np.testing.assert_allclose(
                result[i], (high - low) / 8 * (1 + (low + high) / 2)
            )

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

    def test_per_bin_comparison_rejects_low_precision_and_missing_histories(self):
        analog = np.zeros((4, 3))
        analog[:, 0] = 1.0
        q = np.zeros((12, 12))
        cov = np.zeros((12, 12))
        for t in range(4):
            q[3 * t, 3 * t] = 0.01
            cov[3 * t, 3 * t] = 0.0025
        coarse = np.full(4, 0.99999)
        fine = np.ones(4)

        def compare():
            return _compare_first_order_time_bins(
                analog,
                q,
                cov,
                coarse,
                fine,
                minimum_effective_histories=30,
                maximum_relative_standard_error=0.1,
                quadrature_rtol=0.02,
                sigma_limit=5.0,
            )

        self.assertTrue(all(row["passed"] for row in compare()))
        q[3, 3] = 0.05
        cov[6, 6] = 0.04
        coarse[3] = 0.95
        rows = compare()
        self.assertFalse(rows[1]["powered"])
        self.assertFalse(rows[2]["precise"])
        self.assertFalse(rows[3]["quadrature_converged"])
        self.assertFalse(any(row["passed"] for row in rows[1:]))
        analog[0, 0] = 0.0
        self.assertIsNone(compare()[0]["relative_photon_standard_error"])
        self.assertFalse(compare()[0]["passed"])


if __name__ == "__main__":
    unittest.main()
