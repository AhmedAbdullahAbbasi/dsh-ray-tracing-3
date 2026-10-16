"""Joint spectral/time sampling and absolute, history-grouped response checks."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
from jax import random

from dsh.observer.response import (
    SpectralResponse,
    fold_spectral_response,
    load_spectral_response,
    response_from_events,
    write_spectral_response,
)
from dsh.observer.scoring import PC_LIGHT_TRAVEL_TIME_S
from dsh.sources.histories import (
    FLUX_CONVENTION,
    HISTORY_UNITS,
    build_spectral_history_source,
    integrated_spectral_density,
    load_spectral_history,
    powerlaw_integral,
    spectral_density,
)
from dsh.sources.models import (
    build_powerlaw_band_source,
    sample_tabulated_band_source,
)

PROVENANCE = dict.fromkeys(
    ("material_sha256", "cloud_sha256", "launch_sha256", "runtime_sha256"), "a" * 64
)


def example_response(**overrides):
    data = {
        "energy_kev": [3.0, 3.0, 8.0],
        "delay_s": [0.5, 1.5, 0.0],
        "sky_x_arcsec": [1.0, 1.0, 3.0],
        "sky_y_arcsec": [0.0, 0.0, 0.0],
        "scattering_order": np.array([1, 2, 1]),
        "history_index": np.array([0, 0, 1]),
        "weight_kev": [1.0, 2.0, 4.0],
        "n_histories": 4,
        "energy_support_kev": (2.0, 10.0),
        "provenance": PROVENANCE,
    }
    return SpectralResponse(**(data | overrides))


class TestSpectralHistories(unittest.TestCase):
    def test_joint_time_energy_distribution_with_changing_slope_and_gap(self):
        source = build_powerlaw_band_source(
            [0, 1, 4, 6], [1, 0, 2], [2, 4, 10], [1.0, 2.0, 3.0]
        )
        sampler = jax.jit(sample_tabulated_band_source, static_argnames=("n_packets",))
        packets = sampler(random.PRNGKey(122), source, n_packets=150_000)
        t, energy = np.asarray(packets.time_index), np.asarray(packets.energy_kev)
        counts = np.bincount(t, minlength=3) / len(t)
        np.testing.assert_allclose(counts, [0.2, 0, 0.8], atol=0.004)
        self.assertAlmostEqual(
            np.mean(energy[t == 0] < 4), np.log(2) / np.log(5), delta=0.008
        )
        self.assertAlmostEqual(
            np.mean(energy[t == 2] < 4), (2**-2 - 4**-2) / (2**-2 - 10**-2), delta=0.006
        )
        self.assertAlmostEqual(float(source.total_fluence), 5, places=6)
        self.assertAlmostEqual(
            np.asarray(packets.weight_observer_fluence).sum(), 5, places=5
        )
        self.assertTrue(np.all(np.asarray(packets.emission_time_s)[t == 2] >= 4))

    def test_broken_spectra_normalize_each_cell_and_integrate_time_exactly(self):
        source = build_spectral_history_source(
            [0, 1, 4], [2, 4, 10], [[1, 2], [3, 0]], [[1, 2.5], [3, 1]]
        )
        roots, weights = np.polynomial.legendre.leggauss(48)
        for b, (low, high) in enumerate(((2, 4), (4, 10))):
            nodes = low + (roots + 1) * (high - low) / 2
            integral = spectral_density(source, nodes) @ (weights * (high - low) / 2)
            np.testing.assert_allclose(
                integral, np.asarray(source.band_flux)[:, b], rtol=1e-12, atol=1e-12
            )
        density = spectral_density(source, [3, 8])
        result = integrated_spectral_density(source, [3, 8], [-1, 0.5], [2, 5])
        np.testing.assert_allclose(result, [density[:, 0].sum(), density[0, 1] / 2])
        np.testing.assert_array_equal(
            spectral_density(source, [1, 11]), np.zeros((2, 2))
        )

    def test_gamma_one_limit_is_stable_on_both_sides(self):
        values = powerlaw_integral(2, 10, np.array([1 - 1e-10, 1, 1 + 1e-10]))
        np.testing.assert_allclose(values, np.log(5), rtol=3e-10)

    def test_excessive_slope_range_cannot_silently_overflow_the_sampler(self):
        for gamma in (-100, 100):
            with (
                self.subTest(gamma=gamma),
                self.assertRaisesRegex(ValueError, "dynamic range"),
            ):
                build_spectral_history_source([0, 1], [2, 10], [[1]], gamma)

    def test_rejects_gaps_wrong_shapes_and_lost_precision(self):
        for flux, gamma in (
            ([[np.nan]], 1),
            ([[-1]], 1),
            ([[0]], 1),
            ([[1]], [[1, 2]]),
            ([[1]], np.nan),
        ):
            with self.subTest(flux=flux, gamma=gamma), self.assertRaises(ValueError):
                build_spectral_history_source([0, 1], [2, 10], flux, gamma)
        with self.assertRaisesRegex(ValueError, "collapse"):
            build_spectral_history_source([1e10, 1e10 + 1], [2, 10], [[1]], 1)

    def test_file_requires_intrinsic_units_epoch_and_hashes_original_bytes(self):
        data = {
            "schema_version": 1,
            "flux_convention": FLUX_CONVENTION,
            "units": HISTORY_UNITS,
            "epoch": "relative test origin",
            "time_scale": "simulation-relative",
            "time_edges_s": [0, 1],
            "energy_edges_kev": [2, 10],
            "band_flux": [[1]],
            "photon_index": [[1]],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            path.write_text(json.dumps(data))
            _, metadata = load_spectral_history(path)
            self.assertEqual(
                metadata["source_history_sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for field, value in (
                ("flux_convention", "absorbed"),
                ("epoch", ""),
                ("units", {}),
            ):
                path.write_text(json.dumps(data | {field: value}))
                with self.assertRaises(ValueError):
                    load_spectral_history(path)


class TestSpectralResponse(unittest.TestCase):
    def test_fractional_delays_and_covariance_group_all_orders_and_zero_histories(self):
        source = build_spectral_history_source([0, 1], [2, 10], [[8]], 0)
        # F_E=1. Contributions per history in the two arrival bins are
        # [0.5, 1.5], [4, 0], [0, 0], [0, 0]. The second event at delay1.5
        # contributes only half its weight to this finite arrival window.
        response = example_response()
        result = fold_spectral_response(response, source, [0, 1, 2], [2, 10], [0, 4])
        grouped = np.array([[0.5, 1.5], [4, 0], [0, 0], [0, 0]])
        expected = grouped.sum(axis=0)
        covariance = 4 / 3 * (grouped.T @ grouped - np.outer(expected, expected) / 4)
        np.testing.assert_allclose(result.fluence.ravel(), expected)
        np.testing.assert_allclose(result.covariance, covariance)
        np.testing.assert_allclose(
            result.standard_error.ravel() ** 2, covariance.diagonal()
        )
        diagonal = fold_spectral_response(
            response,
            source,
            [0, 1, 2],
            [2, 10],
            [0, 4],
            covariance=False,
            history_block_size=1,
        )
        np.testing.assert_allclose(diagonal.standard_error, result.standard_error)
        self.assertIsNone(diagonal.covariance)

    def test_energy_and_radius_bins_order_selection_and_window_loss(self):
        source = build_spectral_history_source([0, 1], [2, 10], [[8]], 0)
        response = example_response()
        all_time = fold_spectral_response(
            response, source, [-1, 4], [2, 4, 10], [0, 2, 3]
        )
        np.testing.assert_allclose(all_time.fluence[0], [[3, 0], [0, 4]])
        first = fold_spectral_response(
            response, source, [-1, 4], [2, 10], [0, 3], scattering_orders=[1]
        )
        self.assertAlmostEqual(float(first.fluence.sum()), 5)
        with self.assertRaisesRegex(ValueError, "support"):
            fold_spectral_response(
                response,
                build_spectral_history_source([0, 1], [1, 10], [[1]], 1),
                [0, 1],
                [1, 10],
                [0, 3],
            )

    def test_exact_energy_samples_preserve_a_jump_without_reinterpolation(self):
        response = example_response(
            energy_kev=[7.1239, 7.1241],
            delay_s=[0, 0],
            sky_x_arcsec=[0, 0],
            sky_y_arcsec=[0, 0],
            scattering_order=np.array([1, 1]),
            history_index=np.array([0, 1]),
            weight_kev=[0.8, 0.2],
            n_histories=2,
        )
        source = build_spectral_history_source([0, 1], [2, 10], [[8]], 0)
        result = fold_spectral_response(
            response, source, [0, 1], [2, 7.124, 10], [0, 1]
        )
        np.testing.assert_allclose(result.fluence.ravel(), [0.8, 0.2])

    def test_absolute_response_removes_proposal_and_packet_fluence_once(self):
        proposal = build_spectral_history_source([0, 2], [2, 10], [[4]], 0)
        packets = SimpleNamespace(
            energy_kev=np.array([3, 8]), weight_observer_fluence=np.array([4, 4])
        )
        events = SimpleNamespace(
            valid=np.array([[True], [False]]),
            energy_kev=np.array([[3], [0]]),
            excess_path_length_pc=np.array([[1 / PC_LIGHT_TRAVEL_TIME_S], [0]]),
            sky_x_arcsec=np.zeros((2, 1)),
            sky_y_arcsec=np.zeros((2, 1)),
            scattering_order=np.array([[1], [0]]),
            weight_observer_fluence=np.array([[2], [0]]),
        )
        response = response_from_events(
            events, packets, proposal, [1, 3], provenance=PROVENANCE
        )
        # q=1/8, G=2/4. G/(N*q)=2 keV, so unit F_E for 1s yields2.
        np.testing.assert_allclose(response.weight_kev, [2])
        np.testing.assert_allclose(response.delay_s, [1])
        for status in ([1, 4], [0, 3], [1, 5]):
            with self.assertRaisesRegex(ValueError, "completed"):
                response_from_events(
                    events, packets, proposal, status, provenance=PROVENANCE
                )

    def test_archive_checksums_and_invariants(self):
        response = example_response()
        with tempfile.TemporaryDirectory() as directory:
            path = write_spectral_response(Path(directory) / "response.npz", response)
            restored = load_spectral_response(path)
            np.testing.assert_array_equal(restored.weight_kev, response.weight_kev)
            self.assertFalse(restored.energy_kev.flags.writeable)
            with np.load(path, allow_pickle=False) as stored:
                altered = {name: stored[name] for name in stored.files}
            altered["weight_kev"] = altered["weight_kev"] * 2
            np.savez(path, **altered)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_spectral_response(path)
        for changes in (
            {"delay_s": [-1, 0, 0]},
            {"history_index": np.array([0, 0, 4])},
            {"provenance": {}},
            {"scattering_order": np.ones(3)},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                example_response(**changes)


if __name__ == "__main__":
    unittest.main()
