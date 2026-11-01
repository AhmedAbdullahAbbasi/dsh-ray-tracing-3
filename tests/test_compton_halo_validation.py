"""Independent paired-moment arithmetic and real benchmark control checks."""

import json
import tempfile
import unittest
from pathlib import Path

import jax
import numpy as np
from jax import random

from dsh.transport.inelastic import interaction_opacities
from dsh.validation.compton_halo import (
    HaloCollector,
    add_moments,
    histogram_indices,
    load_checkpoint,
    mean_covariance,
    merge_seed_estimates,
    ratio_and_error,
    save_checkpoint,
    verify_reference,
)

ROOT = Path(__file__).resolve().parents[1]


class PairedMomentTests(unittest.TestCase):
    def test_mean_covariance_matches_centered_history_calculation(self):
        h = np.array([[1.0, 2.0], [2.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
        centered = h - h.mean(axis=0)
        actual = mean_covariance(h.sum(axis=0), h.T @ h, 4)
        np.testing.assert_allclose(actual, centered.T @ centered / 12, rtol=1e-14)
        ratio, error = ratio_and_error(
            h[:, 1].mean(), h[:, 0].mean(), actual[1, 1], actual[0, 0], actual[0, 1]
        )
        self.assertEqual(ratio, 2.0)
        self.assertEqual(error, 0.0)
        self.assertGreater(
            ratio_and_error(1.5, 0.75, actual[1, 1], actual[0, 0], 0.0)[1], 0.0
        )

    def test_independent_seed_merge_counts_zero_scorers_once(self):
        a = np.array([[1.0, 2.0], [0.0, 0.0], [2.0, 0.0], [0.0, 0.0]])
        b = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0], [1.0, 0.0]])
        states = [
            {"histories": np.asarray(len(h)), "sum": h.sum(axis=0), "cross": h.T @ h}
            for h in (a, b)
        ]
        mean, covariance = merge_seed_estimates(states)
        h = np.vstack([a, b])
        np.testing.assert_allclose(mean, h.mean(axis=0), rtol=1e-14)
        np.testing.assert_allclose(
            covariance, np.cov(h, rowvar=False, ddof=1) / 8, rtol=1e-14
        )
        states[1]["histories"] = np.asarray(3)
        with self.assertRaisesRegex(ValueError, "equal"):
            merge_seed_estimates(states)

    def test_maximum_energy_error_is_not_added_across_chunks(self):
        a = {"max_energy_error_kev": np.asarray(1e-6), "sum": np.array([1.0, 2.0])}
        b = {"max_energy_error_kev": np.asarray(2e-6), "sum": np.array([3.0, 4.0])}
        actual = add_moments(a, b)
        self.assertEqual(float(actual["max_energy_error_kev"]), 2e-6)
        np.testing.assert_array_equal(actual["sum"], [4.0, 6.0])

    def test_histogram_last_edge_is_included_without_clipping_outside(self):
        index, inside = histogram_indices(
            np.array([0.0, 1.0, 2.0]), np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        )
        np.testing.assert_array_equal(index, [-1, 0, 1, 1, 2])
        np.testing.assert_array_equal(inside, [False, True, True, True, False])

    def test_bad_counts_shapes_and_ratio_denominator_are_rejected(self):
        with self.assertRaises(ValueError):
            mean_covariance([1.0], np.ones((1, 1)), 1)
        with self.assertRaises(ValueError):
            mean_covariance([1.0, 2.0], np.ones((1, 1)), 2)
        with self.assertRaises(ValueError):
            ratio_and_error(1.0, 0.0, 1.0, 1.0, 0.0)


class ReferenceTests(unittest.TestCase):
    def test_raw_reference_and_single_switch_match(self):
        manifest, images = verify_reference(
            ROOT, ROOT / "benchmarks/compton/halo_4p9_2026-11-01/reference.json"
        )
        self.assertEqual(int(images["on_dust"].sum()), 5981)
        self.assertEqual(int(images["off_dust"].sum()), 5976)
        self.assertEqual(manifest["source_history_count"], 100000000)

    def test_modified_image_is_rejected_before_statistics(self):
        manifest = json.loads(
            (ROOT / "benchmarks/compton/halo_4p9_2026-11-01/reference.json").read_text()
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = next(iter(manifest["files_sha256"]))
            file = root / first
            file.parent.mkdir(parents=True)
            file.write_bytes(b"changed image")
            path = root / "reference.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                verify_reference(root, path)


class CheckpointTests(unittest.TestCase):
    def test_identity_and_history_count_are_checked_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            state = {"histories": np.asarray(4), "sum": np.array([1.0, 2.0])}
            save_checkpoint(path, state, 42, "a" * 64)
            actual = load_checkpoint(path, 42, "a" * 64, 4)
            np.testing.assert_array_equal(actual["sum"], state["sum"])
            for seed, fingerprint, count in (
                (43, "a" * 64, 4),
                (42, "b" * 64, 4),
                (42, "a" * 64, 5),
            ):
                with (
                    self.subTest(seed=seed, fingerprint=fingerprint, count=count),
                    self.assertRaises(ValueError),
                ):
                    load_checkpoint(path, seed, fingerprint, count)


class HaloCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (ROOT / "configs/compton/halo_validation.json").read_text()
        )
        cls.config["chunk_size"] = 1024
        cls.collector = HaloCollector(cls.config, ROOT)
        cls.state = cls.collector.collect(random.PRNGKey(982600), 1024, 912983)

    @classmethod
    def tearDownClass(cls):
        jax.clear_caches()

    def test_positive_control_preserves_dust_and_absorption_at_exact_energy_node(self):
        support, dust, compton, absorption = interaction_opacities(
            self.collector.physics, self.collector.off_control, np.asarray(4.9)
        )
        self.assertTrue(bool(support))
        self.assertTrue(np.isfinite([dust, compton, absorption]).all())
        self.assertGreater(float(compton), 0.0)
        self.assertLess(float(compton) * 1e22, 1e-12)
        self.assertGreater(float(self.state["absorption_sum_kev"][0]), 0.0)
        self.assertEqual(int(self.state["analog_compton_count"][0]), 0)
        self.assertGreater(float(self.state["sum"][:9].sum()), 0.0)

    def test_grouped_images_spectra_times_and_prefix_components_close(self):
        state = self.state
        total = state["sum"].reshape(5, -1).sum(axis=1)
        np.testing.assert_allclose(
            total, state["image"].sum(axis=(1, 2)), rtol=1e-12, atol=1e-25
        )
        np.testing.assert_allclose(
            total, state["energy_sum"].sum(axis=1), rtol=1e-12, atol=1e-25
        )
        np.testing.assert_allclose(
            total, state["time_sum"].sum(axis=1), rtol=1e-12, atol=1e-25
        )
        np.testing.assert_allclose(total[4], total[1] + total[3], rtol=1e-12)
        np.testing.assert_array_equal(state["statuses"][:, [0, 4, 5, 6, 7, 8]], 0)
        np.testing.assert_array_equal(state["statuses"].sum(axis=1), 1024)
        self.assertEqual(int(state["unsupported_observer_count"]), 0)
        self.assertLess(float(state["max_energy_error_kev"]), 2e-5)
