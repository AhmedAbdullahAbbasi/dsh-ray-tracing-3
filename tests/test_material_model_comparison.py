"""Different-model statistics, cache identity, and controlled Mie composition."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from reflex_adapter.compare_reflex import (
    image_edges,
    normalize_camera_counts,
    radial_pixel_map,
)
from scripts.compare_material_models import (
    CASES,
    ROOT,
    analyze,
    geometry_equal,
    hash_file,
    load_case_material,
    verified_cache,
)


class TestMaterialModelComparison(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (ROOT / "reflex_adapter/comparison_config.json").read_text()
        )

    def test_geometry_comparison_changes_only_independent_seeds(self):
        self.assertTrue(geometry_equal(self.config, self.config | {"seeds": [1, 2, 3]}))
        for name in (
            "energy_kev",
            "nh_cm2",
            "packets_per_seed",
            "camera_aperture_radius_pc",
        ):
            with self.subTest(name=name):
                self.assertFalse(
                    geometry_equal(
                        self.config, self.config | {name: self.config[name] * 2}
                    )
                )

    def test_full_mie_composition_preserves_hybrid_inventory_and_total_opacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            full, full_info = load_case_material("wd01_mie", Path(temporary))
        hybrid, hybrid_info = load_case_material("wd01_hybrid", Path("."))
        self.assertEqual(
            full_info["audit"]["elements"], hybrid_info["audit"]["elements"]
        )
        from scripts.compare_wd01_reflex import material_diagnostics

        first = material_diagnostics(full, self.config)
        second = material_diagnostics(hybrid, self.config)
        for name in ("scattering_cm2_per_h", "absorption_cm2_per_h"):
            self.assertAlmostEqual(first[name] / second[name], 1, places=7)
        self.assertNotEqual(
            full_info["definition_sha256"], hybrid_info["definition_sha256"]
        )

    def test_mrn_runtime_is_historical_model_and_all_case_seeds_are_independent(self):
        _, details = load_case_material("mrn_rg", Path("."))
        self.assertEqual(details["legacy_v1_4p9_cross_sections_match_rtol"], 1e-7)
        seeds = [s for case in CASES.values() for s in case["seeds"]]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_null_profile_gives_zero_statistic_without_faking_numerical_quality(self):
        config = self.config | {"packets_per_seed": 4}
        n = len(image_edges(config)) - 1
        reference = np.ones((n, n))
        scale = normalize_camera_counts(config)
        image = reference / scale
        radial = np.bincount(radial_pixel_map(config).ravel(), image.ravel())
        weights = np.array([0, 1, 1, 2]) / 4
        per_history_radial = weights[:, None] * radial
        order = np.array([image.sum(), 0, 0])
        per_history_order = weights[:, None] * order
        runs = [
            {
                "seed": seed,
                "packets": 4,
                "image": np.stack([image, image * 0, image * 0]),
                "image_q": np.sum(weights**2) * image**2,
                "radial": radial,
                "radial_q": per_history_radial.T @ per_history_radial,
                "order_sum": order,
                "order_q": per_history_order.T @ per_history_order,
                "statuses": np.array([0, 4, 0, 0, 0, 0, 0]),
            }
            for seed in (1, 2, 3)
        ]
        result, arrays = analyze(config, runs, reference)
        self.assertAlmostEqual(result["halo_ratio"], 1, places=12)
        self.assertLess(result["radial_chi2"], 1e-20)
        np.testing.assert_allclose(arrays["radial_ratio"], 1, rtol=1e-12)
        self.assertFalse(result["numerical_quality_passed"])
        expected_ref_cov = (
            np.diag(result["annulus_halo_counts"])
            - np.outer(result["annulus_halo_counts"], result["annulus_halo_counts"])
            / config["reflex_photons"]
        )
        np.testing.assert_allclose(
            arrays["radial_reference_covariance"], expected_ref_cov, rtol=1e-12
        )

    def test_cache_rejects_changed_material_input_before_loading_histories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            material = root / "material.json"
            material.write_text('{"material":1}\n')
            actual = dict.fromkeys(
                ("material_sha256", "cloud_sha256", "launch_sha256", "runtime_sha256"),
                "a" * 64,
            )
            provenance = dict(
                actual,
                config=self.config,
                input_sha256={"material.json": hash_file(material)},
            )
            (cache / "provenance.json").write_text(json.dumps(provenance))
            material.write_text('{"material":2}\n')
            with (
                patch("scripts.compare_material_models.ROOT", root),
                patch.dict(CASES, {"test": {"cached": "cache"}}),
                self.assertRaisesRegex(ValueError, "cached input differs"),
            ):
                verified_cache("test", self.config, actual)


if __name__ == "__main__":
    unittest.main()
