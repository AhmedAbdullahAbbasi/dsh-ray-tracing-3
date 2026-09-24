"""Verify scalar moment pooling without running photon transport."""

import math
import unittest
from copy import deepcopy

from scripts.merge_spectral_observer_validation import merge_reports, pool_scalar


class TestSpectralReportMerge(unittest.TestCase):
    def test_pooled_moments_match_raw_histories_including_zero_scores(self):
        first = [0.0, 2.0, 0.0, 3.0]
        second = [0.0, 0.0, 4.0, 0.0, 5.0, 0.0]
        groups = (first, second)
        rows = []
        for group in groups:
            mean = sum(group) / len(group)
            square = sum(value**2 for value in group) / len(group) ** 2
            rows.append({"sum": mean, "neff": mean**2 / square})
        measured, error, effective = pool_scalar(rows, [4, 6], "sum", "neff")
        combined = first + second
        expected = sum(combined) / len(combined)
        variance = sum((value - expected) ** 2 for value in combined)
        variance /= len(combined) * (len(combined) - 1)
        squared = sum(value**2 for value in combined) / len(combined) ** 2
        self.assertAlmostEqual(measured, expected)
        self.assertAlmostEqual(error, math.sqrt(variance))
        self.assertAlmostEqual(effective, expected**2 / squared)

    def test_reused_seed_is_rejected_before_pooling(self):
        report = {
            "stage": "9F_continuous_spectrum_absorbed_observer",
            "git_head": "frozen",
            "scattering_table_sha256": "sca",
            "absorption_table_sha256": "abs",
            "config": {
                "seeds": [912, 319, 141],
                "max_interactions": 32,
                "sigma_limit": 5.0,
                "minimum_effective_histories": 30,
                "quadrature_rtol": 0.02,
                "maximum_time_bin_relative_error": 0.1,
                "spectral_nodes": 6,
            },
            "case": {
                "spectrum": {"photon_index": 1.7},
                "time_edges_s": [0, 1],
                "pivot_energy_kev": 5.35,
                "pivot_tau_scattering": 1.5,
                "column_cm2": 3e23,
                "quadrature": {},
                "bands": [],
                "checks": {
                    "all_band_checks": False,
                    "source_probabilities_sum_to_one": True,
                },
            },
        }
        with self.assertRaisesRegex(ValueError, "reuse random seeds"):
            merge_reports(report, deepcopy(report))


if __name__ == "__main__":
    unittest.main()
