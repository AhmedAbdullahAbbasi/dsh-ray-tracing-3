"""Stage 9B controlled observer-escape physics checks."""

from __future__ import annotations

import unittest

import numpy as np

from dsh.physics.absorption import load_photoelectric_absorption_table
from dsh.physics.materials import load_2_10_material_tables
from dsh.physics.newdust import load_newdust_scattering_table
from dsh.validation.escape import (
    RATIO_TOLERANCE,
    _max_relative_error,
    controlled_cloud,
    controlled_columns,
    controlled_history,
    radial_reference_column_cm2,
    run_escape_attenuation_case,
)


class TestStage9BEscape(unittest.TestCase):
    def test_independent_shell_column_and_foreground_selection(self):
        _, transported = controlled_history(3.3)
        positions = np.asarray(transported.interactions.position_pc)
        for kind, expected in (
            ("vacuum", (0.0, 0.0, 0.0, 0.0)),
            ("base", (1.85e21, 1.8e21, 3.7e21, 9.0e20)),
            ("foreground", (1.185e22, 1.8e21, 3.7e21, 1.09e22)),
            ("behind", (1.85e21, 1.8e21, 3.7e21, 9.0e20)),
        ):
            columns = controlled_columns(kind)
            cloud = controlled_cloud(columns)
            actual = [
                radial_reference_column_cm2(
                    positions[p, event],
                    columns,
                    x_edges_arcsec=np.asarray(cloud.x_edges_arcsec),
                    y_edges_arcsec=np.asarray(cloud.y_edges_arcsec),
                    z_edges_kpc=np.asarray(cloud.z_edges_kpc),
                )
                for p in range(2)
                for event in range(2)
            ]
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e8)

    def test_actual_material_escape_weight_and_paired_ratios(self):
        scattering = load_newdust_scattering_table()
        absorption = load_photoelectric_absorption_table()
        report = run_escape_attenuation_case(scattering, absorption, 3.3)
        self.assertTrue(report["all_passed"], report)
        self.assertEqual(len(report["scenarios"]), 8)
        self.assertTrue(
            all(
                case["checks"]["absorption_slot_masked"] for case in report["scenarios"]
            )
        )
        self.assertLess(
            report["paired_checks"]["absorption_ratio_base"], RATIO_TOLERANCE
        )
        self.assertLess(
            report["paired_checks"]["behind_source_path_both"], RATIO_TOLERANCE
        )

    def test_continuous_energy_uses_both_interpolated_opacities(self):
        scattering, absorption, _ = load_2_10_material_tables()
        report = run_escape_attenuation_case(scattering, absorption, 5.35)
        self.assertTrue(report["all_passed"], report)
        self.assertGreater(report["sigma_scattering_cm2_per_h"], 0.0)
        self.assertGreater(report["sigma_absorption_cm2_per_h"], 0.0)

    def test_ratio_gate_detects_extra_incoming_attenuation(self):
        true_ratio = np.exp(-0.16)
        double_applied_ratio = np.exp(-0.32)
        self.assertGreater(
            _max_relative_error(double_applied_ratio, true_ratio), RATIO_TOLERANCE
        )


if __name__ == "__main__":
    unittest.main()
