"""Independent shell geometry and event-history checks for Stage 9C."""

import math
import unittest

import numpy as np
from jax import random

from dsh.physics.newdust import load_newdust_scattering_table
from dsh.validation.multiple_scattering import (
    analytic_shell_column,
    analytic_world_boundary,
    run_multiple_scattering_case,
)


class TestMultipleScatteringValidation(unittest.TestCase):
    def test_shell_chord_and_crossing(self):
        origin = np.array([10_000.0, 0.0, 0.0])
        forward = np.array([-1.0, 0.0, 0.0])
        bound, status = analytic_world_boundary(origin, forward)
        self.assertEqual(status, 1)
        self.assertAlmostEqual(bound, 10_000)
        self.assertAlmostEqual(
            analytic_shell_column(origin, forward, bound, 1e22), 1e22
        )
        self.assertAlmostEqual(
            analytic_shell_column(origin, forward, 5_500, 1e22), 5e21
        )
        self.assertAlmostEqual(
            analytic_shell_column(np.array([4_500.0, 0, 0]), forward, 4_500, 1e22),
            5e21,
        )
        angle = 0.2
        oblique = np.array([-math.cos(angle), math.sin(angle), 0.0])
        bound, _ = analytic_world_boundary(origin, oblique)
        self.assertGreater(analytic_shell_column(origin, oblique, bound, 1e22), 1e22)

    def test_actual_transport_repeats_and_matches_conditional_hazards(self):
        case = run_multiple_scattering_case(
            random.PRNGKey(23),
            load_newdust_scattering_table(),
            energy_kev=3.3,
            target_tau=2.4,
            packets=1_024,
            chunk_size=512,
            min_expected_events=20,
        )
        self.assertTrue(case["all_passed"], case)
        self.assertGreater(case["flight_rows"][2]["observed_scatters"], 50)
        self.assertEqual(case["invalid_histories"], 0)
        self.assertEqual(case["status_counts"][3], 0)


if __name__ == "__main__":
    unittest.main()
