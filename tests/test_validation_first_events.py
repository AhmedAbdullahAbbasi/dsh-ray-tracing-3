"""Independent shell geometry and first-event checks using real material tables."""

import math
import unittest

from jax import random

from dsh.physics.absorption import load_photoelectric_absorption_table
from dsh.physics.newdust import load_newdust_scattering_table
from dsh.validation.first_events import (
    LOCATION_FRACTIONS,
    first_event_reference,
    run_first_event_case,
    sphere_chord,
)


class TestFirstEventValidation(unittest.TestCase):
    def test_independent_shell_and_competing_risks(self):
        entrance, exit_ = sphere_chord(10.0, 0.0)
        self.assertAlmostEqual(entrance, 5.0)
        self.assertAlmostEqual(exit_, 6.0)
        oblique = sphere_chord(10.0, 0.2)
        self.assertGreater(oblique[1] - oblique[0], 1.0)
        ref = first_event_reference(1e22, 2e-23, 1e-23, 0.0)
        self.assertAlmostEqual(ref.no_event, math.exp(-0.3))
        self.assertAlmostEqual(ref.first_scatter / ref.first_absorb, 2.0)
        self.assertAlmostEqual(ref.no_event + ref.first_scatter + ref.first_absorb, 1.0)
        for fraction, cdf in zip(LOCATION_FRACTIONS, ref.location_cdf, strict=True):
            self.assertAlmostEqual(
                cdf, -math.expm1(-0.3 * fraction) / -math.expm1(-0.3)
            )
        empty = first_event_reference(0.0, 0.0, 0.0, 0.2)
        self.assertEqual((empty.no_event, empty.location_cdf), (1.0, ()))

    def test_actual_tables_competing_first_events_and_positions(self):
        report = run_first_event_case(
            random.PRNGKey(11),
            load_newdust_scattering_table(),
            load_photoelectric_absorption_table(),
            energy_index=0,
            mode="both",
            target_tau_sca_3p3=1.0,
            packets_per_ray=512,
            chunk_size=256,
            sigma_limit=5.0,
        )
        self.assertTrue(report["all_passed"], report["rays"])
        self.assertEqual(report["sigma_abs_cm2_per_h"] > 0, True)
        self.assertEqual(report["sigma_sca_cm2_per_h"] > 0, True)
        self.assertGreater(
            report["rays"][1]["reference_column_cm2"],
            report["rays"][0]["reference_column_cm2"],
        )


if __name__ == "__main__":
    unittest.main()
