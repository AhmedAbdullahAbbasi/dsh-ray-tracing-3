"""Independent angle-law and RefleX transmission checks for the new provider."""
import json
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from dsh.observer.scoring import scattering_phase_pdf_per_sr
from dsh.transport.kernel import _sample_scattering_angle
from reflex_adapter.reflex_materials import load_reflex_materials

ROOT = Path(__file__).parent


class RefleXMaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scattering, cls.absorption, cls.physics = load_reflex_materials()
        cls.scale = cls.scattering.metadata["angular_scale_rad"][1]

    def test_transmission_against_independent_reflex_pencil_runs(self):
        rows = json.loads((ROOT / "evidence/transmission_results.json").read_text())
        sca = self.scattering.scattering_cross_section_cm2_per_h[1]
        absorption = self.absorption.absorption_cross_section_cm2_per_h[1]
        for row in rows:
            sigma = sca if row["case"] == "scattering_only" else absorption if row["case"] == "absorption_only" else sca + absorption
            probability = np.exp(-row["nh_cm2"] * sigma)
            se = np.sqrt(probability * (1-probability) / row["packets"])
            self.assertLess(abs(row["unscattered_counts"]/row["packets"]-probability), 4*se)

    def test_phase_evaluator_matches_analytic_solid_angle_density(self):
        angles = np.r_[0.0, np.geomspace(1e-7, 0.01, 300)]
        correction = np.ones_like(angles)
        np.divide(angles, np.sin(angles), out=correction, where=angles != 0)
        expected = correction / (np.pi*self.scale**2) / (1+(angles/self.scale)**2)**2
        expected /= np.pi**2/(np.pi**2+self.scale**2)
        actual = np.asarray(scattering_phase_pdf_per_sr(self.physics, 4.9, jnp.asarray(angles)))
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=0)

    def test_transport_inverse_cdf_matches_analytic_angle_probabilities(self):
        count = 100000
        keys = random.split(random.PRNGKey(95341), count)
        row = self.physics.scattering_angle_cdf[1]
        sampled = np.asarray(jax.jit(jax.vmap(lambda key: _sample_scattering_angle(
            key, self.physics.scattering_angle_rad, row)))(keys))
        for multiplier in [0.2, 0.5, 1, 2, 5, 10]:
            threshold = self.scale*multiplier
            probability = threshold**2/(threshold**2+self.scale**2)
            probability /= np.pi**2/(np.pi**2+self.scale**2)
            se = np.sqrt(probability*(1-probability)/count)
            self.assertLess(abs(np.mean(sampled <= threshold)-probability), 5*se + 2e-6)

    def test_zero_absorption_control_preserves_scattering_law(self):
        _, _, control = load_reflex_materials(absorption_enabled=False)
        np.testing.assert_array_equal(control.absorption_cross_section_cm2_per_h, 0)
        np.testing.assert_array_equal(control.scattering_cross_section_cm2_per_h,
                                      self.physics.scattering_cross_section_cm2_per_h)
        np.testing.assert_array_equal(control.scattering_angle_cdf, self.physics.scattering_angle_cdf)


if __name__ == "__main__":
    unittest.main()
