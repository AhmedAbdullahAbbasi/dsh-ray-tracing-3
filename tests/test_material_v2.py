"""Check the independently generated, bundled 2–10 keV material pair."""

from __future__ import annotations

import hashlib
import json
import unittest

import jax.numpy as jnp
import numpy as np

from dsh.physics.absorption import load_photoelectric_absorption_table
from dsh.physics.materials import (
    DEFAULT_2_10_ABSORPTION,
    DEFAULT_2_10_GRID,
    DEFAULT_2_10_SCATTERING,
    load_2_10_material_tables,
)
from dsh.physics.newdust import load_newdust_scattering_table
from dsh.transport.kernel import _physics_at_energy


class BundledV2MaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scattering, cls.absorption, cls.physics = load_2_10_material_tables()

    def test_supplied_grid_and_cross_section_provenance(self):
        s, a = self.scattering, self.absorption
        self.assertEqual(s.energy_kev.size, 130)
        self.assertEqual((s.energy_kev[0], s.energy_kev[-1]), (2.0, 10.0))
        self.assertTrue(np.array_equal(s.energy_kev, a.energy_kev))
        with DEFAULT_2_10_GRID.open(encoding="utf-8") as stream:
            grid = json.load(stream)
        self.assertEqual(grid["producer_version"], "12.14.0")
        self.assertEqual(a.metadata["producer_version"], "12.14.0")
        self.assertLess(a.metadata["validation_max_relative_difference"], 5e-7)
        self.assertEqual(len(grid["edge_intervals_kev_and_relative_error"]), 4)
        digest = hashlib.sha256(DEFAULT_2_10_GRID.read_bytes()).hexdigest()
        self.assertEqual(s.metadata["shared_energy_grid_sha256"], digest)
        self.assertEqual(a.metadata["shared_energy_grid_sha256"], digest)
        self.assertEqual(
            hashlib.sha256(DEFAULT_2_10_SCATTERING.read_bytes()).hexdigest(),
            s.metadata["table_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(DEFAULT_2_10_ABSORPTION.read_bytes()).hexdigest(),
            a.metadata["table_sha256"],
        )

    def test_v1_anchor_regression_and_runtime_opacity(self):
        old_s = load_newdust_scattering_table()
        old_a = load_photoelectric_absorption_table()
        indices = np.searchsorted(self.scattering.energy_kev, old_s.energy_kev)
        np.testing.assert_array_equal(
            self.scattering.energy_kev[indices], old_s.energy_kev
        )
        np.testing.assert_allclose(
            self.scattering.scattering_cross_section_cm2_per_h[indices],
            old_s.scattering_cross_section_cm2_per_h,
            rtol=5e-9,
        )
        np.testing.assert_array_equal(
            self.absorption.absorption_cross_section_cm2_per_h[indices],
            old_a.absorption_cross_section_cm2_per_h,
        )
        for node in (2.0, 3.3, 4.9, 6.9, 10.0):
            index = np.searchsorted(self.scattering.energy_kev, node)
            supported, sca, absorbed, cdf = _physics_at_energy(
                self.physics, jnp.asarray(node)
            )
            self.assertTrue(bool(supported))
            self.assertAlmostEqual(
                float(sca / self.scattering.scattering_cross_section_cm2_per_h[index]),
                1.0,
                places=5,
            )
            self.assertAlmostEqual(
                float(
                    absorbed / self.absorption.absorption_cross_section_cm2_per_h[index]
                ),
                1.0,
                places=5,
            )
            self.assertEqual(cdf.shape, (8193,))


if __name__ == "__main__":
    unittest.main()
