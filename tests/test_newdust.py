"""Regression tests for the host-side NewDust scattering adapter."""

import unittest

import numpy as np

from dsh.physics.newdust import (
    build_dust_physics_from_newdust,
    load_newdust_scattering_table,
)


class TestNewDustScatteringTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = load_newdust_scattering_table()

    def test_v1_grid_and_physical_normalization(self):
        np.testing.assert_array_equal(self.table.energy_kev, [3.3, 4.9, 6.9])
        self.assertEqual(self.table.scattering_angle_rad[0], 0.0)
        self.assertEqual(self.table.scattering_angle_rad[-1], np.pi)
        self.assertTrue(
            np.all(np.diff(self.table.scattering_cross_section_cm2_per_h) < 0.0)
        )
        np.testing.assert_allclose(
            self.table.scattering_cross_section_cm2_per_h,
            [1.25278789e-23, 5.68215948e-24, 2.86554650e-24],
            rtol=2.0e-7,
        )

    def test_absorption_must_be_supplied_explicitly(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            build_dust_physics_from_newdust(self.table, [0.0])

        absorption = np.array([4.0e-23, 2.0e-23, 1.0e-23])
        physics = build_dust_physics_from_newdust(self.table, absorption)
        np.testing.assert_allclose(
            np.asarray(physics.absorption_cross_section_cm2_per_h),
            absorption,
        )


if __name__ == "__main__":
    unittest.main()
