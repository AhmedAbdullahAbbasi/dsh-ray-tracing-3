"""Regression tests for the host-side NewDust scattering adapter."""

import unittest

import numpy as np

from dsh.physics.newdust import (
    build_dust_physics_from_newdust,
    legacy_screen_kernel_arcsec2,
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

    def test_reconstructs_supplied_legacy_e1_screen_files(self):
        # Reference values are from the supplied x_0_007 ... x_0_010 files.
        # They are exact columns 7--10 of int_E1.mat (3.3 keV).
        angles = np.array([1.0, 10.0, 100.0, 500.0, 1000.0, 1500.0])
        references = {
            0.007: [
                2.186117e-06,
                2.168618e-06,
                1.022494e-06,
                8.205921e-09,
                7.238576e-10,
                1.739696e-10,
            ],
            0.008: [
                2.190527e-06,
                2.172956e-06,
                1.023098e-06,
                8.193523e-09,
                7.227589e-10,
                1.737016e-10,
            ],
            0.009: [
                2.194949e-06,
                2.177308e-06,
                1.023702e-06,
                8.181131e-09,
                7.216607e-10,
                1.734338e-10,
            ],
            0.010: [
                2.199385e-06,
                2.181673e-06,
                1.024303e-06,
                8.168745e-09,
                7.205631e-10,
                1.731661e-10,
            ],
        }
        for fractional_distance, expected in references.items():
            actual = legacy_screen_kernel_arcsec2(
                self.table,
                energy_kev=3.3,
                observed_angle_arcsec=angles,
                fractional_distance_from_observer=fractional_distance,
            )
            np.testing.assert_allclose(actual, expected, rtol=8.0e-6)

    def test_legacy_adapter_rejects_extrapolation(self):
        with self.assertRaisesRegex(ValueError, "energy"):
            legacy_screen_kernel_arcsec2(
                self.table,
                energy_kev=2.0,
                observed_angle_arcsec=10.0,
                fractional_distance_from_observer=0.5,
            )
        with self.assertRaisesRegex(ValueError, "fractional"):
            legacy_screen_kernel_arcsec2(
                self.table,
                energy_kev=3.3,
                observed_angle_arcsec=10.0,
                fractional_distance_from_observer=1.0,
            )


if __name__ == "__main__":
    unittest.main()
