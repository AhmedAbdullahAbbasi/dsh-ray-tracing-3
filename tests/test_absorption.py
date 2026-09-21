"""Physics and provenance tests for photoelectric absorption tables."""

import unittest

import numpy as np

from dsh.physics.absorption import (
    PhotoelectricAbsorptionTable,
    load_photoelectric_absorption_table,
    monochromatic_transmission,
)
from dsh.physics.newdust import (
    build_dust_physics_from_tables,
    load_newdust_scattering_table,
)


class TestPhotoelectricAbsorptionTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.absorption = load_photoelectric_absorption_table()

    def test_v1_grid_and_xspec_cross_sections(self):
        np.testing.assert_array_equal(self.absorption.energy_kev, [3.3, 4.9, 6.9])
        np.testing.assert_allclose(
            self.absorption.absorption_cross_section_cm2_per_h,
            [
                7.751834566734242e-24,
                2.535061388430872e-24,
                9.375508448922497e-25,
            ],
            rtol=5.0e-8,
            atol=0.0,
        )
        self.assertFalse(self.absorption.metadata["source_spectrum_dependent"])
        self.assertEqual(self.absorption.metadata["abundance_command"], "wilm")
        self.assertEqual(self.absorption.metadata["tbabs_version"], 2)

    def test_beer_lambert_coefficient_is_column_independent(self):
        columns = np.array([1.0e20, 1.0e22, 5.0e23])
        transmission = monochromatic_transmission(self.absorption, columns)
        inferred = -np.log(transmission) / columns[:, None]
        np.testing.assert_allclose(
            inferred,
            np.broadcast_to(
                self.absorption.absorption_cross_section_cm2_per_h,
                inferred.shape,
            ),
            rtol=1.0e-12,
            atol=0.0,
        )

    def test_rejects_invalid_hydrogen_columns(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            monochromatic_transmission(self.absorption, -1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            monochromatic_transmission(self.absorption, np.nan)

    def test_combines_only_identical_energy_grids(self):
        scattering = load_newdust_scattering_table()
        physics = build_dust_physics_from_tables(scattering, self.absorption)
        np.testing.assert_allclose(
            np.asarray(physics.energy_kev),
            scattering.energy_kev,
            rtol=2.0e-7,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(physics.absorption_cross_section_cm2_per_h),
            self.absorption.absorption_cross_section_cm2_per_h,
            rtol=2.0e-7,
        )

        shifted = PhotoelectricAbsorptionTable(
            energy_kev=self.absorption.energy_kev + 0.01,
            absorption_cross_section_cm2_per_h=(
                self.absorption.absorption_cross_section_cm2_per_h
            ),
            metadata=self.absorption.metadata,
        )
        with self.assertRaisesRegex(ValueError, "identical energy grids"):
            build_dust_physics_from_tables(scattering, shifted)


if __name__ == "__main__":
    unittest.main()
