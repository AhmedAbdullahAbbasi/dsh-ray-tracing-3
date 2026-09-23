"""Regression and integration checks for the offline 2–10 keV material path."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dsh.physics.absorption import load_photoelectric_absorption_table
from dsh.physics.material_grid import (
    initial_material_grid,
    load_material_grid,
    refine_material_grid,
)
from dsh.physics.newdust import (
    build_dust_physics_from_tables,
    load_newdust_scattering_table,
)
from dsh.physics.rg_drude import default_angle_grid, gaussian_rg_drude_table
from scripts.generate_rg_drude_table import generate as generate_scattering
from scripts.generate_tbabs_table import generate as generate_absorption


class ScatteringRegressionTests(unittest.TestCase):
    def test_matches_frozen_newdust_at_three_energies(self):
        reference = load_newdust_scattering_table()
        # geomspace may differ by a few ulps across NumPy/libm platforms.
        np.testing.assert_allclose(
            default_angle_grid(),
            reference.scattering_angle_rad,
            rtol=8 * np.finfo(np.float64).eps,
            atol=0,
        )
        differential, sigma, cdf = gaussian_rg_drude_table(
            reference.energy_kev, reference.scattering_angle_rad
        )
        np.testing.assert_allclose(
            differential,
            reference.differential_cross_section_cm2_per_sr_per_h,
            rtol=5e-9,
            atol=1e-35,
        )
        np.testing.assert_allclose(
            sigma, reference.scattering_cross_section_cm2_per_h, rtol=5e-9
        )
        np.testing.assert_allclose(cdf, reference.scattering_angle_cdf, atol=2e-14)

    def test_2_and_10_kev_against_independent_xdust_reference(self):
        # Pinned xdust make_MRN_RGDrude, na=100, linear radii, md=2.32475e-4.
        energy = np.array([2.0, 10.0])
        angle = default_angle_grid()
        differential, sigma, _ = gaussian_rg_drude_table(energy, angle)
        np.testing.assert_allclose(
            differential[:, 0],
            [9.171842688289175e-18, 9.171842687999307e-18],
            rtol=5e-9,
        )
        np.testing.assert_allclose(sigma, [3.410711225e-23, 1.364286796e-24], rtol=1e-8)

    def test_scattering_interpolation_at_coarsest_grid_intervals(self):
        ends = np.array([2.0, 2.1, 4.0, 4.1, 9.9, 10.0])
        angle = default_angle_grid()
        differential, sigma, cdf = gaussian_rg_drude_table(ends, angle)
        middle = np.sqrt(ends[::2] * ends[1::2])
        actual_differential, actual_sigma, actual_cdf = gaussian_rg_drude_table(
            middle, angle
        )
        predicted = np.sqrt(differential[::2] * differential[1::2])
        relevant = actual_differential > actual_differential.max(axis=1)[:, None] * 1e-6
        self.assertLess(
            np.max(np.abs(predicted[relevant] / actual_differential[relevant] - 1)),
            1e-3,
        )
        self.assertLess(
            np.max(np.abs(np.sqrt(sigma[::2] * sigma[1::2]) / actual_sigma - 1)), 1e-6
        )
        self.assertLess(np.max(np.abs((cdf[::2] + cdf[1::2]) / 2 - actual_cdf)), 2e-4)

    def test_shared_grid_generates_loadable_scattering(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            grid = folder / "grid.json"
            grid.write_text(
                json.dumps({"schema_version": 1, "energy_kev": [2, 3.3, 10]}),
                encoding="utf-8",
            )
            table_path = folder / "scattering.npz"
            generate_scattering(table_path, grid)
            table = load_newdust_scattering_table(table_path)
            np.testing.assert_array_equal(table.energy_kev, [2, 3.3, 10])
            self.assertEqual(table.metadata["producer"].split()[0], "DSH")

            def fake_xspec(executable, energy, nh22, half_width):
                sigma = 1e-22 / energy**3
                return sigma, float(np.exp(-nh22 * 1e22 * sigma)), "mock-xspec"

            absorption_path = folder / "absorption.npz"
            with (
                patch(
                    "scripts.generate_tbabs_table.resolve_xspec", return_value="xspec"
                ),
                patch("scripts.generate_tbabs_table._extract_one_energy", fake_xspec),
            ):
                generate_absorption(
                    absorption_path,
                    "xspec",
                    load_material_grid(grid),
                    grid_sha256=table.metadata["shared_energy_grid_sha256"],
                )
            absorption = load_photoelectric_absorption_table(absorption_path)
            self.assertEqual(
                absorption.metadata["shared_energy_grid_sha256"],
                table.metadata["shared_energy_grid_sha256"],
            )
            physics = build_dust_physics_from_tables(table, absorption)
            self.assertEqual(physics.energy_kev.shape[0], 3)


class GridRefinementTests(unittest.TestCase):
    def test_preserves_frozen_anchors_and_localizes_a_sharp_edge(self):
        seed = initial_material_grid()
        for anchor in (3.3, 4.9, 6.9):
            self.assertIn(anchor, seed)

        def absorption(energy: float) -> float:
            return energy**-3 * (1.0 + 0.4 * (energy >= 7.112))

        energy, edges, smooth_error = refine_material_grid(
            absorption,
            initial_energy_kev=seed,
            relative_tolerance=1e-3,
            minimum_interval_kev=1e-4,
        )
        self.assertGreater(len(edges), 0)
        self.assertTrue(any(lo <= 7.112 <= hi for lo, hi, _ in edges))
        self.assertTrue(all(hi - lo <= 1e-4 for lo, hi, _ in edges))
        self.assertLessEqual(smooth_error, 1e-3)
        self.assertTrue(all(anchor in energy for anchor in (3.3, 4.9, 6.9)))

    def test_rejects_malformed_or_incomplete_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.json"
            path.write_text(
                json.dumps({"schema_version": 1, "energy_kev": [2, 3.3, 9]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "2 to 10"):
                load_material_grid(path)


if __name__ == "__main__":
    unittest.main()
