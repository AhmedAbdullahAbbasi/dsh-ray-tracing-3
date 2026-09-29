"""External intrinsic photon fluxes must survive I/O and sampling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import jax
import numpy as np
from astropy.io import fits

from dsh.sources.cells import build_source_cells, sample_source_cells
from dsh.sources.source_fits import SourceFluxFile, load_source_fits, write_source_fits


class SourceInputTests(unittest.TestCase):
    def setUp(self):
        self.source = SourceFluxFile(
            time_edges_s=np.array([0.0, 2.0, 5.0]),
            continuum_energy_edges_kev=np.array([2.0, 4.0, 10.0]),
            continuum_flux=np.array([[2.0, 3.0], [1.0, 4.0]]),
            continuum_shape=("FLAT", "POWERLAW"),
            continuum_photon_index=np.array([[0.0, 1.0], [0.0, 2.1]]),
            line_energy_kev=np.array([6.4]),
            line_flux=np.array([[1.0], [2.0]]),
            line_labels=("FeKa",),
            mjdref=60500.0,
            timesys="TT",
        )

    def test_fits_round_trip_and_selected_fluence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_source_fits(Path(directory) / "source.fits", self.source)
            restored = load_source_fits(path)
            np.testing.assert_array_equal(
                restored.continuum_flux, self.source.continuum_flux
            )
            np.testing.assert_array_equal(restored.line_flux, self.source.line_flux)
            self.assertEqual(restored.mjdref, 60500.0)
            self.assertEqual(restored.timesys, "TT")
            self.assertEqual(restored.continuum_fluence, 25.0)
            self.assertEqual(restored.line_fluence, 8.0)
            self.assertEqual(
                float(build_source_cells(restored, components="lines").total_fluence),
                8.0,
            )
            self.assertEqual(
                float(
                    build_source_cells(restored, components="continuum").total_fluence
                ),
                25.0,
            )

            with fits.open(path, mode="update") as hdus:
                hdus[0].header["FLU_LINE"] = 999.0
            with self.assertRaisesRegex(ValueError, "fluence closure"):
                load_source_fits(path)

    def test_mixed_sampling_respects_photon_fluence_and_support(self):
        cells = build_source_cells(self.source)
        packets = jax.jit(sample_source_cells, static_argnames=("n_packets",))(
            jax.random.PRNGKey(9), cells, n_packets=30000
        )
        energy = np.asarray(packets.energy_kev)
        time = np.asarray(packets.emission_time_s)
        index = np.asarray(packets.spectral_bin_index)
        np.testing.assert_allclose(
            np.sum(np.asarray(packets.weight_observer_fluence)),
            33.0,
            rtol=2e-6,
        )
        self.assertTrue(np.all((energy >= 2.0) & (energy <= 10.0)))
        self.assertTrue(np.all((time >= 0.0) & (time < 5.0)))
        self.assertTrue(np.all(energy[index == 2] == 6.4))
        self.assertAlmostEqual(float(np.mean(index == 2)), 8.0 / 33.0, delta=0.012)
        self.assertAlmostEqual(float(np.mean(index == 0)), 7.0 / 33.0, delta=0.012)

    def test_rejects_ambiguous_input_and_empty_selection(self):
        with self.assertRaisesRegex(ValueError, "no lines"):
            build_source_cells(
                SourceFluxFile(
                    np.array([0.0, 1.0]),
                    np.array([2.0, 4.0]),
                    np.array([[1.0]]),
                    ("FLAT",),
                    np.array([[0.0]]),
                    np.array([]),
                    np.empty((1, 0)),
                ),
                components="lines",
            )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            SourceFluxFile(
                np.array([0.0, 1.0]),
                np.array([]),
                np.empty((1, 0)),
                (),
                np.empty((1, 0)),
                np.array([6.4]),
                np.array([[-1.0]]),
                ("line",),
            )

    def test_line_only_source_with_dark_first_interval(self):
        source = SourceFluxFile(
            np.array([0.0, 2.0, 3.0]),
            np.array([]),
            np.empty((2, 0)),
            (),
            np.empty((2, 0)),
            np.array([6.4]),
            np.array([[0.0], [2.0]]),
            ("FeKa",),
        )
        with tempfile.TemporaryDirectory() as directory:
            restored = load_source_fits(
                write_source_fits(Path(directory) / "line.fits", source)
            )
            cells = build_source_cells(restored)
            self.assertEqual(float(cells.total_fluence), 2.0)
            self.assertEqual(int(cells.time_index[0]), 1)
            self.assertAlmostEqual(float(cells.energy_low_kev[0]), 6.4, places=5)


if __name__ == "__main__":
    unittest.main()
