"""Tests for weighted four-dimensional observer-event binning."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from dsh.geometry.coordinates import ARCSEC_TO_RAD
from dsh.observer.binning import (
    add_binned_observer_products,
    bin_observer_events,
    build_observer_bin_geometry,
    fluence_surface_brightness_per_sr,
    mean_flux_surface_brightness_per_sr_s,
)
from dsh.observer.scoring import ObserverEventResult


def _test_events():
    shape = (2, 4)
    valid = np.array([[True, True, True, True], [True, True, False, True]])
    sky_x = np.array([[-2.0, -1.0, 0.0, 2.0], [2.1, 1.0, -1.0, 1.0]])
    sky_y = np.array([[-2.0, -1.0, 0.0, 2.0], [0.0, 0.0, -1.0, -1.0]])
    energy = np.array([[2.0, 3.0, 4.0, 6.0], [3.0, 3.0, 3.0, 5.0]])
    arrival = np.array([[0.0, 5.0, 10.0, 20.0], [5.0, -1.0, 5.0, 15.0]])
    order = np.array([[1, 1, 2, 3], [1, 2, 1, 1]], dtype=np.int32)
    weight = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 100.0, 7.0]])
    zeros = jnp.zeros(shape, dtype=jnp.float32)
    return ObserverEventResult(
        valid=jnp.asarray(valid),
        sky_x_arcsec=jnp.asarray(sky_x, dtype=jnp.float32),
        sky_y_arcsec=jnp.asarray(sky_y, dtype=jnp.float32),
        energy_kev=jnp.asarray(energy, dtype=jnp.float32),
        arrival_time_s=jnp.asarray(arrival, dtype=jnp.float32),
        excess_path_length_pc=zeros,
        scattering_angle_rad=zeros,
        scattering_order=jnp.asarray(order),
        escape_column_cm2=zeros,
        escape_optical_depth=zeros,
        transmission=zeros,
        phase_pdf_per_sr=zeros,
        weight_observer_fluence=jnp.asarray(weight, dtype=jnp.float32),
        time_index=jnp.zeros(shape, dtype=jnp.int32),
        spectral_bin_index=jnp.zeros(shape, dtype=jnp.int32),
    )


class TestObserverBinning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = build_observer_bin_geometry(
            sky_x_edges_arcsec=[-2.0, 0.0, 2.0],
            sky_y_edges_arcsec=[-2.0, 0.0, 2.0],
            energy_edges_kev=[2.0, 4.0, 6.0],
            arrival_time_edges_s=[0.0, 10.0, 20.0],
        )
        cls.events = _test_events()

    def test_jitted_binning_uses_t_e_y_x_order_and_histogram_edges(self):
        products = jax.jit(bin_observer_events)(self.events, self.geometry)
        self.assertEqual(products.total_fluence.shape, (2, 2, 2, 2))

        expected_total = np.zeros((2, 2, 2, 2), dtype=np.float32)
        expected_first = np.zeros_like(expected_total)
        expected_multiple = np.zeros_like(expected_total)
        expected_count = np.zeros_like(expected_total, dtype=np.int32)
        expected_total[0, 0, 0, 0] = 3.0
        expected_first[0, 0, 0, 0] = 3.0
        expected_count[0, 0, 0, 0] = 2
        expected_total[1, 1, 1, 1] = 7.0
        expected_multiple[1, 1, 1, 1] = 7.0
        expected_count[1, 1, 1, 1] = 2
        expected_total[1, 1, 0, 1] = 7.0
        expected_first[1, 1, 0, 1] = 7.0
        expected_count[1, 1, 0, 1] = 1

        np.testing.assert_array_equal(products.total_fluence, expected_total)
        np.testing.assert_array_equal(products.first_scatter_fluence, expected_first)
        np.testing.assert_array_equal(
            products.multiple_scatter_fluence, expected_multiple
        )
        np.testing.assert_array_equal(products.event_count, expected_count)

    def test_weight_and_event_closure(self):
        products = bin_observer_events(self.events, self.geometry)
        self.assertEqual(int(products.valid_event_count), 7)
        self.assertEqual(int(products.binned_event_count), 5)
        self.assertEqual(int(products.unbinned_event_count), 2)
        self.assertEqual(float(products.valid_weight_observer_fluence), 28.0)
        self.assertEqual(float(products.binned_weight_observer_fluence), 17.0)
        self.assertEqual(float(products.unbinned_weight_observer_fluence), 11.0)
        self.assertEqual(int(products.outside_sky_event_count), 1)
        self.assertEqual(int(products.outside_energy_event_count), 0)
        self.assertEqual(int(products.outside_arrival_time_event_count), 1)
        self.assertEqual(float(products.outside_sky_weight_observer_fluence), 5.0)
        self.assertEqual(float(products.outside_energy_weight_observer_fluence), 0.0)
        self.assertEqual(
            float(products.outside_arrival_time_weight_observer_fluence), 6.0
        )
        self.assertEqual(float(jnp.sum(products.total_fluence)), 17.0)
        self.assertEqual(float(jnp.sum(products.first_scatter_fluence)), 10.0)
        self.assertEqual(float(jnp.sum(products.multiple_scatter_fluence)), 7.0)
        np.testing.assert_array_equal(
            products.total_fluence,
            products.first_scatter_fluence + products.multiple_scatter_fluence,
        )

    def test_exact_solid_angles_and_surface_brightness_units(self):
        one_arcsec = build_observer_bin_geometry(
            sky_x_edges_arcsec=[-0.5, 0.5],
            sky_y_edges_arcsec=[-0.5, 0.5],
            energy_edges_kev=[3.0, 4.0],
            arrival_time_edges_s=[0.0, 2.0],
        )
        expected_small_angle = ARCSEC_TO_RAD**2
        self.assertAlmostEqual(
            float(one_arcsec.sky_pixel_solid_angle_sr[0, 0]) / expected_small_angle,
            1.0,
            places=6,
        )

        products = bin_observer_events(self.events, self.geometry)
        brightness = fluence_surface_brightness_per_sr(
            products.total_fluence, self.geometry
        )
        reconstructed_fluence = (
            brightness * self.geometry.sky_pixel_solid_angle_sr[None, None, :, :]
        )
        np.testing.assert_allclose(
            reconstructed_fluence, products.total_fluence, rtol=2.0e-7
        )

        mean_brightness = mean_flux_surface_brightness_per_sr_s(
            products.total_fluence, self.geometry
        )
        durations = jnp.diff(self.geometry.arrival_time_edges_s)
        np.testing.assert_allclose(
            mean_brightness
            * durations[:, None, None, None]
            * self.geometry.sky_pixel_solid_angle_sr[None, None, :, :],
            products.total_fluence,
            rtol=2.0e-7,
        )

    def test_chunked_accumulation_matches_one_batch(self):
        full = bin_observer_events(self.events, self.geometry)
        first_chunk = jax.tree.map(lambda value: value[:1], self.events)
        second_chunk = jax.tree.map(lambda value: value[1:], self.events)
        left = bin_observer_events(first_chunk, self.geometry)
        right = bin_observer_events(second_chunk, self.geometry)
        combined = jax.jit(add_binned_observer_products)(left, right)

        for field in full._fields:
            np.testing.assert_array_equal(
                np.asarray(getattr(combined, field)),
                np.asarray(getattr(full, field)),
            )

    def test_rejects_invalid_edges_and_inconsistent_shapes(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            build_observer_bin_geometry(
                [-1.0, 0.0, 0.0], [-1.0, 1.0], [2.0, 3.0], [0.0, 1.0]
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            build_observer_bin_geometry(
                [-1.0, 1.0], [-1.0, 1.0], [0.0, 3.0], [0.0, 1.0]
            )
        with self.assertRaisesRegex(ValueError, r"\+/-90"):
            build_observer_bin_geometry(
                [-400_000.0, 0.0], [-1.0, 1.0], [2.0, 3.0], [0.0, 1.0]
            )

        bad_events = self.events._replace(
            energy_kev=jnp.ones((1, 4), dtype=jnp.float32)
        )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            bin_observer_events(bad_events, self.geometry)
        with self.assertRaisesRegex(ValueError, "shape"):
            fluence_surface_brightness_per_sr(jnp.zeros((1, 1, 1, 1)), self.geometry)


if __name__ == "__main__":
    unittest.main()
