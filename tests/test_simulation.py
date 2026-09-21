"""Tests for the complete ideal-observer DSH simulation pipeline."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from dsh.geometry.clouds import build_angular_distance_cloud
from dsh.observer.binning import (
    bin_observer_events,
    build_observer_bin_geometry,
)
from dsh.observer.scoring import score_peeloff_events
from dsh.physics.absorption import load_photoelectric_absorption_table
from dsh.physics.newdust import (
    build_dust_physics_from_tables,
    load_newdust_scattering_table,
)
from dsh.pipeline import (
    TRANSPORT_STATUS_LABELS,
    add_ideal_observer_simulation_results,
    run_tabulated_source_to_observer_chunked,
    simulate_tabulated_source_to_observer,
    simulate_variable_powerlaw_source_to_observer,
)
from dsh.sources.launch import build_cloud_launch_geometry, sample_source_launches
from dsh.sources.models import (
    build_tabulated_band_source,
    build_variable_powerlaw_source,
    sample_tabulated_band_source,
)
from dsh.transport.kernel import transport_photon_batch


class TestIdealObserverSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scattering = load_newdust_scattering_table()
        absorption = load_photoelectric_absorption_table()
        cls.physics = build_dust_physics_from_tables(scattering, absorption)
        radial_column = np.asarray([7.0e21, 8.0e21, 7.0e21])
        cls.cloud = build_angular_distance_cloud(
            np.broadcast_to(radial_column[:, None, None], (3, 3, 3)).copy(),
            x_centers_arcsec=[-400.0, 0.0, 400.0],
            y_centers_arcsec=[-400.0, 0.0, 400.0],
            z_centers_kpc=[2.0, 4.0, 6.0],
            source_distance_kpc=10.0,
        )
        cls.launch_geometry = build_cloud_launch_geometry(cls.cloud)
        cls.source = build_tabulated_band_source(
            time_edges_s=[0.0, 100.0, 300.0],
            band_flux=[
                [3.0e-3, 2.0e-3, 1.0e-3],
                [1.0e-3, 1.0e-3, 1.0e-3],
            ],
            effective_energy_kev=[3.3, 4.9, 6.9],
        )
        cls.bin_geometry = build_observer_bin_geometry(
            sky_x_edges_arcsec=[-600.0, 0.0, 600.0],
            sky_y_edges_arcsec=[-600.0, 0.0, 600.0],
            energy_edges_kev=[3.0, 4.0, 6.0, 7.5],
            arrival_time_edges_s=[0.0, 1.0e8, 2.0e8],
        )

    def test_jitted_runner_matches_explicit_pipeline(self):
        key = random.PRNGKey(401)
        n_packets = 48
        max_interactions = 4
        runner = jax.jit(
            simulate_tabulated_source_to_observer,
            static_argnames=("n_packets", "max_interactions"),
        )
        result = runner(
            key,
            self.source,
            self.launch_geometry,
            self.cloud,
            self.physics,
            self.bin_geometry,
            n_packets=n_packets,
            max_interactions=max_interactions,
        )

        source_key, pipeline_key = random.split(key)
        launch_key, transport_key = random.split(pipeline_key)
        packets = sample_tabulated_band_source(source_key, self.source, n_packets)
        launched = sample_source_launches(launch_key, packets, self.launch_geometry)
        transported = transport_photon_batch(
            transport_key,
            launched.position_pc,
            launched.momentum_kev,
            self.cloud,
            self.physics,
            max_interactions=max_interactions,
        )
        events = score_peeloff_events(launched, transported, self.cloud, self.physics)
        expected_products = bin_observer_events(events, self.bin_geometry)

        for field in expected_products._fields:
            np.testing.assert_array_equal(
                np.asarray(getattr(result.products, field)),
                np.asarray(getattr(expected_products, field)),
            )
        expected_status = np.bincount(
            np.asarray(transported.status),
            minlength=len(TRANSPORT_STATUS_LABELS),
        )
        np.testing.assert_array_equal(
            result.diagnostics.transport_status_count, expected_status
        )
        self.assertEqual(int(result.diagnostics.source_packet_count), n_packets)
        self.assertEqual(
            int(result.diagnostics.analog_interaction_count),
            int(jnp.sum(transported.n_interactions)),
        )
        self.assertEqual(
            int(result.diagnostics.analog_scattering_count),
            int(jnp.sum(transported.n_scatter)),
        )
        self.assertEqual(
            int(result.diagnostics.scored_observer_event_count),
            int(jnp.sum(events.valid)),
        )

    def test_chunked_runner_preserves_one_source_fluence(self):
        total_packets = 70
        progress = []
        result = run_tabulated_source_to_observer_chunked(
            random.PRNGKey(402),
            self.source,
            self.launch_geometry,
            self.cloud,
            self.physics,
            self.bin_geometry,
            total_packets=total_packets,
            chunk_size=32,
            max_interactions=4,
            progress_callback=lambda completed, total: progress.append(
                (completed, total)
            ),
        )
        diagnostics = result.diagnostics
        products = result.products
        self.assertEqual(int(diagnostics.source_packet_count), total_packets)
        np.testing.assert_allclose(
            diagnostics.source_fluence,
            self.source.total_fluence,
            rtol=2.0e-7,
        )
        self.assertEqual(
            int(jnp.sum(diagnostics.transport_status_count)), total_packets
        )
        self.assertEqual(
            int(diagnostics.scored_observer_event_count),
            int(products.valid_event_count),
        )
        np.testing.assert_allclose(
            diagnostics.scored_observer_fluence,
            products.valid_weight_observer_fluence,
            rtol=2.0e-7,
        )
        np.testing.assert_allclose(
            products.valid_weight_observer_fluence,
            products.binned_weight_observer_fluence
            + products.unbinned_weight_observer_fluence,
            rtol=2.0e-7,
        )
        self.assertEqual(progress, [(32, 70), (64, 70), (70, 70)])

    def test_result_addition_is_jittable_and_additive(self):
        runner = jax.jit(
            simulate_tabulated_source_to_observer,
            static_argnames=("n_packets", "max_interactions"),
        )
        left = runner(
            random.PRNGKey(403),
            self.source,
            self.launch_geometry,
            self.cloud,
            self.physics,
            self.bin_geometry,
            n_packets=16,
            max_interactions=3,
        )
        right = runner(
            random.PRNGKey(404),
            self.source,
            self.launch_geometry,
            self.cloud,
            self.physics,
            self.bin_geometry,
            n_packets=16,
            max_interactions=3,
        )
        combined = jax.jit(add_ideal_observer_simulation_results)(left, right)
        expected = jax.tree.map(
            lambda left_value, right_value: left_value + right_value,
            left,
            right,
        )
        for value, expected_value in zip(
            jax.tree.leaves(combined),
            jax.tree.leaves(expected),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(value), np.asarray(expected_value))

    def test_variable_powerlaw_entry_point_is_jittable(self):
        source = build_variable_powerlaw_source(
            time_edges_s=[0.0, 100.0],
            photon_flux=[2.0e-3],
            energy_min_kev=3.3,
            energy_max_kev=6.9,
            photon_index=2.0,
        )
        runner = jax.jit(
            simulate_variable_powerlaw_source_to_observer,
            static_argnames=("n_packets", "max_interactions"),
        )
        result = runner(
            random.PRNGKey(406),
            source,
            self.launch_geometry,
            self.cloud,
            self.physics,
            self.bin_geometry,
            n_packets=16,
            max_interactions=3,
        )
        self.assertEqual(int(result.diagnostics.source_packet_count), 16)
        self.assertEqual(int(jnp.sum(result.diagnostics.transport_status_count)), 16)
        np.testing.assert_allclose(
            result.diagnostics.source_fluence,
            source.total_fluence,
            rtol=2.0e-7,
        )

    def test_rejects_invalid_chunk_sizes(self):
        arguments = (
            random.PRNGKey(405),
            self.source,
            self.launch_geometry,
            self.cloud,
            self.physics,
            self.bin_geometry,
        )
        with self.assertRaisesRegex(ValueError, "total_packets"):
            run_tabulated_source_to_observer_chunked(
                *arguments, total_packets=0, chunk_size=8
            )
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            run_tabulated_source_to_observer_chunked(
                *arguments, total_packets=8, chunk_size=0
            )


if __name__ == "__main__":
    unittest.main()
