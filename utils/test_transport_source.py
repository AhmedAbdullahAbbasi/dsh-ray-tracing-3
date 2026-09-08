"""Integration tests for physical source packets and JAX transport."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from utils.source import (
    SourcePackets,
    build_variable_powerlaw_source,
    sample_variable_powerlaw_source,
)
from utils.transport import KILLED, simulate_source_packets


class TestSourcePacketTransport(unittest.TestCase):
    def setUp(self):
        self.density_grid = jnp.zeros((4, 4, 4), dtype=jnp.float32)
        self.box_min = jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float32)
        self.box_max = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)

    def test_packet_metadata_survives_jitted_transport(self):
        source = build_variable_powerlaw_source(
            time_edges_s=[0.0, 1.0, 3.0],
            photon_flux=[1.0, 2.0],
            energy_min_kev=2.0,
            energy_max_kev=10.0,
            photon_index=2.0,
        )
        packets = sample_variable_powerlaw_source(
            random.PRNGKey(2), source, n_packets=512
        )
        transport_jit = jax.jit(
            simulate_source_packets,
            static_argnums=(2, 3),
            static_argnames=("scattering_model", "grain_radius_um"),
        )

        result = transport_jit(
            random.PRNGKey(8),
            packets,
            1,
            16,
            self.density_grid,
            self.box_min,
            self.box_max,
            0.25,
            scattering_model="isotropic",
            grain_radius_um=0.1,
        )

        np.testing.assert_array_equal(
            np.asarray(result.energy_kev), np.asarray(packets.energy_kev)
        )
        np.testing.assert_array_equal(
            np.asarray(result.emission_time_s),
            np.asarray(packets.emission_time_s),
        )
        np.testing.assert_array_equal(
            np.asarray(result.weight_observer_fluence),
            np.asarray(packets.weight_observer_fluence),
        )
        np.testing.assert_array_equal(
            np.asarray(result.time_index), np.asarray(packets.time_index)
        )
        np.testing.assert_array_equal(
            np.asarray(result.spectral_bin_index),
            np.asarray(packets.spectral_bin_index),
        )
        self.assertTrue(np.all(np.asarray(result.status) == KILLED))
        self.assertTrue(np.all(np.asarray(result.n_scatter) == 0))
        self.assertEqual(result.path.shape, (512, 2, 3))

    def test_mie_conversion_does_not_replace_physical_energy(self):
        packets = SourcePackets(
            energy_kev=jnp.array([2.0, 8.0], dtype=jnp.float32),
            emission_time_s=jnp.array([10.0, 20.0], dtype=jnp.float32),
            weight_observer_fluence=jnp.array([0.5, 0.5], dtype=jnp.float32),
            time_index=jnp.array([0, 1], dtype=jnp.int32),
            spectral_bin_index=jnp.array([-1, -1], dtype=jnp.int32),
        )

        transport_jit = jax.jit(
            simulate_source_packets,
            static_argnames=(
                "n_bounces",
                "n_substeps",
                "scattering_model",
                "grain_radius_um",
            ),
        )
        result = transport_jit(
            random.PRNGKey(4),
            packets,
            n_bounces=1,
            n_substeps=8,
            density_grid=self.density_grid,
            box_min=self.box_min,
            box_max=self.box_max,
            beam_radius=0.0,
            scattering_model="mie",
            grain_radius_um=0.1,
        )

        np.testing.assert_array_equal(np.asarray(result.energy_kev), [2.0, 8.0])
        self.assertAlmostEqual(
            float(np.asarray(result.weight_observer_fluence).sum()), 1.0
        )

    def test_rejects_mismatched_packet_shapes(self):
        packets = SourcePackets(
            energy_kev=jnp.ones(2),
            emission_time_s=jnp.ones(2),
            weight_observer_fluence=jnp.ones(1),
            time_index=jnp.zeros(2, dtype=jnp.int32),
            spectral_bin_index=jnp.zeros(2, dtype=jnp.int32),
        )

        with self.assertRaisesRegex(ValueError, "same length"):
            simulate_source_packets(
                random.PRNGKey(0),
                packets,
                1,
                1,
                self.density_grid,
                self.box_min,
                self.box_max,
                0.0,
            )

    def test_rejects_nonpositive_mie_grain_radius(self):
        packets = SourcePackets(
            energy_kev=jnp.ones(1),
            emission_time_s=jnp.zeros(1),
            weight_observer_fluence=jnp.ones(1),
            time_index=jnp.zeros(1, dtype=jnp.int32),
            spectral_bin_index=jnp.full(1, -1, dtype=jnp.int32),
        )

        with self.assertRaisesRegex(ValueError, "grain_radius_um"):
            simulate_source_packets(
                random.PRNGKey(0),
                packets,
                1,
                1,
                self.density_grid,
                self.box_min,
                self.box_max,
                0.0,
                scattering_model="mie",
                grain_radius_um=0.0,
            )


if __name__ == "__main__":
    unittest.main()
