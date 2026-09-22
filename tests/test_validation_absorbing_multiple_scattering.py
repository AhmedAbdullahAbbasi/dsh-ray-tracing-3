"""Reference probability and actual repeated transport with TBabs enabled."""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from dsh.geometry.clouds import build_angular_distance_cloud
from dsh.observer.scoring import score_peeloff_events
from dsh.physics.absorption import load_photoelectric_absorption_table
from dsh.physics.newdust import (
    build_dust_physics_from_tables,
    load_newdust_scattering_table,
)
from dsh.sources.launch import LaunchedSourcePackets
from dsh.transport.kernel import ABSORBED, DUST_SCATTERING, transport_photon_batch
from dsh.validation.absorbing_multiple_scattering import (
    competing_flight_probabilities,
    run_absorbing_multiple_scattering_case,
)
from dsh.validation.multiple_scattering import analytic_shell_column


class TestAbsorbingMultipleScattering(unittest.TestCase):
    def test_independent_finite_flight_competing_risks(self):
        p = competing_flight_probabilities(1e22, 2e-23, 1e-23)
        self.assertAlmostEqual(p["escape"], math.exp(-0.3))
        self.assertAlmostEqual(p["scatter"] / p["absorb"], 2.0)
        self.assertAlmostEqual(sum(p.values()), 1.0)
        empty = competing_flight_probabilities(0, 2e-23, 1e-23)
        self.assertEqual(empty, {"scatter": 0.0, "absorb": 0.0, "escape": 1.0})
        self.assertEqual(
            competing_flight_probabilities(1e22, 0.0, 0.0),
            {"scatter": 0.0, "absorb": 0.0, "escape": 1.0},
        )

    def test_actual_absorption_after_scattering_and_order_probabilities(self):
        report = run_absorbing_multiple_scattering_case(
            random.PRNGKey(102),
            load_newdust_scattering_table(),
            load_photoelectric_absorption_table(),
            energy_kev=3.3,
            target_tau_scattering=2.4,
            packets=2_048,
            chunk_size=512,
            min_expected_category=5,
        )
        self.assertTrue(report["all_passed"], report)
        self.assertEqual(report["invalid_histories"], 0)
        self.assertGreater(report["absorption_after_two_scatters"], 10)
        self.assertGreater(report["status_counts"][3], 0)

    def test_real_absorbed_histories_score_earlier_scattering_orders(self):
        scattering = load_newdust_scattering_table()
        absorption = load_photoelectric_absorption_table()
        energy = 3.3
        column = 2.4 / scattering.scattering_cross_section_cm2_per_h[0]
        cloud = build_angular_distance_cloud(
            np.full((2, 2, 2), column / 2),
            x_centers_arcsec=[-150_000.0, 150_000.0],
            y_centers_arcsec=[-150_000.0, 150_000.0],
            z_centers_kpc=[4.25, 4.75],
            source_distance_kpc=10,
        )
        n = 256
        position = jnp.broadcast_to(jnp.asarray([10_000.0, 0.0, 0.0]), (n, 3))
        momentum = jnp.broadcast_to(jnp.asarray([energy, -energy, 0.0, 0.0]), (n, 4))
        physics = build_dust_physics_from_tables(scattering, absorption)
        transport = jax.jit(
            transport_photon_batch, static_argnames=("max_interactions",)
        )(random.PRNGKey(919), position, momentum, cloud, physics, max_interactions=4)
        launched = LaunchedSourcePackets(
            position_pc=position,
            momentum_kev=momentum,
            launch_pdf_per_sr=jnp.full(n, 1.0 / (4.0 * math.pi)),
            isotropic_importance=jnp.ones(n),
            emission_time_s=jnp.zeros(n),
            weight_observer_fluence=jnp.ones(n),
            time_index=jnp.zeros(n, dtype=jnp.int32),
            spectral_bin_index=jnp.zeros(n, dtype=jnp.int32),
        )
        scored = jax.jit(score_peeloff_events)(launched, transport, cloud, physics)
        kinds = np.asarray(transport.interactions.interaction_type)
        valid = np.asarray(transport.interactions.valid)
        score_mask = np.asarray(scored.valid)
        self.assertTrue(np.array_equal(score_mask, valid & (kinds == DUST_SCATTERING)))
        absorbed_after_two = (np.asarray(transport.status) == ABSORBED) & (
            np.asarray(transport.n_scatter) >= 2
        )
        self.assertGreater(int(absorbed_after_two.sum()), 0)
        self.assertTrue(np.all(score_mask[absorbed_after_two, :2]))
        self.assertTrue(
            np.all(np.asarray(scored.weight_observer_fluence)[~score_mask] == 0)
        )
        samples = np.argwhere(score_mask)[:32]
        for i, j in samples:
            point = np.asarray(transport.interactions.position_pc[i, j], dtype=float)
            radius = np.linalg.norm(point)
            escape = analytic_shell_column(point, -point / radius, radius, column)
            tau = escape * (
                scattering.scattering_cross_section_cm2_per_h[0]
                + absorption.absorption_cross_section_cm2_per_h[0]
            )
            self.assertAlmostEqual(
                float(scored.escape_column_cm2[i, j]) / escape, 1.0, delta=1e-4
            )
            self.assertAlmostEqual(
                float(scored.transmission[i, j]), math.exp(-tau), delta=2e-5
            )


if __name__ == "__main__":
    unittest.main()
