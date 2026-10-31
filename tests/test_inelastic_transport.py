"""Controlled acceptance of evolving-energy transport and virtual photons."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from dsh.geometry.clouds import build_angular_distance_cloud
from dsh.geometry.rays import integrate_ray_column_cm2
from dsh.physics.cold_compton import SOURCE_PATH, build_cold_compton_table
from dsh.physics.compton import (
    build_free_electron_compton_table,
    to_compton_physics,
)
from dsh.physics.dust import (
    build_dust_physics_table,
    phase_cdf_from_differential_cross_section,
)
from dsh.physics.material_definitions import load_material_definition
from dsh.sources.launch import LaunchedSourcePackets
from dsh.transport.inelastic import (
    COMPTON_SCATTERING,
    ENERGY_CUTOFF,
    UNSUPPORTED_CONTINUATION,
    interaction_opacities,
    transport_inelastic_batch,
    transport_photon_inelastic,
)
from dsh.transport.kernel import (
    ABSORBED,
    DUST_SCATTERING,
    INVALID_ENERGY,
    INVALID_STATE,
    MAX_INTERACTIONS,
    PHOTOELECTRIC_ABSORPTION,
    REACHED_OBSERVER_PLANE,
    transport_photon_batch,
)


def cloud(column=2e21):
    return build_angular_distance_cloud(
        np.full((2, 3, 3), column / 2),
        x_centers_arcsec=[-60000.0, 0.0, 60000.0],
        y_centers_arcsec=[-60000.0, 0.0, 60000.0],
        z_centers_kpc=[4.0, 6.0],
        source_distance_kpc=10.0,
    )


def dust_table(scattering=0.0, absorption=0.0, *, energies=(1.0, 10.0)):
    energies = np.asarray(energies)
    scattering = np.broadcast_to(scattering, energies.shape)
    absorption = np.broadcast_to(absorption, energies.shape)
    theta = np.linspace(0, np.pi, 513)
    sigma, cdf = phase_cdf_from_differential_cross_section(
        theta, np.ones((len(energies), len(theta)))
    )
    differential = (
        None
        if np.any(scattering == 0)
        else scattering[:, None] / sigma[:, None] * np.ones_like(cdf)
    )
    return build_dust_physics_table(
        energies, scattering, absorption, theta, cdf, differential
    )


def compton_table(electrons=1.0, energies=(1.0, 2.0, 3.3, 4.9, 6.9, 10.0)):
    return to_compton_physics(
        build_free_electron_compton_table(
            energies,
            electrons_per_h=electrons,
            material_definition_sha256="b" * 64,
            cosine_nodes=257,
        )
    )


def launch(n=1, energy=4.9, transverse=0.0, weight=1.0):
    axis = jnp.array([-np.sqrt(1 - transverse**2), transverse, 0.0])
    position = jnp.broadcast_to(jnp.array([10000.0, 0.0, 0.0]), (n, 3))
    momentum = jnp.broadcast_to(jnp.r_[energy, energy * axis], (n, 4))
    return LaunchedSourcePackets(
        position,
        momentum,
        jnp.ones(n),
        jnp.full(n, 1 / (4 * np.pi)),
        jnp.zeros(n),
        jnp.full(n, weight),
        jnp.zeros(n, dtype=jnp.int32),
        jnp.zeros(n, dtype=jnp.int32),
    )


def run(packets, scene, dust, compton, *, cutoff=1.0, steps=8, seed=123):
    return jax.jit(transport_inelastic_batch, static_argnames=("max_interactions",))(
        random.PRNGKey(seed),
        packets.position_pc,
        packets.momentum_kev,
        scene,
        dust,
        compton,
        energy_cutoff_kev=cutoff,
        max_interactions=steps,
    )


class TransportTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        jax.clear_caches()

    def test_disabled_path_is_bitwise_equal_to_accepted_transport(self):
        p, scene, dust = launch(64), cloud(4e22), dust_table(1e-22, 1e-22)
        reference = jax.jit(
            transport_photon_batch, static_argnames=("max_interactions",)
        )(
            random.PRNGKey(123),
            p.position_pc,
            p.momentum_kev,
            scene,
            dust,
            max_interactions=5,
        )
        actual = run(p, scene, dust, None, steps=5)
        for field in reference._fields:
            if field != "interactions":
                np.testing.assert_array_equal(
                    getattr(actual, field), getattr(reference, field)
                )
        for field in reference.interactions._fields:
            np.testing.assert_array_equal(
                getattr(actual.interactions, field),
                getattr(reference.interactions, field),
            )
        np.testing.assert_array_equal(actual.n_compton, 0)

    def test_zero_column_crosses_observer_with_no_events(self):
        p = launch(8)
        r = run(p, cloud(0), dust_table(), compton_table(), steps=3)
        np.testing.assert_array_equal(r.status, REACHED_OBSERVER_PLANE)
        np.testing.assert_array_equal(r.interactions.valid, False)
        np.testing.assert_array_equal(r.momentum_kev, p.momentum_kev)

    def test_first_event_competition_and_transmission(self):
        n = 16000
        p, scene, dust, c = (
            launch(n),
            cloud(),
            dust_table(1e-22, 2e-22),
            compton_table(300.0),
        )
        r = run(p, scene, dust, c, steps=1)
        _, sd, sc, sa = interaction_opacities(dust, c, jnp.asarray(4.9))
        sigma = np.array([float(sd), float(sc), float(sa)])
        event_probability = -np.expm1(-2e21 * sigma.sum())
        kind = np.asarray(r.interactions.interaction_type)[:, 0]
        probabilities = np.r_[
            np.exp(-2e21 * sigma.sum()), event_probability * sigma / sigma.sum()
        ]
        measured = [
            (kind == value).mean()
            for value in [
                0,
                DUST_SCATTERING,
                COMPTON_SCATTERING,
                PHOTOELECTRIC_ABSORPTION,
            ]
        ]
        errors = np.sqrt(probabilities * (1 - probabilities) / n)
        np.testing.assert_array_less(
            abs(np.array(measured) - probabilities), 5 * errors + 0.001
        )

    def test_recoil_null_momenta_and_energy_accounting(self):
        p = launch(512)
        r = run(p, cloud(3e24), dust_table(1e-25, 2e-25), compton_table(), steps=10)
        rec = r.interactions
        selected = np.asarray(rec.valid)
        incoherent = selected & (np.asarray(rec.interaction_type) == COMPTON_SCATTERING)
        self.assertGreater(incoherent.sum(), 30)
        incoming, outgoing = (
            np.asarray(rec.incoming_momentum_kev),
            np.asarray(rec.outgoing_momentum_kev),
        )
        mu = np.sum(incoming[..., 1:] * outgoing[..., 1:], axis=-1) / np.maximum(
            incoming[..., 0] * outgoing[..., 0], 1e-30
        )
        expected = incoming[..., 0] / (1 + incoming[..., 0] / 510.99895069 * (1 - mu))
        np.testing.assert_allclose(
            outgoing[..., 0][incoherent], expected[incoherent], rtol=4e-7
        )
        np.testing.assert_allclose(
            np.linalg.norm(outgoing[..., 1:], axis=-1)[selected],
            outgoing[..., 0][selected],
            rtol=4e-7,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            r.deposited_energy_kev + r.momentum_kev[:, 0],
            p.momentum_kev[:, 0],
            rtol=5e-7,
            atol=3e-6,
        )
        np.testing.assert_allclose(
            r.recoil_energy_kev + r.absorbed_energy_kev,
            r.deposited_energy_kev,
            rtol=3e-7,
        )
        np.testing.assert_array_equal(r.n_scatter, r.n_dust + r.n_compton)
        np.testing.assert_array_equal(
            np.asarray(rec.scattering_order)[selected],
            (np.asarray(rec.dust_order) + np.asarray(rec.compton_order))[selected],
        )

    def test_cutoff_keeps_residual_energy_and_prior_records(self):
        r = run(
            launch(128, transverse=0.1),
            cloud(2e24),
            dust_table(),
            compton_table(),
            cutoff=4.89,
            steps=6,
        )
        cut = np.asarray(r.status) == ENERGY_CUTOFF
        self.assertGreater(cut.sum(), 40)
        self.assertTrue(np.all(np.asarray(r.n_compton)[cut] >= 1))
        np.testing.assert_allclose(
            np.asarray(r.cutoff_energy_kev)[cut],
            np.asarray(r.momentum_kev)[cut, 0],
            rtol=1e-7,
        )
        self.assertTrue(np.all(np.asarray(r.cutoff_energy_kev)[cut] < 4.89))
        np.testing.assert_array_equal(np.asarray(r.absorbed_energy_kev)[cut], 0)

    def test_downscatter_outside_support_is_a_distinct_incomplete_status(self):
        r = run(
            launch(128, energy=2.0),
            cloud(2e24),
            dust_table(energies=(2.0, 10.0)),
            compton_table(energies=(2.0, 4.9, 10.0)),
            cutoff=1.0,
            steps=4,
        )
        lost = np.asarray(r.status) == UNSUPPORTED_CONTINUATION
        self.assertGreater(lost.sum(), 40)
        self.assertTrue(np.all(np.asarray(r.unsupported_energy_kev)[lost] < 2))
        np.testing.assert_array_equal(
            np.asarray(r.interactions.valid)[lost].sum(axis=1), 1
        )

    def test_invalid_inputs_and_initial_energy(self):
        scene, d, c = cloud(), dust_table(), compton_table()
        for momentum, expected in [
            ([0.5, -0.5, 0, 0], INVALID_ENERGY),
            ([5, -3, 0, 0], INVALID_STATE),
            ([np.nan, -5, 0, 0], INVALID_STATE),
        ]:
            r = transport_photon_inelastic(
                random.PRNGKey(0),
                jnp.array([10000.0, 0.0, 0.0]),
                jnp.array(momentum),
                scene,
                d,
                c,
                energy_cutoff_kev=1.0,
                max_interactions=2,
            )
            self.assertEqual(int(r.status), expected)
            self.assertFalse(np.asarray(r.interactions.valid).any())

    def test_absorption_recomputed_after_recoil_across_an_edge(self):
        d = dust_table(
            0.0, [1e-19, 1e-19, 0.0, 0.0, 0.0], energies=[2.0, 4.84, 4.85, 4.9, 10.0]
        )
        r = run(launch(1024), cloud(3e24), d, compton_table(), steps=3)
        rec = r.interactions
        selected = np.asarray(rec.valid)[:, 1] & (
            np.asarray(rec.incoming_momentum_kev)[:, 1, 0] < 4.84
        )
        self.assertGreater(selected.sum(), 100)
        self.assertGreater(
            (
                np.asarray(rec.interaction_type)[selected, 1]
                == PHOTOELECTRIC_ABSORPTION
            ).mean(),
            0.995,
        )
        np.testing.assert_array_equal(np.asarray(r.status)[selected], ABSORBED)

    def test_dust_opacity_recomputed_after_recoil(self):
        d = dust_table(
            [1e-19, 1e-19, 0.0, 0.0, 0.0], 0.0, energies=[2.0, 4.84, 4.85, 4.9, 10.0]
        )
        r = run(launch(1024), cloud(3e24), d, compton_table(), steps=3)
        rec = r.interactions
        selected = np.asarray(rec.valid)[:, 1] & (
            np.asarray(rec.incoming_momentum_kev)[:, 1, 0] < 4.84
        )
        self.assertGreater(selected.sum(), 100)
        self.assertGreater(
            (np.asarray(rec.interaction_type)[selected, 1] == DUST_SCATTERING).mean(),
            0.995,
        )

    def test_compton_opacity_recomputed_on_second_flight(self):
        # High-energy FREE-electron control makes recoil large enough to
        # distinguish a frozen opacity from the correct energy-dependent one.
        c = compton_table(energies=np.geomspace(1, 1000, 33))
        scene, d, p = (
            cloud(1e25),
            dust_table(energies=[1.0, 1000.0]),
            launch(4096, energy=500.0),
        )
        r = run(p, scene, d, c, steps=2)
        first = np.asarray(r.interactions.valid)[:, 0]
        pos = r.interactions.position_pc[:, 0]
        outgoing = r.interactions.outgoing_momentum_kev[:, 0]
        directions = outgoing[:, 1:] / jnp.maximum(outgoing[:, 0, None], 1e-30)
        fallback = jnp.array([-1.0, 0.0, 0.0])
        directions = jnp.where(jnp.asarray(first[:, None]), directions, fallback)
        from dsh.transport.kernel import _transport_boundary_pc

        distances, _ = jax.vmap(
            lambda x, axis: _transport_boundary_pc(x, axis, 10000.0)
        )(pos, directions)
        column = np.asarray(
            jax.vmap(
                lambda x, axis, length: integrate_ray_column_cm2(scene, x, axis, length)
            )(pos, directions, distances)
        )[first]
        energies = np.asarray(outgoing)[first, 0]
        sigma = np.exp(
            np.interp(
                np.log(energies),
                np.log(np.asarray(c.energy_kev)),
                np.log(np.asarray(c.cross_section_cm2_per_h)),
            )
        )
        probability = -np.expm1(-column * sigma)
        observed = np.asarray(r.interactions.valid)[first, 1]
        error = np.sqrt(np.sum(probability * (1 - probability)))
        self.assertLess(abs(np.sum(observed - probability)), 5 * error + 1)
        frozen_sigma = np.exp(
            np.interp(
                np.log(500.0),
                np.log(np.asarray(c.energy_kev)),
                np.log(np.asarray(c.cross_section_cm2_per_h)),
            )
        )
        frozen = -np.expm1(-column * frozen_sigma)
        self.assertGreater(np.sum(probability - frozen), 3 * error)

    def test_safety_limit_is_not_reported_as_escape_or_absorption(self):
        r = run(launch(16), cloud(1e29), dust_table(1e-24), compton_table(), steps=1)
        np.testing.assert_array_equal(r.status, MAX_INTERACTIONS)
        np.testing.assert_array_equal(r.n_scatter, 1)
        np.testing.assert_allclose(
            r.deposited_energy_kev + r.momentum_kev[:, 0], 4.9, rtol=5e-7
        )
        np.testing.assert_array_equal(r.cutoff_energy_kev, 0.0)

    def test_inactive_padding_is_exactly_zero(self):
        r = run(
            launch(16), cloud(1e25), dust_table(0.0, 1e-19), compton_table(), steps=5
        )
        np.testing.assert_array_equal(r.status, ABSORBED)
        unused = ~np.asarray(r.interactions.valid)
        for values in r.interactions:
            np.testing.assert_array_equal(np.asarray(values)[unused], 0)

    def test_initial_cutoff_is_separate_from_absorption(self):
        r = run(launch(8), cloud(), dust_table(), compton_table(), cutoff=5.0, steps=2)
        np.testing.assert_array_equal(r.status, ENERGY_CUTOFF)
        np.testing.assert_array_equal(r.n_interactions, 0)
        np.testing.assert_array_equal(r.deposited_energy_kev, 0)
        np.testing.assert_allclose(r.cutoff_energy_kev, 4.9, rtol=1e-7)

    def test_invalid_opacity_requests_never_return_clipped_data(self):
        supported, sd, sc, sa = interaction_opacities(
            dust_table(1e-22, 2e-22),
            compton_table(),
            jnp.array([np.nan, 0.5, 4.9, 11.0]),
        )
        np.testing.assert_array_equal(supported, [False, False, True, False])
        for opacity in [sd, sc, sa]:
            np.testing.assert_array_equal(np.asarray(opacity)[[0, 1, 3]], 0)
            self.assertGreater(float(opacity[2]), 0)

    def test_bound_material_provider_connects_to_real_transport(self):
        material = load_material_definition(
            SOURCE_PATH.parent / "reflex_reference_definition.json"
        )
        c = to_compton_physics(
            build_cold_compton_table(
                material, [2, 3.3, 4.9, 6.9, 10], grain_electrons="omit"
            ).table
        )
        r = run(
            launch(128),
            cloud(3e24),
            dust_table(energies=[2.0, 10.0]),
            c,
            cutoff=2.0,
            steps=8,
        )
        self.assertGreater(int(np.asarray(r.n_compton).sum()), 80)
        np.testing.assert_array_equal(r.n_dust, 0)
        np.testing.assert_allclose(
            r.deposited_energy_kev + r.momentum_kev[:, 0], 4.9, rtol=8e-7
        )

    def test_shape_and_static_limit_validation(self):
        p, d, c, scene = launch(), dust_table(), compton_table(), cloud()
        for steps in [0, True]:
            with self.assertRaises(ValueError):
                transport_photon_inelastic(
                    random.PRNGKey(0),
                    p.position_pc[0],
                    p.momentum_kev[0],
                    scene,
                    d,
                    c,
                    energy_cutoff_kev=1.0,
                    max_interactions=steps,
                )
        with self.assertRaises(ValueError):
            transport_inelastic_batch(
                random.PRNGKey(0),
                jnp.empty((0, 3)),
                jnp.empty((0, 4)),
                scene,
                d,
                c,
                energy_cutoff_kev=1.0,
            )
        for bad in [0.0, float("nan")]:
            r = run(p, scene, d, c, cutoff=bad, steps=1)
            np.testing.assert_array_equal(r.status, INVALID_STATE)

    def test_packet_and_table_precision_can_differ(self):
        p = launch(16)
        p = p._replace(
            position_pc=p.position_pc.astype(jnp.float32),
            momentum_kev=p.momentum_kev.astype(jnp.float32),
        )
        r = run(p, cloud(2e24), dust_table(), compton_table(), steps=4)
        np.testing.assert_allclose(
            r.momentum_kev[:, 0] + r.deposited_energy_kev, 4.9, rtol=3e-7
        )


if __name__ == "__main__":
    unittest.main()
