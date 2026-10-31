"""Independent geometry, energy, escape-opacity and component scoring tests."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from dsh.observer.binning import build_observer_bin_geometry
from dsh.observer.inelastic import (
    COMPTON_ONLY,
    DUST_ONLY,
    MIXED,
    bin_inelastic_events,
    compton_observer_phase_and_energy,
    score_inelastic_peeloff,
)
from dsh.observer.scoring import PC_LIGHT_TRAVEL_TIME_S, score_peeloff_events
from dsh.physics.cold_compton import SOURCE_PATH, build_cold_compton_table
from dsh.physics.compton import evaluate_compton, to_compton_physics
from dsh.physics.material_definitions import load_material_definition
from dsh.transport.inelastic import (
    COMPTON_SCATTERING,
    ENERGY_CUTOFF,
    InelasticInteractionRecord,
    InelasticTransportResult,
)
from dsh.transport.kernel import ABSORBED, DUST_SCATTERING, PHOTOELECTRIC_ABSORPTION
from tests.test_inelastic_transport import cloud, compton_table, dust_table, launch, run


def semantic_history():
    """Hand-built D,C,D,A history isolating the observer contract."""
    p = launch(1, energy=5.0, weight=2.0)
    e_cont = 5 / (1 + 5 / 510.99895069 * 0.5)
    positions = jnp.array(
        [
            [
                [5000.0, 100.0, 0.0],
                [4500.0, 0.0, 0.0],
                [4000.0, 0.0, 80.0],
                [3500.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ]
    )
    axes = np.array(
        [[-1.0, 0, 0], [0.0, 1, 0], [-np.sqrt(0.75), 0.5, 0], [-1.0, 0, 0], [0, 0, 0]]
    )
    energy = np.array([5.0, 5.0, e_cont, e_cont, 0.0])
    incoming = jnp.asarray(np.c_[energy, axes * energy[:, None]][None])
    outgoing = incoming.at[0, 0].set(jnp.array([5.0, 0.0, 5.0, 0.0]))
    outgoing = outgoing.at[0, 1].set(incoming[0, 2])
    outgoing = outgoing.at[0, 2].set(incoming[0, 3])
    outgoing = outgoing.at[0, 3].set(0.0)
    rec = InelasticInteractionRecord(
        jnp.array([[True, True, True, True, False]]),
        jnp.array(
            [
                [
                    DUST_SCATTERING,
                    COMPTON_SCATTERING,
                    DUST_SCATTERING,
                    PHOTOELECTRIC_ABSORPTION,
                    0,
                ]
            ],
            dtype=jnp.int32,
        ),
        positions,
        incoming,
        outgoing,
        jnp.array([[5001.0, 5502.0, 6003.0, 6504.0, 0.0]]),
        jnp.array([[1.0, 2.0, 3.0, 4.0, 0.0]]),
        jnp.array([[1, 2, 3, 3, 0]], dtype=jnp.int32),
        jnp.array([[1, 1, 2, 2, 0]], dtype=jnp.int32),
        jnp.array([[0, 1, 1, 1, 0]], dtype=jnp.int32),
        jnp.array([[0.0, 5 - e_cont, 0.0, 0.0, 0.0]]),
        jnp.array([[0.0, 5 - e_cont, 0.0, e_cont, 0.0]]),
    )
    r = InelasticTransportResult(
        positions[:, 3],
        jnp.zeros((1, 4)),
        jnp.array([6504.0]),
        jnp.array([4.0]),
        jnp.array([5.0]),
        jnp.array([4]),
        jnp.array([3]),
        jnp.array([ABSORBED]),
        rec,
        jnp.array([2]),
        jnp.array([1]),
        jnp.array([5 - e_cont]),
        jnp.array([e_cont]),
        jnp.array([5.0]),
        jnp.array([0.0]),
        jnp.array([0.0]),
        jnp.array([2.0]),
    )
    return p, r


class ObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p, cls.r = semantic_history()
        cls.scene, cls.dust, cls.compton = (
            cloud(2e23),
            dust_table(2e-25, 3e-25),
            compton_table(),
        )
        cls.scored = jax.jit(score_inelastic_peeloff)(
            cls.p, cls.r, cls.scene, cls.dust, cls.compton
        )

    def test_virtual_energy_differs_from_continuation_and_source(self):
        s = self.scored
        np.testing.assert_array_equal(
            s.events.valid, [[True, True, True, False, False]]
        )
        expected = 5 / (1 + 5 / 510.99895069)
        self.assertAlmostEqual(float(s.events.energy_kev[0, 1]), expected, delta=1e-6)
        self.assertGreater(
            abs(float(self.r.interactions.outgoing_momentum_kev[0, 1, 0]) - expected),
            0.01,
        )
        np.testing.assert_array_equal(s.source_energy_kev[0, :3], 5.0)
        self.assertAlmostEqual(
            float(s.events.energy_kev[0, 2]),
            float(self.r.interactions.incoming_momentum_kev[0, 2, 0]),
            delta=1e-6,
        )

    def test_classification_is_per_event_prefix(self):
        np.testing.assert_array_equal(
            self.scored.component, [[DUST_ONLY, MIXED, MIXED, 0, 0]]
        )
        changed = self.r._replace(n_dust=jnp.array([99]), n_compton=jnp.array([99]))
        actual = jax.jit(score_inelastic_peeloff)(
            self.p, changed, self.scene, self.dust, self.compton
        )
        np.testing.assert_array_equal(actual.component, self.scored.component)
        np.testing.assert_array_equal(
            actual.events.weight_observer_fluence,
            self.scored.events.weight_observer_fluence,
        )

    def test_missing_nonzero_dust_phase_is_explicitly_flagged(self):
        missing = self.dust._replace(
            differential_cross_section_cm2_per_sr_per_h=jnp.array([])
        )
        actual = jax.jit(score_inelastic_peeloff)(
            self.p, self.r, self.scene, missing, self.compton
        )
        np.testing.assert_array_equal(
            actual.unsupported_phase, [[True, False, True, False, False]]
        )
        np.testing.assert_array_equal(
            actual.events.valid, [[False, True, False, False, False]]
        )

    def test_channel_phase_without_double_opacity_factor(self):
        i = 1
        phase = float(evaluate_compton(self.compton, 5.0, 0.0).phase_pdf_per_sr)
        np.testing.assert_allclose(
            float(self.scored.events.phase_pdf_per_sr[0, i]), phase, rtol=2e-6
        )
        distance = 4500.0
        weight = (
            2
            * (10000 / distance) ** 2
            * phase
            * float(self.scored.events.transmission[0, i])
        )
        self.assertAlmostEqual(
            float(self.scored.events.weight_observer_fluence[0, i]),
            weight,
            delta=weight * 3e-6,
        )

    def test_escape_uses_observer_energy_for_all_three_opacities(self):
        grid = np.array([2.0, 10.0])
        dust = dust_table(
            2e-25 * (grid / 2) ** -2, 4e-25 * (grid / 2) ** -3, energies=grid
        )
        s = score_inelastic_peeloff(self.p, self.r, self.scene, dust, self.compton)
        e = 5 / (1 + 5 / 510.99895069)
        sigma_c = float(evaluate_compton(self.compton, e, 0.0).cross_section_cm2_per_h)
        sigma = 2e-25 * (e / 2) ** -2 + 4e-25 * (e / 2) ** -3 + sigma_c
        column = 0.375 * 2e23
        self.assertAlmostEqual(
            float(s.events.escape_column_cm2[0, 1]) / column, 1, delta=2e-6
        )
        self.assertAlmostEqual(
            float(s.events.escape_optical_depth[0, 1]) / (column * sigma), 1, delta=3e-6
        )

    def test_compton_extinction_also_attenuates_dust_scores(self):
        off = score_peeloff_events(self.p, self.r, self.scene, self.dust)
        on = self.scored.events
        for index in [0, 2]:
            energy = float(on.energy_kev[0, index])
            sigma = float(
                evaluate_compton(self.compton, energy, 0.0).cross_section_cm2_per_h
            )
            expected = np.exp(-float(on.escape_column_cm2[0, index]) * sigma)
            self.assertAlmostEqual(
                float(
                    on.weight_observer_fluence[0, index]
                    / off.weight_observer_fluence[0, index]
                ),
                expected,
                delta=3e-6,
            )

    def test_geometry_and_timing_remain_geometric(self):
        s = self.scored.events
        pos = np.asarray(self.r.interactions.position_pc)[0, :3]
        expected_x = np.arctan2(pos[:, 1], pos[:, 0]) * 206264.80624709636
        np.testing.assert_allclose(
            s.sky_x_arcsec[0, :3], expected_x, rtol=3e-6, atol=1e-6
        )
        r_minus_z = (pos[:, 1] ** 2 + pos[:, 2] ** 2) / (
            np.linalg.norm(pos, axis=1) + pos[:, 0]
        )
        expected_delay = (
            np.array([1.0, 2.0, 3.0]) + r_minus_z
        ) * PC_LIGHT_TRAVEL_TIME_S
        np.testing.assert_allclose(s.arrival_time_s[0, :3], expected_delay, rtol=2e-6)

    def test_absorption_does_not_cancel_earlier_observer_scores(self):
        self.assertEqual(int(self.r.status[0]), ABSORBED)
        self.assertTrue(
            np.all(np.asarray(self.scored.events.weight_observer_fluence)[0, :3] > 0)
        )
        np.testing.assert_array_equal(
            self.scored.events.weight_observer_fluence[0, 3:], 0.0
        )

    def test_last_supported_score_survives_continuation_cutoff(self):
        p, scene, d, c = (
            launch(128, transverse=0.1),
            cloud(2e24),
            dust_table(),
            compton_table(),
        )
        r = run(p, scene, d, c, cutoff=4.89, steps=6)
        s = jax.jit(score_inelastic_peeloff)(p, r, scene, d, c)
        selected = (np.asarray(r.status) == ENERGY_CUTOFF) & np.asarray(s.events.valid)[
            :, 0
        ]
        self.assertGreater(selected.sum(), 30)
        self.assertTrue(
            np.all(np.asarray(s.events.weight_observer_fluence)[selected, 0] > 0)
        )
        self.assertTrue(np.all(np.asarray(s.events.energy_kev)[selected, 0] > 4.89))
        self.assertTrue(np.all(np.asarray(r.momentum_kev)[selected, 0] < 4.89))
        np.testing.assert_array_equal(
            np.asarray(s.component)[selected, 0], COMPTON_ONLY
        )

    def test_unsupported_escape_is_reported_and_not_extrapolated(self):
        p, scene = launch(128, energy=2.0, transverse=0.1), cloud(2e24)
        d, c = (
            dust_table(energies=(2.0, 10.0)),
            compton_table(energies=(2.0, 4.9, 10.0)),
        )
        r = run(p, scene, d, c, cutoff=1.0, steps=3)
        s = jax.jit(score_inelastic_peeloff)(p, r, scene, d, c)
        self.assertGreater(int(s.unsupported_escape.sum()), 40)
        self.assertFalse(np.asarray(s.events.valid).any())
        self.assertTrue(
            np.all(
                np.asarray(s.unsupported_escape_energy_kev)[
                    np.asarray(s.unsupported_escape)
                ]
                < 2
            )
        )
        np.testing.assert_array_equal(s.events.weight_observer_fluence, 0.0)

    def test_disabled_observer_is_bitwise_equal(self):
        p, scene, d = launch(32, transverse=0.1), cloud(2e23), dust_table(1e-23, 1e-24)
        r = run(p, scene, d, None, steps=4)
        off = jax.jit(score_peeloff_events)(p, r, scene, d)
        new = jax.jit(score_inelastic_peeloff)(p, r, scene, d, None)
        for field in off._fields:
            np.testing.assert_array_equal(
                getattr(off, field), getattr(new.events, field)
            )

    def test_component_binning_uses_observer_energy(self):
        g = build_observer_bin_geometry(
            [-10000, 10000], [-10000, 10000], [4.8, 4.96, 4.99, 5.1], [0, 1e10]
        )
        b = jax.jit(bin_inelastic_events)(self.scored, g)
        np.testing.assert_array_equal(
            np.asarray(b.total.event_count).reshape(-1), [1, 1, 1]
        )
        np.testing.assert_allclose(
            b.total.total_fluence,
            b.dust_only.total_fluence
            + b.compton_only.total_fluence
            + b.mixed.total_fluence,
            rtol=2e-7,
        )

    def test_total_history_moments_keep_cross_component_terms(self):
        g = build_observer_bin_geometry(
            [-10000, 10000], [-10000, 10000], [4.0, 6.0], [0, 1e10]
        )
        b = bin_inelastic_events(self.scored, g)
        dw, mw = (
            float(b.dust_only.total_fluence.sum()),
            float(b.mixed.total_fluence.sum()),
        )
        total_q = float(b.total.total_fluence_squared.sum())
        separate_q = float(
            b.dust_only.total_fluence_squared.sum()
            + b.mixed.total_fluence_squared.sum()
        )
        self.assertAlmostEqual(total_q / (dw + mw) ** 2, 1, delta=5e-7)
        self.assertAlmostEqual((total_q - separate_q) / (2 * dw * mw), 1, delta=2e-6)

    def test_tiny_bound_angle_is_preserved_in_float32(self):
        m = load_material_definition(
            SOURCE_PATH.parent / "reflex_reference_definition.json"
        )
        c = to_compton_physics(
            build_cold_compton_table(m, [2, 4.9, 10], grain_electrons="omit").table
        )
        supported, phase, outgoing = jax.jit(compton_observer_phase_and_energy)(
            c, jnp.asarray(4.9, dtype=jnp.float32), jnp.asarray(1e-5, dtype=jnp.float32)
        )
        self.assertTrue(bool(supported))
        self.assertGreater(float(phase), 0.0)
        _, phase2, _ = compton_observer_phase_and_energy(
            c, jnp.asarray(4.9, dtype=jnp.float32), jnp.asarray(2e-5, dtype=jnp.float32)
        )
        self.assertAlmostEqual(float(phase2 / phase), 4, delta=1e-4)
        self.assertLessEqual(float(outgoing), 4.9 + 1e-6)

    def test_angle_evaluator_agrees_with_stage1_at_resolved_angles(self):
        angles = jnp.linspace(0.05, 3.0, 100)
        support, phase, _ = compton_observer_phase_and_energy(self.compton, 4.9, angles)
        reference = evaluate_compton(self.compton, 4.9, jnp.cos(angles))
        self.assertTrue(np.asarray(support).all())
        np.testing.assert_allclose(phase, reference.phase_pdf_per_sr, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
