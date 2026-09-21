"""Physics tests for next-event observer scoring of DSH scatterings."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from dsh.geometry.clouds import build_angular_distance_cloud
from dsh.geometry.coordinates import sky_position_pc
from dsh.observer.scoring import (
    PC_LIGHT_TRAVEL_TIME_S,
    scattering_phase_pdf_per_sr,
    score_peeloff_events,
)
from dsh.physics.absorption import load_photoelectric_absorption_table
from dsh.physics.dust import build_dust_physics_table
from dsh.physics.newdust import (
    build_dust_physics_from_tables,
    load_newdust_scattering_table,
)
from dsh.sources.launch import (
    LaunchedSourcePackets,
    build_cloud_launch_geometry,
    sample_source_launches,
)
from dsh.sources.models import SourcePackets
from dsh.transport.kernel import (
    ABSORBED,
    DUST_SCATTERING,
    NO_INTERACTION,
    PHOTOELECTRIC_ABSORPTION,
    PhotonInteractionRecord,
    PhotonTransportResult,
    transport_photon_batch,
)


class TestPeeloffObserver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scattering = load_newdust_scattering_table()
        cls.absorption = load_photoelectric_absorption_table()
        cls.physics = build_dust_physics_from_tables(cls.scattering, cls.absorption)
        radial_column = np.array([1.0e21, 2.0e21, 3.0e21])
        cls.cloud = build_angular_distance_cloud(
            np.broadcast_to(radial_column[:, None, None], (3, 3, 3)).copy(),
            x_centers_arcsec=[-500.0, 0.0, 500.0],
            y_centers_arcsec=[-500.0, 0.0, 500.0],
            z_centers_kpc=[1.0, 3.0, 5.0],
            source_distance_kpc=10.0,
        )
        cls.launched, cls.transported = cls._synthetic_history()

    @classmethod
    def _synthetic_history(cls):
        energy = 3.3
        source = np.array([10_000.0, 0.0, 0.0], dtype=np.float64)
        positions = np.array(
            [
                sky_position_pc(5.0, 50.0, -20.0),
                sky_position_pc(3.0, 80.0, 10.0),
                sky_position_pc(2.0, 30.0, 20.0),
            ],
            dtype=np.float64,
        )

        starts = np.vstack((source, positions[:-1]))
        displacements = positions - starts
        distances = np.linalg.norm(displacements, axis=1)
        directions = displacements / distances[:, None]
        cumulative_path = np.cumsum(distances)
        cumulative_excess = np.cumsum(distances * (1.0 + directions[:, 0]))
        incoming = np.column_stack((np.full(3, energy), energy * directions))
        outgoing = np.vstack((incoming[1], incoming[2], np.zeros(4)))

        max_interactions = 4
        valid = np.array([[True, True, True, False]])
        interaction_type = np.array(
            [
                [
                    DUST_SCATTERING,
                    DUST_SCATTERING,
                    PHOTOELECTRIC_ABSORPTION,
                    NO_INTERACTION,
                ]
            ],
            dtype=np.int32,
        )
        position_record = np.zeros((1, max_interactions, 3), dtype=np.float32)
        position_record[0, :3] = positions
        incoming_record = np.zeros((1, max_interactions, 4), dtype=np.float32)
        incoming_record[0, :3] = incoming
        outgoing_record = np.zeros((1, max_interactions, 4), dtype=np.float32)
        outgoing_record[0, :3] = outgoing
        cumulative_path_record = np.zeros((1, max_interactions), dtype=np.float32)
        cumulative_path_record[0, :3] = cumulative_path
        cumulative_excess_record = np.zeros((1, max_interactions), dtype=np.float32)
        cumulative_excess_record[0, :3] = cumulative_excess

        records = PhotonInteractionRecord(
            valid=jnp.asarray(valid),
            interaction_type=jnp.asarray(interaction_type),
            position_pc=jnp.asarray(position_record),
            incoming_momentum_kev=jnp.asarray(incoming_record),
            outgoing_momentum_kev=jnp.asarray(outgoing_record),
            cumulative_path_length_pc=jnp.asarray(cumulative_path_record),
            cumulative_excess_path_length_pc=jnp.asarray(cumulative_excess_record),
            scattering_order=jnp.asarray([[1, 2, 2, 0]], dtype=jnp.int32),
        )
        transported = PhotonTransportResult(
            position_pc=jnp.asarray(positions[-1][None, :], dtype=jnp.float32),
            momentum_kev=jnp.zeros((1, 4), dtype=jnp.float32),
            path_length_pc=jnp.asarray([cumulative_path[-1]], dtype=jnp.float32),
            excess_path_length_pc=jnp.asarray(
                [cumulative_excess[-1]], dtype=jnp.float32
            ),
            deposited_energy_kev=jnp.asarray([energy], dtype=jnp.float32),
            n_interactions=jnp.asarray([3], dtype=jnp.int32),
            n_scatter=jnp.asarray([2], dtype=jnp.int32),
            status=jnp.asarray([ABSORBED], dtype=jnp.int32),
            interactions=records,
        )

        launch_pdf = np.float32(2.0e5)
        launched = LaunchedSourcePackets(
            position_pc=jnp.asarray(source[None, :], dtype=jnp.float32),
            momentum_kev=jnp.asarray(incoming[0][None, :], dtype=jnp.float32),
            launch_pdf_per_sr=jnp.asarray([launch_pdf]),
            isotropic_importance=jnp.asarray(
                [1.0 / (4.0 * np.pi * launch_pdf)], dtype=jnp.float32
            ),
            emission_time_s=jnp.asarray([1234.0], dtype=jnp.float32),
            weight_observer_fluence=jnp.asarray([2.0], dtype=jnp.float32),
            time_index=jnp.asarray([7], dtype=jnp.int32),
            spectral_bin_index=jnp.asarray([0], dtype=jnp.int32),
        )
        return launched, transported

    def test_jitted_scorer_returns_exact_geometry_attenuation_and_weight(self):
        scorer = jax.jit(score_peeloff_events)
        result = scorer(self.launched, self.transported, self.cloud, self.physics)

        np.testing.assert_array_equal(result.valid, [[True, True, False, False]])
        np.testing.assert_allclose(
            np.asarray(result.sky_x_arcsec)[0, :2], [50.0, 80.0], atol=2.0e-5
        )
        np.testing.assert_allclose(
            np.asarray(result.sky_y_arcsec)[0, :2], [-20.0, 10.0], atol=2.0e-5
        )
        np.testing.assert_array_equal(
            np.asarray(result.scattering_order)[0], [1, 2, 0, 0]
        )
        np.testing.assert_allclose(
            np.asarray(result.escape_column_cm2)[0, :2],
            [4.5e21, 2.0e21],
            rtol=2.0e-6,
        )

        sigma_total = (
            self.scattering.scattering_cross_section_cm2_per_h[0]
            + self.absorption.absorption_cross_section_cm2_per_h[0]
        )
        expected_tau = np.array([4.5e21, 2.0e21]) * sigma_total
        np.testing.assert_allclose(
            np.asarray(result.escape_optical_depth)[0, :2],
            expected_tau,
            rtol=3.0e-6,
        )
        np.testing.assert_allclose(
            np.asarray(result.transmission)[0, :2],
            np.exp(-expected_tau),
            rtol=3.0e-6,
        )

        source = np.asarray(self.launched.position_pc)[0].astype(np.float64)
        event_positions = np.asarray(self.transported.interactions.position_pc)[
            0, :2
        ].astype(np.float64)
        incoming = np.asarray(self.transported.interactions.incoming_momentum_kev)[
            0, :2, 1:
        ].astype(np.float64)
        incoming /= np.linalg.norm(incoming, axis=1, keepdims=True)
        to_observer = -event_positions / np.linalg.norm(
            event_positions, axis=1, keepdims=True
        )
        expected_angles = np.arctan2(
            np.linalg.norm(np.cross(incoming, to_observer), axis=1),
            np.sum(incoming * to_observer, axis=1),
        )
        np.testing.assert_allclose(
            np.asarray(result.scattering_angle_rad)[0, :2],
            expected_angles,
            rtol=2.0e-6,
        )

        segment_starts = np.vstack((source, event_positions[:1]))
        cumulative_path = np.cumsum(
            np.linalg.norm(event_positions - segment_starts, axis=1)
        )
        exact_excess = (
            cumulative_path
            + np.linalg.norm(event_positions, axis=1)
            - np.linalg.norm(source)
        )
        np.testing.assert_allclose(
            np.asarray(result.excess_path_length_pc)[0, :2],
            exact_excess,
            rtol=3.0e-3,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            np.asarray(result.arrival_time_s)[0, :2],
            1234.0 + exact_excess * PC_LIGHT_TRAVEL_TIME_S,
            rtol=3.0e-3,
        )

        expected_phase = np.asarray(
            scattering_phase_pdf_per_sr(
                self.physics,
                np.full(2, 3.3),
                expected_angles,
            )
        )
        expected_weight = (
            2.0
            * (10_000.0 / np.linalg.norm(event_positions, axis=1)) ** 2
            * (1.0 / 2.0e5)
            * expected_phase
            * np.exp(-expected_tau)
        )
        np.testing.assert_allclose(
            np.asarray(result.phase_pdf_per_sr)[0, :2],
            expected_phase,
            rtol=3.0e-6,
        )
        np.testing.assert_allclose(
            np.asarray(result.weight_observer_fluence)[0, :2],
            expected_weight,
            rtol=5.0e-6,
        )
        np.testing.assert_array_equal(np.asarray(result.time_index)[0], [7, 7, 0, 0])
        np.testing.assert_array_equal(
            np.asarray(result.spectral_bin_index)[0], [0, 0, 0, 0]
        )

    def test_absorption_and_unused_slots_are_exactly_zero(self):
        result = score_peeloff_events(
            self.launched, self.transported, self.cloud, self.physics
        )
        for field in result._fields:
            if field == "valid":
                continue
            values = np.asarray(getattr(result, field))
            np.testing.assert_array_equal(values[0, 2:], 0)

    def test_actual_outgoing_direction_does_not_change_peeloff_score(self):
        changed_records = self.transported.interactions._replace(
            outgoing_momentum_kev=jnp.full_like(
                self.transported.interactions.outgoing_momentum_kev, 99.0
            )
        )
        changed_transport = self.transported._replace(interactions=changed_records)
        original = score_peeloff_events(
            self.launched, self.transported, self.cloud, self.physics
        )
        changed = score_peeloff_events(
            self.launched, changed_transport, self.cloud, self.physics
        )
        for field in original._fields:
            np.testing.assert_array_equal(
                np.asarray(getattr(original, field)),
                np.asarray(getattr(changed, field)),
            )

    def test_launched_transport_history_scores_end_to_end(self):
        n_packets = 512
        packets = SourcePackets(
            energy_kev=jnp.full((n_packets,), 3.3, dtype=jnp.float32),
            emission_time_s=jnp.linspace(0.0, 1000.0, n_packets),
            weight_observer_fluence=jnp.full(
                (n_packets,), 1.0 / n_packets, dtype=jnp.float32
            ),
            time_index=jnp.arange(n_packets, dtype=jnp.int32) % 4,
            spectral_bin_index=jnp.zeros(n_packets, dtype=jnp.int32),
        )
        geometry = build_cloud_launch_geometry(self.cloud)
        launched = jax.jit(sample_source_launches)(
            random.PRNGKey(51), packets, geometry
        )
        transport_jit = jax.jit(
            transport_photon_batch, static_argnames=("max_interactions",)
        )
        transported = transport_jit(
            random.PRNGKey(52),
            launched.position_pc,
            launched.momentum_kev,
            self.cloud,
            self.physics,
            max_interactions=4,
        )
        observed = jax.jit(score_peeloff_events)(
            launched, transported, self.cloud, self.physics
        )

        expected_valid = np.asarray(transported.interactions.valid) & (
            np.asarray(transported.interactions.interaction_type) == DUST_SCATTERING
        )
        valid = np.asarray(observed.valid)
        np.testing.assert_array_equal(valid, expected_valid)
        self.assertGreater(valid.sum(), 0)
        np.testing.assert_array_equal(
            np.asarray(observed.scattering_order)[valid],
            np.asarray(transported.interactions.scattering_order)[valid],
        )
        self.assertTrue(
            np.all(
                np.asarray(observed.arrival_time_s)[valid]
                >= np.broadcast_to(
                    np.asarray(launched.emission_time_s)[:, None], valid.shape
                )[valid]
            )
        )
        weights = np.asarray(observed.weight_observer_fluence)[valid]
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights >= 0.0))
        self.assertTrue(np.any(weights > 0.0))

    def test_phase_density_matches_intrinsic_table_nodes(self):
        angle_indices = np.array([0, 1, 31, 257, 1023, -1])
        angles = self.scattering.scattering_angle_rad[angle_indices]
        energies = self.scattering.energy_kev
        actual = np.asarray(
            scattering_phase_pdf_per_sr(
                self.physics,
                energies[:, None],
                angles[None, :],
            )
        )
        expected = (
            self.scattering.differential_cross_section_cm2_per_sr_per_h[
                :, angle_indices
            ]
            / self.scattering.scattering_cross_section_cm2_per_h[:, None]
        )
        np.testing.assert_allclose(actual, expected, rtol=3.0e-6)

    def test_phase_density_is_normalized_at_grid_and_intermediate_energies(self):
        angle = self.scattering.scattering_angle_rad
        energies = np.array([3.3, 4.0, 4.9, 5.8, 6.9])
        phase_pdf = np.asarray(
            scattering_phase_pdf_per_sr(
                self.physics,
                energies[:, None],
                angle[None, :],
            )
        )
        integrand = 2.0 * np.pi * np.sin(angle)[None, :] * phase_pdf
        integrated = np.sum(
            0.5 * (integrand[:, 1:] + integrand[:, :-1]) * np.diff(angle),
            axis=1,
        )
        np.testing.assert_allclose(integrated, 1.0, rtol=3.0e-6)

    def test_missing_differential_table_is_rejected(self):
        transport_only_physics = build_dust_physics_table(
            energy_kev=[3.3, 4.9],
            scattering_cross_section_cm2_per_h=[1.0e-23, 5.0e-24],
            absorption_cross_section_cm2_per_h=[2.0e-23, 1.0e-23],
            scattering_angle_rad=[0.0, np.pi],
            scattering_angle_cdf=[[0.0, 1.0], [0.0, 1.0]],
        )
        with self.assertRaisesRegex(ValueError, "differential cross-section"):
            score_peeloff_events(
                self.launched,
                self.transported,
                self.cloud,
                transport_only_physics,
            )

    def test_rejects_packet_and_interaction_shape_mismatch(self):
        bad_launch = self.launched._replace(weight_observer_fluence=jnp.ones(2))
        with self.assertRaisesRegex(ValueError, "metadata"):
            score_peeloff_events(bad_launch, self.transported, self.cloud, self.physics)


if __name__ == "__main__":
    unittest.main()
