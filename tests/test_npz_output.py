"""Round-trip test for the complete reproducibility NPZ payload."""

import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from dsh.geometry.clouds import build_angular_distance_cloud
from dsh.io.npz_output import write_ideal_observer_npz
from dsh.observer.binning import (
    bin_observer_events,
    build_observer_bin_geometry,
)
from dsh.observer.scoring import ObserverEventResult
from dsh.physics.absorption import load_photoelectric_absorption_table
from dsh.physics.newdust import (
    build_dust_physics_from_tables,
    load_newdust_scattering_table,
)
from dsh.pipeline import IdealObserverDiagnostics, IdealObserverSimulationResult
from dsh.sources.launch import build_cloud_launch_geometry
from dsh.sources.models import build_powerlaw_band_source, build_tabulated_band_source


class TestIdealObserverNpz(unittest.TestCase):
    def test_complete_npz_product_round_trip(self):
        scattering = load_newdust_scattering_table()
        physics = build_dust_physics_from_tables(
            scattering, load_photoelectric_absorption_table()
        )
        cloud = build_angular_distance_cloud(
            np.full((2, 2, 2), 1.0e20),
            [-5.0, 5.0],
            [-5.0, 5.0],
            [2.0, 4.0],
            10.0,
        )
        launch_geometry = build_cloud_launch_geometry(cloud)
        source = build_tabulated_band_source(
            [0.0, 100.0],
            [[2.0e-2, 1.2e-2, 6.0e-3]],
            scattering.energy_kev,
        )
        bin_geometry = build_observer_bin_geometry(
            [-10.0, 0.0, 10.0],
            [-10.0, 0.0, 10.0],
            [2.5, 4.1, 5.9, 7.9],
            [0.0, 100.0, 200.0],
        )
        shape = (1, 1)
        zeros = jnp.zeros(shape, dtype=jnp.float32)
        events = ObserverEventResult(
            valid=jnp.ones(shape, dtype=bool),
            sky_x_arcsec=jnp.asarray([[-5.0]], dtype=jnp.float32),
            sky_y_arcsec=jnp.asarray([[5.0]], dtype=jnp.float32),
            energy_kev=jnp.asarray([[3.3]], dtype=jnp.float32),
            arrival_time_s=jnp.asarray([[50.0]], dtype=jnp.float32),
            excess_path_length_pc=zeros,
            scattering_angle_rad=zeros,
            scattering_order=jnp.ones(shape, dtype=jnp.int32),
            escape_column_cm2=zeros,
            escape_optical_depth=zeros,
            transmission=jnp.ones(shape, dtype=jnp.float32),
            phase_pdf_per_sr=zeros,
            weight_observer_fluence=jnp.asarray([[2.0]], dtype=jnp.float32),
            time_index=jnp.zeros(shape, dtype=jnp.int32),
            spectral_bin_index=jnp.zeros(shape, dtype=jnp.int32),
        )
        diagnostics = IdealObserverDiagnostics(
            source_packet_count=jnp.asarray(1, dtype=jnp.int32),
            source_fluence=jnp.asarray(3.8, dtype=jnp.float32),
            transport_status_count=jnp.asarray([0, 1, 0, 0, 0, 0, 0], dtype=jnp.int32),
            analog_interaction_count=jnp.asarray(1, dtype=jnp.int32),
            analog_scattering_count=jnp.asarray(1, dtype=jnp.int32),
            scored_observer_event_count=jnp.asarray(1, dtype=jnp.int32),
            scored_observer_fluence=jnp.asarray(2.0, dtype=jnp.float32),
        )
        result = IdealObserverSimulationResult(
            bin_observer_events(events, bin_geometry), diagnostics
        )
        metadata = {
            "material_tables": "v1",
            "scattering_table_sha256": scattering.metadata["table_sha256"],
            "absorption_table_sha256": (
                load_photoelectric_absorption_table().metadata["table_sha256"]
            ),
            "packets": 1,
            "chunk_size": 1,
            "max_interactions": 8,
            "seed": 12,
            "cloud_description": "test cloud",
            "source_model": "constant-flare",
            "peak_band_fluxes": [0.020, 0.012, 0.006],
            "baseline_band_fluxes": [0.0, 0.0, 0.0],
            "decay_time_days": None,
            "decay_start_days": None,
            "decay_duration_days": None,
            "source_time_bin_days": None,
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complete.npz"
            returned = write_ideal_observer_npz(
                path,
                result,
                bin_geometry,
                source,
                cloud,
                physics,
                launch_geometry,
                run_metadata=metadata,
            )

            self.assertEqual(returned, path)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(int(archive["output_schema_version"]), 5)
                self.assertEqual(str(archive["material_tables"]), "v1")
                self.assertEqual(
                    str(archive["scattering_table_sha256"]),
                    metadata["scattering_table_sha256"],
                )
                self.assertEqual(
                    str(archive["absorption_table_sha256"]),
                    metadata["absorption_table_sha256"],
                )
                self.assertEqual(int(archive["requested_packet_count"]), 1)
                self.assertEqual(str(archive["source_model"]), "constant-flare")
                self.assertTrue(np.isnan(archive["source_decay_time_days"]))
                self.assertAlmostEqual(float(archive["total_fluence"].sum()), 2.0)
                np.testing.assert_allclose(
                    archive["physics_energy_kev"],
                    scattering.energy_kev,
                    rtol=1.0e-6,
                )

            powerlaw = build_powerlaw_band_source(
                [0.0, 100.0], [0.038], [2, 4, 6, 10], 1.7
            )
            powerlaw_path = Path(directory) / "powerlaw.npz"
            write_ideal_observer_npz(
                powerlaw_path,
                result,
                bin_geometry,
                powerlaw,
                cloud,
                physics,
                launch_geometry,
                run_metadata={
                    **metadata,
                    "source_spectrum": "hard-state-powerlaw",
                    "peak_band_fluxes": np.asarray(powerlaw.band_flux[0]),
                },
            )
            with np.load(powerlaw_path, allow_pickle=False) as archive:
                np.testing.assert_allclose(
                    archive["source_energy_edges_kev"], [2.0, 4.0, 6.0, 10.0]
                )
                self.assertAlmostEqual(float(archive["source_photon_index"]), 1.7)
                self.assertEqual(str(archive["source_spectrum"]), "hard-state-powerlaw")
                self.assertAlmostEqual(float(archive["source_total_fluence"]), 3.8)


if __name__ == "__main__":
    unittest.main()
