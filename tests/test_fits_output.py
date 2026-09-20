"""Round-trip tests for complete ideal-observer FITS products."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import jax.numpy as jnp
import numpy as np

from utils.absorption import load_photoelectric_absorption_table
from utils.clouds import build_angular_distance_cloud
from utils.fits_output import write_ideal_observer_fits
from utils.newdust import (
    build_dust_physics_from_tables,
    load_newdust_scattering_table,
)
from utils.observer import ObserverEventResult
from utils.observer_binning import (
    bin_observer_events,
    build_observer_bin_geometry,
)
from utils.simulation import (
    IdealObserverDiagnostics,
    IdealObserverSimulationResult,
)
from utils.source import build_tabulated_band_source
from utils.source_launch import build_cloud_launch_geometry
from utils.voxel_transport import REACHED_OBSERVER_PLANE


@unittest.skipUnless(
    importlib.util.find_spec("astropy") is not None,
    "Astropy is optional outside FITS-output runs",
)
class TestIdealObserverFits(unittest.TestCase):
    def test_complete_fits_product_round_trip(self):
        from astropy.io import fits

        scattering = load_newdust_scattering_table()
        absorption = load_photoelectric_absorption_table()
        physics = build_dust_physics_from_tables(scattering, absorption)
        cloud = build_angular_distance_cloud(
            np.full((2, 2, 2), 1.0e20),
            x_centers_arcsec=[-5.0, 5.0],
            y_centers_arcsec=[-5.0, 5.0],
            z_centers_kpc=[2.0, 4.0],
            source_distance_kpc=10.0,
        )
        launch_geometry = build_cloud_launch_geometry(cloud)
        source = build_tabulated_band_source(
            time_edges_s=[0.0, 100.0],
            band_flux=[[2.0e-2, 1.2e-2, 6.0e-3]],
            effective_energy_kev=scattering.energy_kev,
        )
        bin_geometry = build_observer_bin_geometry(
            sky_x_edges_arcsec=[-10.0, 0.0, 10.0],
            sky_y_edges_arcsec=[-10.0, 0.0, 10.0],
            energy_edges_kev=[2.5, 4.1, 5.9, 7.9],
            arrival_time_edges_s=[0.0, 100.0, 200.0],
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
        products = bin_observer_events(events, bin_geometry)
        diagnostics = IdealObserverDiagnostics(
            source_packet_count=jnp.asarray(1, dtype=jnp.int32),
            source_fluence=jnp.asarray(3.8, dtype=jnp.float32),
            transport_status_count=jnp.asarray(
                [0, 1, 0, 0, 0, 0, 0], dtype=jnp.int32
            ),
            analog_interaction_count=jnp.asarray(1, dtype=jnp.int32),
            analog_scattering_count=jnp.asarray(1, dtype=jnp.int32),
            scored_observer_event_count=jnp.asarray(1, dtype=jnp.int32),
            scored_observer_fluence=jnp.asarray(2.0, dtype=jnp.float32),
        )
        result = IdealObserverSimulationResult(products, diagnostics)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complete.fits"
            returned = write_ideal_observer_fits(
                path,
                result,
                bin_geometry,
                source,
                cloud,
                physics,
                launch_geometry,
                run_metadata={
                    "packets": 1,
                    "chunk_size": 1,
                    "max_interactions": 8,
                    "seed": 12,
                    "cloud_description": "test cloud",
                },
            )
            self.assertEqual(returned, path)
            with fits.open(path, checksum=True) as hdul:
                hdul.verify("exception")
                expected_extensions = {
                    "TOTAL4D",
                    "FIRST4D",
                    "MULTI4D",
                    "EVENT4D",
                    "SURFBRIT",
                    "CLOUDNH",
                    "CLOUDDEN",
                    "SOURCE",
                    "PHYSICS",
                    "SCATCDF",
                    "DSIGMA",
                    "DIAGNOSTICS",
                    "STATUS",
                }
                self.assertTrue(
                    expected_extensions.issubset(
                        {hdu.name for hdu in hdul}
                    )
                )
                self.assertEqual(hdul[0].data.shape, (2, 2))
                self.assertEqual(hdul["TOTAL4D"].data.shape, (2, 3, 2, 2))
                self.assertAlmostEqual(float(hdul[0].data.sum()), 2.0)
                self.assertAlmostEqual(
                    float(hdul["TOTAL4D"].data.sum()), 2.0
                )
                self.assertEqual(len(hdul["SOURCE"].data), 3)
                self.assertEqual(
                    int(hdul["STATUS"].data["COUNT"].sum()), 1
                )
                self.assertEqual(hdul[0].header["NPACKETS"], 1)
                self.assertEqual(
                    hdul["STATUS"].data["STATUS_CODE"][1],
                    REACHED_OBSERVER_PLANE,
                )
                self.assertEqual(
                    hdul["DSIGMA"].data.shape,
                    np.asarray(
                        physics.differential_cross_section_cm2_per_sr_per_h
                    ).shape,
                )


if __name__ == "__main__":
    unittest.main()
