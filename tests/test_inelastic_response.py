"""Initial-energy reweighting, outgoing-energy bins, covariance and archive v2."""

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dsh.observer.inelastic_response import InelasticSpectralResponse
from dsh.observer.response import (
    fold_spectral_response,
    load_spectral_response,
    write_spectral_response,
)
from dsh.sources.histories import build_spectral_history_source


def response():
    # Free-electron high-energy redistribution control, independent of the
    # 2--10 keV cold provider. History zero scores D then C (mixed); one scores C.
    return InelasticSpectralResponse(
        energy_kev=np.array([500.0, 250.0, 250.0]),
        source_energy_kev=np.full(3, 500.0),
        incident_energy_kev=np.full(3, 500.0),
        delay_s=np.zeros(3),
        sky_x_arcsec=np.full(3, 0.5),
        sky_y_arcsec=np.zeros(3),
        scattering_order=np.array([1, 2, 1]),
        dust_order=np.array([1, 1, 0]),
        compton_order=np.array([0, 1, 1]),
        interaction_type=np.array([1, 3, 3]),
        history_index=np.array([0, 0, 1]),
        weight_kev=np.ones(3),
        n_histories=4,
        energy_support_kev=(400.0, 600.0),
        provenance=dict.fromkeys(
            ("material_sha256", "cloud_sha256", "launch_sha256", "runtime_sha256"),
            "a" * 64,
        ),
    )


class InelasticResponseTests(unittest.TestCase):
    def setUp(self):
        self.response = response()
        self.source = build_spectral_history_source([0, 1], [400, 600], [[200]], 2.0)

    def fold(self, **kwargs):
        return fold_spectral_response(
            self.response, self.source, [0, 1], [200, 300, 600], [0, 1], **kwargs
        )

    def test_initial_energy_density_and_outgoing_energy_bins_are_distinct(self):
        result = self.fold()
        # Integral E^-2 from 400 to 600 is 1/1200; density at 500 is .96.
        np.testing.assert_allclose(result.fluence, [[[1.92], [0.96]]], rtol=2e-6)

    def test_history_covariance_includes_mixed_prefix_cross_terms(self):
        result = self.fold()
        grouped = np.array([[0.96, 0.96], [0.96, 0], [0, 0], [0, 0]])
        total = grouped.sum(axis=0)
        expected = 4 / 3 * (grouped.T @ grouped - np.outer(total, total) / 4)
        np.testing.assert_allclose(result.covariance, expected, rtol=3e-6)

    def test_component_filters_partition_fluence(self):
        parts = [self.fold(component=value, covariance=False) for value in (1, 2, 3)]
        np.testing.assert_allclose(
            sum(part.fluence for part in parts), self.fold().fluence, rtol=1e-14
        )
        for value in (0, 4, True, "mixed"):
            with self.assertRaisesRegex(ValueError, "component"):
                self.fold(component=value)

    def test_delays_use_emission_time_intervals(self):
        self.response = dataclasses.replace(self.response, delay_s=np.full(3, 0.5))
        np.testing.assert_allclose(self.fold().fluence, [[[0.96], [0.48]]], rtol=2e-6)

    def test_order_selection_still_applies_to_combined_orders(self):
        np.testing.assert_allclose(
            self.fold(scattering_orders=[2]).fluence, [[[0.96], [0]]], rtol=2e-6
        )

    def test_schema_two_roundtrip_includes_all_energy_and_prefix_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response.npz"
            write_spectral_response(path, self.response)
            actual = load_spectral_response(path)
            self.assertIsInstance(actual, InelasticSpectralResponse)
            for name in self.response.__dataclass_fields__:
                if name not in ("provenance", "energy_support_kev"):
                    np.testing.assert_array_equal(
                        getattr(actual, name), getattr(self.response, name)
                    )
            with np.load(path) as data:
                self.assertEqual(
                    json.loads(str(data["metadata_json"]))["schema_version"], 2
                )
                payload = {key: data[key] for key in data.files}
            payload["source_energy_kev"] = np.full(3, 510.0)
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_spectral_response(path)

    def test_invalid_prefixes_upscatter_and_initial_support_are_rejected(self):
        for change in (
            {"dust_order": np.array([1, 0, 0])},
            {"energy_kev": np.full(3, 501.0)},
            {"incident_energy_kev": np.full(3, 501.0)},
            {"source_energy_kev": np.full(3, 300.0)},
            {"compton_order": np.array([0.0, 1.0, 1.0])},
            {"interaction_type": np.array([1, 2, 3])},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                dataclasses.replace(self.response, **change)

    def test_empty_response_remains_valid_and_retains_zero_score_histories(self):
        changes = {
            name: np.asarray(getattr(self.response, name))[:0]
            for name in self.response.__dataclass_fields__
            if isinstance(getattr(self.response, name), np.ndarray)
        }
        self.response = dataclasses.replace(self.response, **changes)
        result = self.fold()
        np.testing.assert_array_equal(result.fluence, 0)
        np.testing.assert_array_equal(result.covariance, 0)
        self.assertEqual(result.n_histories, 4)
