"""Material inputs must preserve the pinned numerical payloads."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dsh.materials import load_material_inputs, packaged_material_paths
from dsh.physics.materials import load_2_10_material_tables


class MaterialInputTests(unittest.TestCase):
    def test_packaged_pair_is_numerically_identical_to_legacy_loader(self):
        scattering_path, absorption_path, grid_path = packaged_material_paths("2-10")
        new = load_material_inputs(
            scattering_path, absorption_path, grid_path=grid_path
        )
        old_scattering, old_absorption, old_physics = load_2_10_material_tables()
        for name in old_physics._fields:
            np.testing.assert_array_equal(
                np.asarray(getattr(new.physics, name)),
                np.asarray(getattr(old_physics, name)),
            )
        np.testing.assert_array_equal(
            new.scattering.scattering_cross_section_cm2_per_h,
            old_scattering.scattering_cross_section_cm2_per_h,
        )
        np.testing.assert_array_equal(
            new.absorption.absorption_cross_section_cm2_per_h,
            old_absorption.absorption_cross_section_cm2_per_h,
        )

    def test_absorption_can_be_replaced_without_changing_scattering(self):
        scattering_path, absorption_path, _ = packaged_material_paths("v1")
        original = load_material_inputs(scattering_path, absorption_path)
        with tempfile.TemporaryDirectory() as directory:
            replacement = Path(directory) / "alternate_absorption.npz"
            with np.load(absorption_path) as archive:
                energy = archive["energy_kev"]
                sigma = archive["absorption_cross_section_cm2_per_h"] * 1.1
            np.savez(
                replacement,
                energy_kev=energy,
                absorption_cross_section_cm2_per_h=sigma,
            )
            metadata = json.loads(absorption_path.with_suffix(".json").read_text())
            metadata["table_sha256"] = hashlib.sha256(
                replacement.read_bytes()
            ).hexdigest()
            replacement.with_suffix(".json").write_text(json.dumps(metadata))
            modified = load_material_inputs(scattering_path, replacement)
            np.testing.assert_array_equal(
                modified.scattering.scattering_cross_section_cm2_per_h,
                original.scattering.scattering_cross_section_cm2_per_h,
            )
            np.testing.assert_allclose(
                modified.absorption.absorption_cross_section_cm2_per_h,
                original.absorption.absorption_cross_section_cm2_per_h * 1.1,
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
