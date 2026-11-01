"""Active portable variant of the frozen material-comparison regression.

The original module is benchmark evidence protected by its published SHA256.
Inherit all its cases, changing only the derived-inventory equality check.
"""

import copy
import math
import tempfile
from pathlib import Path

from scripts.compare_material_models import load_case_material
from scripts.compare_wd01_reflex import material_diagnostics
from tests import test_material_model_comparison as frozen


def assert_same_element_inventory(case, actual, expected):
    """Keep abundance budgets/fields exact; allow <=32 ulps of their budget.

    Gas and solid values are derived independently on each platform. Scale
    their absolute allowance to the element's total, including near depletion.
    This matches the existing material audit's own 32-ulp closure policy.
    """
    case.assertEqual(actual.keys(), expected.keys())
    for element, target in expected.items():
        row = actual[element]
        case.assertEqual(row.keys(), target.keys(), element)
        total = target["total_atoms_per_h"]
        case.assertEqual(row["total_atoms_per_h"], total, element)
        for field in ("grain_atoms_per_h", "gas_atoms_per_h"):
            case.assertTrue(math.isfinite(row[field]), (element, field))
            case.assertAlmostEqual(
                row[field],
                target[field],
                delta=32 * math.ulp(total),
                msg=f"{element}.{field}",
            )
        if target["gas_fraction"] is None:
            case.assertIsNone(row["gas_fraction"], element)
        else:
            case.assertTrue(math.isfinite(row["gas_fraction"]), element)
            case.assertAlmostEqual(
                row["gas_fraction"],
                target["gas_fraction"],
                delta=32 * math.ulp(1.0),
                msg=f"{element}.gas_fraction",
            )


class TestPortableMaterialModelComparison(frozen.TestMaterialModelComparison):
    def test_full_mie_composition_preserves_hybrid_inventory_and_total_opacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            full, full_info = load_case_material("wd01_mie", Path(temporary))
        hybrid, hybrid_info = load_case_material("wd01_hybrid", Path("."))
        assert_same_element_inventory(
            self, full_info["audit"]["elements"], hybrid_info["audit"]["elements"]
        )
        first = material_diagnostics(full, self.config)
        second = material_diagnostics(hybrid, self.config)
        for name in ("scattering_cm2_per_h", "absorption_cm2_per_h"):
            self.assertAlmostEqual(first[name] / second[name], 1, places=7)
        self.assertNotEqual(
            full_info["definition_sha256"], hybrid_info["definition_sha256"]
        )

    def test_inventory_comparison_accepts_roundoff_but_rejects_changed_material(self):
        expected = {
            "C": {
                "total_atoms_per_h": 2.4e-4,
                "grain_atoms_per_h": 0.00012000000000000003,
                "gas_atoms_per_h": 1.2e-4,
                "gas_fraction": 0.5,
            }
        }
        actual = copy.deepcopy(expected)
        actual["C"]["grain_atoms_per_h"] = 0.00011999999999999999
        assert_same_element_inventory(self, actual, expected)
        for field, value in (
            ("total_atoms_per_h", math.nextafter(2.4e-4, math.inf)),
            ("grain_atoms_per_h", 1.2e-4 * (1 + 1e-10)),
            ("gas_atoms_per_h", 1.2e-4 * (1 + 1e-10)),
            ("gas_fraction", 0.5 + 1e-10),
            ("grain_atoms_per_h", math.nan),
        ):
            changed = copy.deepcopy(expected)
            changed["C"][field] = value
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(AssertionError),
            ):
                assert_same_element_inventory(self, changed, expected)
        changed = copy.deepcopy(expected)
        changed["C"]["unexpected"] = 0.0
        with self.assertRaises(AssertionError):
            assert_same_element_inventory(self, changed, expected)
