"""Guard the numerical boundary and independently implemented references."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "dsh"


def imported_modules(path):
    relative = path.relative_to(PACKAGE.parent).with_suffix("")
    parts = relative.parts
    package = ".".join(parts[:-1])
    if parts[-1] == "__init__":
        package = ".".join(parts[:-1])
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (name.name for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
            yield importlib.util.resolve_name(name, package) if node.level else name


class PackageBoundaryTests(unittest.TestCase):
    def test_core_imports_only_numerical_dependencies(self):
        allowed = {"__future__", "collections", "jax", "numpy"}
        for path in (PACKAGE / "core").rglob("*.py"):
            for module in imported_modules(path):
                with self.subTest(path=path.relative_to(PACKAGE), module=module):
                    self.assertTrue(
                        module == "dsh.contracts"
                        or module == "dsh.core"
                        or module.startswith("dsh.core.")
                        or module.split(".")[0] in allowed,
                        "core must not load files, configuration, models, or products",
                    )

    def test_contracts_do_not_import_a_subsystem(self):
        for module in imported_modules(PACKAGE / "contracts.py"):
            self.assertFalse(module == "dsh" or module.startswith("dsh."))

    def test_production_does_not_import_validation(self):
        for path in PACKAGE.rglob("*.py"):
            if "validation" in path.relative_to(PACKAGE).parts:
                continue
            for module in imported_modules(path):
                with self.subTest(path=path.relative_to(PACKAGE), module=module):
                    self.assertFalse(
                        module == "dsh.validation"
                        or module.startswith("dsh.validation.")
                    )

    def test_stage9f_references_do_not_import_production_rays_or_scorer(self):
        # The validation harness may exercise core; these reference algorithms
        # must keep their independent host integration and absorption scorer.
        forbidden = {
            "dsh.core.geometry.rays",
            "dsh.geometry.rays",
            "dsh.core.observer.scoring",
            "dsh.observer.scoring",
            "dsh.core.transport.kernel",
            "dsh.transport.kernel",
        }
        for name in ("absorbed_observer", "heterogeneous_observer"):
            path = PACKAGE / "validation" / f"{name}.py"
            for module in imported_modules(path):
                with self.subTest(reference=name, module=module):
                    self.assertNotIn(module, forbidden)

    def test_configuration_parsing_does_not_import_execution(self):
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json,sys; import dsh.config; "
                "print(json.dumps(sorted(n for n in sys.modules if n.startswith('dsh'))))",
            ],
            cwd=PACKAGE.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        modules = json.loads(process.stdout)
        self.assertIn("dsh.config.load", modules)
        for name in ("dsh.core", "dsh.build", "dsh.run", "dsh.io", "dsh.materials"):
            self.assertFalse(
                any(n == name or n.startswith(name + ".") for n in modules)
            )

    def test_legacy_imports_share_canonical_module_identity(self):
        # Module identity also preserves assignments made by existing mocks.
        pairs = (
            ("dsh.geometry.rays", "dsh.core.geometry.rays"),
            ("dsh.transport.kernel", "dsh.core.transport.kernel"),
            ("dsh.observer.scoring", "dsh.core.observer.scoring"),
            ("dsh.pipeline", "dsh.core.pipeline"),
            ("dsh.physics.materials", "dsh.materials.registry"),
            ("dsh.sources.source_fits", "dsh.sources.format"),
            ("scripts.extract_configured_line_snapshots", "dsh.products.snapshots"),
        )
        for old, new in pairs:
            with self.subTest(old=old, new=new):
                self.assertIs(
                    importlib.import_module(old), importlib.import_module(new)
                )


if __name__ == "__main__":
    unittest.main()
