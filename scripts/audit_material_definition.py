"""Audit an explicit chemical material definition, without running transport.

Usage: python -m scripts.audit_material_definition definition.json [--output audit.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dsh.physics.material_definitions import load_material_definition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("definition", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    definition = load_material_definition(args.definition)
    report = definition.audit()
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
