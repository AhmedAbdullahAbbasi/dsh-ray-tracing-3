"""Input boundary for independently versioned scattering and absorption tables."""

from .inputs import MaterialInputs, load_material_inputs, packaged_material_paths
from .scattering import (
    ScatteringTable,
    build_material_from_tables,
    load_scattering_table,
)

__all__ = [
    "MaterialInputs",
    "load_material_inputs",
    "packaged_material_paths",
    "ScatteringTable",
    "load_scattering_table",
    "build_material_from_tables",
]
