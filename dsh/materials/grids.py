"""Build a shared 2–10 keV table axis from an absorption evaluator.

The transport kernel interpolates positive cross sections in log(E), log(sigma).
Midpoint refinement therefore checks that same interpolation. A true
absorption edge cannot be interpolated continuously; intervals around edges
are narrowed to the stated energy resolution and reported explicitly.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

ANCHOR_ENERGIES_KEV = (3.3, 4.9, 6.9)


def load_material_grid(path: str | Path) -> np.ndarray:
    """Read the same generated grid in both independent table generators."""

    with Path(path).open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported material energy-grid schema")
    energy = np.asarray(metadata.get("energy_kev", []), dtype=np.float64)
    if (
        energy.ndim != 1
        or energy.size < 2
        or not np.all(np.isfinite(energy))
        or energy[0] != 2.0
        or energy[-1] != 10.0
        or np.any(np.diff(energy) <= 0.0)
    ):
        raise ValueError("material energy grid must increase from 2 to 10 keV")
    return energy


def initial_material_grid(step_kev: float = 0.1) -> np.ndarray:
    """Seed a regular scan, including the three existing validation energies."""

    if not np.isfinite(step_kev) or step_kev <= 0.0 or 8.0 / step_kev > 2048:
        raise ValueError("step_kev must be finite and between 8/2048 and 8")
    count = int(np.ceil(8.0 / step_kev))
    return np.unique(np.r_[np.linspace(2.0, 10.0, count + 1), ANCHOR_ENERGIES_KEV])


def refine_material_grid(
    evaluate: Callable[[float], float],
    *,
    initial_energy_kev=None,
    relative_tolerance: float = 5e-3,
    minimum_interval_kev: float = 1e-4,
    maximum_nodes: int = 2048,
):
    """Return (energy, narrow edge intervals, max smooth midpoint error).

    Intervals with interpolation error above tolerance are split unless they
    are already narrower than minimum_interval_kev; such intervals are marked
    as edge intervals, where interpolation remains approximate. The evaluator
    must return a positive finite intrinsic absorption cross section.
    """

    energy = np.asarray(
        initial_material_grid() if initial_energy_kev is None else initial_energy_kev,
        dtype=np.float64,
    )
    if (
        energy.ndim != 1
        or energy.size < 2
        or not np.all(np.isfinite(energy))
        or energy[0] != 2.0
        or energy[-1] != 10.0
        or np.any(np.diff(energy) <= 0.0)
        or not np.isfinite(relative_tolerance)
        or not 0.0 < relative_tolerance < 1.0
        or not np.isfinite(minimum_interval_kev)
        or minimum_interval_kev <= 0.0
        or maximum_nodes < energy.size
    ):
        raise ValueError("invalid initial grid or refinement limits")

    cache: dict[float, float] = {}

    def checked(energy_kev: float) -> float:
        if energy_kev not in cache:
            value = float(evaluate(energy_kev))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "absorption evaluator must return finite positive values"
                )
            cache[energy_kev] = value
        return cache[energy_kev]

    for value in energy:
        checked(float(value))
    intervals = [
        (float(lo), float(hi)) for lo, hi in zip(energy[:-1], energy[1:], strict=True)
    ]
    accepted = []
    unresolved = []
    max_smooth_error = 0.0
    while intervals:
        lo, hi = intervals.pop()
        middle = float(np.sqrt(lo * hi))
        predicted = np.sqrt(checked(lo) * checked(hi))
        actual = checked(middle)
        error = abs(predicted / actual - 1.0)
        if error <= relative_tolerance:
            accepted.append((lo, hi))
            max_smooth_error = max(max_smooth_error, error)
        elif hi - lo <= minimum_interval_kev:
            accepted.append((lo, hi))
            unresolved.append((lo, hi, error))
        else:
            if len(accepted) + len(intervals) + 3 > maximum_nodes:
                raise RuntimeError("energy refinement exceeded maximum_nodes")
            intervals.extend([(middle, hi), (lo, middle)])

    nodes = np.unique(np.array([x for interval in accepted for x in interval]))
    if nodes.size > maximum_nodes:
        raise RuntimeError("energy refinement exceeded maximum_nodes")
    return nodes, sorted(unresolved), max_smooth_error
