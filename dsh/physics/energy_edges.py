"""One-sided material edges and spectral quadrature split at discontinuities."""

from __future__ import annotations

import numpy as np


def edge_intervals(energy_kev, edge_energies_kev, *, maximum_relative_gap=1e-4):
    """Mark narrowly bracketed edges without blending their two sides.

    Each edge occupies one interval with separate left/right samples. Within
    that tiny interval the corresponding endpoint is held on its own side;
    other intervals retain the established interpolation. The edge itself
    takes the high-energy value. This resolves the jump exactly at the stored
    threshold, with a bounded energy offset in the one-sided samples.
    """
    energy = np.asarray(energy_kev, dtype=float)
    edges = np.asarray(edge_energies_kev, dtype=float)
    if (
        energy.ndim != 1
        or energy.size < 2
        or np.any(~np.isfinite(energy))
        or np.any(energy <= 0)
        or np.any(np.diff(energy) <= 0)
        or edges.ndim != 1
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0)
        or not 0 < maximum_relative_gap <= 1e-3
    ):
        raise ValueError("invalid material energy grid or edge list")
    interval = np.zeros(energy.size - 1)
    for edge in edges:
        upper = int(np.searchsorted(energy, edge, side="right"))
        if (
            upper == 0
            or upper == energy.size
            or energy[upper - 1] == edge
            or interval[upper - 1] != 0
        ):
            raise ValueError("each edge needs its own strict left/right energy bracket")
        if (energy[upper] - energy[upper - 1]) / edge > maximum_relative_gap:
            raise ValueError(
                "edge bracket is too wide for one-sided endpoint interpolation"
            )
        interval[upper - 1] = edge
    # The production default is float32: distinct nodes and the threshold's
    # position within its bracket must survive conversion to that dtype.
    single = energy.astype(np.float32)
    if np.any(np.diff(single) <= 0):
        raise ValueError("material energy nodes collapse in float32")
    for index in np.flatnonzero(interval):
        threshold = np.float32(interval[index])
        if not single[index] < threshold < single[index + 1]:
            raise ValueError("edge bracket collapses in float32")
    return interval


def bracketed_edge_grid(energy_kev, edge_energies_kev, *, relative_half_width=2e-6):
    """Insert float32-resolved samples on each side of every declared edge."""
    energy = np.asarray(energy_kev, dtype=float)
    edges = np.asarray(edge_energies_kev, dtype=float)
    if not 8 * np.finfo(np.float32).eps <= relative_half_width <= 4e-5:
        raise ValueError(
            "edge half width must resolve float32 and keep narrow brackets"
        )
    kept = energy.copy()
    samples = []
    for edge in edges:
        lo, hi = edge * (1 - relative_half_width), edge * (1 + relative_half_width)
        kept = kept[(kept < lo) | (kept > hi)]
        samples.extend((lo, hi))
    grid = np.unique(np.r_[kept, samples])
    edge_intervals(grid, edges)
    return grid


def integrate_spectrum(function, energy_bounds_kev, *, breakpoints_kev=(), order=24):
    """Gauss-integrate a scalar spectrum, splitting at every supplied feature.

    Breakpoints can include jumps, native solid-state structure and spectral
    breaks. No quadrature panel crosses a breakpoint. This is host validation
    and spectral folding, independent of the Monte Carlo energy sampler.
    """
    bounds = np.asarray(energy_bounds_kev, dtype=float)
    points = np.asarray(breakpoints_kev, dtype=float)
    if (
        bounds.shape != (2,)
        or np.any(~np.isfinite(bounds))
        or not 0 < bounds[0] < bounds[1]
        or points.ndim != 1
        or np.any(~np.isfinite(points))
        or not isinstance(order, int)
        or not 2 <= order <= 256
    ):
        raise ValueError("invalid spectral interval, breakpoints or quadrature order")
    boundaries = np.unique(
        np.r_[bounds, points[(points > bounds[0]) & (points < bounds[1])]]
    )
    node, weight = np.polynomial.legendre.leggauss(order)
    low, high = boundaries[:-1, None], boundaries[1:, None]
    energy = ((low + high) / 2 + (high - low) / 2 * node).ravel()
    values = np.asarray(function(energy), dtype=float)
    if values.shape != energy.shape or np.any(~np.isfinite(values)):
        raise ValueError("spectral integrand must return one finite value per energy")
    return float(np.sum(values.reshape(low.size, order) * weight * (high - low) / 2))
