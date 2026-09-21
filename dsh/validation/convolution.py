"""Exact continuous-time reference for an impulse-response Monte Carlo sample."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class ImpulseConvolutionStatistics(NamedTuple):
    impulse_fluence: np.ndarray
    expected_flare_fluence: np.ndarray
    conditional_variance: np.ndarray
    expected_window_fluence: float
    conditional_window_variance: float


def convolve_scored_impulses(
    delays_s,
    observer_weights,
    source_time_edges_s,
    source_cell_fluence,
    arrival_time_edges_s,
) -> ImpulseConvolutionStatistics:
    """Convolve actual scattering delays with a piecewise-constant flare.

    Each scored impulse carries its observer weight. The source CDF is linear
    inside its tabulated time cells. Integrating that CDF between arrival-bin
    edges gives an exact continuous-time prediction without rounding delays or
    emission times to a time-bin center. Variance is conditional on the scored
    photon histories: one independent source emission time is sampled per
    photon, and the photon has at most one scored scattering event.
    """

    delays = np.asarray(delays_s, dtype=np.float64)
    weights = np.asarray(observer_weights, dtype=np.float64)
    source_edges = np.asarray(source_time_edges_s, dtype=np.float64)
    source_fluence = np.asarray(source_cell_fluence, dtype=np.float64)
    arrival_edges = np.asarray(arrival_time_edges_s, dtype=np.float64)
    if delays.ndim != 1 or weights.shape != delays.shape:
        raise ValueError("delays and observer weights must be matching vectors")
    if (
        source_edges.ndim != 1
        or source_edges.size < 2
        or source_fluence.shape != (source_edges.size - 1,)
        or arrival_edges.ndim != 1
        or arrival_edges.size < 2
    ):
        raise ValueError("source and arrival bins must have matching 1D axes")
    if (
        not np.all(np.isfinite(delays))
        or not np.all(np.isfinite(weights))
        or np.any(delays < 0.0)
        or np.any(weights < 0.0)
        or not np.all(np.isfinite(source_edges))
        or not np.all(np.isfinite(source_fluence))
        or np.any(source_fluence < 0.0)
        or not np.all(np.isfinite(arrival_edges))
        or np.any(np.diff(source_edges) <= 0.0)
        or np.any(np.diff(arrival_edges) <= 0.0)
        or not source_fluence.sum() > 0.0
    ):
        raise ValueError(
            "input values must be finite with ordered axes and positive source fluence"
        )

    source_cdf = np.r_[0.0, np.cumsum(source_fluence, dtype=np.float64)]
    source_cdf /= source_cdf[-1]
    source_cdf[-1] = 1.0
    shifted_edges = arrival_edges[None, :] - delays[:, None]
    probabilities = np.diff(
        np.interp(shifted_edges, source_edges, source_cdf, left=0.0, right=1.0),
        axis=1,
    )
    weighted_probabilities = weights[:, None] * probabilities
    expected = weighted_probabilities.sum(axis=0, dtype=np.float64)
    variance = ((weights[:, None] ** 2) * probabilities * (1.0 - probabilities)).sum(
        axis=0, dtype=np.float64
    )
    # The union of the arrival bins is one interval. Summing rounded per-bin
    # probabilities can yield 1 - 1e-16 even when the source lies entirely
    # within that interval. Its Bernoulli variance must be exactly zero in
    # that case, so compute the window probability from its CDF endpoints.
    window_probability = (
        np.interp(
            shifted_edges[:, -1], source_edges, source_cdf, left=0.0, right=1.0
        )
        - np.interp(
            shifted_edges[:, 0], source_edges, source_cdf, left=0.0, right=1.0
        )
    )
    return ImpulseConvolutionStatistics(
        impulse_fluence=np.histogram(delays, bins=arrival_edges, weights=weights)[0],
        expected_flare_fluence=expected,
        conditional_variance=variance,
        expected_window_fluence=float(
            np.sum(weights * window_probability, dtype=np.float64)
        ),
        conditional_window_variance=float(
            np.sum(weights**2 * window_probability * (1.0 - window_probability))
        ),
    )
