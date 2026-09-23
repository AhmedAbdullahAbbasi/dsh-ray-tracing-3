"""Read-only physical and sampling-adequacy checks of a saved flare run."""

from __future__ import annotations

import math

import numpy as np

DAY_S = 86_400.0


def snapshot_slice(edges_s, start_day, exposure_days):
    """Require both requested arrival boundaries to be explicit bin edges."""

    edges = np.asarray(edges_s, dtype=np.float64)
    if edges.ndim != 1 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("arrival edges must be finite and strictly increasing")
    begin = np.flatnonzero(np.isclose(edges, start_day * DAY_S, rtol=0, atol=0.05))
    end = np.flatnonzero(
        np.isclose(edges, (start_day + exposure_days) * DAY_S, rtol=0, atol=0.05)
    )
    if begin.size != 1 or end.size != 1 or end[0] <= begin[0]:
        raise ValueError(
            f"snapshot [{start_day}, {start_day + exposure_days}) days "
            "is not bounded by arrival-bin edges"
        )
    return slice(int(begin[0]), int(end[0]))


def inspect_flare_arrays(
    *,
    counts,
    total,
    first,
    multiple,
    arrival_edges_s,
    status_counts,
    expected_packets,
    first_day=3,
    separation_days=3,
    exposure_days=1,
    coarse_factor=20,
    min_snapshot_events=2_000,
    min_supported_cell_events=10,
    min_supported_event_fraction=0.8,
):
    """Screen three sky images; do not treat positive pixels as convergence.

    Counts are an *upper bound* on the effective weighted sample size in each
    pixel. A ten-event 20-arcsec cell has at best a 32% Poisson relative error.
    These configurable defaults screen basic morphology, not publication
    quality or independent-realization convergence.
    """

    if (
        expected_packets <= 0
        or separation_days <= 0
        or exposure_days <= 0
        or first_day < 0
        or coarse_factor <= 0
        or min_snapshot_events <= 0
        or min_supported_cell_events <= 0
        or not 0 < min_supported_event_fraction <= 1
    ):
        raise ValueError("invalid snapshot schedule or adequacy thresholds")
    count = np.asarray(counts)
    weight = np.asarray(total)
    first_weight = np.asarray(first)
    multi_weight = np.asarray(multiple)
    status = np.asarray(status_counts)
    edges = np.asarray(arrival_edges_s, dtype=np.float64)
    if (
        count.ndim != 4
        or count.shape != weight.shape
        or count.shape != first_weight.shape
        or count.shape != multi_weight.shape
        or edges.shape != (count.shape[0] + 1,)
        or status.shape != (7,)
    ):
        raise ValueError("incompatible observer cube, time-edge, or status shapes")
    height, width = count.shape[2:]
    if height % coarse_factor or width % coarse_factor:
        raise ValueError("coarse factor must divide both sky image axes")
    if (
        np.any(~np.isfinite(count))
        or np.any(count != np.floor(count))
        or np.any(count < 0)
        or np.any(~np.isfinite(status))
        or np.any(status != np.floor(status))
        or np.any(~np.isfinite(weight))
        or np.any(weight < 0)
        or np.any(first_weight < 0)
        or np.any(multi_weight < 0)
    ):
        raise ValueError("invalid event counts or fluence")
    if not np.allclose(weight, first_weight + multi_weight, rtol=2e-6, atol=1e-10):
        raise ValueError("first and multiple fluence do not close to total")
    if not np.all(np.isfinite(first_weight)) or not np.all(np.isfinite(multi_weight)):
        raise ValueError("nonfinite first or multiple fluence")

    terminal_clean = bool(
        status.sum() == expected_packets
        and np.all(status >= 0)
        and status[0] == 0
        and status[4] == 0
        and status[5] == 0
        and status[6] == 0
    )
    rows = []
    for j in range(3):
        day = first_day + j * separation_days
        selection = snapshot_slice(edges, day, exposure_days)
        image_counts = count[selection].sum(axis=(0, 1), dtype=np.int64)
        image_weight = weight[selection].sum(axis=(0, 1), dtype=np.float64)
        multi_fluence = multi_weight[selection].sum(dtype=np.float64)
        n_events = int(image_counts.sum())
        coarse = image_counts.reshape(
            height // coarse_factor,
            coarse_factor,
            width // coarse_factor,
            coarse_factor,
        ).sum(axis=(1, 3), dtype=np.int64)
        occupied = coarse[coarse > 0]
        supported = int(coarse[coarse >= min_supported_cell_events].sum())
        fraction_supported = supported / n_events if n_events else 0.0
        count_check = n_events >= min_snapshot_events
        support_check = fraction_supported >= min_supported_event_fraction
        rows.append(
            {
                "start_day": day,
                "stop_day": day + exposure_days,
                "event_count": n_events,
                "fluence_ph_cm2": float(image_weight.sum()),
                "multiple_scatter_fluence_fraction": (
                    float(multi_fluence / image_weight.sum())
                    if image_weight.sum() > 0
                    else 0.0
                ),
                "native_occupied_pixels": int(np.count_nonzero(image_counts)),
                "native_pixels_with_two_or_more_events": int(
                    np.count_nonzero(image_counts >= 2)
                ),
                "coarse_occupied_cells": int(occupied.size),
                "coarse_median_events_in_occupied_cells": (
                    float(np.median(occupied)) if occupied.size else 0.0
                ),
                "coarse_max_events": int(coarse.max()),
                "events_in_supported_coarse_cells": supported,
                "supported_event_fraction": fraction_supported,
                "count_check": bool(count_check),
                "spatial_support_check": bool(support_check),
                "passed": bool(count_check and support_check),
            }
        )
    checks = {
        "clean_transport": terminal_clean,
        "snapshots_meet_count_and_spatial_screen": all(row["passed"] for row in rows),
    }
    minimum_scale = (
        max(
            (
                math.ceil(min_snapshot_events / row["event_count"])
                if row["event_count"]
                else None
            )
            for row in rows
            if row["event_count"] > 0
        )
        if all(row["event_count"] for row in rows)
        else None
    )
    return {
        "status_counts": status.tolist(),
        "snapshot_rows": rows,
        "count_only_packet_multiplier_lower_bound": minimum_scale,
        "checks": checks,
        "all_passed": all(checks.values()),
    }
