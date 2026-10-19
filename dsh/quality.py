"""Numerical completion and window losses are separate acceptance questions."""

import numpy as np

from .pipeline import TRANSPORT_STATUS_LABELS


def evaluate_quality(result, requested_packets):
    counts = np.asarray(result.diagnostics.transport_status_count)
    valid_counts = (
        counts.shape == (len(TRANSPORT_STATUS_LABELS),)
        and np.all(np.isfinite(counts))
        and np.all(counts >= 0)
    )
    accounting_ok = bool(
        valid_counts
        and int(counts.sum()) == requested_packets
        and int(result.diagnostics.source_packet_count) == requested_packets
        and int(result.products.history_count) == requested_packets
    )
    failures = int(counts[0] + counts[4:].sum()) if valid_counts else requested_packets
    finite_products = all(
        np.all(np.isfinite(np.asarray(value)))
        for value in (*result.products, *result.diagnostics)
    )
    return {
        "numerical_complete": bool(accounting_ok and failures == 0 and finite_products),
        "packet_accounting_ok": accounting_ok,
        "finite_products": bool(finite_products),
        "numerical_failure_histories": failures,
        "terminal_counts": {
            name: int(value)
            for name, value in zip(TRANSPORT_STATUS_LABELS, counts, strict=True)
        }
        if valid_counts
        else {},
        "scored_fluence_ph_cm2": float(result.diagnostics.scored_observer_fluence),
        "binned_fluence_ph_cm2": float(result.products.binned_weight_observer_fluence),
        "window_loss_fluence_ph_cm2": float(
            result.products.unbinned_weight_observer_fluence
        ),
        "window_losses_are_numerical_failures": False,
        "sampling_status": "Monte Carlo convergence is not established by numerical completion.",
    }
