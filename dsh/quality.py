"""Numerical completion and window losses are separate acceptance questions."""

import numpy as np

from .pipeline import TRANSPORT_STATUS_LABELS


def evaluate_quality(result, requested_packets):
    labels = getattr(result, "status_labels", TRANSPORT_STATUS_LABELS)
    counts = np.asarray(result.diagnostics.transport_status_count)
    valid_counts = (
        counts.shape == (len(labels),)
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
    report = {
        "numerical_complete": bool(accounting_ok and failures == 0 and finite_products),
        "packet_accounting_ok": accounting_ok,
        "finite_products": bool(finite_products),
        "numerical_failure_histories": failures,
        "terminal_counts": {
            name: int(value) for name, value in zip(labels, counts, strict=True)
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
    if hasattr(result, "inelastic_diagnostics"):
        extra = result.inelastic_diagnostics
        finite = all(
            np.all(np.isfinite(np.asarray(value)))
            for group in result.components
            for value in group
        ) and all(np.isfinite(value) for value in extra)
        energy_ok = bool(extra.max_energy_closure_error_kev < 2e-5)
        unsupported = int(
            extra.unsupported_observer_escape_count
            + extra.unsupported_observer_phase_count
        )
        component_ok = all(
            int(group.history_count) == requested_packets for group in result.components
        ) and np.allclose(
            result.products.total_fluence,
            sum(group.total_fluence for group in result.components[1:]),
            rtol=5e-6,
            atol=1e-20,
        )
        report.update(
            compton_diagnostics={
                name: int(value) if "count" in name else float(value)
                for name, value in zip(extra._fields, extra, strict=True)
            },
            energy_accounting_ok=energy_ok,
            component_accounting_ok=bool(component_ok),
            unsupported_observer_events=unsupported,
            analog_energy_sum_convention="unweighted simulated photons; not an isotropic-source calorimetric estimate",
        )
        report["finite_products"] &= bool(finite)
        report["numerical_complete"] &= bool(
            finite and energy_ok and component_ok and unsupported == 0
        )
    return report
