"""Memory-bounded configured Compton simulation with prefix component products."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from .observer.inelastic import bin_inelastic_events, score_inelastic_peeloff
from .pipeline import (
    TRANSPORT_STATUS_LABELS,
    IdealObserverDiagnostics,
    _validated_chunk_counts,
)
from .sources.launch import sample_source_launches
from .sources.models import sample_tabulated_band_source
from .transport.inelastic import transport_inelastic_batch

INELASTIC_STATUS_LABELS = TRANSPORT_STATUS_LABELS + (
    "continuation below declared energy cutoff",
    "continuation outside common material support",
)


class InelasticDiagnostics(NamedTuple):
    analog_dust_count: jnp.ndarray
    analog_compton_count: jnp.ndarray
    source_energy_sum_kev: jnp.ndarray
    deposited_energy_sum_kev: jnp.ndarray
    recoil_energy_sum_kev: jnp.ndarray
    absorbed_energy_sum_kev: jnp.ndarray
    residual_energy_sum_kev: jnp.ndarray
    escaping_energy_sum_kev: jnp.ndarray
    cutoff_energy_sum_kev: jnp.ndarray
    unsupported_energy_sum_kev: jnp.ndarray
    safety_limit_energy_sum_kev: jnp.ndarray
    max_energy_closure_error_kev: jnp.ndarray
    unsupported_observer_escape_count: jnp.ndarray
    unsupported_observer_phase_count: jnp.ndarray


class InelasticSimulationResult(NamedTuple):
    products: object
    diagnostics: IdealObserverDiagnostics
    components: object
    inelastic_diagnostics: InelasticDiagnostics

    @property
    def status_labels(self):
        return INELASTIC_STATUS_LABELS


def simulate_inelastic_packets(
    key,
    packets,
    launch_geometry,
    cloud,
    physics,
    compton,
    bins,
    *,
    energy_cutoff_kev,
    max_interactions,
):
    launch_key, transport_key = random.split(key)
    launched = sample_source_launches(launch_key, packets, launch_geometry)
    result = transport_inelastic_batch(
        transport_key,
        launched.position_pc,
        launched.momentum_kev,
        cloud,
        physics,
        compton,
        energy_cutoff_kev=energy_cutoff_kev,
        max_interactions=max_interactions,
    )
    scored = score_inelastic_peeloff(launched, result, cloud, physics, compton)
    components = bin_inelastic_events(scored, bins)
    diagnostics = IdealObserverDiagnostics(
        jnp.asarray(packets.energy_kev.size, dtype=jnp.int32),
        jnp.sum(packets.weight_observer_fluence),
        jnp.zeros(9, dtype=jnp.int32).at[result.status].add(1),
        jnp.sum(result.n_interactions, dtype=jnp.int32),
        jnp.sum(result.n_scatter, dtype=jnp.int32),
        jnp.sum(scored.events.valid, dtype=jnp.int32),
        jnp.sum(scored.events.weight_observer_fluence),
    )
    remaining = result.momentum_kev[:, 0]
    energy = InelasticDiagnostics(
        jnp.sum(result.n_dust, dtype=jnp.int32),
        jnp.sum(result.n_compton, dtype=jnp.int32),
        jnp.sum(result.source_energy_kev),
        jnp.sum(result.deposited_energy_kev),
        jnp.sum(result.recoil_energy_kev),
        jnp.sum(result.absorbed_energy_kev),
        jnp.sum(remaining),
        jnp.sum(jnp.where((result.status == 1) | (result.status == 2), remaining, 0)),
        jnp.sum(result.cutoff_energy_kev),
        jnp.sum(result.unsupported_energy_kev),
        jnp.sum(jnp.where(result.status == 4, remaining, 0)),
        jnp.max(
            jnp.abs(result.source_energy_kev - result.deposited_energy_kev - remaining)
        ),
        jnp.sum(scored.unsupported_escape, dtype=jnp.int32),
        jnp.sum(scored.unsupported_phase, dtype=jnp.int32),
    )
    return InelasticSimulationResult(components.total, diagnostics, components, energy)


def run_inelastic_source_chunked(
    key,
    source,
    launch_geometry,
    cloud,
    physics,
    compton,
    bins,
    *,
    total_packets,
    chunk_size,
    energy_cutoff_kev,
    max_interactions=64,
    progress_callback=None,
):
    total, chunk = _validated_chunk_counts(total_packets, chunk_size)
    sample = jax.jit(sample_tabulated_band_source, static_argnames=("n_packets",))
    simulate = jax.jit(
        simulate_inelastic_packets, static_argnames=("max_interactions",)
    )
    accumulated = None
    for index, start in enumerate(range(0, total, chunk)):
        size = min(chunk, total - start)
        source_key, pipeline_key = random.split(random.fold_in(key, index))
        packets = sample(source_key, source, n_packets=size)
        packets = packets._replace(
            weight_observer_fluence=jnp.full(
                (size,),
                source.total_fluence / total,
                dtype=packets.weight_observer_fluence.dtype,
            )
        )
        current = simulate(
            pipeline_key,
            packets,
            launch_geometry,
            cloud,
            physics,
            compton,
            bins,
            energy_cutoff_kev=energy_cutoff_kev,
            max_interactions=max_interactions,
        )
        current = jax.tree.map(
            lambda value: np.asarray(value).astype(
                np.int64 if np.issubdtype(value.dtype, np.integer) else np.float64
            ),
            current,
        )
        if accumulated is None:
            accumulated = current
        else:
            maximum = max(
                accumulated.inelastic_diagnostics.max_energy_closure_error_kev,
                current.inelastic_diagnostics.max_energy_closure_error_kev,
            )
            accumulated = jax.tree.map(lambda a, b: a + b, accumulated, current)
            accumulated = accumulated._replace(
                inelastic_diagnostics=accumulated.inelastic_diagnostics._replace(
                    max_energy_closure_error_kev=maximum
                )
            )
        if progress_callback is not None:
            progress_callback(start + size, total)
    return accumulated
