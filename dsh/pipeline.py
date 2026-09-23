"""Compose the Version-1 source-to-observer DSH simulation pipeline.

This module joins the independently validated physical layers without adding
instrument effects:

``source -> launch -> voxel transport -> peel-off scoring -> DSH binning``.

The single-batch functions are JAX-jittable.  The chunked functions are host
coordinators: they run fixed-size compiled batches and retain only accumulated
observer products and scalar diagnostics.  Every chunk packet is assigned
``source.total_fluence / total_packets``.  This detail is essential because
calling a source sampler independently for every chunk would otherwise count
the complete source fluence once per chunk.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from .geometry.clouds import AngularDistanceCloud
from .observer.binning import (
    BinnedObserverProducts,
    ObserverBinGeometry,
    add_binned_observer_products,
    bin_observer_events,
)
from .observer.scoring import score_peeloff_events
from .physics.dust import DustPhysicsTable
from .sources.launch import SourceLaunchGeometry, sample_source_launches
from .sources.models import (
    SourcePackets,
    TabulatedBandSource,
    VariablePowerLawSource,
    sample_tabulated_band_source,
    sample_variable_powerlaw_source,
)
from .transport.kernel import STATUS_NAMES, transport_photon_batch

TRANSPORT_STATUS_LABELS = tuple(
    STATUS_NAMES[index] for index in range(len(STATUS_NAMES))
)


class IdealObserverDiagnostics(NamedTuple):
    """Additive diagnostics for one or more simulated packet batches.

    ``transport_status_count`` follows :data:`TRANSPORT_STATUS_LABELS`.
    Source and observer fluences are in ``ph cm^-2``.  Observer fluence is a
    next-event Monte Carlo sum and is not expected to equal source fluence.
    """

    source_packet_count: jnp.ndarray
    source_fluence: jnp.ndarray
    transport_status_count: jnp.ndarray
    analog_interaction_count: jnp.ndarray
    analog_scattering_count: jnp.ndarray
    scored_observer_event_count: jnp.ndarray
    scored_observer_fluence: jnp.ndarray


class IdealObserverSimulationResult(NamedTuple):
    """Ideal-observer DSH products plus transport and scoring diagnostics."""

    products: BinnedObserverProducts
    diagnostics: IdealObserverDiagnostics


def _validate_packet_weights(packets: SourcePackets):
    weights = jnp.asarray(packets.weight_observer_fluence)
    if weights.ndim != 1 or weights.shape[0] <= 0:
        raise ValueError("packet weights must be a nonempty one-dimensional array")


def simulate_source_packets_to_observer(
    key,
    packets: SourcePackets,
    launch_geometry: SourceLaunchGeometry,
    cloud: AngularDistanceCloud,
    physics: DustPhysicsTable,
    bin_geometry: ObserverBinGeometry,
    *,
    max_interactions: int = 64,
) -> IdealObserverSimulationResult:
    """Run already sampled source packets through the complete V1 pipeline.

    The supplied key is split into independent launch and transport streams.
    Observer scoring and binning are deterministic.  Intermediate launch,
    history, and event arrays are released after this function returns; only
    accumulated products and diagnostics remain.

    ``max_interactions`` controls fixed JAX shapes and must be static when the
    function is passed to :func:`jax.jit`.
    """

    _validate_packet_weights(packets)
    launch_key, transport_key = random.split(key)
    launched = sample_source_launches(launch_key, packets, launch_geometry)
    transported = transport_photon_batch(
        transport_key,
        launched.position_pc,
        launched.momentum_kev,
        cloud,
        physics,
        max_interactions=max_interactions,
    )
    events = score_peeloff_events(launched, transported, cloud, physics)
    products = bin_observer_events(events, bin_geometry)

    status_count = (
        jnp.zeros((len(TRANSPORT_STATUS_LABELS),), dtype=jnp.int32)
        .at[transported.status]
        .add(1)
    )
    valid_event_weight = jnp.where(events.valid, events.weight_observer_fluence, 0.0)
    diagnostics = IdealObserverDiagnostics(
        source_packet_count=jnp.asarray(packets.energy_kev.shape[0], dtype=jnp.int32),
        source_fluence=jnp.sum(packets.weight_observer_fluence),
        transport_status_count=status_count,
        analog_interaction_count=jnp.sum(transported.n_interactions, dtype=jnp.int32),
        analog_scattering_count=jnp.sum(transported.n_scatter, dtype=jnp.int32),
        scored_observer_event_count=jnp.sum(events.valid, dtype=jnp.int32),
        scored_observer_fluence=jnp.sum(valid_event_weight),
    )
    return IdealObserverSimulationResult(
        products=products,
        diagnostics=diagnostics,
    )


def simulate_tabulated_source_to_observer(
    key,
    source: TabulatedBandSource,
    launch_geometry: SourceLaunchGeometry,
    cloud: AngularDistanceCloud,
    physics: DustPhysicsTable,
    bin_geometry: ObserverBinGeometry,
    *,
    n_packets: int,
    max_interactions: int = 64,
) -> IdealObserverSimulationResult:
    """Sample a tabulated-band source and run the complete V1 pipeline.

    ``n_packets`` and ``max_interactions`` must be static under JIT.
    """

    source_key, pipeline_key = random.split(key)
    packets = sample_tabulated_band_source(source_key, source, n_packets)
    return simulate_source_packets_to_observer(
        pipeline_key,
        packets,
        launch_geometry,
        cloud,
        physics,
        bin_geometry,
        max_interactions=max_interactions,
    )


def simulate_variable_powerlaw_source_to_observer(
    key,
    source: VariablePowerLawSource,
    launch_geometry: SourceLaunchGeometry,
    cloud: AngularDistanceCloud,
    physics: DustPhysicsTable,
    bin_geometry: ObserverBinGeometry,
    *,
    n_packets: int,
    max_interactions: int = 64,
) -> IdealObserverSimulationResult:
    """Sample a variable power-law source and run the complete V1 pipeline.

    ``n_packets`` and ``max_interactions`` must be static under JIT.
    """

    source_key, pipeline_key = random.split(key)
    packets = sample_variable_powerlaw_source(source_key, source, n_packets)
    return simulate_source_packets_to_observer(
        pipeline_key,
        packets,
        launch_geometry,
        cloud,
        physics,
        bin_geometry,
        max_interactions=max_interactions,
    )


def add_ideal_observer_simulation_results(
    left: IdealObserverSimulationResult,
    right: IdealObserverSimulationResult,
) -> IdealObserverSimulationResult:
    """Add products and diagnostics from independent packet batches."""

    if (
        left.diagnostics.transport_status_count.shape
        != right.diagnostics.transport_status_count.shape
    ):
        raise ValueError("transport status-count shapes must match")
    products = add_binned_observer_products(left.products, right.products)
    diagnostics = jax.tree.map(
        lambda left_value, right_value: left_value + right_value,
        left.diagnostics,
        right.diagnostics,
    )
    return IdealObserverSimulationResult(
        products=products,
        diagnostics=diagnostics,
    )


def _validated_chunk_counts(total_packets, chunk_size):
    total = np.asarray(total_packets)
    chunk = np.asarray(chunk_size)
    if total.ndim != 0 or not np.issubdtype(total.dtype, np.integer) or int(total) <= 0:
        raise ValueError("total_packets must be one positive integer")
    if chunk.ndim != 0 or not np.issubdtype(chunk.dtype, np.integer) or int(chunk) <= 0:
        raise ValueError("chunk_size must be one positive integer")
    return int(total), int(chunk)


def _run_source_to_observer_chunked(
    key,
    source,
    sampler: Callable,
    launch_geometry: SourceLaunchGeometry,
    cloud: AngularDistanceCloud,
    physics: DustPhysicsTable,
    bin_geometry: ObserverBinGeometry,
    *,
    total_packets: int,
    chunk_size: int,
    max_interactions: int,
    progress_callback: Callable[[int, int], None] | None,
) -> IdealObserverSimulationResult:
    total, chunk = _validated_chunk_counts(total_packets, chunk_size)
    if max_interactions <= 0:
        raise ValueError("max_interactions must be positive")

    sample_jit = jax.jit(sampler, static_argnames=("n_packets",))
    pipeline_jit = jax.jit(
        simulate_source_packets_to_observer,
        static_argnames=("max_interactions",),
    )
    result = None
    completed = 0
    chunk_index = 0
    while completed < total:
        current_size = min(chunk, total - completed)
        chunk_key = random.fold_in(key, chunk_index)
        source_key, pipeline_key = random.split(chunk_key)
        packets = sample_jit(source_key, source, n_packets=current_size)
        packet_weight = source.total_fluence / total
        packets = packets._replace(
            weight_observer_fluence=jnp.full(
                (current_size,),
                packet_weight,
                dtype=packets.weight_observer_fluence.dtype,
            )
        )
        current = pipeline_jit(
            pipeline_key,
            packets,
            launch_geometry,
            cloud,
            physics,
            bin_geometry,
            max_interactions=max_interactions,
        )
        # Persist sums on the host in float64/int64. JAX's default float32 is
        # appropriate for each bounded batch, not millions of chunk additions.
        current = jax.tree.map(
            lambda value: np.asarray(value).astype(
                np.int64 if np.issubdtype(value.dtype, np.integer) else np.float64
            ),
            current,
        )
        result = (
            current
            if result is None
            else add_ideal_observer_simulation_results(result, current)
        )
        # GPU dispatch is asynchronous.  Synchronizing once per chunk makes
        # the coordinator genuinely memory bounded and ensures that progress
        # callbacks report completed work rather than merely queued work.
        result = jax.block_until_ready(result)
        completed += current_size
        chunk_index += 1
        if progress_callback is not None:
            progress_callback(completed, total)
    return result


def run_tabulated_source_to_observer_chunked(
    key,
    source: TabulatedBandSource,
    launch_geometry: SourceLaunchGeometry,
    cloud: AngularDistanceCloud,
    physics: DustPhysicsTable,
    bin_geometry: ObserverBinGeometry,
    *,
    total_packets: int,
    chunk_size: int,
    max_interactions: int = 64,
    progress_callback: Callable[[int, int], None] | None = None,
) -> IdealObserverSimulationResult:
    """Run a tabulated source in memory-bounded packet chunks on the host."""

    return _run_source_to_observer_chunked(
        key,
        source,
        sample_tabulated_band_source,
        launch_geometry,
        cloud,
        physics,
        bin_geometry,
        total_packets=total_packets,
        chunk_size=chunk_size,
        max_interactions=max_interactions,
        progress_callback=progress_callback,
    )


def run_variable_powerlaw_source_to_observer_chunked(
    key,
    source: VariablePowerLawSource,
    launch_geometry: SourceLaunchGeometry,
    cloud: AngularDistanceCloud,
    physics: DustPhysicsTable,
    bin_geometry: ObserverBinGeometry,
    *,
    total_packets: int,
    chunk_size: int,
    max_interactions: int = 64,
    progress_callback: Callable[[int, int], None] | None = None,
) -> IdealObserverSimulationResult:
    """Run a power-law source in memory-bounded packet chunks on the host."""

    return _run_source_to_observer_chunked(
        key,
        source,
        sample_variable_powerlaw_source,
        launch_geometry,
        cloud,
        physics,
        bin_geometry,
        total_packets=total_packets,
        chunk_size=chunk_size,
        max_interactions=max_interactions,
        progress_callback=progress_callback,
    )
