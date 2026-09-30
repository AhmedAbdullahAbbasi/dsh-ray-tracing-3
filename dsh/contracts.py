"""Stable numerical contracts shared by inputs, transport, and products.

These pytrees contain the same fields and defaults as the validated V1 types.
Source/file metadata and configuration parsing stay outside numerical kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

CARTESIAN_AXIS_ORDER = ("line_of_sight", "sky_x", "sky_y")
CLOUD_AXIS_ORDER = ("distance", "sky_y", "sky_x")
OBSERVER_AXIS_ORDER = ("arrival_time", "energy", "sky_y", "sky_x")
SOURCE_FORMAT_VERSION = 1
MATERIAL_FORMAT_VERSION = 1
LEGACY_OUTPUT_SCHEMA_VERSION = 6
FILE_INPUT_OUTPUT_SCHEMA_VERSION = 7
LINE = 0
FLAT = 1
POWERLAW = 2

ACTIVE = 0

REACHED_OBSERVER_PLANE = 1

ESCAPED_OUTER_BOUNDARY = 2

ABSORBED = 3

MAX_INTERACTIONS = 4

INVALID_ENERGY = 5

INVALID_STATE = 6

NO_INTERACTION = 0

DUST_SCATTERING = 1

PHOTOELECTRIC_ABSORPTION = 2

STATUS_NAMES = {
    ACTIVE: "active",
    REACHED_OBSERVER_PLANE: "reached the observer plane",
    ESCAPED_OUTER_BOUNDARY: "escaped through the outer transport boundary",
    ABSORBED: "absorbed",
    MAX_INTERACTIONS: "reached the numerical interaction safety limit",
    INVALID_ENERGY: "energy lies outside the dust table",
    INVALID_STATE: "position or photon four-momentum is invalid",
}

INTERACTION_NAMES = {
    NO_INTERACTION: "unused record slot",
    DUST_SCATTERING: "dust scattering",
    PHOTOELECTRIC_ABSORPTION: "photoelectric absorption",
}


class SightlineGeometry(NamedTuple):
    """JAX-ready central source and observer geometry in parsecs."""

    observer_position_pc: jnp.ndarray
    source_position_pc: jnp.ndarray
    source_distance_pc: jnp.ndarray
    source_to_observer_direction: jnp.ndarray


class AngularDistanceCloud(NamedTuple):
    """JAX-ready physical cloud scene on a native ``(z, y, x)`` grid.

    All axes are stored in increasing order even when the input FITS WCS has a
    negative increment.  ``delta_nh_cm2`` is the column carried by each radial
    voxel.  ``n_h_cm3`` is the corresponding radial-average volume density.
    """

    delta_nh_cm2: jnp.ndarray
    n_h_cm3: jnp.ndarray
    x_edges_arcsec: jnp.ndarray
    y_edges_arcsec: jnp.ndarray
    z_edges_kpc: jnp.ndarray
    radial_bin_width_cm: jnp.ndarray
    source_distance_kpc: jnp.ndarray


class RaySegments(NamedTuple):
    """Fixed-size piecewise-constant representation of a finite ray."""

    start_distance_pc: jnp.ndarray
    stop_distance_pc: jnp.ndarray
    n_h_cm3: jnp.ndarray
    column_cm2: jnp.ndarray
    inside_cloud: jnp.ndarray


class DustPhysicsTable(NamedTuple):
    """Fixed-shape JAX arrays for energy-dependent DSH interactions."""

    energy_kev: jnp.ndarray
    scattering_cross_section_cm2_per_h: jnp.ndarray
    absorption_cross_section_cm2_per_h: jnp.ndarray
    scattering_angle_rad: jnp.ndarray
    scattering_angle_cdf: jnp.ndarray
    differential_cross_section_cm2_per_sr_per_h: jnp.ndarray


class TabulatedBandSource(NamedTuple):
    """JAX-ready piecewise-constant, band-integrated source table.

    Time scales, reference epochs, and flux conventions are host metadata.
    File adapters validate them before creating the numerical source cells.
    """

    time_edges_s: jnp.ndarray
    effective_energy_kev: jnp.ndarray
    band_flux: jnp.ndarray
    cell_fluence: jnp.ndarray
    flat_cdf: jnp.ndarray
    total_fluence: jnp.ndarray
    energy_edges_kev: jnp.ndarray | None = None
    photon_index: jnp.ndarray | None = None


class SourcePackets(NamedTuple):
    """Spectral-temporal properties sampled for a batch of photon packets."""

    energy_kev: jnp.ndarray
    emission_time_s: jnp.ndarray
    weight_observer_fluence: jnp.ndarray
    time_index: jnp.ndarray
    spectral_bin_index: jnp.ndarray


class VariablePowerLawSource(NamedTuple):
    """JAX-ready time-variable source with a fixed power-law spectrum.

    ``photon_flux`` is the photon flux integrated from ``energy_min_kev`` to
    ``energy_max_kev`` in each time interval. The spectrum inside that band is
    proportional to ``E**(-photon_index)`` and is sampled analytically rather
    than approximated with energy bins.
    """

    time_edges_s: jnp.ndarray
    photon_flux: jnp.ndarray
    time_bin_fluence: jnp.ndarray
    time_cdf: jnp.ndarray
    total_fluence: jnp.ndarray
    energy_min_kev: jnp.ndarray
    energy_max_kev: jnp.ndarray
    photon_index: jnp.ndarray


class ObservationWindow(NamedTuple):
    """Observer time interval, in seconds relative to the source epoch."""

    start_s: jnp.ndarray
    stop_s: jnp.ndarray


class SourceCells(NamedTuple):
    """Positive-fluence cells; all fields are numerical JAX pytrees."""

    time_edges_s: jnp.ndarray
    start_s: jnp.ndarray
    stop_s: jnp.ndarray
    energy_low_kev: jnp.ndarray
    energy_high_kev: jnp.ndarray
    photon_index: jnp.ndarray
    kind: jnp.ndarray
    time_index: jnp.ndarray
    spectral_bin_index: jnp.ndarray
    cell_fluence: jnp.ndarray
    flat_cdf: jnp.ndarray
    total_fluence: jnp.ndarray


class SourceLaunchGeometry(NamedTuple):
    """Fixed rectangular launch cone for a centered point source.

    ``slope_x_bounds`` and ``slope_y_bounds`` describe directions
    ``normalize((-1, u, v))``.  ``launch_solid_angle_sr`` is the exact solid
    angle of that spherical rectangle; it is metadata for validation and
    normalization checks rather than a small-angle approximation.
    """

    source_position_pc: jnp.ndarray
    source_distance_pc: jnp.ndarray
    slope_x_bounds: jnp.ndarray
    slope_y_bounds: jnp.ndarray
    slope_area: jnp.ndarray
    launch_solid_angle_sr: jnp.ndarray


class LaunchedSourcePackets(NamedTuple):
    """Source packet metadata plus JAX-ready initial transport states.

    ``launch_pdf_per_sr`` is normalized over the launch cone.
    ``isotropic_importance`` is the ratio of the physical isotropic emission
    density to that sampling density, ``1 / (4*pi*q)``.  Neither quantity is
    folded into ``weight_observer_fluence``.
    """

    position_pc: jnp.ndarray
    momentum_kev: jnp.ndarray
    launch_pdf_per_sr: jnp.ndarray
    isotropic_importance: jnp.ndarray
    emission_time_s: jnp.ndarray
    weight_observer_fluence: jnp.ndarray
    time_index: jnp.ndarray
    spectral_bin_index: jnp.ndarray


class PhotonInteractionRecord(NamedTuple):
    """Fixed-size history emitted by one photon transport.

    Every field has leading dimension ``max_interactions``.  Slots for which
    ``valid`` is false are exactly zero and must not be interpreted as
    physical events.  ``scattering_order`` is one-based for scattering
    events.  For a terminal absorption it is the number of scatterings that
    occurred before absorption.

    The incoming and outgoing momenta use the same
    ``(E, p_los, p_sky_x, p_sky_y)`` convention as the main transport API.
    The outgoing momentum of an absorption event is zero.
    """

    valid: jnp.ndarray
    interaction_type: jnp.ndarray
    position_pc: jnp.ndarray
    incoming_momentum_kev: jnp.ndarray
    outgoing_momentum_kev: jnp.ndarray
    cumulative_path_length_pc: jnp.ndarray
    cumulative_excess_path_length_pc: jnp.ndarray
    scattering_order: jnp.ndarray


class PhotonTransportResult(NamedTuple):
    """Fixed-shape result for one transported photon."""

    position_pc: jnp.ndarray
    momentum_kev: jnp.ndarray
    path_length_pc: jnp.ndarray
    excess_path_length_pc: jnp.ndarray
    deposited_energy_kev: jnp.ndarray
    n_interactions: jnp.ndarray
    n_scatter: jnp.ndarray
    status: jnp.ndarray
    interactions: PhotonInteractionRecord


class ObserverEventResult(NamedTuple):
    """Fixed-shape virtual photons contributed by scattering events.

    Every field except ``valid`` has zero in unused or non-scattering slots.
    Scalar event fields have shape ``(n_packets, max_interactions)``.
    ``weight_observer_fluence`` is the contribution to an image bin in
    ``ph cm^-2``; division by the bin's sky solid angle produces surface
    brightness per steradian.
    """

    valid: jnp.ndarray
    sky_x_arcsec: jnp.ndarray
    sky_y_arcsec: jnp.ndarray
    energy_kev: jnp.ndarray
    arrival_time_s: jnp.ndarray
    excess_path_length_pc: jnp.ndarray
    scattering_angle_rad: jnp.ndarray
    scattering_order: jnp.ndarray
    escape_column_cm2: jnp.ndarray
    escape_optical_depth: jnp.ndarray
    transmission: jnp.ndarray
    phase_pdf_per_sr: jnp.ndarray
    weight_observer_fluence: jnp.ndarray
    time_index: jnp.ndarray
    spectral_bin_index: jnp.ndarray


class ObserverBinGeometry(NamedTuple):
    """Validated observer-product axes and exact sky-pixel solid angles."""

    sky_x_edges_arcsec: jnp.ndarray
    sky_y_edges_arcsec: jnp.ndarray
    energy_edges_kev: jnp.ndarray
    arrival_time_edges_s: jnp.ndarray
    sky_pixel_solid_angle_sr: jnp.ndarray


class BinnedObserverProducts(NamedTuple):
    """Weighted DSH cubes and Monte Carlo closure diagnostics.

    The three fluence arrays and ``event_count`` have shape
    ``(n_time, n_energy, n_sky_y, n_sky_x)``.  Fluence units are
    ``ph cm^-2`` per four-dimensional bin.  ``valid_*`` totals refer to all
    input events marked valid; ``binned_*`` totals include only events inside
    every requested axis.  Their difference is reported as ``unbinned_*``.
    The ``outside_*`` fields are marginal diagnostics: an event outside more
    than one axis appears in each relevant field, so those values need not sum
    to the unique unbinned total.
    """

    total_fluence: jnp.ndarray
    first_scatter_fluence: jnp.ndarray
    multiple_scatter_fluence: jnp.ndarray
    event_count: jnp.ndarray
    valid_event_count: jnp.ndarray
    binned_event_count: jnp.ndarray
    unbinned_event_count: jnp.ndarray
    valid_weight_observer_fluence: jnp.ndarray
    binned_weight_observer_fluence: jnp.ndarray
    unbinned_weight_observer_fluence: jnp.ndarray
    outside_sky_event_count: jnp.ndarray
    outside_energy_event_count: jnp.ndarray
    outside_arrival_time_event_count: jnp.ndarray
    outside_sky_weight_observer_fluence: jnp.ndarray
    outside_energy_weight_observer_fluence: jnp.ndarray
    outside_arrival_time_weight_observer_fluence: jnp.ndarray
    history_count: jnp.ndarray
    total_fluence_squared: jnp.ndarray
    time_image_fluence_squared: jnp.ndarray
    time_order_fluence_sum: jnp.ndarray
    time_order_fluence_cross: jnp.ndarray


class IdealObserverDiagnostics(NamedTuple):
    """Additive diagnostics for one or more simulated packet batches.

    ``transport_status_count`` follows the integer order in :data:`STATUS_NAMES`.
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


# A model-independent name for the numerical material interface.
Material = DustPhysicsTable


@dataclass(frozen=True)
class RunPlan:
    """Validated arrays and shape/settings needed to execute one simulation.

    Host-only plan: strings, paths, input digests, and file-format objects do
    not enter the numerical pipeline. PRNG splitting and weights are preserved.
    """

    source: SourceCells
    cloud: AngularDistanceCloud
    material: Material
    launch: SourceLaunchGeometry
    observer: ObserverBinGeometry
    packets: int
    chunk_size: int
    max_interactions: int
    seed: int
