"""Photon transport through a voxel grid of densities, parallelized with JAX.

Geometry
--------
The simulation domain is one axis-aligned box, `box_min` to `box_max`,
chopped into a regular grid of voxels (see `voxels.py`), each with its
own density -- a local scattering coefficient, in units of 1/length.
There is no single hard object anymore; "an object" is simply wherever
that density field is nonzero. A photon enters as a beam (see
`_launch_state`) and, exactly as in a classic continuous-medium MCRT
slab code, its free path to the next scattering event is a random
*optical depth* (`sampling.sample_free_path`) rather than a fixed
physical distance -- except now the medium's density varies from voxel
to voxel along the way, in full 3D, instead of being uniform along one
axis.

Every step, exactly one of two things can happen:

  - it accumulates enough optical depth to **scatter**, at whatever
    physical point along its path that happens (found by marching
    through the grid -- see `photon_step`), and picks a new,
    isotropically random direction, or
  - it runs out of box before accumulating that much optical depth,
    and **leaves the domain**: killed, removed from the simulation.

Scatter, or die leaving the box -- still the only two things that can
happen, exactly as in the previous (single hard-object) version. No
absorption.

Parallelization
----------------
Same JAX pattern as ever, just with one more layer of nesting:
`photon_step` advances one photon through *one* scattering event, but
finding that event now itself requires a fixed-length inner
`jax.lax.scan` that marches through the grid in `n_substeps` small
steps, accumulating optical depth as it goes (see below). That inner
scan is wrapped in the same outer `jax.lax.scan` over `n_bounces`
possible events as before, which `jax.vmap` then runs for every photon
in the batch at once, and `jax.jit` compiles into one kernel.

`simulate_source_packets` is the physical-source entry point. It keeps
source-sampled energy in keV, emission time, and statistical fluence weight
separate from the legacy toy Mie model's dimensionless phase parameter.
"""

from typing import NamedTuple

from jax import lax, random, vmap
import jax.numpy as jnp

from .sampling import (sample_free_path, sample_isotropic_mu, sample_azimuth, sample_mu_in_cone,
                        sample_rayleigh_mu, sample_henyey_greenstein_mu, mie_asymmetry_from_size_parameter,
                        size_parameter_from_energy_kev)
from .raytracing import ray_box_intersection, direction_from_axis_mu_phi, orthonormal_basis, sample_disk_point
from .source import SourcePackets
from .voxels import voxel_index

# Photon status codes.
ACTIVE = 0
KILLED = 1

STATUS_NAMES = {
    ACTIVE: "active (ran out of bounce budget -- increase n_bounces)",
    KILLED: "killed (left the simulation domain)",
}

_DEFAULT_BEAM_DIRECTION = jnp.array([1.0, 0.0, 0.0])

# Valid `scattering_model` choices for `photon_step`/`run_one_photon`/`simulate_photons` --
# see `_sample_scattering_mu` for what each one actually draws.
SCATTERING_MODELS = ("isotropic", "rayleigh", "mie", "rayleigh_mie")


class SourceTransportResult(NamedTuple):
    """Transport outputs with the physical source metadata kept per packet.

    ``energy_kev``, ``emission_time_s``, and ``weight_observer_fluence`` are
    copied from :class:`utils.source.SourcePackets`; the transport does not
    reinterpret or modify them. ``path`` contains positions inside the
    simulation domain and therefore is not yet a complete source-to-observer
    light-travel path.
    """

    position: jnp.ndarray
    direction: jnp.ndarray
    energy_kev: jnp.ndarray
    emission_time_s: jnp.ndarray
    weight_observer_fluence: jnp.ndarray
    time_index: jnp.ndarray
    spectral_bin_index: jnp.ndarray
    n_scatter: jnp.ndarray
    status: jnp.ndarray
    path: jnp.ndarray


def _sample_scattering_mu(key_mu, energy, scattering_model, mie_g_max, rayleigh_mie_transition_x):
    """Draw a scattering-angle cosine using whichever phase function `scattering_model` selects.

    `scattering_model` is a plain Python string, not a traced value --
    picking a phase function is a *compile-time* choice (like
    `n_bounces` or `n_substeps`), so this is an ordinary Python
    `if`/`elif`, not a `jnp.where` chain, and `scattering_model` must be
    passed as a static argument when this ends up under `jax.jit`.

    `"rayleigh_mie"` is the one case that *is* a per-photon runtime
    decision -- which regime a given photon is in depends on its own
    `energy` (reused here as a dimensionless grain-size parameter, see
    `sampling.mie_asymmetry_from_size_parameter`), which varies across
    the batch. So both candidate draws are computed and `jnp.where`
    picks the right one per photon, the same "compute both branches,
    select with jnp.where" pattern used everywhere else in this file.
    """
    if scattering_model == "isotropic":
        return sample_isotropic_mu(key_mu)
    elif scattering_model == "rayleigh":
        return sample_rayleigh_mu(key_mu)
    elif scattering_model == "mie":
        g = mie_asymmetry_from_size_parameter(energy, mie_g_max)
        return sample_henyey_greenstein_mu(key_mu, g)
    elif scattering_model == "rayleigh_mie":
        g = mie_asymmetry_from_size_parameter(energy, mie_g_max)
        mu_rayleigh = sample_rayleigh_mu(key_mu)
        mu_mie = sample_henyey_greenstein_mu(key_mu, g)
        return jnp.where(energy < rayleigh_mie_transition_x, mu_rayleigh, mu_mie)
    else:
        raise ValueError(f"unknown scattering_model {scattering_model!r}, expected one of {SCATTERING_MODELS}")


def _march_to_scattering_point(pos, direction, tau_target, t_exit, density_grid, box_min, voxel_size, n_substeps):
    """Walk from `pos` along `direction`, accumulating optical depth, up to `t_exit` away.

    This is the heart of what changed from a single hard object to a
    voxel grid: instead of one analytic yes/no intersection test, we
    take `n_substeps` fixed-size steps of length `ds = t_exit /
    n_substeps`, look up the local density at each one, and add up
    density * ds -- a Riemann-sum approximation of the optical depth
    integral integral(density ds) along the ray. As soon as that running
    total would reach `tau_target`, we've found the scattering point;
    interpolate its exact distance within that last small step. `ds`
    smaller (i.e. `n_substeps` larger) makes this a better
    approximation of the true continuous integral, at the cost of more
    compute -- the same "increase the resolution" tradeoff as any
    numerical integration.

    Returns (scattered, t_scatter): whether the target was reached
    before `t_exit`, and if so, at what distance along the ray.
    """
    ds = t_exit / n_substeps
    n_voxels = density_grid.shape[0]

    def march_step(carry, i):
        tau_accum, scattered, t_scatter = carry
        t_start = i * ds
        sample_pos = pos + direction * t_start
        idx = voxel_index(sample_pos, box_min, voxel_size, n_voxels)
        rho = density_grid[idx[0], idx[1], idx[2]]

        tau_after = tau_accum + rho * ds
        crosses_now = (~scattered) & (tau_after >= tau_target)
        t_cross = t_start + (tau_target - tau_accum) / jnp.maximum(rho, 1e-30)

        return (tau_after, scattered | crosses_now, jnp.where(crosses_now, t_cross, t_scatter)), None

    init = (jnp.array(0.0), jnp.array(False), jnp.array(0.0))
    (_, scattered, t_scatter), _ = lax.scan(march_step, init, jnp.arange(n_substeps))
    return scattered, t_scatter


def photon_step(state, box_min, box_max, density_grid, n_substeps,
                 scattering_model="isotropic", mie_g_max=0.85, rayleigh_mie_transition_x=1.0):
    """Advance one photon by (up to) one event: a scatter, or leaving the domain.

    `state` is (pos, direction, energy, n_scatter, key, status):
      pos       -- current 3D position, shape (3,)
      direction -- current unit direction vector, shape (3,)
      energy    -- a per-photon weight, carried along unchanged in the "isotropic" model;
                   under "mie"/"rayleigh_mie" it doubles as a dimensionless grain-size
                   parameter that picks the scattering phase function (see
                   `_sample_scattering_mu`) -- still not consumed/changed by absorption,
                   there still isn't any in this model
      n_scatter -- how many times this photon has scattered so far
      key       -- this photon's JAX PRNG key
      status    -- ACTIVE or KILLED

    `scattering_model` picks the angular distribution of a scattering
    event -- `"isotropic"` (the original, default, and the only model
    that ignores `energy` entirely), `"rayleigh"` (exact small-grain
    phase function), `"mie"` (Henyey-Greenstein, the standard stand-in
    for large-grain Mie scattering), or `"rayleigh_mie"` (per-photon
    switch between the two based on its own `energy`/size parameter,
    against `rayleigh_mie_transition_x`). See `_sample_scattering_mu`
    and `sampling.py` for the actual phase functions.

    If the photon is already KILLED, every `jnp.where` below leaves its
    state untouched, which is what lets the outer `lax.scan` run a
    fixed number of steps per photon even though real trajectories
    have different lengths.
    """
    pos, direction, energy, n_scatter, key, status = state
    key, key_tau, key_mu, key_phi = random.split(key, 4)
    is_active = status == ACTIVE

    voxel_size = (box_max - box_min) / density_grid.shape[0]
    tau_target = sample_free_path(key_tau)

    # How far along this straight segment until it would leave the box, if it never scatters first?
    _, t_exit, _ = ray_box_intersection(pos, direction, box_min, box_max)
    t_exit = jnp.maximum(t_exit, 0.0)

    scattered, t_scatter = _march_to_scattering_point(
        pos, direction, tau_target, t_exit, density_grid, box_min, voxel_size, n_substeps)
    hits_voxel = is_active & scattered
    leaves_domain = is_active & ~scattered

    # If it scatters: move to that point, then pick a new direction from the chosen phase
    # function. Any fixed reference axis gives the same result in the lab frame by symmetry
    # (see sampling.sample_isotropic_mu) -- using the photon's own incoming direction as that
    # axis needs no extra state to carry around, and is also exactly the axis a real phase
    # function's scattering angle is measured from, so the same trick still works unchanged
    # for the anisotropic (Rayleigh/Mie) models, not just the isotropic one.
    scatter_point = pos + direction * jnp.where(hits_voxel, t_scatter, 0.0)
    mu = _sample_scattering_mu(key_mu, energy, scattering_model, mie_g_max, rayleigh_mie_transition_x)
    phi = sample_azimuth(key_phi)
    scattered_direction = direction_from_axis_mu_phi(direction, mu, phi)

    # If it leaves the domain instead: move to the exit point on the boundary.
    exit_point = pos + direction * jnp.where(leaves_domain, t_exit, 0.0)

    pos_next = jnp.where(hits_voxel, scatter_point, jnp.where(leaves_domain, exit_point, pos))
    direction_next = jnp.where(hits_voxel, scattered_direction, direction)
    n_scatter_next = n_scatter + hits_voxel.astype(jnp.int32)
    status_next = jnp.where(leaves_domain, KILLED, status)

    return (pos_next, direction_next, energy, n_scatter_next, key, status_next)


def run_one_photon(key, n_bounces, n_substeps, box_min, box_max, density_grid,
                    start_pos, start_direction, start_energy,
                    scattering_model="isotropic", mie_g_max=0.85, rayleigh_mie_transition_x=1.0):
    """Run a single photon for up to `n_bounces` possible scattering events.

    Wraps `photon_step` in `lax.scan`, and also records the photon's
    position after every step, so the full path is available
    afterwards for plotting -- with no need for a second, separate
    simulation just to draw a picture. `scattering_model`/`mie_g_max`/
    `rayleigh_mie_transition_x` are passed straight through to every
    `photon_step` call -- see there for what they mean.
    """
    init_state = (start_pos, start_direction, start_energy, jnp.array(0, jnp.int32), key, jnp.array(ACTIVE))

    def scan_body(state, _):
        new_state = photon_step(state, box_min, box_max, density_grid, n_substeps,
                                 scattering_model, mie_g_max, rayleigh_mie_transition_x)
        return new_state, new_state[0]  # new_state[0] is the new position

    final_state, path = lax.scan(scan_body, init_state, xs=None, length=n_bounces)
    pos, direction, energy, n_scatter, _, status = final_state
    full_path = jnp.concatenate([start_pos[None, :], path], axis=0)
    return pos, direction, energy, n_scatter, status, full_path


def _launch_state(key, beam_radius, beam_direction, beam_divergence, aim_point,
                   box_min, box_max, energy_min, energy_max):
    """Pick where a photon starts, which way it's initially headed, and how much energy it carries.

    Same idea as the previous (single hard-object) version: launch
    from a point far outside the domain and compute exactly where that
    ray first crosses the domain boundary, so the photon starts
    already sitting inside/on the box -- just using `ray_box_intersection`
    instead of a sphere formula, since the domain is a box now.
    """
    key_disk, key_mu, key_phi, key_energy = random.split(key, 4)

    y, z = sample_disk_point(key_disk, beam_radius)
    beam_t1, beam_t2 = orthonormal_basis(beam_direction)
    offset = y * beam_t1 + z * beam_t2

    mu = sample_mu_in_cone(key_mu, jnp.cos(beam_divergence))
    phi = sample_azimuth(key_phi)
    direction = direction_from_axis_mu_phi(beam_direction, mu, phi)

    launch_distance = 2.0 * jnp.linalg.norm(box_max - box_min)  # generous -- see Section 10's caveat on divergence
    far_outside = aim_point - beam_direction * launch_distance + offset
    t_entry, _, _ = ray_box_intersection(far_outside, direction, box_min, box_max)
    start_pos = far_outside + direction * t_entry

    energy = energy_min + (energy_max - energy_min) * random.uniform(key_energy)
    return start_pos, direction, energy


def _launch_geometry(key, beam_radius, beam_direction, beam_divergence, aim_point,
                     box_min, box_max):
    """Sample only a launch position and direction for one source packet.

    Source-driven simulations already have physical energies, so their launch
    step must not draw or overwrite the packet energy. This helper mirrors the
    geometric part of :func:`_launch_state` while leaving all spectral and
    temporal properties in ``SourcePackets``.
    """
    key_disk, key_mu, key_phi = random.split(key, 3)

    y, z = sample_disk_point(key_disk, beam_radius)
    beam_t1, beam_t2 = orthonormal_basis(beam_direction)
    offset = y * beam_t1 + z * beam_t2

    mu = sample_mu_in_cone(key_mu, jnp.cos(beam_divergence))
    phi = sample_azimuth(key_phi)
    direction = direction_from_axis_mu_phi(beam_direction, mu, phi)

    launch_distance = 2.0 * jnp.linalg.norm(box_max - box_min)
    far_outside = aim_point - beam_direction * launch_distance + offset
    t_entry, _, _ = ray_box_intersection(far_outside, direction, box_min, box_max)
    start_pos = far_outside + direction * t_entry
    return start_pos, direction


def simulate_photons(key, n_photons, n_bounces, n_substeps, density_grid, box_min, box_max, beam_radius,
                      beam_direction=_DEFAULT_BEAM_DIRECTION, beam_divergence=0.0, aim_point=None,
                      energy_min=1.0, energy_max=1.0,
                      scattering_model="isotropic", mie_g_max=0.85, rayleigh_mie_transition_x=1.0):
    """Run `n_photons` independent photon trajectories in parallel.

    As always, the only line that mentions `vmap` is this one: it
    turns `run_one_photon`, written for a single photon, into a
    batched function with no other code changes.

    `density_grid` (see `voxels.py`) is what makes this a scene rather
    than empty space -- an (n_voxels, n_voxels, n_voxels) array of
    scattering coefficients spanning the box from `box_min` to
    `box_max`. `n_substeps` controls the resolution of the ray-marching
    integration within each possible scattering event (Section 2 shows
    how to check it's fine enough). `beam_direction`/`beam_divergence`/
    `aim_point` control how photons are created, `energy_min`/
    `energy_max` give each one a random weight (both 1.0, the default,
    is a plain photon count).

    `scattering_model` picks the scattering angle's phase function --
    `"isotropic"` (the original, default, uniform over the sphere,
    ignores `energy`), `"rayleigh"` (exact small-grain phase function),
    `"mie"` (Henyey-Greenstein, the standard approximation for
    large-grain Mie scattering), or `"rayleigh_mie"` (each photon picks
    Rayleigh or Mie based on its own `energy`, reused as a dimensionless
    grain-size parameter, against `rayleigh_mie_transition_x`; `mie_g_max`
    caps how forward-peaked the Mie/HG branch gets). See
    `transport._sample_scattering_mu` and `sampling.py` for the actual
    math and the honest caveats about what this is (and isn't) modeling.

    IMPORTANT if this is jitted: `scattering_model` selects a phase
    function via a plain Python `if`/`elif`, not a `jnp.where` -- it
    must be passed as a **static** argument, e.g.
    `jax.jit(simulate_photons, static_argnums=(1, 2, 3), static_argnames=("scattering_model",))`.

    Returns, each with a leading `n_photons` axis: final position,
    final direction, final energy, number of scatterings, final
    status, and the full recorded path (shape (n_photons, n_bounces + 1, 3)).
    """
    if aim_point is None:
        aim_point = 0.5 * (box_min + box_max)  # the box's own center, by default
    keys = random.split(key, n_photons)

    def run(k):
        k_launch, k_photon = random.split(k)
        start_pos, start_direction, start_energy = _launch_state(
            k_launch, beam_radius, beam_direction, beam_divergence, aim_point,
            box_min, box_max, energy_min, energy_max)
        return run_one_photon(k_photon, n_bounces, n_substeps, box_min, box_max, density_grid,
                               start_pos, start_direction, start_energy,
                               scattering_model, mie_g_max, rayleigh_mie_transition_x)

    pos, direction, energy, n_scatter, status, path = vmap(run)(keys)
    return pos, direction, energy, n_scatter, status, path


def simulate_source_packets(
    key,
    packets: SourcePackets,
    n_bounces,
    n_substeps,
    density_grid,
    box_min,
    box_max,
    beam_radius,
    beam_direction=_DEFAULT_BEAM_DIRECTION,
    beam_divergence=0.0,
    aim_point=None,
    scattering_model="isotropic",
    grain_radius_um=0.1,
    mie_g_max=0.85,
    rayleigh_mie_transition_x=1.0,
):
    """Transport a pre-sampled physical source packet batch.

    This is the bridge between :mod:`utils.source` and the original JAX
    transport. Unlike :func:`simulate_photons`, it never draws an energy:
    every packet keeps its source-sampled energy in keV, emission time, and
    observer-fluence weight.

    The old toy Mie implementation expects a dimensionless grain size
    parameter rather than keV. For ``"mie"`` and ``"rayleigh_mie"`` only,
    this function converts each physical energy to
    ``2*pi*grain_radius/wavelength`` internally. The converted value is used
    solely by the phase-function placeholder and is not returned as photon
    energy.

    ``n_bounces`` and ``n_substeps`` must be static when jitted. Because the
    scattering model selects Python control flow, ``scattering_model`` and
    ``grain_radius_um`` must also be static arguments. For example::

        jax.jit(
            simulate_source_packets,
            static_argnums=(2, 3),
            static_argnames=("scattering_model", "grain_radius_um"),
        )

    Arrival-time filtering is intentionally not done here. The returned
    emission times must later be combined with a full geometric excess-path
    delay before applying an observer window.
    """
    packet_fields = (
        packets.energy_kev,
        packets.emission_time_s,
        packets.weight_observer_fluence,
        packets.time_index,
        packets.spectral_bin_index,
    )
    if any(field.ndim != 1 for field in packet_fields):
        raise ValueError("all SourcePackets fields must be one-dimensional")

    n_packets = packets.energy_kev.shape[0]
    if n_packets <= 0:
        raise ValueError("SourcePackets cannot be empty")
    if any(field.shape != (n_packets,) for field in packet_fields[1:]):
        raise ValueError("all SourcePackets fields must have the same length")
    if scattering_model not in SCATTERING_MODELS:
        raise ValueError(
            f"unknown scattering_model {scattering_model!r}, expected one of {SCATTERING_MODELS}"
        )
    if scattering_model in ("mie", "rayleigh_mie") and grain_radius_um <= 0.0:
        raise ValueError("grain_radius_um must be positive for Mie scattering")

    if aim_point is None:
        aim_point = 0.5 * (box_min + box_max)

    if scattering_model in ("mie", "rayleigh_mie"):
        phase_parameter = size_parameter_from_energy_kev(
            packets.energy_kev, grain_radius_um
        )
    else:
        phase_parameter = packets.energy_kev

    keys = random.split(key, n_packets)

    def run(k, packet_phase_parameter):
        k_launch, k_photon = random.split(k)
        start_pos, start_direction = _launch_geometry(
            k_launch,
            beam_radius,
            beam_direction,
            beam_divergence,
            aim_point,
            box_min,
            box_max,
        )
        return run_one_photon(
            k_photon,
            n_bounces,
            n_substeps,
            box_min,
            box_max,
            density_grid,
            start_pos,
            start_direction,
            packet_phase_parameter,
            scattering_model,
            mie_g_max,
            rayleigh_mie_transition_x,
        )

    position, direction, _, n_scatter, status, path = vmap(run)(
        keys, phase_parameter
    )
    return SourceTransportResult(
        position=position,
        direction=direction,
        energy_kev=packets.energy_kev,
        emission_time_s=packets.emission_time_s,
        weight_observer_fluence=packets.weight_observer_fluence,
        time_index=packets.time_index,
        spectral_bin_index=packets.spectral_bin_index,
        n_scatter=n_scatter,
        status=status,
        path=path,
    )
