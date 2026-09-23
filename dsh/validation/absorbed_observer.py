"""Independent host references for absorbed, repeated-scattering halo fluence.

Restricted controlled experiment: a uniform 4--5 kpc spherical shell, a
10 kpc centered point source, and a compact launch cone fully inside the cloud
frustum. Reference geometry is double-precision sphere chords; observer phase
interpolation and weights are evaluated here without the production scorer.
"""

from __future__ import annotations

import math

import numpy as np

from ..geometry.coordinates import ARCSEC_TO_RAD
from .multiple_scattering import (
    INNER_KPC,
    OUTER_KPC,
    PC_TO_CM,
    SOURCE_KPC,
    analytic_shell_column,
)

DAY_S = 86_400
PC_LIGHT_S = 3.0856775814913673e18 / 299_792_45800.0


def host_material(physics, energy):
    grid = np.asarray(physics.energy_kev, dtype=np.float64)
    e = float(energy)
    if e < grid[0] or e > grid[-1]:
        raise ValueError("energy outside the material table")
    frac_index = int(np.clip(np.searchsorted(grid, e, side="right"), 1, len(grid) - 1))
    low, high = frac_index - 1, frac_index
    fraction = (math.log(e) - math.log(grid[low])) / (
        math.log(grid[high]) - math.log(grid[low])
    )
    sca = np.asarray(physics.scattering_cross_section_cm2_per_h, dtype=np.float64)
    absorption = np.asarray(
        physics.absorption_cross_section_cm2_per_h, dtype=np.float64
    )

    def interpolate_sigma(values):
        if values[low] <= 0 or values[high] <= 0:
            return float(values[low] + fraction * (values[high] - values[low]))
        return math.exp(
            math.log(values[low])
            + fraction * (math.log(values[high]) - math.log(values[low]))
        )

    return low, high, fraction, interpolate_sigma(sca), interpolate_sigma(absorption)


def host_phase(physics, energy, angle):
    """NumPy phase interpolation directly from stored intrinsic differential law."""
    low, high, f, _, _ = host_material(physics, energy)
    angles = np.asarray(physics.scattering_angle_rad, dtype=np.float64)
    alpha = np.clip(np.asarray(angle, dtype=np.float64), angles[0], angles[-1])
    index = np.clip(
        np.searchsorted(angles, alpha, side="right") - 1, 0, len(angles) - 2
    )
    x0, x1 = angles[index], angles[index + 1]
    fraction = np.where(
        x0 > 0,
        np.log(
            np.maximum(alpha, np.finfo(float).tiny)
            / np.maximum(x0, np.finfo(float).tiny)
        )
        / np.log(x1 / np.maximum(x0, np.finfo(float).tiny)),
        (alpha - x0) / (x1 - x0),
    )
    dsigma = np.asarray(
        physics.differential_cross_section_cm2_per_sr_per_h, dtype=np.float64
    )
    sigma = np.asarray(physics.scattering_cross_section_cm2_per_h, dtype=np.float64)
    result = 0.0
    for row, weight in ((low, 1 - f), (high, f)):
        p0, p1 = dsigma[row, index], dsigma[row, index + 1]
        linear = p0 + fraction * (p1 - p0)
        log_value = np.exp(
            np.log(np.maximum(p0, np.finfo(float).tiny))
            + fraction
            * (
                np.log(np.maximum(p1, np.finfo(float).tiny))
                - np.log(np.maximum(p0, np.finfo(float).tiny))
            )
        )
        result = (
            result
            + weight * np.where((p0 > 0) & (p1 > 0), log_value, linear) / sigma[row]
        )
    return result


def _entry_distance(direction, radius_pc):
    source_pc = SOURCE_KPC * 1000.0
    dot = source_pc * direction[..., 0]
    discr = dot * dot - (source_pc**2 - radius_pc**2)
    return -dot - np.sqrt(np.maximum(discr, 0.0))


def first_order_quadrature(
    physics, energy, central_column_cm2, bounds, time_edges_s, n_slope=72, n_depth=32
):
    """Absolute first-order time-bin fluence for unit observer-equivalent source fluence.

    Gauss--Legendre integration is in the two tangent slopes and flight length;
    no production ray traversal, binning, launch sampler, or scorer is used.
    """
    _, _, _, sigma_sca, sigma_abs = host_material(physics, energy)
    roots, weights = np.polynomial.legendre.leggauss(n_slope)
    u = (bounds[0][0] + bounds[0][1]) / 2 + roots * (bounds[0][1] - bounds[0][0]) / 2
    v = (bounds[1][0] + bounds[1][1]) / 2 + roots * (bounds[1][1] - bounds[1][0]) / 2
    uu, vv = np.meshgrid(u, v, indexing="ij")
    jacobian = (1 + uu**2 + vv**2) ** -1.5
    d = np.stack((-np.ones_like(uu), uu, vv), axis=-1) * np.sqrt(jacobian)[
        ..., None
    ] ** (2 / 3)
    entry = _entry_distance(d, OUTER_KPC * 1000.0)
    exit_shell = _entry_distance(d, INNER_KPC * 1000.0)
    source = np.array([SOURCE_KPC * 1000.0, 0, 0])
    tnodes, tweights = np.polynomial.legendre.leggauss(n_depth)
    density = central_column_cm2 / (1000.0 * PC_TO_CM)
    solid_angle = weights[:, None, None] * weights[None, :, None]
    solid_angle *= (bounds[0][1] - bounds[0][0]) * (bounds[1][1] - bounds[1][0]) / 4

    def delay_at(path):
        point = source + path[..., None] * d
        return (
            path + np.linalg.norm(point, axis=-1) - SOURCE_KPC * 1000.0
        ) * PC_LIGHT_S

    def path_at_delay(target):
        lower, upper = entry.copy(), exit_shell.copy()
        for _ in range(42):
            mid = (lower + upper) * 0.5
            too_early = delay_at(mid) < target
            lower = np.where(too_early, mid, lower)
            upper = np.where(too_early, upper, mid)
        return np.where(
            target <= delay_at(entry),
            entry,
            np.where(target >= delay_at(exit_shell), exit_shell, (lower + upper) * 0.5),
        )

    boundaries = [path_at_delay(t) for t in time_edges_s]
    totals = []
    for beginning, ending in zip(boundaries[:-1], boundaries[1:], strict=True):
        path = beginning[..., None] + (tnodes + 1) * (ending - beginning)[..., None] / 2
        pos = source + path[..., None] * d[..., None, :]
        radius = np.linalg.norm(pos, axis=-1)
        toward_observer = -pos / radius[..., None]
        cosine = np.sum(d[..., None, :] * toward_observer, axis=-1)
        sine = np.linalg.norm(np.cross(d[..., None, :], toward_observer), axis=-1)
        phase = host_phase(physics, energy, np.arctan2(sine, cosine))
        incoming = density * (path - entry[..., None]) * PC_TO_CM
        outgoing = density * (radius - INNER_KPC * 1000.0) * PC_TO_CM
        contribution = (
            solid_angle
            * tweights[None, None, :]
            * (ending - beginning)[..., None]
            / 2
            * jacobian[..., None]
            * density
            * PC_TO_CM
            * sigma_sca
            * (SOURCE_KPC * 1000.0 / radius) ** 2
            * phase
            * np.exp(-(sigma_sca + sigma_abs) * (incoming + outgoing))
        )
        totals.append(float(np.sum(contribution, dtype=np.float64)))
    return np.asarray(totals)


def first_order_radial_quadrature(
    physics, energy, central_column_cm2, bounds, time_edges_s, n_radius=48, n_depth=32
):
    """First-order shell fluence, integrating the symmetric launch cone by radius.

    For a centered rectangular cone the shell, source, and observer are
    rotationally symmetric. At each slope radius we integrate the exactly
    accessible azimuth of the rectangle. Radial intervals are split wherever
    a time-bin edge crosses either shell surface, so narrow time bins cannot
    be smeared across a discontinuity in the angular integration.
    """
    if (
        len(bounds) != 2
        or any(len(pair) != 2 or pair[0] != -pair[1] or pair[1] <= 0 for pair in bounds)
        or n_radius < 2
        or n_depth < 2
    ):
        raise ValueError("radial quadrature requires a centered rectangular cone")
    edges = np.asarray(time_edges_s, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("time edges must increase")

    a, b = float(bounds[0][1]), float(bounds[1][1])
    r_max = math.hypot(a, b)
    source = np.array([SOURCE_KPC * 1000.0, 0.0, 0.0])
    density = central_column_cm2 / (1000.0 * PC_TO_CM)
    _, _, _, sigma_sca, sigma_abs = host_material(physics, energy)

    def direction(radius):
        radius = np.asarray(radius)
        return (
            np.stack((-np.ones_like(radius), radius, np.zeros_like(radius)), axis=-1)
            / np.sqrt(1 + radius**2)[..., None]
        )

    def delay_at(path, direction_vector):
        point = source + path[..., None] * direction_vector
        return (path + np.linalg.norm(point, axis=-1) - source[0]) * PC_LIGHT_S

    # Each surface's arrival delay grows monotonically with slope radius.
    # Split at both surfaces because the allowed flight-length interval changes
    # its formula at these radii for every nontrivial time boundary.
    radial_breaks = [0.0, a, b, r_max]
    for shell_radius in (OUTER_KPC * 1000.0, INNER_KPC * 1000.0):
        max_delay = delay_at(
            _entry_distance(direction(r_max), shell_radius), direction(r_max)
        )
        for edge in edges[1:-1]:
            if not 0.0 < edge < max_delay:
                continue
            low, high = 0.0, r_max
            for _ in range(52):
                middle = (low + high) * 0.5
                d_middle = direction(middle)
                delay = delay_at(_entry_distance(d_middle, shell_radius), d_middle)
                if delay < edge:
                    low = middle
                else:
                    high = middle
            radial_breaks.append((low + high) * 0.5)

    roots, weights = np.polynomial.legendre.leggauss(n_radius)
    segments = np.unique(radial_breaks)
    radius = np.concatenate(
        [
            (lo + hi) / 2 + (hi - lo) / 2 * roots
            for lo, hi in zip(segments[:-1], segments[1:], strict=True)
        ]
    )
    radial_weight = np.concatenate(
        [
            (hi - lo) / 2 * weights
            for lo, hi in zip(segments[:-1], segments[1:], strict=True)
        ]
    )
    lower_angle = np.arccos(np.minimum(a / radius, 1.0))
    upper_angle = np.arcsin(np.minimum(b / radius, 1.0))
    azimuth_width = 4 * np.maximum(upper_angle - lower_angle, 0.0)
    solid_angle = radial_weight * radius * azimuth_width / (1 + radius**2) ** 1.5

    d = direction(radius)
    entry = _entry_distance(d, OUTER_KPC * 1000.0)
    exit_shell = _entry_distance(d, INNER_KPC * 1000.0)

    def path_at_delay(target):
        low, high = entry.copy(), exit_shell.copy()
        for _ in range(52):
            middle = (low + high) * 0.5
            too_early = delay_at(middle, d) < target
            low = np.where(too_early, middle, low)
            high = np.where(too_early, high, middle)
        return np.where(
            target <= delay_at(entry, d),
            entry,
            np.where(target >= delay_at(exit_shell, d), exit_shell, (low + high) * 0.5),
        )

    boundaries = [path_at_delay(t) for t in edges]
    depth_roots, depth_weights = np.polynomial.legendre.leggauss(n_depth)
    totals = []
    for beginning, ending in zip(boundaries[:-1], boundaries[1:], strict=True):
        length = (ending - beginning)[:, None] / 2
        path = beginning[:, None] + (depth_roots[None, :] + 1) * length
        pos = source + path[..., None] * d[:, None, :]
        distance = np.linalg.norm(pos, axis=-1)
        toward_observer = -pos / distance[..., None]
        cosine = np.sum(d[:, None, :] * toward_observer, axis=-1)
        sine = np.linalg.norm(np.cross(d[:, None, :], toward_observer), axis=-1)
        phase = host_phase(physics, energy, np.arctan2(sine, cosine))
        incoming = density * (path - entry[:, None]) * PC_TO_CM
        outgoing = density * (distance - INNER_KPC * 1000.0) * PC_TO_CM
        integrand = (
            solid_angle[:, None]
            * depth_weights[None, :]
            * length
            * density
            * PC_TO_CM
            * sigma_sca
            * (source[0] / distance) ** 2
            * phase
            * np.exp(-(sigma_sca + sigma_abs) * (incoming + outgoing))
        )
        totals.append(float(np.sum(integrand, dtype=np.float64)))
    return np.asarray(totals)


def score_scattering_only_histories(
    launched,
    transported,
    physics,
    energy,
    central_column_cm2,
    time_edges_s,
    sky_x_edges_arcsec,
    sky_y_edges_arcsec,
):
    """Host peel-off estimator with accumulated deterministic absorption.

    Each photon is grouped before second moments are accumulated; the
    underlying scattering-only transport samples only scattering optical depth.
    """
    # A scalar preserves the monoenergetic benchmark; a vector permits a
    # continuous spectrum with an independent material lookup per photon.
    scalar_energy = np.asarray(energy, dtype=np.float64).ndim == 0
    energies = np.broadcast_to(
        np.asarray(energy, dtype=np.float64),
        (len(launched.launch_pdf_per_sr),),
    )
    if not np.all(np.isfinite(energies)):
        raise ValueError("reference photon energies must be finite")
    material = host_material(physics, float(energies[0])) if scalar_energy else None
    positions = np.asarray(transported.interactions.position_pc, dtype=np.float64)
    momenta = np.asarray(
        transported.interactions.incoming_momentum_kev, dtype=np.float64
    )
    valid = np.asarray(transported.interactions.valid)
    kind = np.asarray(transported.interactions.interaction_type)
    order = np.asarray(transported.interactions.scattering_order)
    q = np.asarray(launched.launch_pdf_per_sr, dtype=np.float64)
    weights = np.asarray(launched.weight_observer_fluence, dtype=np.float64)
    n = len(q)
    per_packet = np.zeros((n, len(time_edges_s) - 1, 3), dtype=np.float64)
    start_source = np.array([SOURCE_KPC * 1000.0, 0.0, 0.0])
    for photon in range(n):
        photon_energy = float(energies[photon])
        _, _, _, sigma_sca, sigma_abs = (
            material if material is not None else host_material(physics, photon_energy)
        )
        previous = start_source
        traveled = 0.0
        incoming_column = 0.0
        for j in range(valid.shape[1]):
            if not valid[photon, j]:
                break
            if kind[photon, j] != 1:  # DUST_SCATTERING
                raise ValueError("scattering-only reference recorded a non-scatter")
            point = positions[photon, j]
            segment = point - previous
            length = float(np.linalg.norm(segment))
            incoming_column += analytic_shell_column(
                previous, segment / length, length, central_column_cm2
            )
            traveled += length
            radius = float(np.linalg.norm(point))
            obsdir = -point / radius
            sky_x = math.atan2(point[1], point[0]) / ARCSEC_TO_RAD
            sky_y = math.atan2(point[2], point[0]) / ARCSEC_TO_RAD
            inside_sky = (
                sky_x_edges_arcsec[0] <= sky_x <= sky_x_edges_arcsec[-1]
                and sky_y_edges_arcsec[0] <= sky_y <= sky_y_edges_arcsec[-1]
            )
            indir = momenta[photon, j, 1:]
            indir /= np.linalg.norm(indir)
            phase = float(
                host_phase(
                    physics,
                    photon_energy,
                    math.atan2(
                        np.linalg.norm(np.cross(indir, obsdir)), np.dot(indir, obsdir)
                    ),
                )
            )
            escape = analytic_shell_column(point, obsdir, radius, central_column_cm2)
            delay = (traveled + radius - SOURCE_KPC * 1000.0) * PC_LIGHT_S
            tbin = int(np.searchsorted(time_edges_s, delay, side="right") - 1)
            if inside_sky and 0 <= tbin < per_packet.shape[1]:
                w = (
                    weights[photon]
                    * (SOURCE_KPC * 1000.0 / radius) ** 2
                    * phase
                    / q[photon]
                    * math.exp(-sigma_sca * escape)
                    * math.exp(-sigma_abs * (incoming_column + escape))
                )
                per_packet[photon, tbin, min(int(order[photon, j]) - 1, 2)] += w
            previous = point
    flattened = per_packet.reshape(n, -1)
    return {
        "sum": per_packet.sum(axis=0),
        "cross": flattened.T @ flattened,
        "histories": n,
        "statuses": np.bincount(np.asarray(transported.status), minlength=7),
    }
