"""Independent double-precision references for the Stage 9F asymmetric scene.

The on-axis source, quadrant boundaries and finite launch cone make the
incoming first-scatter flight stay in one sky quadrant.  Its intersections
with radial shells and the radial observer column can then be integrated
analytically.  Repeated-scattering flights can cross sky cells: their host
column reference clips each segment to the cloud's physical boundaries using
NumPy, without calling the production JAX ray integrator or observer scorer.
"""

from __future__ import annotations

import math

import numpy as np

ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)
PC_TO_CM = 3.0856775814913673e18
PC_LIGHT_S = PC_TO_CM / 299_792_45800.0
SOURCE_PC = 10_000.0
SKY_EDGES = np.array([-1800.0, 0.0, 1800.0])
RADIAL_EDGES_KPC = np.arange(2.0, 9.0)
LAUNCH_BOUNDS = ((-0.0012, 0.0012), (-0.0012, 0.0012))
TIME_EDGES_S = (0.0, 365.0 * 86400.0)


def scene_columns(central_column_cm2):
    """Three separated asymmetric slabs, with vacuum radial cells between."""
    if not math.isfinite(central_column_cm2) or central_column_cm2 <= 0:
        raise ValueError("central column must be positive and finite")
    columns = np.zeros((6, 2, 2), dtype=np.float64)
    columns[0] = central_column_cm2 * np.array([[0.28, 0.12], [0.16, 0.35]])
    columns[2] = central_column_cm2 * np.array([[0.65, 0.90], [1.25, 1.60]])
    columns[4] = central_column_cm2 * np.array([[0.12, 0.20], [0.24, 0.10]])
    return columns


def _sphere_entry(direction, radius_pc):
    projection = SOURCE_PC * direction[..., 0]
    return -projection - np.sqrt(
        np.maximum(projection**2 - (SOURCE_PC**2 - radius_pc**2), 0.0)
    )


def _radial_column(position, columns):
    """Observer column on a fixed sky sightline by shell overlap."""
    radius_kpc = np.linalg.norm(position, axis=-1) / 1000.0
    widths = np.diff(RADIAL_EDGES_KPC)
    overlap = np.clip(
        radius_kpc[..., None] - RADIAL_EDGES_KPC[:-1], 0.0, widths
    ) / widths
    return np.sum(overlap * columns, axis=-1)


def first_order_quadrants(physics, energy, columns, *, n_slope, n_depth):
    """Absolute first-order fluence per (sky_y, sky_x) quadrant.

    Independent Gauss--Legendre integration in the two launch slopes and in
    each occupied shell.  The slope domains are split exactly at sky x/y=0,
    and the depth domain is split at every occupied radial-shell surface.
    """
    from .absorbed_observer import host_material, host_phase

    if columns.shape != (6, 2, 2) or min(n_slope, n_depth) < 2:
        raise ValueError("invalid scene or quadrature order")
    _, _, _, sigma_sca, sigma_abs = host_material(physics, energy)
    roots, weights = np.polynomial.legendre.leggauss(n_slope)
    depth_roots, depth_weights = np.polynomial.legendre.leggauss(n_depth)
    result = np.zeros((2, 2), dtype=np.float64)
    for iy in range(2):
        for ix in range(2):
            u0, u1 = (
                (LAUNCH_BOUNDS[0][0], 0.0) if ix == 0 else (0.0, LAUNCH_BOUNDS[0][1])
            )
            v0, v1 = (
                (LAUNCH_BOUNDS[1][0], 0.0) if iy == 0 else (0.0, LAUNCH_BOUNDS[1][1])
            )
            u = (u0 + u1) / 2 + (u1 - u0) * roots / 2
            v = (v0 + v1) / 2 + (v1 - v0) * roots / 2
            uu, vv = np.meshgrid(u, v, indexing="xy")
            direction = np.stack((-np.ones_like(uu), uu, vv), axis=-1)
            direction /= np.linalg.norm(direction, axis=-1)[..., None]
            jacobian = (1 + uu**2 + vv**2) ** -1.5
            path_edges = np.stack(
                [_sphere_entry(direction, r * 1000.0) for r in RADIAL_EDGES_KPC]
            )
            density = columns[:, iy, ix] / (
                np.diff(RADIAL_EDGES_KPC) * 1000.0 * PC_TO_CM
            )
            slope_weight = weights[None, :] * weights[:, None] * (
                (u1 - u0) * (v1 - v0) / 4
            )
            for shell in np.flatnonzero(density):
                start = path_edges[shell + 1]
                stop = path_edges[shell]
                length = (stop - start) / 2
                path = start[..., None] + (depth_roots + 1) * length[..., None]
                point = (np.array([SOURCE_PC, 0.0, 0.0])
                         + path[..., None] * direction[..., None, :])
                radius = np.linalg.norm(point, axis=-1)
                observer = -point / radius[..., None]
                incoming_direction = direction[..., None, :]
                angle = np.arctan2(
                    np.linalg.norm(np.cross(incoming_direction, observer), axis=-1),
                    np.sum(incoming_direction * observer, axis=-1),
                )
                phase = host_phase(physics, energy, angle)
                incoming = np.zeros_like(path)
                for previous in np.flatnonzero(density):
                    traveled = np.clip(
                        path - path_edges[previous + 1][..., None],
                        0.0,
                        (path_edges[previous] - path_edges[previous + 1])[..., None],
                    )
                    incoming += density[previous] * traveled * PC_TO_CM
                outgoing = _radial_column(point, columns[:, iy, ix])
                delay = (path + radius - SOURCE_PC) * PC_LIGHT_S
                integrand = (
                    slope_weight[..., None]
                    * jacobian[..., None]
                    * length[..., None]
                    * depth_weights
                    * density[shell] * PC_TO_CM * sigma_sca
                    * (SOURCE_PC / radius) ** 2
                    * phase
                    * np.exp(-(sigma_sca + sigma_abs) * (incoming + outgoing))
                )
                result[iy, ix] += float(np.sum(
                    np.where((delay >= TIME_EDGES_S[0]) &
                             (delay <= TIME_EDGES_S[-1]), integrand, 0.0),
                    dtype=np.float64,
                ))
    return result


def host_ray_column(origin_pc, direction, distance_pc, columns):
    """Double-precision voxel integral for arbitrarily directed actual flights.

    Intersect the segment with each spherical and angular cell surface; sample
    the density between consecutive crossings.  This host implementation is
    independent of ``dsh.geometry.rays`` and works on the input column cube.
    """
    origin = np.asarray(origin_pc, dtype=np.float64)
    ray = np.array(direction, dtype=np.float64, copy=True)
    columns = np.asarray(columns, dtype=np.float64)
    if (origin.shape != (3,) or ray.shape != (3,) or columns.shape != (6, 2, 2)
            or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(ray))
            or not math.isfinite(distance_pc) or distance_pc < 0):
        raise ValueError("invalid ray or column cube")
    norm = float(np.linalg.norm(ray))
    if norm == 0:
        raise ValueError("ray direction must be nonzero")
    ray /= norm
    cuts = [0.0, float(distance_pc)]
    projection = float(np.dot(origin, ray))
    for radial_edge in RADIAL_EDGES_KPC * 1000.0:
        discriminant = projection**2 - (float(np.dot(origin, origin)) - radial_edge**2)
        if discriminant >= 0:
            root = math.sqrt(discriminant)
            cuts.extend((-projection - root, -projection + root))
    for axis in (1, 2):
        for sky_edge in SKY_EDGES:
            slope = math.tan(sky_edge * ARCSEC_TO_RAD)
            denominator = ray[axis] - slope * ray[0]
            if abs(denominator) > 1e-20:
                cuts.append((slope * origin[0] - origin[axis]) / denominator)
    inside = sorted(c for c in cuts if math.isfinite(c) and 0 < c < distance_pc)
    boundaries = [0.0, *inside, float(distance_pc)]
    total = 0.0
    for a, b in zip(boundaries[:-1], boundaries[1:], strict=True):
        if b <= a:
            continue
        point = origin + (a + b) * 0.5 * ray
        radius = float(np.linalg.norm(point) / 1000.0)
        if point[0] <= 0:
            continue
        x = math.atan2(point[1], point[0]) / ARCSEC_TO_RAD
        y = math.atan2(point[2], point[0]) / ARCSEC_TO_RAD
        iz = int(np.searchsorted(RADIAL_EDGES_KPC, radius, side="right") - 1)
        ix = int(np.searchsorted(SKY_EDGES, x, side="right") - 1)
        iy = int(np.searchsorted(SKY_EDGES, y, side="right") - 1)
        if 0 <= iz < 6 and 0 <= iy < 2 and 0 <= ix < 2:
            total += columns[iz, iy, ix] * (b - a) / (
                1000.0 * (RADIAL_EDGES_KPC[iz + 1] - RADIAL_EDGES_KPC[iz])
            )
    return total


def reference_history_weights(launched, transported, physics, energy, columns):
    """Scattering-only paths with explicit absorption on all previous flights.

    Returns per-launched-photon scores of shape (n, y=2, x=2, order=3).
    """
    from .absorbed_observer import host_material

    low, high, fraction, sigma_sca, sigma_abs = host_material(physics, energy)
    angle_grid = np.asarray(physics.scattering_angle_rad, dtype=np.float64)
    phase_rows = np.asarray(
        physics.differential_cross_section_cm2_per_sr_per_h
    )[[low, high]].astype(np.float64)
    scattering_rows = np.asarray(
        physics.scattering_cross_section_cm2_per_h
    )[[low, high]].astype(np.float64)

    def phase_at(angle):
        alpha = np.clip(angle, angle_grid[0], angle_grid[-1])
        index = int(np.clip(np.searchsorted(angle_grid, alpha, side="right") - 1,
                            0, len(angle_grid) - 2))
        x0, x1 = angle_grid[index:index + 2]
        if x0 > 0:
            t = math.log(max(alpha, np.finfo(float).tiny) / x0) / math.log(x1 / x0)
        else:
            t = (alpha - x0) / (x1 - x0)
        terms = []
        for row, sigma in zip(phase_rows, scattering_rows, strict=True):
            p0, p1 = row[index:index + 2]
            value = (math.exp(math.log(p0) + t * math.log(p1 / p0))
                     if p0 > 0 and p1 > 0 else p0 + t * (p1 - p0))
            terms.append(value / sigma)
        return (1 - fraction) * terms[0] + fraction * terms[1]
    positions = np.asarray(transported.interactions.position_pc, dtype=np.float64)
    valid = np.asarray(transported.interactions.valid)
    kinds = np.asarray(transported.interactions.interaction_type)
    orders = np.asarray(transported.interactions.scattering_order)
    momentum = np.asarray(
        transported.interactions.incoming_momentum_kev, dtype=np.float64
    )
    start = np.asarray(launched.position_pc, dtype=np.float64)
    pdf = np.asarray(launched.launch_pdf_per_sr, dtype=np.float64)
    weight = np.asarray(launched.weight_observer_fluence, dtype=np.float64)
    out = np.zeros((len(valid), 2, 2, 3), dtype=np.float64)
    for i in range(len(valid)):
        previous = start[i]
        path = 0.0
        incoming_column = 0.0
        for j in range(valid.shape[1]):
            if not valid[i, j]:
                break
            if kinds[i, j] != 1:
                raise ValueError("scattering-only history contains absorption")
            point = positions[i, j]
            segment = point - previous
            length = float(np.linalg.norm(segment))
            if length <= 0:
                raise ValueError("zero-length physical scattering flight")
            incoming_column += host_ray_column(previous, segment, length, columns)
            path += length
            radius = float(np.linalg.norm(point))
            observer = -point / radius
            x = math.atan2(point[1], point[0]) / ARCSEC_TO_RAD
            y = math.atan2(point[2], point[0]) / ARCSEC_TO_RAD
            delay = (path + radius - SOURCE_PC) * PC_LIGHT_S
            ix = int(np.searchsorted(SKY_EDGES, x, side="right") - 1)
            iy = int(np.searchsorted(SKY_EDGES, y, side="right") - 1)
            if (
                0 <= ix < 2
                and 0 <= iy < 2
                and TIME_EDGES_S[0] <= delay <= TIME_EDGES_S[-1]
            ):
                incoming = momentum[i, j, 1:]
                incoming /= np.linalg.norm(incoming)
                angle = math.atan2(
                    float(np.linalg.norm(np.cross(incoming, observer))),
                    float(np.dot(incoming, observer)),
                )
                phase = phase_at(angle)
                escape = host_ray_column(point, observer, radius, columns)
                out[i, iy, ix, min(int(orders[i, j]) - 1, 2)] += (
                    weight[i] * (SOURCE_PC / radius) ** 2 * phase / pdf[i]
                    * math.exp(-sigma_sca * escape)
                    * math.exp(-sigma_abs * (incoming_column + escape))
                )
            previous = point
    return out
