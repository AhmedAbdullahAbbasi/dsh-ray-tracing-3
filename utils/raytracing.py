"""Pure geometry: ray-box intersection, and turning angles into 3D vectors.

The simulation domain is now an axis-aligned box -- the extent of the
voxel grid (see `voxels.py`) -- rather than a sphere with one hard
object in it, so the geometry needed is a ray-box intersection instead
of a ray-sphere one. Everything a photon can hit is now the voxel grid
itself, cell by cell (see `transport.photon_step`); this module only
answers "where does this ray cross the domain's outer boundary?" and
"how do I turn a (cosine, azimuth) pair into an actual direction?".
"""

import jax.numpy as jnp
from jax import random


def ray_box_intersection(origin, direction, box_min, box_max):
    """Solve for where a ray enters and exits an axis-aligned box.

    The standard "slab method": treat the box as the intersection of
    three axis-aligned slabs (one pair of parallel planes per axis),
    find where the ray crosses each pair, and combine. `direction`
    must be a unit vector (any nonzero vector actually works here, unlike
    the sphere formula, but the rest of this simulation assumes unit
    vectors throughout).

    Returns (t_near, t_far, hits): `t_near` is where the ray enters the
    box, `t_far` where it exits (both can be negative -- an
    intersection "behind" the ray's origin); `hits` is whether the ray
    crosses the box at all. Relies on IEEE division-by-zero producing
    +/-inf (not a JAX quirk -- ordinary float semantics) for a
    direction component of exactly 0, which correctly means "this axis
    never constrains the intersection".
    """
    inv_direction = 1.0 / direction
    t1 = (box_min - origin) * inv_direction
    t2 = (box_max - origin) * inv_direction
    t_near = jnp.max(jnp.minimum(t1, t2))
    t_far = jnp.min(jnp.maximum(t1, t2))
    hits = t_near <= t_far
    return t_near, t_far, hits


def orthonormal_basis(n):
    """Build two unit vectors (t1, t2) so that (t1, t2, n) is a right-handed orthonormal frame.

    Needed to turn a "how much toward this axis, how much around it"
    direction into an actual (x, y, z) vector -- `n` is whatever axis
    a random direction should be built relative to, and t1/t2 span the
    plane perpendicular to it. Used both for a photon's new direction
    after scattering (any fixed axis works -- see `sample_isotropic_mu`)
    and for spreading an injected beam into a cone (`n` = the beam's
    central direction).

    Uses the branchless construction from Duff, Burgess, Christensen,
    Hery, Kensler, Liani & Villemin, "Building an Orthonormal Basis,
    Revisited" (2017) -- ordinary "pick an arbitrary axis and cross
    it with n" recipes have a singularity as n approaches that axis,
    which is fatal once this runs under `jax.vmap` across many
    different axes at once.
    """
    sign = jnp.where(n[2] >= 0.0, 1.0, -1.0)
    a = -1.0 / (sign + n[2])
    b = n[0] * n[1] * a
    t1 = jnp.array([1.0 + sign * n[0] * n[0] * a, sign * b, -sign * n[0]])
    t2 = jnp.array([b, sign + n[1] * n[1] * a, -n[1]])
    return t1, t2


def direction_from_axis_mu_phi(axis, mu, phi):
    """Build a unit direction at cosine `mu` from `axis`, at azimuth `phi` around it.

    This is the one formula behind two different-looking things in
    this simulation: a photon's new direction after scattering off the
    medium (any fixed `axis` works for isotropic scattering -- see
    `sampling.sample_isotropic_mu`) and a photon being launched with
    some beam divergence (`axis` = the beam's central direction). Both
    are just "pick a direction some angle away from a reference axis".
    """
    t1, t2 = orthonormal_basis(axis)
    sin_theta = jnp.sqrt(jnp.maximum(0.0, 1.0 - mu * mu))
    return sin_theta * jnp.cos(phi) * t1 + sin_theta * jnp.sin(phi) * t2 + mu * axis


def sample_disk_point(key, radius):
    """Draw a point (y, z) uniformly (by area) inside a disk of the given radius.

    Used once per photon to place it somewhere in the cross-section of
    the incoming beam. Sampling r as radius*sqrt(xi) (rather than just
    radius*xi) is what makes the distribution uniform *by area* instead
    of bunching points up near the center -- area within radius r grows
    as r^2, so its CDF needs the square root to invert.
    """
    key_r, key_theta = random.split(key)
    r = radius * jnp.sqrt(random.uniform(key_r))
    theta = 2.0 * jnp.pi * random.uniform(key_theta)
    return r * jnp.cos(theta), r * jnp.sin(theta)
