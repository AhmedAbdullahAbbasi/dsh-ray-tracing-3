"""Stable three-dimensional direction-vector helpers."""

import jax.numpy as jnp


def orthonormal_basis(n):
    """Build two unit vectors (t1, t2) so that (t1, t2, n) is a right-handed orthonormal frame.

    Needed to turn a "how much toward this axis, how much around it"
    direction into an actual (x, y, z) vector -- `n` is whatever axis
    a random direction should be built relative to, and t1/t2 span the
    plane perpendicular to it. Used both for a photon's new direction
    after scattering and for sampling directions around a chosen axis.

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
