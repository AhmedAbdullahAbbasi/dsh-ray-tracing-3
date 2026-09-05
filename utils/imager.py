"""Turning photon trajectories into an image on a virtual detector screen.

Every other plot in this project draws the simulation from the outside (a 3D
scene, a histogram of scattering counts). This is the one place that asks the
question a real telescope actually asks: of all the photons that were fired,
which ones would have landed on a detector sitting somewhere downstream, and
where on it?

The detector is a square, centered at (x, y, z) = (`screen_x`, 0, 0) and lying
in the y-z plane (i.e. perpendicular to the x axis) -- `side_width` (its full
side length) is the one thing you have to supply; everything else about the
scene comes from `simulate_photons`'s own output.

Two things `path` alone doesn't give us, which is why `direction` and `status`
are needed too:

- `path` only records positions *inside* the simulation box. If the screen
  sits beyond the box (the usual case -- see the notebook's dust-scattering-
  halo section, where the box is a thin slab around x=0 and the screen is at
  x=10), a killed photon's last recorded position is just where it crossed the
  box boundary, not where it hits the screen -- its `direction` at that point
  is what carries it the rest of the way, in a straight line through the empty
  space this simulation doesn't otherwise model.
- A photon that's still `ACTIVE` (ran out of `n_bounces` without leaving the
  box) has no defined onward path at all, so it can't be placed on the screen
  and is simply excluded.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .transport import ACTIVE


@jax.jit
def compute_screen_hits(path, direction, status, screen_x=10.0):
    """Find where each photon's real trajectory crosses the plane x = `screen_x`, if it does.

    Runs entirely on-device in JAX (`path`/`direction`/`status` are exactly
    `simulate_photons`'s own outputs -- no `np.asarray` copy back to the CPU
    needed before this runs, unlike a plain NumPy version would need), and
    fully vectorized over *both* photons and bounces at once, with no Python
    loop: every recorded segment of `path` (shape (n_photons, n_bounces+1, 3))
    is checked for a plane crossing simultaneously (an (n_photons, n_bounces)
    boolean array), and each photon's *first* crossing is picked out with one
    `argmax` along the bounce axis (an `argmax` over booleans returns the
    index of the first `True`) -- this covers a screen sitting inside the
    simulation box.

    For a screen beyond the box (the usual case), a photon that never crosses
    the plane while still inside gets one more, separately-computed segment
    checked: its last recorded position, extended forward along its final
    `direction` (only killed photons have a meaningful "final direction" to
    extend -- still-`ACTIVE` photons are excluded entirely).

    Returns an (n_photons, 2) array of (y, z) hit coordinates, with `nan` rows
    for photons that never reach the screen at all (e.g. still active, or
    moving away from/parallel to it).
    """
    p0, p1 = path[:, :-1, :], path[:, 1:, :]  # every segment, all photons and bounces at once
    x0, x1 = p0[:, :, 0], p1[:, :, 0]
    denom = jnp.where(x1 != x0, x1 - x0, 1.0)  # dodge the x0==x1 division; masked out by `crosses` below
    frac = (screen_x - x0) / denom
    crosses = (x1 != x0) & (frac >= 0.0) & (frac <= 1.0)  # (n_photons, n_bounces)
    seg_y = p0[:, :, 1] + frac * (p1[:, :, 1] - p0[:, :, 1])
    seg_z = p0[:, :, 2] + frac * (p1[:, :, 2] - p0[:, :, 2])

    any_cross = jnp.any(crosses, axis=1)
    first_idx = jnp.argmax(crosses, axis=1)  # first True per photon; meaningless where any_cross is False
    hit_y_inside = jnp.take_along_axis(seg_y, first_idx[:, None], axis=1)[:, 0]
    hit_z_inside = jnp.take_along_axis(seg_z, first_idx[:, None], axis=1)[:, 0]

    # Photons that never crossed the plane while inside the box: if they were
    # killed (i.e. actually left the domain), extend their last position
    # forward along their final direction -- the straight line they'd
    # continue along through the empty space outside the box.
    last_pos = path[:, -1, :]
    dx = jnp.where(direction[:, 0] != 0.0, direction[:, 0], 1.0)
    t = (screen_x - last_pos[:, 0]) / dx
    reaches = ~any_cross & (status != ACTIVE) & (direction[:, 0] != 0.0) & (t > 0.0)
    hit_y_ext = last_pos[:, 1] + t * direction[:, 1]
    hit_z_ext = last_pos[:, 2] + t * direction[:, 2]

    hit_y = jnp.where(any_cross, hit_y_inside, jnp.where(reaches, hit_y_ext, jnp.nan))
    hit_z = jnp.where(any_cross, hit_z_inside, jnp.where(reaches, hit_z_ext, jnp.nan))
    return jnp.stack([hit_y, hit_z], axis=1)


def render(path, direction, status, side_width, screen_x=10.0, bins=200, ax=None, cmap="inferno",
           log_scale=True):
    """Image the photons that land inside a square detector on the x = `screen_x` plane.

    The square is centered at (x, y, z) = (`screen_x`, 0, 0), `side_width` on
    a side, lying in the y-z plane. Builds a 2D histogram of hit positions
    (`compute_screen_hits`) restricted to that square and draws it as an
    image -- the closest thing in this project to what a real detector/CCD
    would record.

    `log_scale=True` (the default) color-scales bin counts logarithmically,
    with empty bins left transparent (`cmin=1`) rather than log(0): a
    realistic halo is many orders of magnitude fainter than the unscattered
    core, so a plain linear scale shows nothing but a single bright dot at
    the center -- exactly what a real telescope image of a faint dust halo
    around a bright point source needs log stretching to reveal, too.

    Returns (ax, n_hit, n_total): the axes drawn on, how many photons landed
    inside the square, and how many were fired in total (`path.shape[0]`) --
    a small, faint `n_hit/n_total` is exactly what a real, faint scattering
    halo looks like.
    """
    # compute_screen_hits runs entirely on-device; only its small (n_photons, 2)
    # result needs to come back to the CPU for matplotlib, not the full path array.
    hits = np.asarray(compute_screen_hits(path, direction, status, screen_x=screen_x))
    half = side_width / 2.0
    inside = (
        np.isfinite(hits[:, 0]) & np.isfinite(hits[:, 1])
        & (np.abs(hits[:, 0]) <= half) & (np.abs(hits[:, 1]) <= half)
    )
    y_hits, z_hits = hits[inside, 0], hits[inside, 1]

    ax = ax or plt.gca()
    norm = LogNorm() if log_scale else None
    ax.hist2d(y_hits, z_hits, bins=bins, range=[[-half, half], [-half, half]],
              cmap=cmap, cmin=1, norm=norm)
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_aspect("equal")
    ax.set_title(f"Detector image at x={screen_x} (side {side_width}): "
                 f"{inside.sum()}/{path.shape[0]} photons")
    return ax, int(inside.sum()), int(path.shape[0])
