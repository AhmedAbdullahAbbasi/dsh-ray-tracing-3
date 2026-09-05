"""The voxel grid: what an "object" is made of in this simulation.

There's no single hard-surfaced shape anymore -- the whole simulation
domain is one axis-aligned box, chopped into a regular grid of
`n_voxels` x `n_voxels` x `n_voxels` cubes ("voxels"), each carrying
its own scalar `density` (a local scattering coefficient, in units of
1/length: how much optical depth a photon accumulates per unit
distance travelled through that cell). An "object" is just wherever
those densities are nonzero -- a solid grain, a diffuse cloud, empty
vacuum, anything a density field can represent, all with exactly the
same photon physics in `transport.py`.

The density grid itself is built once, up front, in plain Python/numpy
(there's no need for it to be a JAX-traced computation -- it's data,
not part of the per-photon simulation), then handed to
`transport.simulate_photons` as an ordinary array.
"""

import numpy as np
import jax.numpy as jnp


def voxel_size_of(box_min, box_max, n_voxels):
    """The physical size of one voxel along each axis (can differ per axis if the box isn't a cube)."""
    return (box_max - box_min) / n_voxels


def voxel_index(pos, box_min, voxel_size, n_voxels):
    """Which voxel (i, j, k) a position falls in, clamped to the grid's valid range.

    Clamping (rather than, say, returning an out-of-bounds index)
    matters because floating-point positions that are meant to sit
    exactly on the domain boundary can land a hair outside it; used
    only from inside `transport.photon_step`, where the position is
    always meant to be within the box anyway.
    """
    idx_f = (pos - box_min) / voxel_size
    return jnp.clip(jnp.floor(idx_f).astype(jnp.int32), 0, n_voxels - 1)


def density_at(density_grid, pos, box_min, voxel_size):
    """Look up the density of the voxel a position falls in."""
    n_voxels = density_grid.shape[0]
    i, j, k = voxel_index(pos, box_min, voxel_size, n_voxels)
    return density_grid[i, j, k]


def _voxel_centers(box_min, box_max, n_voxels):
    """The (n_voxels,) array of voxel-center coordinates along each axis, as plain numpy."""
    box_min, box_max = np.asarray(box_min), np.asarray(box_max)
    voxel_size = (box_max - box_min) / n_voxels
    offsets = (np.arange(n_voxels) + 0.5)
    return [box_min[a] + offsets * voxel_size[a] for a in range(3)]


def make_sphere_density_grid(n_voxels, box_min, box_max, center, radius, density):
    """Build a density grid with a uniform-density solid sphere sitting in it.

    The most direct voxelized equivalent of the single hard dust grain
    from the previous version of this simulation -- every voxel whose
    center falls within `radius` of `center` gets `density`, everything
    else is empty (density 0).
    """
    cx, cy, cz = _voxel_centers(box_min, box_max, n_voxels)
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
    dist2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
    grid = np.where(dist2 <= radius ** 2, density, 0.0)
    return jnp.asarray(grid)


def _downsample_mean(array, n_voxels):
    """Downsample a 3D array to (n_voxels, n_voxels, n_voxels) by block-averaging.

    `np.array_split` groups each axis's indices into `n_voxels`
    contiguous, near-equal-sized chunks, so this works regardless of
    whether the input shape divides evenly by `n_voxels` -- unlike a
    plain `reshape`, which would require that.
    """
    splits = [np.array_split(np.arange(n), n_voxels) for n in array.shape]
    out = np.empty((n_voxels, n_voxels, n_voxels), dtype=np.float64)
    for i, idx_i in enumerate(splits[0]):
        for j, idx_j in enumerate(splits[1]):
            for k, idx_k in enumerate(splits[2]):
                out[i, j, k] = array[np.ix_(idx_i, idx_j, idx_k)].mean()
    return out


def from_fits_cube(cube, n_voxels=30, box_min=None, box_max=None, peak_density=0.3):
    """Turn a loaded FITS density cube (`utils.fits_cube.load_cube`'s output) into a `simulate_photons`-ready density grid.

    Two things this cube isn't, that `density_grid` needs to be:

    - a *cube* of voxels, equal count per axis -- the FITS cube is
      (n_dist, n_y, n_x) = (200, 500, 500), not equal;
    - in the units `photon_step` expects -- a local scattering
      coefficient, 1/length. The FITS cube instead holds `DELTA_NH`,
      the hydrogen column density (cm^-2) contributed by one
      line-of-sight distance bin, and this file gives no scattering
      cross-section to convert that into an actual opacity.

    So this is a deliberate simplification, not a physical unit
    conversion: block-average the cube down to `(n_voxels,) * 3`,
    reorder axes from the FITS's (z, y, x) to the simulation's
    (x, y, z) (matching `voxel_index`'s axis convention), and rescale
    so the field's peak equals `peak_density` -- preserving *where*
    the real cloud complexes are denser or sparser relative to each
    other, landing in the same toy-scale range as
    `make_sphere_density_grid`'s examples. If you have a real
    cross-section to turn NH into optical depth, apply it before or
    instead of this rescaling.

    `box_min`/`box_max` default to the same symmetric box used
    elsewhere in the notebook ([-10, -10, -10] to [10, 10, 10]) rather
    than the cube's real (and very non-cubic: ~500 arcsec x 500 arcsec
    x 10 kpc) physical extent, since converting the angular x/y axes
    to a physical length requires assuming a single distance, and a
    non-cubic box would badly distort the resampled grid's aspect
    ratio. Pass your own `box_min`/`box_max` if you've worked out a
    distance to convert with and want the real proportions instead.

    Returns `(density_grid, box_min, box_max)`, ready to hand to
    `simulate_photons`/`simulate_photons_jit` in that order.
    """
    density = cube["density"]  # (n_z, n_y, n_x)
    grid = _downsample_mean(density, n_voxels)
    grid = np.transpose(grid, (2, 1, 0))  # (z, y, x) -> (x, y, z)
    grid = grid / grid.max() * peak_density

    if box_min is None:
        box_min = jnp.array([-10.0, -10.0, -10.0])
    if box_max is None:
        box_max = jnp.array([10.0, 10.0, 10.0])

    return jnp.asarray(grid), jnp.asarray(box_min), jnp.asarray(box_max)


def make_slab_density_grid(n_voxels, box_min, box_max, axis, x_min, x_max, density):
    """Build a density grid with a uniform-density slab spanning the full box except along `axis`.

    A voxelized version of the very first (1D) version of this
    notebook's plane-parallel slab: full density between `x_min` and
    `x_max` along `axis` (0, 1, or 2 for x/y/z), empty everywhere else.
    Useful mainly as a sanity check -- see Section 2 of the notebook --
    since a straight beam through a uniform slab has a known analytic
    answer (the Beer-Lambert law) to check the voxel marching against.
    """
    centers = _voxel_centers(box_min, box_max, n_voxels)
    X, Y, Z = np.meshgrid(*centers, indexing="ij")
    coord = [X, Y, Z][axis]
    grid = np.where((coord >= x_min) & (coord < x_max), density, 0.0)
    return jnp.asarray(grid)
