"""Turning real simulation output into a handful of paths worth plotting.

Earlier versions of this simulation only tracked a scalar optical
depth per photon, so drawing an actual 3D picture meant re-deriving
trajectories with a second, separate piece of code. That's no longer
true: `transport.simulate_photons` now tracks real 3D positions and
records the full path of every photon as part of the batched
simulation itself (see the `path` output of `run_one_photon`). So the
only job left here is picking a few of those paths out and trimming
them for plotting -- no re-simulation, no duplicate physics.
"""

import numpy as np


def extract_example_paths(path, status, n_examples=16, n_scatter=None):
    """Pull the first `n_examples` photon paths out of simulation output, trimmed to length.

    `path` (from `simulate_photons`) has shape (n_photons, n_bounces+1,
    3): every photon's position after every scan step, including
    frozen repeats of its final position once it's been killed (so
    that all photons have the same fixed array length, as JAX
    requires). This finds where each path actually stops moving and
    cuts the frozen tail off, so a plotted line ends exactly where the
    photon did instead of sitting on top of itself.

    Pass `n_scatter` (`simulate_photons`'s own `n_scatter` output) to
    restrict this to photons that actually scattered at least once --
    otherwise the first `n_examples` photons are taken as they come,
    which in a sparse medium can be mostly (or entirely) straight
    unscattered lines, not very interesting to look at. Left as
    `None` (the default) keeps the old, unfiltered behavior.

    Returns a list of (positions, status) pairs -- `positions` an
    (n_i, 3) array, `status` one of `transport.STATUS_NAMES`.
    """
    path = np.asarray(path)
    status = np.asarray(status)

    candidate_indices = np.arange(path.shape[0])
    if n_scatter is not None:
        candidate_indices = candidate_indices[np.asarray(n_scatter) >= 1]
    candidate_indices = candidate_indices[:n_examples]

    examples = []
    for i in candidate_indices:
        p = path[i]
        step_lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
        moved = np.nonzero(step_lengths > 1e-9)[0]
        last_moving_step = int(moved[-1]) + 1 if moved.size else 0
        examples.append((p[: last_moving_step + 1], int(status[i])))
    return examples
