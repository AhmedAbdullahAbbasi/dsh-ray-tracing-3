"""Small matplotlib helpers, kept out of the notebook to keep it readable.

None of these do anything physics-related -- they just take arrays the
notebook already computed and draw them.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .transport import ACTIVE

# Shared outcome -> color/label mapping, reused by several plots below.
UNSCATTERED = "unscattered"   # n_scatter == 0: flew straight through, never scattered
SCATTERED = "scattered"       # n_scatter >= 1: scattered at least once in the medium

_OUTCOME_COLOR = {
    UNSCATTERED: "#4C72B0",
    SCATTERED: "#C44E52",
    ACTIVE: "#8172B2",
}
_OUTCOME_LABEL = {
    UNSCATTERED: "unscattered (flew straight through)",
    SCATTERED: "scattered ≥ 1 time in the medium",
    ACTIVE: "still active (increase n_bounces)",
}


def _outcome_key(n_scatter_i, status_i):
    if status_i == ACTIVE:
        return ACTIVE
    return SCATTERED if n_scatter_i >= 1 else UNSCATTERED


def plot_isotropic_mu_check(samples, ax=None):
    """Histogram of sampled direction cosines vs. the flat pdf.

    `mu` here is the cosine to *some* fixed reference axis -- any axis
    works for a genuinely isotropic direction (`photon_step` uses the
    photon's own incoming direction, for convenience), which is exactly
    why this check doesn't need to reference the scattering geometry
    at all, only the raw draw.
    """
    ax = ax or plt.gca()
    samples = np.asarray(samples)
    ax.hist(samples, bins=60, range=(-1, 1), density=True,
             alpha=0.6, label="sampled (JAX)")
    ax.axhline(0.5, color="k", linestyle="--", label="analytic: uniform")
    ax.set_xlabel(r"direction cosine $\mu$ (to an arbitrary reference axis)")
    ax.set_ylabel("probability density")
    ax.set_title("Isotropic scattering-angle sampling")
    ax.legend()
    ax.set_ylim(0, 1)


def plot_rayleigh_mu_check(samples, ax=None):
    """Histogram of sampled Rayleigh direction cosines vs. the analytic dipole phase function.

    Same idea as `plot_isotropic_mu_check`, for `sampling.sample_rayleigh_mu`: unlike isotropic
    scattering's flat p(mu) = 1/2, Rayleigh's p(mu) = (3/8)(1 + mu^2) favors mu near +-1 (forward/
    backward) over mu near 0 (sideways) -- the classic dipole shape.
    """
    ax = ax or plt.gca()
    samples = np.asarray(samples)
    ax.hist(samples, bins=60, range=(-1, 1), density=True,
             alpha=0.6, label="sampled (JAX)")
    mu = np.linspace(-1, 1, 200)
    ax.plot(mu, 3 / 8 * (1 + mu ** 2), "k--", label=r"analytic: $\frac{3}{8}(1+\mu^2)$")
    ax.set_xlabel(r"direction cosine $\mu$ (to the incoming direction)")
    ax.set_ylabel("probability density")
    ax.set_title("Rayleigh scattering-angle sampling")
    ax.legend()


def plot_henyey_greenstein_mu_check(samples, g, ax=None):
    """Histogram of sampled Henyey-Greenstein direction cosines vs. the analytic phase function.

    Same idea as `plot_isotropic_mu_check`, for `sampling.sample_henyey_greenstein_mu` at a given
    asymmetry `g` -- the standard stand-in for Mie scattering used by the `"mie"`/`"rayleigh_mie"`
    scattering models. Larger `g` should show a sharper pile-up near mu = 1 (forward scattering).
    """
    ax = ax or plt.gca()
    samples = np.asarray(samples)
    ax.hist(samples, bins=60, range=(-1, 1), density=True,
             alpha=0.6, label="sampled (JAX)")
    mu = np.linspace(-1, 1, 400)
    analytic = (1 - g ** 2) / (2 * (1 + g ** 2 - 2 * g * mu) ** 1.5)
    ax.plot(mu, analytic, "k--", label=r"analytic Henyey-Greenstein")
    ax.set_xlabel(r"direction cosine $\mu$ (to the incoming direction)")
    ax.set_ylabel("probability density")
    ax.set_title(f"Henyey-Greenstein scattering-angle sampling (g = {g:.2f})")
    ax.legend()


def plot_optical_depth_check(tau_values, simulated_fractions, ax=None):
    """Simulated P(scatters >= 1) through a uniform slab vs. the exact Beer-Lambert law.

    The voxel ray-marching in `transport.photon_step` approximates a
    continuous optical-depth integral with a Riemann sum over
    `n_substeps` small steps -- this checks that approximation against
    the one case with a clean closed-form answer: a photon fired
    straight through a slab of *uniform* density (built with
    `voxels.make_slab_density_grid`) survives without scattering with
    probability exp(-tau), so it scatters at least once with
    probability 1 - exp(-tau), regardless of how that slab happens to
    be chopped into voxels.
    """
    ax = ax or plt.gca()
    tau_values = np.asarray(tau_values)
    ax.plot(tau_values, simulated_fractions, "o", label="simulated (JAX)", color="#4C72B0")
    x = np.linspace(0.0, max(tau_values.max(), 1e-6), 200)
    ax.plot(x, 1.0 - np.exp(-x), "k--", label=r"analytic: $1-e^{-\tau}$")
    ax.set_xlabel(r"optical depth $\tau$ through the slab")
    ax.set_ylabel(r"P(photon scatters $\geq 1$ time)")
    ax.set_title("Voxel ray-marching check: the Beer-Lambert law")
    ax.legend()


def plot_n_scatter_histogram(n_scatter, ax=None):
    """Bar chart of how many times each photon scattered before it left the domain."""
    ax = ax or plt.gca()
    n_scatter = np.asarray(n_scatter)
    max_n = min(int(n_scatter.max()), 8) if n_scatter.size else 0
    counts = [np.mean(n_scatter == k) for k in range(max_n)]
    counts.append(np.mean(n_scatter >= max_n))
    labels = [str(k) for k in range(max_n)] + [f"{max_n}+"]

    bars = ax.bar(labels, counts, color="#55A868")
    for bar, frac in zip(bars, counts):
        if frac > 0.001:
            ax.text(bar.get_x() + bar.get_width() / 2, frac + 0.01, f"{frac:.3f}",
                     ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("number of times scattered")
    ax.set_ylabel("fraction of photons")
    ax.set_title(f"N = {n_scatter.size:,} photons")


def plot_exit_angle_comparison(direction, n_scatter, ax=None):
    """Histogram of the angle between a photon's final direction and the beam axis.

    Split into unscattered photons (angle is exactly 0 -- they never
    changed direction) and scattered photons (spread out over a much
    wider range, since even a single isotropic scattering event
    randomizes the direction with little memory of the original beam
    axis).
    """
    ax = ax or plt.gca()
    direction = np.asarray(direction)
    n_scatter = np.asarray(n_scatter)

    beam_axis = np.array([1.0, 0.0, 0.0])
    angle_deg = np.degrees(np.arccos(np.clip(direction @ beam_axis, -1.0, 1.0)))

    unscattered = n_scatter == 0
    scattered = ~unscattered

    if scattered.sum():
        ax.hist(angle_deg[scattered], bins=40, range=(0, 180), density=True,
                 alpha=0.6, color=_OUTCOME_COLOR[SCATTERED], label=_OUTCOME_LABEL[SCATTERED])
    if unscattered.sum():
        ax.axvline(0.0, color=_OUTCOME_COLOR[UNSCATTERED], linewidth=3,
                    label=f"{_OUTCOME_LABEL[UNSCATTERED]} ({unscattered.mean():.1%})")
    ax.set_xlabel("angle between final direction and the beam axis (degrees)")
    ax.set_ylabel("probability density (scattered photons)")
    ax.set_title("Where photons end up heading")
    ax.legend()


def _voxel_centers_grid(box_min, box_max, n_voxels):
    box_min, box_max = np.asarray(box_min), np.asarray(box_max)
    voxel_size = (box_max - box_min) / n_voxels
    offsets = np.arange(n_voxels) + 0.5
    axes = [box_min[a] + offsets * voxel_size[a] for a in range(3)]
    return np.meshgrid(*axes, indexing="ij")


def plot_voxel_grid(ax, density_grid, box_min, box_max, density_threshold_frac=0.05,
                     max_points=8000, seed=0, alpha=0.6, color="#8B5E3C"):
    """Scatter-plot the voxels whose density is above a threshold, sized by density.

    matplotlib has no real volume renderer, so this is the simple,
    honest substitute: one point per occupied voxel (center),
    subsampled if there are more than `max_points` of them (a fine
    grid can easily have tens of thousands of occupied cells --
    drawing all of them is slow and unreadable), sized so denser voxels
    stand out a little more than sparse ones.
    """
    density_grid = np.asarray(density_grid)
    n_voxels = density_grid.shape[0]
    d_max = density_grid.max()
    if d_max <= 0:
        return

    X, Y, Z = _voxel_centers_grid(box_min, box_max, n_voxels)
    mask = density_grid > density_threshold_frac * d_max
    xs, ys, zs, ds = X[mask], Y[mask], Z[mask], density_grid[mask]

    if xs.size > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(xs.size, size=max_points, replace=False)
        xs, ys, zs, ds = xs[keep], ys[keep], zs[keep], ds[keep]

    sizes = 14.0 + 70.0 * (ds / d_max)
    ax.scatter(xs, ys, zs, s=sizes, c=color, alpha=alpha, linewidths=0)


def _box_edges(box_min, box_max):
    box_min, box_max = np.asarray(box_min), np.asarray(box_max)
    corners = np.array([[x, y, z]
                         for x in (box_min[0], box_max[0])
                         for y in (box_min[1], box_max[1])
                         for z in (box_min[2], box_max[2])])
    edges = [(corners[i], corners[j])
             for i in range(8) for j in range(i + 1, 8)
             if bin(i ^ j).count("1") == 1]  # differ in exactly one coordinate -> an edge
    return edges


def plot_box_wireframe(ax, box_min, box_max, color="lightgray", alpha=0.4, linewidth=0.7):
    """Draw the 12 edges of the domain's bounding box."""
    for p1, p2 in _box_edges(box_min, box_max):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                 color=color, alpha=alpha, linewidth=linewidth)


def plot_3d_scene(example_paths, density_grid, box_min, box_max, ax=None,
                   density_threshold_frac=0.05, voxel_max_points=8000, voxel_alpha=0.6,
                   voxel_color="#8B5E3C"):
    """Draw the voxel density grid, the domain boundary, and a handful of real example photon paths in 3D.

    `example_paths` is the output of `geometry.extract_example_paths`:
    a list of (positions, status) pairs, taken directly from the same
    batched simulation used for the statistics elsewhere in the
    notebook -- not a separate re-simulation. Pass the same
    `density_grid`/`box_min`/`box_max` used in that `simulate_photons`
    call, or the drawn scene won't match the paths.

    `density_threshold_frac`/`voxel_max_points`/`voxel_alpha`/`voxel_color`
    are forwarded straight to `plot_voxel_grid`, so the cloud's
    visibility can be tuned per call -- e.g. a lower
    `density_threshold_frac` or a higher `voxel_alpha`/`voxel_max_points`
    to bring out a faint or sparse density field -- without editing
    this file.
    """
    ax = ax or plt.gcf().add_subplot(111, projection="3d")
    box_min, box_max = np.asarray(box_min), np.asarray(box_max)

    plot_voxel_grid(ax, density_grid, box_min, box_max,
                     density_threshold_frac=density_threshold_frac,
                     max_points=voxel_max_points, alpha=voxel_alpha, color=voxel_color)
    plot_box_wireframe(ax, box_min, box_max)

    seen_categories = set()
    for positions, status in example_paths:
        # A path with only 2 points (launch point -> exit point) went straight through
        # without ever scattering; more points means at least one scatter in between.
        category = ACTIVE if status == ACTIVE else (UNSCATTERED if len(positions) <= 2 else SCATTERED)
        seen_categories.add(category)
        color = _OUTCOME_COLOR[category]
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                 color=color, linewidth=1.4, alpha=0.85)
        if len(positions) > 2:
            ax.scatter(*positions[1:-1].T, color=color, s=14, alpha=0.7)  # scattering points
        ax.scatter(*positions[-1], color=color, s=30, edgecolor="k", linewidth=0.4, zorder=5)
        ax.scatter(*positions[0], color="black", marker="*", s=90, zorder=6)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D scene: voxel density grid, domain boundary, and example photon paths")
    ax.set_xlim(box_min[0], box_max[0])
    ax.set_ylim(box_min[1], box_max[1])
    ax.set_zlim(box_min[2], box_max[2])
    ax.set_box_aspect(tuple(box_max - box_min))  # proportional to the box's actual shape, not forced to a cube

    legend_handles = [Line2D([0], [0], color="black", marker="*", linestyle="",
                              markersize=11, label="photon enters here")]
    for category in (UNSCATTERED, SCATTERED, ACTIVE):
        if category in seen_categories:
            legend_handles.append(Line2D([0], [0], color=_OUTCOME_COLOR[category], lw=2,
                                          label=_OUTCOME_LABEL[category]))
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
