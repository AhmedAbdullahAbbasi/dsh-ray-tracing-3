"""Derived intensity views with each sky pixel's own area and exact bin edges."""

import numpy as np


def plot_arrays(plan, result):
    """Return interval-averaged differential intensity and scored MC events."""
    edges = np.asarray(plan.bins.arrival_time_edges_s, dtype=float)
    start, stop = 0, len(edges) - 1
    if plan.config.plots.epoch_days:
        original = plan.config.observer.arrival.edges_days
        start, stop = [
            next(
                i
                for i, x in enumerate(original)
                if np.isclose(x, day, rtol=0, atol=1e-9)
            )
            for day in plan.config.plots.epoch_days
        ]
    x = np.asarray(plan.bins.sky_x_edges_arcsec, dtype=float)
    y = np.asarray(plan.bins.sky_y_edges_arcsec, dtype=float)
    energy = np.asarray(plan.bins.energy_edges_kev, dtype=float)
    area = np.diff(y)[:, None] * np.diff(x)[None, :]
    image = np.asarray(result.products.total_fluence)[start:stop].sum(
        axis=(0, 1), dtype=float
    )
    image /= area * (edges[stop] - edges[start]) * (energy[-1] - energy[0])
    events = np.asarray(result.products.event_count)[start:stop].sum(
        axis=(0, 1), dtype=np.int64
    )
    return image, events


def save_plots(plan, result, directory, prefix=""):
    image, events = plot_arrays(plan, result)
    paths = []
    if plan.config.plots.save_scored_event_map:
        path = directory / (prefix + "scored_mc_events.npy")
        np.save(path, events, allow_pickle=False)
        paths.append(path)
    if not plan.config.plots.enabled:
        return paths
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    for scale in plan.config.plots.scales:
        positive = image[image > 0]
        norm = None
        display = image
        if scale == "log" and positive.size:
            low, high = positive.min(), positive.max()
            norm = LogNorm(vmin=low, vmax=max(high, low * (1 + 1e-6)))
            display = np.ma.masked_less_equal(image, 0)
        figure, axis = plt.subplots()
        artist = axis.pcolormesh(
            np.asarray(plan.bins.sky_x_edges_arcsec),
            np.asarray(plan.bins.sky_y_edges_arcsec),
            display,
            shading="flat",
            norm=norm,
        )
        axis.set(
            xlabel="Sky x [arcsec]", ylabel="Sky y [arcsec]", title=plan.config.name
        )
        axis.set_aspect("equal")
        figure.colorbar(
            artist,
            ax=axis,
            label="Intensity [ph cm$^{-2}$ s$^{-1}$ keV$^{-1}$ arcsec$^{-2}$]",
        )
        path = directory / (prefix + "intensity_" + scale + ".png")
        figure.savefig(path, dpi=plan.config.plots.dpi, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths
