"""Derived views with explicit bins, per-pixel areas and selected epochs."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlotSelection:
    times: slice
    energies: slice
    ys: slice
    xs: slice
    x_edges: object
    y_edges: object
    duration_s: float
    width_kev: float


def plot_selection(plots, bins, source=None, *, declared_days=None):
    """Validate/select whole stored bins without running or plotting transport."""
    time = np.asarray(bins.arrival_time_edges_s, dtype=float)
    start, stop = 0, len(time) - 1
    if plots.epoch_days:
        for edge, position in zip(plots.epoch_days, (0, 1), strict=True):
            days = time / 86400 if declared_days is None else np.asarray(declared_days)
            hits = np.flatnonzero(np.isclose(days, edge, rtol=0, atol=1e-9))
            if hits.size != 1:
                raise ValueError(
                    "plots.epoch_days: bounds must align with stored bins at runtime precision"
                )
            if position == 0:
                start = int(hits[0])
            else:
                stop = int(hits[0])
        if stop <= start:
            raise ValueError("plots.epoch_days: must contain at least one complete bin")
    energy = np.asarray(bins.energy_edges_kev, dtype=float)
    first, end = 0, len(energy) - 1
    if plots.energy_selection == "nearest":
        centers = (energy[:-1] + energy[1:]) / 2
        if source is not None:
            labels = np.asarray(source.effective_energy_kev)
            source_edges = source.energy_edges_kev
            if len(labels) == len(centers) and (
                source_edges is None or np.array_equal(source_edges, energy)
            ):
                centers = labels
        first = int(np.argmin(np.abs(centers - plots.energy_kev)))
        end = first + 1
    x = np.asarray(bins.sky_x_edges_arcsec, dtype=float)
    y = np.asarray(bins.sky_y_edges_arcsec, dtype=float)
    xl, xr, yl, yr = 0, len(x) - 1, 0, len(y) - 1
    if plots.crop == "centered":
        half = plots.fov_arcsec / 2
        xl, xr = (
            int(np.searchsorted(x, -half)),
            int(np.searchsorted(x, half, side="right")) - 1,
        )
        yl, yr = (
            int(np.searchsorted(y, -half)),
            int(np.searchsorted(y, half, side="right")) - 1,
        )
        if xr <= xl or yr <= yl:
            raise ValueError(
                "plots.fov_arcsec: crop must contain at least one whole sky pixel on each axis"
            )
    return PlotSelection(
        slice(start, stop),
        slice(first, end),
        slice(yl, yr),
        slice(xl, xr),
        x[xl : xr + 1],
        y[yl : yr + 1],
        time[stop] - time[start],
        energy[end] - energy[first],
    )


def plot_arrays(plan, result):
    selection = plot_selection(
        plan.config.plots,
        plan.bins,
        getattr(plan, "source", None),
        declared_days=plan.config.observer.arrival.edges_days,
    )
    selection_tuple = (selection.times, selection.energies, selection.ys, selection.xs)
    image = np.asarray(result.products.total_fluence)[selection_tuple].sum(
        axis=(0, 1), dtype=float
    )
    if plan.config.plots.quantity == "intensity":
        area = np.diff(selection.y_edges)[:, None] * np.diff(selection.x_edges)[None, :]
        image /= area * selection.duration_s
        if plan.config.plots.density_per_kev:
            image /= selection.width_kev
    events = np.asarray(result.products.event_count)[selection_tuple].sum(
        axis=(0, 1), dtype=np.int64
    )
    return image, events


def save_plots(plan, result, directory, prefix=""):
    image, events = plot_arrays(plan, result)
    paths = []
    plots = plan.config.plots
    if plots.save_scored_event_map:
        path = directory / (prefix + "scored_mc_events.npy")
        np.save(path, events, allow_pickle=False)
        paths.append(path)
    if not plots.enabled:
        return paths
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    selection = plot_selection(
        plots,
        plan.bins,
        plan.source,
        declared_days=plan.config.observer.arrival.edges_days,
    )
    unit = "Fluence [ph cm$^{-2}$]"
    if plots.quantity == "intensity":
        spectral = " keV$^{-1}$" if plots.density_per_kev else ""
        unit = "Intensity [ph cm$^{-2}$ s$^{-1}$" + spectral + " arcsec$^{-2}$]"
    views = [("intensity_", image, unit)]
    if plots.save_count_plots:
        views.append(("scored_events_", events, "Scored Monte Carlo events per pixel"))
    for stem, values, label in views:
        for scale in plots.scales:
            positive = values[values > 0]
            norm, display = None, values
            if scale == "log" and positive.size:
                low, high = positive.min(), positive.max()
                norm = LogNorm(vmin=low, vmax=max(high, low * (1 + 1e-6)))
                display = np.ma.masked_less_equal(values, 0)
            figure, axis = plt.subplots()
            artist = axis.pcolormesh(
                selection.x_edges, selection.y_edges, display, shading="flat", norm=norm
            )
            axis.set(
                xlabel="Sky x [arcsec]", ylabel="Sky y [arcsec]", title=plan.config.name
            )
            axis.set_aspect("equal")
            figure.colorbar(artist, ax=axis, label=label)
            path = directory / (prefix + stem + scale + ".png")
            figure.savefig(path, dpi=plots.dpi, bbox_inches="tight")
            plt.close(figure)
            paths.append(path)
    return paths
