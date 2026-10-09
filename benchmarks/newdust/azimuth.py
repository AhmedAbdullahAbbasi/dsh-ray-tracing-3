"""First-history azimuthal checks on the same finite-pixel masks as the reference."""

import csv

import numpy as np
from scipy.stats import chi2


def analyze_azimuth(reference, image, squares, edges, histories, output):
    centers = (edges[1:] + edges[:-1]) / 2
    x, y = np.meshgrid(centers, centers)
    radius = np.hypot(x, y)
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)
    basis = np.stack(
        [np.cos(2 * phi), np.sin(2 * phi), np.cos(4 * phi), np.sin(4 * phi)]
    )
    radial_edges = [0.0, 40.0, 80.0, 160.0, 320.0, float(edges[-1])]
    sector = np.minimum((phi / (2 * np.pi) * 16).astype(int), 15)
    pixel_area = (edges[1] - edges[0]) ** 2
    summaries = []
    rows = []

    def se(total, square):
        return np.sqrt(
            histories / (histories - 1) * max(square - total * total / histories, 0)
        )

    for lo, hi in zip(radial_edges[:-1], radial_edges[1:], strict=True):
        if hi <= lo:
            continue
        mask = (radius >= lo) & (radius < hi)
        ref = reference[mask].sum()
        value = image[mask].sum()
        square = squares[mask].sum()
        neff = value * value / square if square > 0 else 0
        b = basis[:, mask]
        ref_mean = (b @ reference[mask]) / ref if ref > 0 else np.zeros(4)
        centered = b - ref_mean[:, None]
        difference = centered @ image[mask]
        covariance = (
            histories
            / (histories - 1)
            * (
                (centered * squares[mask]) @ centered.T
                - np.outer(difference, difference) / histories
            )
        )
        rank = int(np.linalg.matrix_rank(covariance, hermitian=True))
        statistic = (
            float(difference @ np.linalg.pinv(covariance, hermitian=True) @ difference)
            if rank
            else 0.0
        )
        p = float(chi2.sf(statistic, rank)) if rank else None
        sector_neff = []
        for index in range(16):
            selected = mask & (sector == index)
            area = selected.sum() * pixel_area
            r = reference[selected].sum()
            m = image[selected].sum()
            s = squares[selected].sum()
            effective = m * m / s if s > 0 else 0
            sector_neff.append(effective)
            rows.append(
                {
                    "radius_lo_arcsec": lo,
                    "radius_hi_arcsec": hi,
                    "phi_lo_deg": index * 22.5,
                    "phi_hi_deg": (index + 1) * 22.5,
                    "pixel_area_arcsec2": area,
                    "newdust_fluence": r,
                    "dsh_fluence": m,
                    "dsh_mc_se": se(m, s),
                    "ratio": m / r if r > 0 else None,
                    "neff": effective,
                }
            )
        resolved = bool(neff >= 300 and min(sector_neff) >= 15 and rank == 4)
        summaries.append(
            {
                "radius_lo_arcsec": lo,
                "radius_hi_arcsec": hi,
                "neff": float(neff),
                "minimum_sector_neff": float(min(sector_neff)),
                "resolved": resolved,
                "harmonic_chi2": statistic,
                "degrees_of_freedom": rank,
                "p_value": p,
                "passed": bool(p is not None and p >= 0.001),
                "modes": [2, 4],
                "mc_mean_harmonics": (b @ image[mask] / value).tolist()
                if value > 0
                else [None] * 4,
                "reference_mean_harmonics": ref_mean.tolist(),
            }
        )
    with (output / "azimuth_profiles.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, summary in zip(axs.flat, summaries, strict=False):
        selected = [
            r for r in rows if r["radius_lo_arcsec"] == summary["radius_lo_arcsec"]
        ]
        ref = np.array([r["newdust_fluence"] for r in selected])
        mc = np.array([r["dsh_fluence"] for r in selected])
        err = np.array([r["dsh_mc_se"] for r in selected])
        angles = np.arange(16) * 22.5 + 11.25
        ax.errorbar(
            angles,
            np.divide(mc, ref, out=np.zeros(16), where=ref > 0),
            yerr=np.divide(err, ref, out=np.zeros(16), where=ref > 0),
            fmt=".",
            color="tab:blue" if summary["resolved"] else "grey",
        )
        ax.axhline(1, color="black", linewidth=1)
        p = summary["p_value"]
        status = (
            f"p={p:.3g}" if summary["resolved"] else "insufficient azimuthal sampling"
        )
        ax.set(
            title=f"{summary['radius_lo_arcsec']:g}–{summary['radius_hi_arcsec']:g} arcsec; {status}",
            xlabel="Azimuth (degrees)",
            ylabel="DSH / NewDust",
            xlim=(0, 360),
        )
    for ax in list(axs.flat)[len(summaries) :]:
        ax.axis("off")
    fig.suptitle(
        "Azimuthal comparison; pixel masks matched; errors from first-history moments"
    )
    fig.savefig(output / "azimuth_comparison.png", dpi=150)
    plt.close(fig)
    resolved = [s for s in summaries if s["resolved"]]
    return {
        "minimum_resolved_windows": 3,
        "resolved_windows": len(resolved),
        "sampling_sufficient": len(resolved) >= 3,
        "resolved_harmonics_passed": all(s["passed"] for s in resolved),
        "p_threshold_per_window": 0.001,
        "sectors": 16,
        "radial_windows": summaries,
        "scope": "m=2 and m=4 in adequately sampled radial windows; full per-pixel convergence is not asserted.",
    }
