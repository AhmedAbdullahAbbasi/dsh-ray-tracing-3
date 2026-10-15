"""Render saved RefleX/DSH comparison arrays without rerunning transport."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output
    report = json.loads((root / "comparison_results.json").read_text())
    with np.load(root / "comparison_arrays.npz") as archive:
        a = dict(archive)
    reference = a["reference_counts"]
    expected = a["expected_counts"]
    edges = a["sky_edges_arcsec"]
    centers = (edges[1:] + edges[:-1]) / 2
    xx, yy = np.meshgrid(centers, centers)
    bins = a["annulus_edges_arcsec"]
    region = np.searchsorted(bins, np.hypot(xx, yy), side="right") - 1
    areas = np.bincount(region.ravel(), minlength=len(bins)-1) * report["config"]["pixel_arcsec"]**2
    n0 = report["source_camera_normalization"]
    observed_radial = a["observed_radial_counts"]
    expected_radial = a["expected_radial_counts"]
    radius = 0.5 * (bins[1:] + bins[:-1])
    mc_se = np.sqrt(np.diag(a["radial_covariance"]))
    observed_brightness = observed_radial / n0 / areas
    predicted_brightness = a["radial_fluence"] / areas
    total_ratio_se = np.sqrt(np.diag(a["radial_count_covariance"])) / observed_radial
    rows = []
    for i in range(len(radius)):
        rows.append(dict(inner_arcsec=bins[i], outer_arcsec=bins[i+1],
            reflex_counts=observed_radial[i], dsh_expected_reflex_counts=expected_radial[i],
            dsh_fluence=a["radial_fluence"][i], dsh_mc_se=mc_se[i],
            dsh_over_reflex=expected_radial[i]/observed_radial[i],
            standardized_reflex_minus_dsh=report["annulus_z"][i]))
    with (root / "radial_profiles.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    factor = 5
    def grouped(image):
        n = len(image) // factor
        return image.reshape(n, factor, n, factor).sum(axis=(1, 3))
    first, second = grouped(reference), grouped(expected)
    norm = LogNorm(vmin=0.5, vmax=max(first.max(), second.max()))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(cmap(0))
    extent = (edges[0], edges[-1], edges[0], edges[-1])
    for axis, image, title in [(ax[0, 0], first, "RefleX: recorded halo photons"),
                               (ax[0, 1], second, "DSH: expected counts in the RefleX cameras")]:
        artist = axis.imshow(np.ma.masked_less_equal(image, 0), origin="lower", extent=extent, cmap=cmap, norm=norm)
        axis.set(title=title, xlabel="Sky x (arcsec)", ylabel="Sky y (arcsec)")
        fig.colorbar(artist, ax=axis, label="Counts per 25-arcsec display pixel")
    ax[1, 0].errorbar(radius, observed_brightness, yerr=np.sqrt(observed_radial)/n0/areas,
                     fmt="o", ms=4, color="black", label="RefleX ± count SE")
    ax[1, 0].errorbar(radius, predicted_brightness, yerr=mc_se/areas,
                     fmt="s", ms=4, color="#006f9f", label="DSH all orders ± history MC SE")
    for order_image, label in [(a["image_by_order"][0], "DSH first scattering"),
                              (a["image_by_order"][1:].sum(axis=0), "DSH second and higher")]:
        summed = np.bincount(region.ravel(), weights=order_image.ravel(), minlength=len(radius))
        ax[1, 0].plot(radius, summed/areas, "--", lw=1, alpha=0.8, label=label)
    ax[1, 0].set(yscale="log", xlabel="Radius (arcsec)",
                 ylabel="Halo/source surface fluence (arcsec⁻²)")
    ax[1, 0].legend(fontsize=8)
    ratio = expected_radial / observed_radial
    ax[1, 1].errorbar(radius, ratio, yerr=total_ratio_se, fmt="o", color="#006f9f", capsize=3)
    ax[1, 1].axhline(1, color="black", lw=1)
    ax[1, 1].set(xlabel="Radius (arcsec)", ylabel="DSH / RefleX (no fitted scale)")
    ax[1, 1].text(0.03, 0.03, f"Aperture difference: {report['absolute_aperture_z']:+.2f} SE\n"
        f"Radial χ² / dof: {report['global_radial_chi2']:.2f} / {report['radial_degrees_of_freedom']}\n"
        f"Radial p = {report['global_radial_pvalue']:.3f}", transform=ax[1, 1].transAxes,
        fontsize=10, va="bottom", bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    fig.suptitle(f"RefleX / DSH: 4.9 keV, NH = 10²² cm⁻², D = 10 kpc\n"
                 f"{report['histories']:,} DSH histories; {int(report['reference_halo_counts']):,} RefleX halo photons; aperture response included", fontsize=13)
    fig.savefig(root / "reflex_comparison.png", dpi=180)
    fig.savefig(root / "reflex_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
