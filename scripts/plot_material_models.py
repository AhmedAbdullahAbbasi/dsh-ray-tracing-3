"""Render images, absolute radial profiles, phase laws and portable data exports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reflex_adapter.compare_reflex import normalize_camera_counts, radial_pixel_map  # noqa: E402
from scripts.plot_wd01_reflex import rebin_counts  # noqa: E402


def plot_comparison(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from matplotlib.colors import LogNorm

    output = Path(output)
    report = json.loads((output / "comparison_results.json").read_text())
    with np.load(output / "comparison_arrays.npz", allow_pickle=False) as data:
        arrays = dict(data)
    with np.load(output / "phase_comparison.npz", allow_pickle=False) as data:
        phase = dict(data)
    config, models = report["config"], report["models"]
    scale = normalize_camera_counts(config) / config["source_fluence_ph_cm2"]
    edges = arrays["annulus_edges_arcsec"]
    centers = (edges[:-1] + edges[1:]) / 2
    radial_map = radial_pixel_map(config)
    area = (
        np.bincount(radial_map.ravel(), minlength=len(centers))
        * config["pixel_arcsec"] ** 2
    )
    reference = arrays["reference_counts"]
    observed = np.bincount(radial_map.ravel(), reference.ravel(), len(centers))
    reference_se = np.sqrt(observed * (1 - observed / config["reflex_photons"]))
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 9),
        layout="constrained",
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1]},
    )
    ax, ratio_ax = axes
    ax.errorbar(
        centers,
        observed / (scale * area),
        yerr=reference_se / (scale * area),
        fmt="o",
        color="#202020",
        ms=5,
        capsize=2,
        label="RefleX analog image ± count SE",
        zorder=6,
    )
    ratio_ax.axhline(1, color="#202020", lw=1)
    for key, result in models.items():
        case = result["model"]
        color = case["color"]
        line = ":" if key == "wd01_mie" else "-"
        marker = "D" if key == "wd01_mie" else "s"
        counts = arrays[f"{key}__radial_counts"]
        se = np.sqrt(np.diag(arrays[f"{key}__radial_model_covariance"]))
        ax.errorbar(
            centers,
            counts / (scale * area),
            yerr=se / (scale * area),
            marker=marker,
            linestyle=line,
            color=color,
            ms=4,
            capsize=2,
            label=case["short_label"] + " ± MC SE",
            alpha=0.95,
        )
        ratio_ax.errorbar(
            centers,
            arrays[f"{key}__radial_ratio"],
            yerr=arrays[f"{key}__radial_ratio_se"],
            marker=marker,
            linestyle=line,
            color=color,
            ms=4,
            capsize=2,
            label=case["short_label"],
            alpha=0.85,
        )
    ax.set(
        yscale="log",
        ylabel="Halo fluence [ph cm$^{-2}$ arcsec$^{-2}$]",
        title="Absolute radial profiles · no fitted scale or column",
    )
    ax.legend(fontsize=9)
    ratio_ax.set(
        xlabel="Annulus midpoint [arcsec]",
        ylabel="Model / RefleX counts",
        title="Ratios with combined model MC and RefleX count SE",
    )
    ratio_ax.set_ylim(bottom=0.65)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"Four DSH material prescriptions versus RefleX · {config['energy_kev']:g} keV\n"
        f"N$_H$={config['nh_cm2']:.0e} cm$^{{-2}}$ · same cloud and camera · 6,000,000 histories per model",
        fontsize=14,
    )
    fig.savefig(output / "model_radial_comparison.png", dpi=180)
    fig.savefig(output / "model_radial_comparison.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.axhline(1, color="#202020", lw=1)
    for key, result in models.items():
        if key == "mrn_rg":
            continue
        case = result["model"]
        ax.errorbar(
            centers,
            arrays[f"{key}__radial_ratio"],
            yerr=arrays[f"{key}__radial_ratio_se"],
            marker="D" if key == "wd01_mie" else "s",
            linestyle=":" if key == "wd01_mie" else "-",
            color=case["color"],
            ms=4,
            capsize=2,
            label=case["short_label"],
        )
    ax.set(
        xlabel="Annulus midpoint [arcsec]",
        ylabel="Model / RefleX counts",
        ylim=(0.7, 1.3),
        title="Resolved radial detail · WD01 prescriptions and matched control",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.savefig(output / "wd01_radial_detail.png", dpi=180)
    fig.savefig(output / "wd01_radial_detail.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout="constrained")
    displays = [("RefleX analog reference", rebin_counts(reference))]
    displays.extend(
        (
            result["model"]["short_label"],
            rebin_counts(arrays[f"{key}__predicted_counts"]),
        )
        for key, result in models.items()
    )
    norm = LogNorm(0.5, max(image.max() for _, image in displays))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(cmap(0))
    sky = arrays["sky_edges_arcsec"]
    for axis, (title, data) in zip(axes.ravel(), displays, strict=False):
        im = axis.imshow(
            np.ma.masked_less_equal(data, 0),
            origin="lower",
            extent=[sky[0], sky[-1], sky[0], sky[-1]],
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        axis.set(title=title, xlabel="Sky X [arcsec]", ylabel="Sky Y [arcsec]")
    text_ax = axes.ravel()[-1]
    text_ax.axis("off")
    lines = ["Absolute aperture counts", f"RefleX: {reference.sum():,.0f}", ""]
    for result in models.values():
        lines.extend(
            [
                result["model"]["short_label"],
                f"{result['halo_counts']:,.1f}   /RefleX = {result['halo_ratio']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "Native pixels: 5 arcsec",
            "Display pixels: 15 arcsec",
            "All scattering orders; halo only",
        ]
    )
    text_ax.text(
        0.02, 0.98, "\n".join(lines), transform=text_ax.transAxes, va="top", fontsize=10
    )
    fig.colorbar(
        im,
        ax=axes.ravel()[:-1].tolist(),
        shrink=0.8,
        pad=0.015,
        label="Counts per 15 × 15 arcsec display pixel",
    )
    fig.suptitle(
        "Material comparison images · one common absolute color scale", fontsize=15
    )
    fig.savefig(output / "model_image_comparison.png", dpi=180)
    fig.savefig(output / "model_image_comparison.pdf")
    plt.close(fig)

    angles = phase["physical_angle_arcsec"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, layout="constrained")
    axes[0].loglog(
        angles,
        phase["reflex_adapter"],
        color="#202020",
        lw=2,
        label="RefleX intrinsic phase law",
    )
    axes[1].axhline(1, color="#202020", lw=1)
    for key, result in models.items():
        if key == "reflex_adapter":
            continue
        case = result["model"]
        style = ":" if key == "wd01_mie" else "-"
        axes[0].loglog(
            angles,
            phase[key],
            color=case["color"],
            linestyle=style,
            label=case["short_label"],
        )
        axes[1].semilogx(
            angles,
            phase[key] / phase["reflex_adapter"],
            color=case["color"],
            linestyle=style,
            label=case["short_label"],
        )
    axes[0].set(
        ylabel="Normalized intrinsic phase [sr$^{-1}$]",
        title="Angular laws at 4.9 keV · independent of total scattering opacity",
    )
    axes[0].legend()
    axes[1].set(
        xlabel="Physical scattering angle [arcsec]", ylabel="Model / RefleX phase"
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(output / "model_phase_comparison.png", dpi=180)
    fig.savefig(output / "model_phase_comparison.pdf")
    plt.close(fig)

    # Separate numerical-approximation diagnostic avoids hiding tiny Mie/ADT
    # differences behind the MRN-versus-RefleX differences.
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.semilogx(
        angles, 100 * (phase["wd01_hybrid"] / phase["wd01_mie"] - 1), color="#c64a26"
    )
    ax.axhline(0, color="#202020", lw=1)
    ax.set(
        xlabel="Physical scattering angle [arcsec]",
        ylabel="Hybrid / full Mie − 1 [%]",
        title="Same WD01/D03 C50/Si90 population · phase interpolation and approximation",
    )
    ax.grid(alpha=0.2)
    fig.savefig(output / "mie_hybrid_phase_difference.png", dpi=180)
    fig.savefig(output / "mie_hybrid_phase_difference.pdf")
    plt.close(fig)

    with (output / "model_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "model",
                "halo_counts",
                "halo_over_reflex",
                "ratio_combined_se",
                "radial_chi2",
                "radial_dof",
                "radial_pvalue",
                "sigma_sca_cm2_per_h",
                "sigma_abs_cm2_per_h",
                "median_physical_angle_arcsec",
                "mc_relative_se",
                "numerical_quality_passed",
            ]
        )
        for key, result in models.items():
            diagnostic = result["diagnostics"]
            writer.writerow(
                [
                    key,
                    result["halo_counts"],
                    result["halo_ratio"],
                    result["halo_ratio_combined_se"],
                    result["radial_chi2"],
                    result["radial_dof"],
                    result["radial_pvalue"],
                    diagnostic["scattering_cm2_per_h"],
                    diagnostic["absorption_cm2_per_h"],
                    diagnostic["physical_angle_quantiles_arcsec"][1],
                    result["aperture_relative_mc_se"],
                    result["numerical_quality_passed"],
                ]
            )
    with (output / "radial_profiles.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "model",
                "inner_arcsec",
                "outer_arcsec",
                "selected_area_arcsec2",
                "reflex_counts",
                "model_counts",
                "model_mc_se",
                "model_over_reflex",
                "ratio_combined_se",
            ]
        )
        for key in models:
            writer.writerows(
                (key, *values)
                for values in zip(
                    edges[:-1],
                    edges[1:],
                    area,
                    observed,
                    arrays[f"{key}__radial_counts"],
                    np.sqrt(np.diag(arrays[f"{key}__radial_model_covariance"])),
                    arrays[f"{key}__radial_ratio"],
                    arrays[f"{key}__radial_ratio_se"],
                    strict=True,
                )
            )
    header = fits.Header()
    n = reference.shape[0]
    for axis in (1, 2):
        header[f"CTYPE{axis}"] = "LINEAR"
        header[f"CUNIT{axis}"] = "arcsec"
        header[f"CRPIX{axis}"] = (n + 1) / 2
        header[f"CRVAL{axis}"] = 0.0
        header[f"CDELT{axis}"] = config["pixel_arcsec"]
    header["ENERGY"] = config["energy_kev"]
    header["NH"] = config["nh_cm2"]
    header["BUNIT"] = "count"
    header["HISTORY"] = "RefleX Y reversed to align increasing DSH Y; halo only."
    hdus = [
        fits.PrimaryHDU(header=header),
        fits.ImageHDU(reference, header=header, name="REFLEX"),
    ]
    for key, result in models.items():
        model_header = header.copy()
        model_header["MODEL"] = key
        model_header["NHIST"] = result["histories"]
        counts = arrays[f"{key}__predicted_counts"]
        hdus.append(fits.ImageHDU(counts, header=model_header, name=key.upper()))
        error = np.sqrt(np.maximum(arrays[f"{key}__image_mc_variance"], 0)) * scale
        hdus.append(
            fits.ImageHDU(error, header=model_header, name=key.upper() + "_MCSE")
        )
    fits.HDUList(hdus).writeto(
        output / "model_halo_images.fits", overwrite=True, checksum=True
    )
    print(
        f"Saved images, radial and phase comparisons, CSV and native FITS in {output}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    plot_comparison(parser.parse_args().output)
