"""Render the different-material comparison without collecting new histories."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reflex_adapter.compare_reflex import radial_pixel_map  # noqa: E402
from scripts.compare_wd01_reflex import independent_ratio  # noqa: E402


def rebin_counts(image, factor=3):
    """Sum disjoint native pixels; display only, never used for radial errors."""
    ny, nx = image.shape
    if ny % factor or nx % factor:
        raise ValueError("display rebinning must divide the image dimensions")
    return image.reshape(ny // factor, factor, nx // factor, factor).sum((1, 3))


def export_fits(output, arrays, summary):
    from astropy.io import fits

    config = summary["config"]
    n = arrays["reference_counts"].shape[0]
    header = fits.Header()
    for axis in (1, 2):
        header[f"CTYPE{axis}"] = "LINEAR"
        header[f"CUNIT{axis}"] = "arcsec"
        header[f"CRPIX{axis}"] = (n + 1) / 2
        header[f"CRVAL{axis}"] = 0.0
        header[f"CDELT{axis}"] = config["pixel_arcsec"]
    header["ENERGY"] = (config["energy_kev"], "keV; monochromatic")
    header["NH"] = (config["nh_cm2"], "H nuclei cm-2; central column")
    header["NHIST"] = summary["histories"]
    header["MATLSHA"] = summary["material"]["definition_sha256"]
    header["HISTORY"] = "RefleX native Y axis reversed to increasing DSH sky Y."
    header["HISTORY"] = "Halo only; orders >=1; source/camera normalization fixed."
    products = (
        ("REFLEX", arrays["reference_counts"], "count"),
        ("WD01", arrays["expected_counts"], "count"),
        ("DIFFERENCE", arrays["expected_counts"] - arrays["reference_counts"], "count"),
        ("FLUENCE", arrays["image_by_order"].sum(axis=0), "ph cm-2"),
        ("FIRST", arrays["image_by_order"][0], "ph cm-2"),
        ("MULTIPLE", arrays["image_by_order"][1:].sum(axis=0), "ph cm-2"),
        ("MCSE", np.sqrt(np.maximum(arrays["image_mc_variance"], 0)), "ph cm-2"),
    )
    hdus = [fits.PrimaryHDU(header=header)]
    for name, data, units in products:
        extension_header = header.copy()
        extension_header["BUNIT"] = units
        hdus.append(fits.ImageHDU(data, header=extension_header, name=name))
    fits.HDUList(hdus).writeto(
        output / "halo_images.fits", overwrite=True, checksum=True
    )


def plot_comparison(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    output = Path(output)
    summary = json.loads((output / "comparison_results.json").read_text())
    with np.load(output / "comparison_arrays.npz", allow_pickle=False) as data:
        arrays = dict(data)
    config = summary["config"]
    observed = arrays["observed_radial_counts"]
    expected = arrays["expected_radial_counts"]
    model_cov = arrays["model_radial_count_covariance"]
    reference_cov = arrays["reference_radial_count_covariance"]
    normalization = summary["source_camera_normalization"]
    annuli = arrays["annulus_edges_arcsec"]
    midpoints = (annuli[1:] + annuli[:-1]) / 2
    pixel_counts = np.bincount(
        radial_pixel_map(config).ravel(), minlength=len(observed)
    )
    # Last annulus is clipped by the square image. Count selected native pixels
    # instead of using pi*(r_outer**2-r_inner**2).
    area = pixel_counts * config["pixel_arcsec"] ** 2
    radial_scale = normalization * area
    ratio, ratio_se = independent_ratio(
        expected, observed, np.diag(model_cov), np.diag(reference_cov)
    )
    baseline = None
    baseline_dir = (
        ROOT / "benchmarks/materials/absorption_edges_2026-10-16/reflex_fresh"
    )
    if (baseline_dir / "comparison_results.json").exists():
        baseline_summary = json.loads(
            (baseline_dir / "comparison_results.json").read_text()
        )
        for key, value in config.items():
            if key != "seeds" and baseline_summary["config"][key] != value:
                raise ValueError(f"baseline comparison differs in {key}")
        with np.load(
            baseline_dir / "comparison_arrays.npz", allow_pickle=False
        ) as data:
            if not np.array_equal(data["reference_counts"], arrays["reference_counts"]):
                raise ValueError("baseline uses a different reference image")
            baseline = data["expected_radial_counts"].copy()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 125})
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 10), layout="constrained")
    extent = [arrays["sky_edges_arcsec"][0], arrays["sky_edges_arcsec"][-1]] * 2
    display_reference = rebin_counts(arrays["reference_counts"])
    display_model = rebin_counts(arrays["expected_counts"])
    norm = LogNorm(vmin=0.5, vmax=max(display_reference.max(), display_model.max()))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(cmap(0))  # Zero reference counts are the darkest color.
    for ax, data, title in (
        (axes[0, 0], display_reference, "RefleX: simulated halo counts"),
        (axes[0, 1], display_model, "WD01/D03: predicted camera counts"),
    ):
        im = ax.imshow(
            np.ma.masked_less_equal(data, 0),
            origin="lower",
            extent=extent,
            norm=norm,
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set(title=title, xlabel="Sky X [arcsec]", ylabel="Sky Y [arcsec]")
    colorbar = fig.colorbar(im, ax=axes[0, :], shrink=0.88, pad=0.015)
    colorbar.set_label("Counts per 15 × 15 arcsec display pixel")
    ax = axes[1, 0]
    ax.errorbar(
        midpoints,
        observed / radial_scale,
        yerr=np.sqrt(np.diag(reference_cov)) / radial_scale,
        fmt="o",
        ms=4,
        color="#242424",
        capsize=2,
        label="RefleX ± count SE",
    )
    ax.errorbar(
        midpoints,
        expected / radial_scale,
        yerr=np.sqrt(np.diag(model_cov)) / radial_scale,
        fmt="s-",
        ms=4,
        color="#c64a26",
        capsize=2,
        label="WD01/D03 ± history MC SE",
    )
    if baseline is not None:
        ax.plot(
            midpoints,
            baseline / radial_scale,
            "--",
            color="#277da1",
            label="DSH with RefleX material (previous run)",
        )
    ax.set(
        yscale="log",
        xlabel="Annulus midpoint [arcsec]",
        ylabel="Mean halo fluence [ph cm$^{-2}$ arcsec$^{-2}$]",
        title="Radial profile · fixed absolute normalization",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    ax = axes[1, 1]
    ax.axhline(1, color="#242424", lw=1)
    ax.errorbar(
        midpoints,
        ratio,
        yerr=ratio_se,
        xerr=np.vstack((midpoints - annuli[:-1], annuli[1:] - midpoints)),
        fmt="s",
        ms=4,
        color="#c64a26",
        capsize=2,
        label="WD01 / RefleX ± combined SE",
    )
    if baseline is not None:
        ax.plot(
            midpoints,
            baseline / observed,
            "--",
            color="#277da1",
            label="Previous DSH RefleX-material / RefleX",
        )
    ax.set(
        xlabel="Annulus midpoint [arcsec]",
        ylabel="Predicted / observed halo counts",
        title="Radial differences with independent errors",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    ax.text(
        0.03,
        0.04,
        f"Aperture ratio = {summary['halo_ratio']:.4f} ± {summary['halo_ratio_se']:.4f}\n"
        f"Radial χ² / dof = {summary['global_radial_chi2']:.2f} / {len(observed)}; "
        f"p = {summary['global_radial_pvalue']:.3g}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    fig.suptitle(
        f"Independent material comparison · {config['energy_kev']:g} keV · "
        f"N$_H$ = {config['nh_cm2']:.0e} cm$^{{-2}}$\n"
        f"WD01 C50/Si90 + D03 Mie–ADT + residual-gas TBnew/Verner · "
        f"{summary['histories']:,} DSH histories",
        fontsize=14,
    )
    fig.savefig(output / "wd01_reflex_comparison.png", dpi=180)
    fig.savefig(output / "wd01_reflex_comparison.pdf")
    plt.close(fig)

    with np.load(output / "phase_comparison.npz", allow_pickle=False) as phase:
        angles = phase["physical_angle_arcsec"]
        wd01_phase = phase["wd01_phase_per_sr"]
        reflex_phase = phase["reflex_phase_per_sr"]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [2, 1]},
    )
    axes[0].loglog(angles, reflex_phase, color="#242424", label="RefleX angular law")
    axes[0].loglog(angles, wd01_phase, color="#c64a26", label="WD01/D03 Mie–ADT")
    axes[0].set(
        ylabel="Normalized intrinsic phase [sr$^{-1}$]",
        title=f"Scattering law at {config['energy_kev']:g} keV · physical deflection angle",
    )
    axes[0].legend()
    axes[1].semilogx(angles, wd01_phase / reflex_phase, color="#c64a26")
    axes[1].axhline(1, color="#242424", lw=1)
    axes[1].set(
        xlabel="Physical scattering angle [arcsec]", ylabel="WD01 / RefleX phase"
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(output / "material_phase_comparison.png", dpi=180)
    fig.savefig(output / "material_phase_comparison.pdf")
    plt.close(fig)
    with (output / "radial_profile.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "inner_arcsec",
                "outer_arcsec",
                "selected_area_arcsec2",
                "reflex_counts",
                "wd01_counts",
                "reflex_count_se",
                "wd01_mc_count_se",
                "wd01_over_reflex",
                "ratio_combined_se",
                "wd01_minus_reflex_z",
            ]
        )
        writer.writerows(
            zip(
                annuli[:-1],
                annuli[1:],
                area,
                observed,
                expected,
                np.sqrt(np.diag(reference_cov)),
                np.sqrt(np.diag(model_cov)),
                ratio,
                ratio_se,
                summary["annulus_difference_z"],
                strict=True,
            )
        )
    export_fits(output, arrays, summary)
    print(
        f"Saved comparison PNG/PDF, phase PNG/PDF, radial CSV and native FITS in {output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    plot_comparison(parser.parse_args().output)
