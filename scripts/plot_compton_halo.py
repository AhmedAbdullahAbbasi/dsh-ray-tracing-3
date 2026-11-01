"""Scientific figures and a transparent fixed-reference Compton comparison report."""

from pathlib import Path

import numpy as np


def write_report(root, report, arrays):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from matplotlib.colors import LogNorm

    root = Path(root)
    config = report["config"]
    edges = np.asarray(config["annulus_edges_arcsec"])
    centers = (edges[1:] + edges[:-1]) / 2
    len(centers)
    arrays["joint_radial_mean"]
    report["normalization_counts_per_ph_cm2"]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axs[0, 0]
    colors = {"off": "#2166ac", "on": "#b2182b"}
    for condition in ("off", "on"):
        predicted = arrays[condition + "__radial_counts"]
        error = np.sqrt(
            np.maximum(np.diag(arrays[condition + "__radial_mc_covariance"]), 0)
        )
        observed = arrays[condition + "__reference_radial_counts"]
        ax.errorbar(
            centers,
            observed,
            yerr=np.sqrt(observed),
            fmt="o" if condition == "off" else "s",
            ms=4,
            color=colors[condition],
            label="RefleX " + condition,
        )
        ax.plot(centers, predicted, color=colors[condition], label="DSH " + condition)
        ax.fill_between(
            centers,
            np.maximum(predicted - error, 1e-3),
            predicted + error,
            color=colors[condition],
            alpha=0.15,
        )
    ax.set(
        yscale="log",
        xlabel="Radius (arcsec)",
        ylabel="Camera counts per annulus",
        title="Absolute profiles; no fitted normalization",
    )
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    ax.errorbar(
        centers,
        100 * (arrays["on_off_radial_ratio"] - 1),
        yerr=100 * arrays["on_off_radial_ratio_mc_se"],
        fmt="o-",
        label="DSH: paired MC uncertainty",
    )
    ref = arrays["on__reference_radial_counts"] / arrays["off__reference_radial_counts"]
    ax.plot(
        centers,
        100 * (ref - 1),
        "s",
        color="#b2182b",
        label="RefleX ratio: covariance unavailable",
    )
    ax.axhline(
        100 * (report["near_forward_no_compton_survival_ratio"] - 1),
        ls="--",
        color="0.4",
        label="Near-forward no-Compton survival",
    )
    ax.set(
        xlabel="Radius (arcsec)",
        ylabel="On/off change (%)",
        title="Different uncertainty information",
    )
    ax.legend(fontsize=7)
    ax = axs[1, 0]
    names = ["on_dust_only", "on_compton_only", "on_mixed"]
    values = [report["components"][name]["expected_camera_counts"] for name in names]
    ax.bar(
        ["Dust only", "Compton only", "Mixed"],
        np.maximum(values, 1e-20),
        color=["#b2182b", "#238b45", "#8856a7"],
    )
    ax.axhline(
        report["reflex_non_dust_outside_core_95_upper_expected_counts"],
        ls="--",
        color="0.4",
        label="RefleX zero non-dust outside core: 95% upper bound",
    )
    ax.set(
        yscale="log",
        ylabel="Expected camera counts",
        title="Modeled components (entire field)",
    )
    ax.legend(fontsize=7)
    ax = axs[1, 1]
    stages = report["convergence"]
    x = [s["histories_per_seed"] * report["seed_count"] for s in stages]
    ax.plot(
        x,
        [100 * s["on_relative_mc_se"] for s in stages],
        "o-",
        label="On fluence relative MC SE",
    )
    ax.plot(
        x,
        [100 * s["on_off_dust_selected_ratio_mc_se"] for s in stages],
        "s-",
        label="Paired ratio MC SE (percentage points)",
    )
    ax.axhline(100 * config["maximum_aperture_relative_mc_se"], ls="--", color="C0")
    ax.axhline(100 * config["maximum_delta_ratio_mc_se"], ls="--", color="C1")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Paired histories across three seeds",
        ylabel="Uncertainty (%)",
        title="Correlated prefixes; fixed final budget",
    )
    ax.legend(fontsize=7)
    fig.suptitle(
        "4.9 keV; $N_H=10^{22}$ cm$^{-2}$; WD01/cold model versus RefleX", fontsize=13
    )
    fig.savefig(root / "comparison.png", dpi=180)
    fig.savefig(root / "comparison.pdf")
    plt.close(fig)
    model = arrays["image"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    extent = [-150, 150, -150, 150]
    norm = LogNorm(
        vmin=0.03,
        vmax=max(
            arrays["off_dust"].max(),
            arrays["on_dust"].max(),
            model[0].max(),
            model[4].max(),
        ),
    )
    fields = [
        arrays["off_dust"],
        arrays["on_dust"],
        arrays["on_dust"] - arrays["off_dust"],
        model[0],
        model[4],
        model[4] - model[0],
    ]
    titles = [
        "RefleX off\nRaw dust-selected counts",
        "RefleX on\nRaw dust-selected counts",
        "RefleX on minus off\nShared seed; covariance unavailable",
        "DSH off\nExpected dust-selected counts",
        "DSH on\nExpected dust-selected counts",
        "DSH on minus off\nPaired expectation",
    ]
    for index, (ax, data, title) in enumerate(
        zip(axes.flat, fields, titles, strict=True)
    ):
        if index % 3 == 2:
            limit = max(abs(data).max(), 1e-9)
            im = ax.imshow(
                data,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
        else:
            im = ax.imshow(
                np.ma.masked_less_equal(data, 0),
                origin="lower",
                extent=extent,
                cmap="magma",
                norm=norm,
            )
        ax.set(title=title, xlabel="Sky x (arcsec)", ylabel="Sky y (arcsec)")
        fig.colorbar(im, ax=ax, label="Counts per 5-arcsec pixel", shrink=0.8)
    fig.savefig(root / "images.png", dpi=180)
    fig.savefig(root / "images.pdf")
    plt.close(fig)
    hdus = [fits.PrimaryHDU(model[4])]
    hdus[0].header["BUNIT"] = "expected camera count"
    hdus[0].header["NREF"] = config["reflex_photons"]
    hdus[0].header["NPAIR"] = report["paired_histories"]
    hdus[0].header["NOISE"] = "MC expectation; no Poisson realization"
    for index, name in enumerate(
        ("OFF_DUST", "ON_DUST", "ON_COMPT", "ON_MIXED", "ON_DUSTSEL")
    ):
        hdus.append(fits.ImageHDU(model[index], name=name))
    hdus.append(fits.ImageHDU(arrays["image_mc_variance"][4], name="ON_MCVAR"))
    fits.HDUList(hdus).writeto(
        root / "model_images.fits", overwrite=True, checksum=True
    )
    header = [
        "r_low_arcsec",
        "r_high_arcsec",
        "reflex_off",
        "reflex_on",
        "model_off",
        "model_on",
        "paired_on_off_ratio",
        "paired_ratio_mc_se",
    ]
    values = np.column_stack(
        [
            edges[:-1],
            edges[1:],
            arrays["off__reference_radial_counts"],
            arrays["on__reference_radial_counts"],
            arrays["off__radial_counts"],
            arrays["on__radial_counts"],
            arrays["on_off_radial_ratio"],
            arrays["on_off_radial_ratio_mc_se"],
        ]
    )
    np.savetxt(
        root / "radial_profiles.csv",
        values,
        delimiter=",",
        header=",".join(header),
        comments="",
    )
    table = "\n".join(
        f"| {condition} | {r['reference_camera_counts']} | {r['predicted_camera_counts']:.2f} | {100 * r['model_relative_mc_se']:.3f}% | {r['radial_predictive_pvalue']:.4g} |"
        for condition, r in report["conditions"].items()
    )
    components = "\n".join(
        f"| {name} | {r['expected_camera_counts']:.6g} | {r['mc_standard_error_counts']:.3g} |"
        for name, r in report["components"].items()
    )
    summary = f"""# Compton halo validation: 4.9 keV, NH=1e22 cm^-2

Numerical completion: **{report["numerical_complete"]}**. Sampling sufficient:
**{report["sampling_sufficient"]}**. Within the predeclared RefleX residual
thresholds: **{report["within_declared_reflex_thresholds"]}**. All declared gates:
**{report["passes_declared_gates"]}**. Pilot mode: **{report["pilot"]}**.

**External model equivalence is not established.** The predeclared p-value
threshold is {config["minimum_global_pvalue"]}; passing it does not imply an
absence of radial residuals at a conventional 5% level. Such tension is flagged
for off: **{report["radial_tension_at_5_percent"]["off"]}**, on:
**{report["radial_tension_at_5_percent"]["on"]}**. Examine the p-values below and
retain material/phase differences as an unresolved control.

This run uses {report["seed_count"]} independent seed pairs, each with
{report["histories_per_seed"]:,} source histories ({report["paired_histories"]:,}
paired histories, twice that many transport realizations). Precision is
{report["precision"]}; all moment accumulation is host float64. The current
analysis/coordinator invocation took {report["elapsed_seconds"]:.1f} s and can
exclude prior collection on resume. Per-seed collection timings are saved in
the JSON report and checkpoint directories. The maximum energy-closure error was
{report["maximum_energy_closure_error_kev"]:.3g} keV, with
{report["unsupported_observer_events"]} unsupported observer contributions.

## Absolute comparison

| Condition | RefleX dust-selected counts | DSH expected counts | DSH relative MC SE | Predictive radial p |
| --- | ---: | ---: | ---: | ---: |
{table}

The fixed source cone, 10,000 camera apertures and 1e8 RefleX source photons
set the normalization. No amplitude is fitted. RefleX DUSTSCAT>0 corresponds
to **dust-only plus mixed** histories in the new pipeline. Raw RefleX images
are reflected in y using the accepted camera convention. The camera's finite
0.1-pc circular aperture is sampled at each event using its observer distance.
The accepted 16-cell cloud representation and camera mapping are retained;
this is not an independent rederivation of their geometry calibration.
The outer 150--215 arcsec bin contains only the square field's corners;
no full-annulus area correction is applied.

The predictive radial test uses the model's full photon-grouped annular MC
covariance plus fixed-N camera multinomial counting variance. Its calibration
uses independent Gaussian finite-MC errors and exact multinomial reference
replicates, separately for each condition. The MC approximation requires the
precision gates; p-values from pilots are descriptive. No Poisson likelihood
is applied to weighted DSH pixel values, and no positive-count pixel cut is used.

## Compton change and internal survival control

The modeled dust-selected on/off ratio is **{report["on_off_dust_selected_ratio"]:.7f}
± {report["on_off_dust_selected_ratio_mc_se"]:.7f}** (one paired MC standard error),
a change of **{100 * (report["on_off_dust_selected_ratio"] - 1):.4f}%**.
The RefleX observed ratio is {report["reflex_on_off_ratio_observed"]:.7f}; the two
RefleX conditions use the same seed and their covariance was not supplied.
Consequently, no significance or equivalence interval is assigned to their
on/off difference, and it is not combined with the simulated pair as if independent.

The simulated dust-only survival ratio is {report["dust_only_survival_ratio"]:.7f}
± {report["dust_only_survival_ratio_mc_se"]:.7f}, compared with the near-forward
single-screen no-Compton survival approximation
{report["near_forward_no_compton_survival_ratio"]:.7f}. The internal survival
check passed: **{report["no_compton_survival_control"]}**. This approximation
is not used as a replacement for the full inelastic simulation.

The nominal direct-beam prediction gives
{report["direct_beam_diagnostic"]["conditions"]["off"]["nominal_straight_path_prediction"]:.1f}
off and {report["direct_beam_diagnostic"]["conditions"]["on"]["nominal_straight_path_prediction"]:.1f}
on camera counts. The direct-dominated central reference remainders are
{report["direct_beam_diagnostic"]["conditions"]["off"]["reference_direct_dominated_core_proxy_counts"]:.0f}
and {report["direct_beam_diagnostic"]["conditions"]["on"]["reference_direct_dominated_core_proxy_counts"]:.0f}.
The on remainder can include unresolved Compton-only photons, so this is a
transmission diagnostic rather than validation against a separated direct image.

{report["off_control"]}

| Component | Expected camera counts | Photon-grouped MC SE |
| --- | ---: | ---: |
{components}

RefleX contains zero non-dust photons outside its four central pixels, giving
a one-sided 95% upper expected count of
{report["reflex_non_dust_outside_core_95_upper_expected_counts"]:.4f}. This limits
Compton-only emission outside the core; it does not limit mixed histories or
emission outside the field. Component values in the table integrate the whole
field and must not be mistaken for a matched outside-core measurement.

## Material and observable limits

The primary DSH material is the declared WD01/D03 C50/Si90 mixture with Mie/ADT
dust scattering, residual gas/H2/grain absorption, and cold Compton grain policy
**{config["grain_electrons"]}**. It uses one common elemental budget. RefleX
uses Wilms abundances with H2=0.2 and Fe dust fraction=0.691. These prescriptions
are close in opacity but are not identical inventories or angular laws. The
DSH Compton cross-section is {report["opacity"]["compton_cm2_per_h"]:.8g} cm2/H;
the exported RefleX effective value is
{report["opacity"]["reflex_compton_cm2_per_h"]:.8g} cm2/H. The physical table is
not rescaled to fit the image. Earlier isolated-H controls showed that the
supplied RefleX angular sample follows KN rather than the bound KN*S law.
Agreement here therefore tests this observable and precision, not exact model identity.

The original images have no emergent-energy or arrival-time selection. Our
model stores observed-energy/time summaries, but there is no matching RefleX
product for a spectral or time-delay validation. Realized lower-support/cutoff
losses, guard limits and unsupported virtual escapes would fail numerical
completion. This does not extend the accepted cold prescription below 2 keV.

The precision gates require <=0.6% aperture MC error, <=0.2 percentage-point
paired ratio error, and annular model errors <=half the reference counting SE.
Convergence-prefix estimates share histories and are correlated. The final
budget is fixed; no significance-based early stopping is used. Low-count
components may remain poorly sampled even when the dust-selected halo meets
these criteria. The reference's ~1.3% marginal halo counting error limits claims
about sub-percent external equivalence.

## Outputs and next controls

![Profiles and convergence](comparison.png)

![Images](images.png)

`comparison_arrays.npz` preserves joint paired annular covariance, photon-grouped
pixel variances, component spectra/time summaries and raw reference arrays.
Raw moment sums are retained in the per-seed checkpoints.
`model_images.fits`, CSV, PNG and PDF products use predetermined absolute
camera normalization. Per-seed moment checkpoints permit exact continuation
with matching collection provenance. See `provenance.json` and `report.json`.

Next: run a control with the exported RefleX opacity and angular inputs to
separate material/phase differences from implementation differences. The
present radial residuals do not identify their cause. Then raise the column
and add other energies with newly matched RefleX runs. Request direct,
dust-only, Compton-only and mixed camera products, observed-energy/time
information and independent seed pairs. Existing low-column 2D images cannot
validate those missing observables or high-column physics.
"""
    (root / "REPORT.md").write_text(summary, encoding="utf-8")
