"""Pool independent Stage 9F spectral reports at photon-history level.

The original reports retain each scalar sum and effective-history count. Those
two values recover its squared photon contributions, including zero-score
histories. Each run's source weights are rescaled to the pooled history count.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def pool_scalar(rows, histories, values, effective):
    """Return pooled mean, standard error, and effective history count."""
    total = sum(histories)
    measured = 0.0
    squared = 0.0
    for row, n in zip(rows, histories, strict=True):
        value = float(row[values])
        count = float(row[effective])
        if value < 0 or count < 0 or (value > 0 and count == 0):
            raise ValueError("invalid scalar sum or effective-history count")
        factor = n / total
        measured += factor * value
        squared += factor**2 * value**2 / count if count else 0.0
    variance = total / (total - 1) * (squared - measured**2 / total)
    if variance < -1e-12 * max(squared, 1e-100):
        raise ValueError("pooled photon variance is negative")
    return measured, math.sqrt(max(variance, 0.0)), (
        measured**2 / squared if squared else 0.0
    )


def _compare(analog, reference, error_analog, error_reference, sigma_limit):
    error = math.hypot(error_analog, error_reference)
    z = (analog - reference) / error if error > 0 else None
    return {
        "analog": analog,
        "reference": reference,
        "standard_error": error,
        "z": z,
        "passed": bool(z is not None and abs(z) <= sigma_limit),
    }


def _matching_inputs(baseline, supplement):
    if baseline.get("stage") != "9F_continuous_spectrum_absorbed_observer" or (
        supplement.get("stage") != baseline["stage"]
    ):
        raise ValueError("reports describe different validation stages")
    for report in (baseline, supplement):
        if not all(
            value for key, value in report["case"]["checks"].items()
            if key != "all_band_checks"
        ):
            raise ValueError("input report failed a global validity check")
    for key in ("git_head", "scattering_table_sha256", "absorption_table_sha256"):
        if baseline[key] != supplement[key]:
            raise ValueError(f"reports have different {key}")
    for key in (
        "max_interactions",
        "sigma_limit",
        "minimum_effective_histories",
        "quadrature_rtol",
        "maximum_time_bin_relative_error",
        "spectral_nodes",
    ):
        if baseline["config"][key] != supplement["config"][key]:
            raise ValueError(f"reports have different {key}")
    for key in (
        "spectrum",
        "time_edges_s",
        "pivot_energy_kev",
        "pivot_tau_scattering",
        "column_cm2",
        "quadrature",
    ):
        if baseline["case"][key] != supplement["case"][key]:
            raise ValueError(f"reports have different {key}")
    seeds_a = baseline["config"]["seeds"]
    seeds_b = supplement["config"]["seeds"]
    if len(set(seeds_a + seeds_b)) != len(seeds_a) + len(seeds_b):
        raise ValueError("pooled reports reuse random seeds")
    if len(baseline["case"]["bands"]) != len(supplement["case"]["bands"]):
        raise ValueError("reports have different energy bands")
    for a, b in zip(
        baseline["case"]["bands"], supplement["case"]["bands"], strict=True
    ):
        for key in (
            "band_kev", "source_probability", "quadrature_coarse", "quadrature_fine"
        ):
            if a[key] != b[key]:
                raise ValueError(f"reports have different {key}")
        for band, report in ((a, baseline), (b, supplement)):
            expected = band["packets_per_seed"] * len(report["config"]["seeds"])
            if sum(band["analog_statuses"]) != expected or (
                sum(band["pure_scattering_statuses"]) != expected
            ):
                raise ValueError("input report failed status-count closure")


def merge_reports(baseline, supplement, merge_bands=(0, 2)):
    """Recompute acceptance gates after pooling selected independent bands."""
    _matching_inputs(baseline, supplement)
    if not merge_bands or len(set(merge_bands)) != len(merge_bands) or any(
        band not in range(len(baseline["case"]["bands"])) for band in merge_bands
    ):
        raise ValueError("invalid band selection")
    config = baseline["config"]
    sigma = config["sigma_limit"]
    minimum = config["minimum_effective_histories"]
    maximum_bin_error = config["maximum_time_bin_relative_error"]
    reports = []
    for index, (a, b) in enumerate(
        zip(baseline["case"]["bands"], supplement["case"]["bands"], strict=True)
    ):
        if index not in merge_bands:
            reports.append(
                {key: value for key, value in a.items() if key != "per_seed"}
            )
            reports[-1]["pooled"] = False
            continue
        n = [
            a["packets_per_seed"] * len(baseline["config"]["seeds"]),
            b["packets_per_seed"] * len(supplement["config"]["seeds"]),
        ]
        bins = []
        for left, right in zip(
            a["first_order_time_bins"], b["first_order_time_bins"], strict=True
        ):
            if left["time_index"] != right["time_index"]:
                raise ValueError("reports have different time bins")
            measured, se, neff = pool_scalar(
                [left, right], n, "analog", "effective_histories"
            )
            reference = left["reference"]
            quadrature_error = left["quadrature_difference"]
            row = _compare(measured, reference, se, quadrature_error, sigma)
            relative = se / measured if measured > 0 else None
            quad_relative = quadrature_error / max(reference, 1e-100)
            row.update(
                time_index=left["time_index"],
                photon_standard_error=se,
                quadrature_difference=quadrature_error,
                quadrature_relative_change=quad_relative,
                effective_histories=neff,
                relative_photon_standard_error=relative,
                powered=neff >= minimum,
                precise=relative is not None and relative <= maximum_bin_error,
                quadrature_converged=quad_relative <= config["quadrature_rtol"],
            )
            row["passed"] = bool(
                row["passed"]
                and row["powered"]
                and row["precise"]
                and row["quadrature_converged"]
            )
            bins.append(row)
        orders = {}
        for order in ("first", "second", "third_plus"):
            left = a["order_comparisons"][order]
            right = b["order_comparisons"][order]
            analog, se_analog, na = pool_scalar(
                [
                    {"value": x["analog"], "neff": x["effective_histories"][0]}
                    for x in (left, right)
                ],
                n,
                "value",
                "neff",
            )
            pure, se_pure, nb = pool_scalar(
                [
                    {"value": x["reference"], "neff": x["effective_histories"][1]}
                    for x in (left, right)
                ],
                n,
                "value",
                "neff",
            )
            row = _compare(analog, pure, se_analog, se_pure, sigma)
            relative = se_analog / analog if analog > 0 else None
            row.update(
                effective_histories=[na, nb],
                relative_analog_standard_error=relative,
                powered=min(na, nb) >= minimum,
                precise=relative is not None
                and relative <= (0.05 if order == "first" else 0.10),
            )
            row["passed"] = bool(row["passed"] and row["powered"] and row["precise"])
            orders[order] = row
        analog_status = [
            x + y
            for x, y in zip(a["analog_statuses"], b["analog_statuses"], strict=True)
        ]
        pure_status = [
            x + y
            for x, y in zip(
                a["pure_scattering_statuses"],
                b["pure_scattering_statuses"],
                strict=True,
            )
        ]
        checks = {
            "first_order_each_time_bin": all(row["passed"] for row in bins),
            "analog_vs_explicit_absorption_each_order": all(
                row["passed"] for row in orders.values()
            ),
            "no_interaction_limit_or_invalid_analog_terminal": not any(
                analog_status[i] for i in (0, 4, 5, 6)
            ),
            "no_interaction_limit_or_invalid_pure_terminal": not any(
                pure_status[i] for i in (0, 4, 5, 6)
            ),
            "status_count_closure": sum(analog_status) == sum(pure_status) == sum(n),
        }
        reports.append(
            {
                "band_kev": a["band_kev"],
                "source_probability": a["source_probability"],
                "pooled": True,
                "histories": sum(n),
                "component_histories": n,
                "first_order_time_bins": bins,
                "order_comparisons": orders,
                "analog_statuses": analog_status,
                "pure_scattering_statuses": pure_status,
                "checks": checks,
                "all_passed": all(checks.values()),
            }
        )
    all_passed = all(row["all_passed"] for row in reports)
    return {
        "stage": "9F_continuous_spectrum_pooled_audit",
        "source_git_head": baseline["git_head"],
        "baseline_seeds": baseline["config"]["seeds"],
        "supplement_seeds": supplement["config"]["seeds"],
        "merge_band_indices": list(merge_bands),
        "source_spectrum": baseline["case"]["spectrum"],
        "bands": reports,
        "all_passed": all_passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("supplement", type=Path)
    parser.add_argument("--merge-bands", nargs="+", type=int, default=[0, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    supplement = json.loads(args.supplement.read_text())
    report = merge_reports(baseline, supplement, args.merge_bands)
    report["inputs"] = [str(args.baseline), str(args.supplement)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f'Saved {args.output}; all_passed={report["all_passed"]}')
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
