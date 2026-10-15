"""Convert the inspected RefleX 3.0 DUMP format into DSH material tables.

NumPy is the only dependency. This is a deliberately narrow FITS reader for
the supplied executable's unscaled 8-field CROSSSECTIONS extension, not a
general FITS reader. The exported angular distribution matches its analytic
Dust_angular sampler, with the negligible theta > pi tail excluded explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_header(raw, offset):
    header = {}
    for pos in range(offset, len(raw), 80):
        card = raw[pos:pos + 80].decode("ascii")
        key = card[:8].strip()
        if key == "END":
            return header, ((pos + 80 + 2879) // 2880) * 2880
        if card[8:10] != "= ":
            continue
        value = card[10:].split("/")[0].strip()
        if value.startswith("'"):
            value = value.strip("'").strip()
        elif value in ("T", "F"):
            value = value == "T"
        else:
            value = float(value.replace("D", "E")) if any(c in value for c in ".EeD") else int(value)
        header[key] = value
    raise ValueError("missing FITS END card")


def read_reflex_dump(path):
    raw = Path(path).read_bytes()
    primary, offset = read_header(raw, 0)
    if primary.get("NAXIS") != 0:
        raise ValueError("expected empty primary HDU")
    header, offset = read_header(raw, offset)
    names = ["PHOTO", "RAYLEIGH", "COMPTON", "DUST", "ALBEDO", "THETA", "PHOTO_ELEM", "RAYLEIGH_ELEM"]
    forms = ["1E"] * 6 + ["31E"] * 2
    if header.get("XTENSION") != "BINTABLE" or header.get("EXTNAME") != "CROSSSECTIONS":
        raise ValueError("expected CROSSSECTIONS binary table")
    if header.get("TFIELDS") != 8 or header.get("NAXIS1") != 272 or header.get("PCOUNT") != 0:
        raise ValueError("unsupported RefleX table layout")
    for index, (name, form) in enumerate(zip(names, forms), 1):
        if header.get(f"TTYPE{index}") != name or header.get(f"TFORM{index}") != form:
            raise ValueError("unsupported field layout")
        if any(f"{prefix}{index}" in header for prefix in ("TSCAL", "TZERO", "TNULL")):
            raise ValueError("scaled/null FITS fields are unsupported")
    if header.get("STEP") != 1.0 or header.get("NAXIS2") != 20000:
        raise ValueError("only the inspected 1-eV, 20000-row RefleX 3.0 grid is supported")
    for flag, expected in {"PHOTO": 1, "DUSTPHOT": 1, "DUSTSCAT": 1, "COMPTON": 0, "RAYLEIGH": 0, "FLUOR": 0}.items():
        if header.get(flag) != expected:
            raise ValueError(f"DUMP must contain absorption plus dust scattering only: {flag}")
    expected_size = offset + 20000 * 272
    if len(raw) < expected_size:
        raise ValueError("truncated FITS data")
    rows = np.frombuffer(raw, dtype=">f4", count=20000 * 68, offset=offset).reshape(20000, 68)
    rows = rows[:, :6].astype(np.float64)
    if not np.all(np.isfinite(rows)) or np.any(rows[:, :4] < 0) or np.any((rows[:, 4] < 0) | (rows[:, 4] > 1)):
        raise ValueError("invalid RefleX opacity/albedo values")
    # Computed energy nodes: 1..10000 eV, then 10100..1010000 eV.
    # DUST is already composition/depletion scaled by DUMP.
    energy_kev = np.concatenate([np.arange(1, 10001),
                                 10000 + 100 * np.arange(1, 10001)]) / 1000.0
    return header, energy_kev, rows


def runtime_fields(energy_kev, rows):
    """Reproduce this executable's get_index/get_all_sigma lookups in 2-10 keV.

    Below 10 keV its index is floor(E_eV/STEP), with no subtraction of one;
    at 10 keV it switches to the coarser branch and index 9999. Export the
    executable's runtime coefficients rather than silently shifting its grid.
    The float32 DUMP introduces at most its own rounding error.
    """
    ev = np.asarray(energy_kev) * 1000.0
    low = ev < 10000.0
    lower = np.where(low, np.floor(ev), 9999 + np.floor((ev - 10000) / 100)).astype(int)
    fraction = np.where(low, ev - np.floor(ev), ((ev - 10000) % 100) / 100)
    return rows[lower] + fraction[:, None] * (rows[lower + 1] - rows[lower])


def phase_integral(angle, differential):
    integrand = 2 * np.pi * np.sin(angle)[None, :] * differential
    steps = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * np.diff(angle)
    cumulative = np.concatenate([np.zeros((len(differential), 1)), np.cumsum(steps, axis=1)], axis=1)
    sigma = cumulative[:, -1]
    return sigma, cumulative / sigma[:, None]


def export_materials(dump_path, output, energies=(3.3, 4.9, 6.9), positive_angles=16384):
    header, _, native = read_reflex_dump(dump_path)
    energy = np.asarray(energies, dtype=np.float64)
    if energy.ndim != 1 or len(energy) < 2 or not np.all(np.isfinite(energy)) or np.any(np.diff(energy) <= 0) or energy[0] < 2 or energy[-1] > 10:
        raise ValueError("supply at least two increasing energies in 2-10 keV")
    if positive_angles < 1024:
        raise ValueError("at least 1024 positive angular nodes are required")
    selected = runtime_fields(energy, native)
    # RefleX interpolates total dust opacity and albedo separately, and then
    # selects absorption versus scattering using that interpolated albedo.
    sigma_sca = selected[:, 3] * selected[:, 4] * 1e-18  # Mb -> cm2/H
    sigma_abs = (selected[:, 0] + selected[:, 3] * (1 - selected[:, 4])) * 1e-18
    scale = selected[:, 5]
    if np.any(scale <= 0) or np.any(sigma_sca <= 0) or np.any(sigma_abs <= 0):
        raise ValueError("positive scattering, absorption and angle scale are required")
    angle = np.concatenate([[0.0], np.geomspace(0.001 * np.pi / 648000, np.pi, positive_angles)])
    angle[-1] = np.pi
    # RefleX samples alpha = theta50 * sqrt(u/(1-u)). Its angle PDF is
    # 2 alpha/theta50^2/(1+(alpha/theta50)^2)^2. Convert this PDF to sr^-1
    # using sin(alpha), retaining the exact DSH solid-angle convention.
    shape = np.empty((len(energy), len(angle)))
    shape[:, 0] = 1 / (np.pi * scale**2)
    middle = angle[1:-1][None, :]
    shape[:, 1:-1] = middle / np.sin(middle) / (np.pi * scale[:, None]**2) / (1 + (middle / scale[:, None])**2)**2
    shape[:, -1] = 0.0  # endpoint singularity; omitted mass is audited below
    norm, _ = phase_integral(angle, shape)
    differential = shape * (sigma_sca / norm)[:, None]
    integrated, cdf = phase_integral(angle, differential)
    cdf[:, 0] = 0.0
    cdf[:, -1] = 1.0
    analytic_cdf = angle[None, :]**2 / (angle[None, :]**2 + scale[:, None]**2)
    analytic_cdf /= analytic_cdf[:, -1:]
    error = np.max(np.abs(cdf - analytic_cdf), axis=1)
    if np.max(error) > 2e-6:
        raise ValueError("angular discretization exceeds 2e-6 CDF tolerance")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    common = {
        "schema_version": 1, "energy_kev": energy.tolist(),
        "producer": "RefleX 3.0 CROSSSECTIONS DUMP plus inspected Dust_angular analytic sampler",
        "source_dump_sha256": digest(dump_path),
        "source_dump_header": header,
        "opacity_conversion": "DUMP cross-sections in Mb per H multiplied by 1e-18; DUST column already depletion-scaled",
        "energy_lookup": "inspected executable get_index/get_all_sigma, including its below-10-keV row offset; native DUMP float32 precision",
        "scope": "initial monochromatic comparison at listed energy nodes; continuous-energy interpolation not validated",
    }
    scattering = output / "reflex_scattering_v1.npz"
    np.savez_compressed(scattering, energy_kev=energy, scattering_angle_rad=angle,
                        differential_cross_section_cm2_per_sr_per_h=differential,
                        scattering_cross_section_cm2_per_h=integrated, scattering_angle_cdf=cdf)
    metadata = dict(common, quantity="intrinsic_differential_cross_section",
                    differential_cross_section_unit="cm2 sr-1 H-1", integrated_cross_section_unit="cm2 H-1",
                    table_sha256=digest(scattering), angular_scale_rad=scale.tolist(),
                    angle_sampler="alpha=theta50*sqrt(u/(1-u)); uniform azimuth",
                    normalization="phase integrated using 2*pi*sin(alpha); opacity preserved",
                    max_cdf_absolute_error=error.tolist(),
                    reflex_probability_alpha_above_pi=(scale**2 / (np.pi**2 + scale**2)).tolist(),
                    tail_policy="condition on alpha<=pi; endpoint set to zero; no exact match claimed for folded ultra-rare angles above pi")
    scattering.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    absorption = output / "reflex_absorption_v1.npz"
    np.savez_compressed(absorption, energy_kev=energy, absorption_cross_section_cm2_per_h=sigma_abs)
    absorption.with_suffix(".json").write_text(json.dumps(dict(common,
        quantity="photoelectric_absorption_cross_section", cross_section_unit="cm2 H-1",
        table_sha256=digest(absorption), composition="depleted gas PHOTO + DUST*(1-ALBEDO); no TBabs addition"), indent=2) + "\n")
    return dict(energy_kev=energy.tolist(), sigma_sca_cm2_per_h=integrated.tolist(),
                sigma_abs_cm2_per_h=sigma_abs.tolist(), theta50_arcsec=(scale * 648000 / np.pi).tolist(),
                max_cdf_absolute_error=error.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("materials"))
    parser.add_argument("--energies", type=float, nargs="+", default=[3.3, 4.9, 6.9])
    args = parser.parse_args()
    print(json.dumps(export_materials(args.dump, args.output, args.energies), indent=2))


if __name__ == "__main__":
    main()
