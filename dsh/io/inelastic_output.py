"""Schema-9 additions to the established ideal-observer NPZ and FITS products."""

import json

import numpy as np


def inelastic_payload(result, compton, metadata):
    payload = {
        "output_schema_version": np.asarray(9),
        "observer_energy_convention": np.asarray(
            "outgoing virtual photon energy; source energy retained separately"
        ),
        "compton_metadata_json": np.asarray(
            json.dumps(metadata["compton"], sort_keys=True, allow_nan=False)
        ),
        "analog_energy_sum_convention": np.asarray(
            "unweighted simulated photons; not an isotropic-source calorimetric estimate"
        ),
    }
    for name in compton._fields:
        payload["compton_" + name] = np.asarray(getattr(compton, name))
    for name in result.inelastic_diagnostics._fields:
        payload["inelastic_" + name] = np.asarray(
            getattr(result.inelastic_diagnostics, name)
        )
    for component in ("dust_only", "compton_only", "mixed"):
        group = getattr(result.components, component)
        for name in group._fields:
            payload[component + "__" + name] = np.asarray(getattr(group, name))
    return payload


def append_inelastic_fits(fits, hdus, result, compton, bins, metadata):
    from .fits_output import _observer_image_hdu

    primary = hdus[0].header
    primary["OUTSCHEM"] = 9
    primary["COMPTON"] = True
    primary["ECUTOFF"] = metadata["compton"]["energy_cutoff_kev"]
    primary["HIERARCH OBSERVER ENERGY"] = "outgoing virtual photon energy"
    for component, prefix in (
        ("dust_only", "DUST"),
        ("compton_only", "COMPT"),
        ("mixed", "MIXED"),
    ):
        group = getattr(result.components, component)
        for field, suffix, unit in (
            ("total_fluence", "4D", "ph cm-2"),
            ("first_scatter_fluence", "1ST", "ph cm-2"),
            ("multiple_scatter_fluence", "MULT", "ph cm-2"),
            ("event_count", "EVT", "count"),
            ("total_fluence_squared", "Q4D", "ph2 cm-4"),
        ):
            hdu = _observer_image_hdu(
                fits,
                np.asarray(getattr(group, field)),
                prefix + suffix,
                bins,
                bunit=unit,
                btype=component + " " + field,
            )
            hdu.header["NHIST"] = int(group.history_count)
            hdus.append(hdu)
        for field, suffix in (
            ("time_image_fluence_squared", "QIMG"),
            ("time_order_fluence_sum", "ORDSUM"),
            ("time_order_fluence_cross", "ORDCROSS"),
        ):
            hdu = fits.ImageHDU(
                np.asarray(getattr(group, field), dtype=np.float64),
                name=prefix + suffix,
            )
            hdu.header["BUNIT"] = "ph cm-2" if suffix == "ORDSUM" else "ph2 cm-4"
            hdus.append(hdu)
    for field in compton._fields:
        hdu = fits.ImageHDU(
            np.asarray(getattr(compton, field)), name="COMPT_" + field.upper()
        )
        hdu.header["BUNIT"] = {
            "energy_kev": "keV",
            "cosine": "1",
            "cross_section_cm2_per_h": "cm2 H-1",
            "phase_pdf_per_sr": "sr-1",
            "cosine_cdf": "1",
        }[field]
        hdus.append(hdu)
    groups = result.components[1:]
    columns = [
        fits.Column(
            name="COMPONENT", format="12A", array=["dust_only", "compton_only", "mixed"]
        )
    ]
    for field in groups[0]._fields:
        value = np.asarray(getattr(groups[0], field))
        if value.ndim == 0:
            columns.append(
                fits.Column(
                    name=field.upper(),
                    format="K" if value.dtype.kind in "iu" else "D",
                    array=[getattr(group, field) for group in groups],
                )
            )
    hdus.append(fits.BinTableHDU.from_columns(columns, name="COMPONENTS"))
    columns = [
        fits.Column(
            name=name.upper(), format="K" if "count" in name else "D", array=[value]
        )
        for name, value in zip(
            result.inelastic_diagnostics._fields,
            result.inelastic_diagnostics,
            strict=True,
        )
    ]
    hdu = fits.BinTableHDU.from_columns(columns, name="COMPT_DIAGNOSTICS")
    hdu.header["ENERUNIT"] = "keV; unweighted analog photon sums"
    hdus.append(hdu)
    text = json.dumps(metadata["compton"], sort_keys=True, allow_nan=False)
    hdus.append(
        fits.BinTableHDU.from_columns(
            [fits.Column(name="JSON", format=f"{len(text)}A", array=[text])],
            name="COMPT_METADATA",
        )
    )
