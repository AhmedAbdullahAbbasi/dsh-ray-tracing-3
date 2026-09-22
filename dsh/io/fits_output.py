"""Write complete ideal-observer DSH runs as multi-extension FITS files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..geometry.clouds import AngularDistanceCloud
from ..observer.binning import ObserverBinGeometry
from ..physics.dust import DustPhysicsTable
from ..pipeline import (
    TRANSPORT_STATUS_LABELS,
    IdealObserverSimulationResult,
)
from ..sources.launch import SourceLaunchGeometry
from ..sources.models import TabulatedBandSource


def _import_fits():
    try:
        from astropy.io import fits
    except ImportError as error:
        raise ImportError(
            "writing FITS output requires Astropy; install it with "
            "'python -m pip install astropy'"
        ) from error
    return fits


def _bin_table_hdu(fits, name, edges, unit):
    values = np.asarray(edges, dtype=np.float64)
    columns = [
        fits.Column(name="LOW", format="D", unit=unit, array=values[:-1]),
        fits.Column(name="HIGH", format="D", unit=unit, array=values[1:]),
        fits.Column(
            name="CENTER",
            format="D",
            unit=unit,
            array=0.5 * (values[:-1] + values[1:]),
        ),
    ]
    return fits.BinTableHDU.from_columns(columns, name=name)


def _add_linear_spatial_wcs(header, geometry: ObserverBinGeometry):
    x_edges = np.asarray(geometry.sky_x_edges_arcsec, dtype=np.float64)
    y_edges = np.asarray(geometry.sky_y_edges_arcsec, dtype=np.float64)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    header["CTYPE1"] = ("XOFFSET", "observer sky-x offset")
    header["CUNIT1"] = "arcsec"
    header["CTYPE2"] = ("YOFFSET", "observer sky-y offset")
    header["CUNIT2"] = "arcsec"
    if x_centers.size == 1 or np.allclose(
        np.diff(x_centers), np.diff(x_centers)[0], rtol=1.0e-10
    ):
        header["CRPIX1"] = 1.0
        header["CRVAL1"] = float(x_centers[0])
        header["CDELT1"] = float(np.diff(x_centers)[0]) if x_centers.size > 1 else 1.0
    if y_centers.size == 1 or np.allclose(
        np.diff(y_centers), np.diff(y_centers)[0], rtol=1.0e-10
    ):
        header["CRPIX2"] = 1.0
        header["CRVAL2"] = float(y_centers[0])
        header["CDELT2"] = float(np.diff(y_centers)[0]) if y_centers.size > 1 else 1.0


def _observer_image_hdu(
    fits,
    data,
    name,
    geometry,
    *,
    bunit,
    btype,
):
    hdu = fits.ImageHDU(np.asarray(data), name=name)
    hdu.header["BUNIT"] = bunit
    hdu.header["BTYPE"] = btype
    hdu.header["AXORDER"] = "T,E,Y,X"
    hdu.header["EAXIS"] = "See ENERGY_BINS extension"
    hdu.header["TAXIS"] = "See TIME_BINS extension"
    _add_linear_spatial_wcs(hdu.header, geometry)
    return hdu


def _source_table_hdu(fits, source: TabulatedBandSource):
    time_edges = np.asarray(source.time_edges_s, dtype=np.float64)
    energy = np.asarray(source.effective_energy_kev, dtype=np.float64)
    flux = np.asarray(source.band_flux, dtype=np.float64)
    fluence = np.asarray(source.cell_fluence, dtype=np.float64)
    n_time, n_energy = flux.shape
    time_index, energy_index = np.indices((n_time, n_energy))
    columns = [
        fits.Column(name="TIME_INDEX", format="J", array=time_index.reshape(-1)),
        fits.Column(name="ENERGY_INDEX", format="J", array=energy_index.reshape(-1)),
        fits.Column(
            name="TIME_LOW",
            format="D",
            unit="s",
            array=time_edges[time_index.reshape(-1)],
        ),
        fits.Column(
            name="TIME_HIGH",
            format="D",
            unit="s",
            array=time_edges[time_index.reshape(-1) + 1],
        ),
        fits.Column(
            name="ENERGY",
            format="D",
            unit="keV",
            array=energy[energy_index.reshape(-1)],
        ),
        fits.Column(
            name="PHOTON_FLUX",
            format="D",
            unit="ph cm-2 s-1",
            array=flux.reshape(-1),
        ),
        fits.Column(
            name="FLUENCE",
            format="D",
            unit="ph cm-2",
            array=fluence.reshape(-1),
        ),
        fits.Column(
            name="SAMPLING_CDF",
            format="D",
            array=np.asarray(source.flat_cdf, dtype=np.float64),
        ),
    ]
    return fits.BinTableHDU.from_columns(columns, name="SOURCE")


def _physics_table_hdu(fits, physics: DustPhysicsTable):
    return fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="ENERGY",
                format="D",
                unit="keV",
                array=np.asarray(physics.energy_kev, dtype=np.float64),
            ),
            fits.Column(
                name="SIGMA_SCA",
                format="D",
                unit="cm2 H-1",
                array=np.asarray(
                    physics.scattering_cross_section_cm2_per_h,
                    dtype=np.float64,
                ),
            ),
            fits.Column(
                name="SIGMA_ABS",
                format="D",
                unit="cm2 H-1",
                array=np.asarray(
                    physics.absorption_cross_section_cm2_per_h,
                    dtype=np.float64,
                ),
            ),
        ],
        name="PHYSICS",
    )


def _diagnostics_table_hdu(fits, result: IdealObserverSimulationResult):
    fields = {}
    for container in (result.products, result.diagnostics):
        for name in container._fields:
            value = np.asarray(getattr(container, name))
            if value.ndim == 0:
                fields[name] = value
    columns = []
    for name, value in fields.items():
        if np.issubdtype(value.dtype, np.integer):
            columns.append(
                fits.Column(name=name.upper(), format="K", array=[int(value)])
            )
        else:
            columns.append(
                fits.Column(name=name.upper(), format="D", array=[float(value)])
            )
    return fits.BinTableHDU.from_columns(columns, name="DIAGNOSTICS")


def _status_table_hdu(fits, result: IdealObserverSimulationResult):
    counts = np.asarray(result.diagnostics.transport_status_count, dtype=np.int64)
    width = max(len(label) for label in TRANSPORT_STATUS_LABELS)
    return fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="STATUS_CODE",
                format="J",
                array=np.arange(counts.size, dtype=np.int32),
            ),
            fits.Column(
                name="STATUS",
                format=f"{width}A",
                array=np.asarray(TRANSPORT_STATUS_LABELS),
            ),
            fits.Column(name="COUNT", format="K", array=counts),
        ],
        name="STATUS",
    )


def write_ideal_observer_fits(
    path,
    result: IdealObserverSimulationResult,
    bin_geometry: ObserverBinGeometry,
    source: TabulatedBandSource,
    cloud: AngularDistanceCloud,
    physics: DustPhysicsTable,
    launch_geometry: SourceLaunchGeometry,
    *,
    run_metadata: Mapping[str, object] | None = None,
    overwrite: bool = True,
):
    """Write images, 4D cubes, physical inputs, and diagnostics to FITS.

    The primary HDU is the energy- and time-integrated ideal-observer fluence
    image, so ordinary FITS viewers can open it directly.  Exact bin edges are
    stored in dedicated table extensions; the energy grid is intentionally
    not represented by a misleading linear WCS.
    """

    fits = _import_fits()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    products = result.products
    total = np.asarray(products.total_fluence, dtype=np.float32)
    first = np.asarray(products.first_scatter_fluence, dtype=np.float32)
    multiple = np.asarray(products.multiple_scatter_fluence, dtype=np.float32)
    event_count = np.asarray(products.event_count, dtype=np.int32)
    integrated_total = np.sum(total, axis=(0, 1), dtype=np.float64).astype(np.float32)

    primary = fits.PrimaryHDU(integrated_total)
    primary.header["BUNIT"] = "ph cm-2"
    primary.header["BTYPE"] = "ideal observer fluence"
    primary.header["SIMVER"] = (1, "DSH ideal-observer simulation version")
    primary.header["IDEALOBS"] = (True, "no instrument response applied")
    primary.header["SRCDIST"] = (
        float(np.asarray(cloud.source_distance_kpc)),
        "source distance [kpc]",
    )
    primary.header["SRCFLUX"] = (
        float(np.sum(np.asarray(source.band_flux)[0])),
        "first source interval total flux [ph cm-2 s-1]",
    )
    primary.header["SRCFLUEN"] = (
        float(np.asarray(source.total_fluence)),
        "source total fluence [ph cm-2]",
    )
    _add_linear_spatial_wcs(primary.header, bin_geometry)
    metadata = dict(run_metadata or {})
    header_mapping = {
        "packets": "NPACKETS",
        "chunk_size": "CHUNKSZ",
        "max_interactions": "MAXINTER",
        "seed": "RANDSEED",
    }
    for metadata_name, header_name in header_mapping.items():
        if metadata_name in metadata:
            primary.header[header_name] = int(metadata[metadata_name])
    if "cloud_description" in metadata:
        primary.header["CLOUD"] = str(metadata["cloud_description"])
    if "source_model" in metadata:
        primary.header["SRCMODEL"] = str(metadata["source_model"])
    material_headers = {
        "material_tables": "MATMODEL",
        "scattering_table_sha256": "SCATSHA",
        "absorption_table_sha256": "ABSSHA",
    }
    for field, keyword in material_headers.items():
        if field in metadata:
            primary.header[keyword] = str(metadata[field])
    decay_header_mapping = {
        "decay_time_days": "DCTAU_D",
        "decay_start_days": "DCSTRT_D",
        "decay_duration_days": "DCDUR_D",
        "source_time_bin_days": "SRCTB_D",
    }
    for metadata_name, header_name in decay_header_mapping.items():
        value = metadata.get(metadata_name)
        if value is not None:
            primary.header[header_name] = float(value)
    primary.header.add_history(
        "Source -> importance launch -> voxel transport -> peel-off -> binning"
    )
    primary.header.add_history(
        "No effective area, PSF, exposure map, redistribution, background, or noise"
    )

    solid_angle = np.asarray(bin_geometry.sky_pixel_solid_angle_sr, dtype=np.float32)
    surface_brightness = integrated_total / solid_angle
    total_nh = np.sum(np.asarray(cloud.delta_nh_cm2, dtype=np.float64), axis=0).astype(
        np.float32
    )

    hdus = [
        primary,
        _observer_image_hdu(
            fits,
            total,
            "TOTAL4D",
            bin_geometry,
            bunit="ph cm-2",
            btype="total fluence per 4D bin",
        ),
        _observer_image_hdu(
            fits,
            first,
            "FIRST4D",
            bin_geometry,
            bunit="ph cm-2",
            btype="first-scatter fluence per 4D bin",
        ),
        _observer_image_hdu(
            fits,
            multiple,
            "MULTI4D",
            bin_geometry,
            bunit="ph cm-2",
            btype="multiple-scatter fluence per 4D bin",
        ),
        _observer_image_hdu(
            fits,
            event_count,
            "EVENT4D",
            bin_geometry,
            bunit="count",
            btype="Monte Carlo event count per 4D bin",
        ),
        fits.ImageHDU(
            np.sum(first, axis=(0, 1), dtype=np.float64).astype(np.float32),
            name="FIRSTIMG",
        ),
        fits.ImageHDU(
            np.sum(multiple, axis=(0, 1), dtype=np.float64).astype(np.float32),
            name="MULTIIMG",
        ),
        fits.ImageHDU(surface_brightness.astype(np.float32), name="SURFBRIT"),
        fits.ImageHDU(solid_angle, name="SOLIDANG"),
        fits.ImageHDU(total_nh, name="TOTALNH"),
        fits.ImageHDU(np.asarray(cloud.delta_nh_cm2, dtype=np.float32), name="CLOUDNH"),
        fits.ImageHDU(np.asarray(cloud.n_h_cm3, dtype=np.float32), name="CLOUDDEN"),
        _bin_table_hdu(fits, "X_BINS", bin_geometry.sky_x_edges_arcsec, "arcsec"),
        _bin_table_hdu(fits, "Y_BINS", bin_geometry.sky_y_edges_arcsec, "arcsec"),
        _bin_table_hdu(fits, "ENERGY_BINS", bin_geometry.energy_edges_kev, "keV"),
        _bin_table_hdu(fits, "TIME_BINS", bin_geometry.arrival_time_edges_s, "s"),
        _bin_table_hdu(fits, "Z_BINS", cloud.z_edges_kpc, "kpc"),
        _source_table_hdu(fits, source),
        _physics_table_hdu(fits, physics),
        fits.ImageHDU(
            np.asarray(physics.scattering_angle_rad, dtype=np.float64),
            name="SCATANGL",
        ),
        fits.ImageHDU(
            np.asarray(physics.scattering_angle_cdf, dtype=np.float32),
            name="SCATCDF",
        ),
        fits.ImageHDU(
            np.asarray(
                physics.differential_cross_section_cm2_per_sr_per_h,
                dtype=np.float32,
            ),
            name="DSIGMA",
        ),
        _diagnostics_table_hdu(fits, result),
        _status_table_hdu(fits, result),
    ]

    for name in ("FIRSTIMG", "MULTIIMG", "SURFBRIT", "SOLIDANG", "TOTALNH"):
        hdu = next(item for item in hdus if item.name == name)
        _add_linear_spatial_wcs(hdu.header, bin_geometry)
    next(item for item in hdus if item.name == "FIRSTIMG").header["BUNIT"] = "ph cm-2"
    next(item for item in hdus if item.name == "MULTIIMG").header["BUNIT"] = "ph cm-2"
    next(item for item in hdus if item.name == "SURFBRIT").header["BUNIT"] = (
        "ph cm-2 sr-1"
    )
    next(item for item in hdus if item.name == "SOLIDANG").header["BUNIT"] = "sr"
    next(item for item in hdus if item.name == "TOTALNH").header["BUNIT"] = "cm-2"
    for name, unit in (("CLOUDNH", "cm-2"), ("CLOUDDEN", "cm-3")):
        hdu = next(item for item in hdus if item.name == name)
        hdu.header["BUNIT"] = unit
        hdu.header["AXORDER"] = "Z,Y,X"
    next(item for item in hdus if item.name == "SCATANGL").header["BUNIT"] = "rad"
    next(item for item in hdus if item.name == "DSIGMA").header["BUNIT"] = (
        "cm2 sr-1 H-1"
    )

    primary.header["SLPXMIN"] = float(np.asarray(launch_geometry.slope_x_bounds)[0])
    primary.header["SLPXMAX"] = float(np.asarray(launch_geometry.slope_x_bounds)[1])
    primary.header["SLPYMIN"] = float(np.asarray(launch_geometry.slope_y_bounds)[0])
    primary.header["SLPYMAX"] = float(np.asarray(launch_geometry.slope_y_bounds)[1])
    primary.header["LAUNCHSR"] = float(
        np.asarray(launch_geometry.launch_solid_angle_sr)
    )

    fits.HDUList(hdus).writeto(output_path, overwrite=overwrite, checksum=True)
    return output_path
