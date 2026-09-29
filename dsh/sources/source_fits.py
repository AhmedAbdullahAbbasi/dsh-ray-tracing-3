"""Canonical intrinsic photon-flux input, independent of spectral producer.

Time is the direct-light arrival time relative to an optional MJD reference.
Within each interval the flux is constant. Continuum values are integrated
over their energy bins; monochromatic lines have integrated photon fluxes.
An observational or XSPEC adapter must make its own absorption, gap and time
conversion choices before writing this format.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _fits_module():
    try:
        from astropy.io import fits
    except ImportError as error:
        raise ImportError("source FITS input requires astropy>=6") from error
    return fits


@dataclass(frozen=True)
class SourceFluxFile:
    """Validated host arrays for one piecewise-constant intrinsic source."""

    time_edges_s: np.ndarray
    continuum_energy_edges_kev: np.ndarray
    continuum_flux: np.ndarray
    continuum_shape: tuple[str, ...]
    continuum_photon_index: np.ndarray
    line_energy_kev: np.ndarray
    line_flux: np.ndarray
    line_labels: tuple[str, ...] = ()
    mjdref: float | None = None
    timesys: str | None = None
    file_sha256: str | None = None

    def __post_init__(self):
        time = np.asarray(self.time_edges_s, dtype=np.float64)
        energy = np.asarray(self.continuum_energy_edges_kev, dtype=np.float64)
        line_energy = np.asarray(self.line_energy_kev, dtype=np.float64)
        continuum = np.asarray(self.continuum_flux, dtype=np.float64)
        line = np.asarray(self.line_flux, dtype=np.float64)
        gamma = np.asarray(self.continuum_photon_index, dtype=np.float64)
        shapes = tuple(str(value).upper() for value in self.continuum_shape)
        labels = tuple(str(value) for value in self.line_labels)

        if time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time)):
            raise ValueError("source time edges must be a finite one-dimensional grid")
        if not np.all(np.diff(time) > 0.0):
            raise ValueError("source time edges must increase strictly")
        if energy.ndim != 1 or (energy.size and (energy.size < 2 or energy[0] <= 0)):
            raise ValueError("continuum energy edges must be empty or positive bins")
        if energy.size and (
            not np.all(np.isfinite(energy)) or not np.all(np.diff(energy) > 0.0)
        ):
            raise ValueError("continuum energy edges must increase strictly")
        if line_energy.ndim != 1 or (
            line_energy.size
            and (
                not np.all(np.isfinite(line_energy))
                or np.any(line_energy <= 0.0)
                or np.any(np.diff(line_energy) <= 0.0)
            )
        ):
            raise ValueError("line energies must be positive, unique and increasing")
        n_time = time.size - 1
        n_cont = max(energy.size - 1, 0)
        n_line = line_energy.size
        if not n_cont and not n_line:
            raise ValueError("source requires continuum or monochromatic lines")
        if (
            continuum.shape != (n_time, n_cont)
            or line.shape != (n_time, n_line)
            or gamma.shape != continuum.shape
        ):
            raise ValueError(
                "source flux/index arrays must match time and energy grids"
            )
        if len(shapes) != n_cont or any(s not in {"FLAT", "POWERLAW"} for s in shapes):
            raise ValueError("each continuum bin needs a FLAT or POWERLAW shape")
        if len(labels) != n_line or any(not label for label in labels):
            raise ValueError("each monochromatic line needs a nonempty label")
        if any(
            not np.all(np.isfinite(values)) or np.any(values < 0.0)
            for values in (continuum, line)
        ):
            raise ValueError("source photon fluxes must be finite and nonnegative")
        if not np.all(np.isfinite(gamma)):
            raise ValueError("continuum photon indices must be finite")
        if not np.sum(continuum) + np.sum(line) > 0.0:
            raise ValueError("source must have positive photon flux")
        if self.mjdref is not None and (
            not np.isfinite(self.mjdref) or not self.timesys
        ):
            raise ValueError("absolute source times require MJDREF and TIMESYS")
        if self.mjdref is None and self.timesys is not None:
            raise ValueError("TIMESYS requires MJDREF")
        for name, value in (
            ("time_edges_s", time),
            ("continuum_energy_edges_kev", energy),
            ("continuum_flux", continuum),
            ("continuum_shape", shapes),
            ("continuum_photon_index", gamma),
            ("line_energy_kev", line_energy),
            ("line_flux", line),
            ("line_labels", labels),
        ):
            object.__setattr__(self, name, value)

    @property
    def continuum_fluence(self) -> float:
        return float(np.sum(self.continuum_flux * np.diff(self.time_edges_s)[:, None]))

    @property
    def line_fluence(self) -> float:
        return float(np.sum(self.line_flux * np.diff(self.time_edges_s)[:, None]))


def write_source_fits(path: str | Path, source: SourceFluxFile) -> Path:
    """Write a canonical source with checksums and independent fluence closure."""

    fits = _fits_module()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU()
    primary.header["DSHSRC"] = 1
    primary.header["FLUXDEF"] = "INTRINSIC"
    primary.header["TFRAME"] = "DIRECT"
    primary.header["TIMEUNIT"] = "s"
    primary.header["FLU_CONT"] = source.continuum_fluence
    primary.header["FLU_LINE"] = source.line_fluence
    if source.mjdref is not None:
        primary.header["MJDREF"] = source.mjdref
        primary.header["TIMESYS"] = source.timesys
    time = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="T_START", format="D", unit="s", array=source.time_edges_s[:-1]
            ),
            fits.Column(
                name="T_STOP", format="D", unit="s", array=source.time_edges_s[1:]
            ),
        ],
        name="TGRID",
    )
    hdus = [primary, time]
    if source.continuum_flux.shape[1]:
        energy = fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name="E_LO",
                    format="D",
                    unit="keV",
                    array=source.continuum_energy_edges_kev[:-1],
                ),
                fits.Column(
                    name="E_HI",
                    format="D",
                    unit="keV",
                    array=source.continuum_energy_edges_kev[1:],
                ),
                fits.Column(
                    name="SHAPE", format="8A", array=np.asarray(source.continuum_shape)
                ),
            ],
            name="EGRID",
        )
        continuum = fits.ImageHDU(source.continuum_flux, name="CFLUX")
        continuum.header["BUNIT"] = "ph cm-2 s-1"
        hdus.extend((energy, continuum))
        if "POWERLAW" in source.continuum_shape:
            hdus.append(fits.ImageHDU(source.continuum_photon_index, name="CGAMMA"))
    if source.line_energy_kev.size:
        width = max(1, *(len(value) for value in source.line_labels))
        lines = fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name="E_LINE", format="D", unit="keV", array=source.line_energy_kev
                ),
                fits.Column(
                    name="LABEL",
                    format=f"{width}A",
                    array=np.asarray(source.line_labels),
                ),
            ],
            name="LINES",
        )
        line = fits.ImageHDU(source.line_flux, name="LFLUX")
        line.header["BUNIT"] = "ph cm-2 s-1"
        hdus.extend((lines, line))
    fits.HDUList(hdus).writeto(destination, overwrite=True, checksum=True)
    return destination


def load_source_fits(path: str | Path) -> SourceFluxFile:
    """Reject noncanonical units, incomplete HDUs and failed fluence closure."""

    fits = _fits_module()
    source_path = Path(path)
    with fits.open(source_path, checksum=True) as hdus:
        if any(hdu.verify_checksum() != 1 or hdu.verify_datasum() != 1 for hdu in hdus):
            raise ValueError("source FITS checksum failed")
        primary = hdus[0].header
        if (
            primary.get("DSHSRC") != 1
            or primary.get("FLUXDEF") != "INTRINSIC"
            or primary.get("TFRAME") != "DIRECT"
            or primary.get("TIMEUNIT") != "s"
        ):
            raise ValueError("unsupported source FITS convention")
        names = {hdu.name for hdu in hdus[1:]}
        continuum_present = {"EGRID", "CFLUX"}.issubset(names)
        lines_present = {"LINES", "LFLUX"}.issubset(names)
        allowed = {"TGRID"}
        if continuum_present:
            allowed.update(("EGRID", "CFLUX"))
            if "CGAMMA" in names:
                allowed.add("CGAMMA")
        if lines_present:
            allowed.update(("LINES", "LFLUX"))
        if names != allowed or not (continuum_present or lines_present):
            raise ValueError("source FITS has missing or unexpected extensions")
        time = hdus["TGRID"].data
        if any(hdus["TGRID"].columns[key].unit != "s" for key in ("T_START", "T_STOP")):
            raise ValueError("source time units must be seconds")
        starts = np.asarray(time["T_START"], dtype=np.float64)
        stops = np.asarray(time["T_STOP"], dtype=np.float64)
        if starts.size == 0 or not np.array_equal(starts[1:], stops[:-1]):
            raise ValueError("source time intervals must be contiguous")
        time_edges = np.concatenate((starts, stops[-1:]))
        if continuum_present:
            grid = hdus["EGRID"].data
            if any(
                hdus["EGRID"].columns[key].unit != "keV" for key in ("E_LO", "E_HI")
            ):
                raise ValueError("continuum energy units must be keV")
            low = np.asarray(grid["E_LO"], dtype=np.float64)
            high = np.asarray(grid["E_HI"], dtype=np.float64)
            if low.size == 0 or not np.array_equal(low[1:], high[:-1]):
                raise ValueError("continuum energy intervals must be contiguous")
            energy_edges = np.concatenate((low, high[-1:]))
            shapes = tuple(value.strip() for value in grid["SHAPE"].astype(str))
            if hdus["CFLUX"].header.get("BUNIT") != "ph cm-2 s-1":
                raise ValueError("continuum flux must be integrated photon flux")
            continuum_flux = np.asarray(hdus["CFLUX"].data, dtype=np.float64)
            if "POWERLAW" in shapes and "CGAMMA" not in names:
                raise ValueError("power-law continuum requires CGAMMA")
            gamma = (
                np.asarray(hdus["CGAMMA"].data, dtype=np.float64)
                if "CGAMMA" in names
                else np.zeros_like(continuum_flux)
            )
        else:
            energy_edges = np.array([], dtype=np.float64)
            continuum_flux = np.empty((starts.size, 0), dtype=np.float64)
            shapes = ()
            gamma = np.empty_like(continuum_flux)
        if lines_present:
            if hdus["LINES"].columns["E_LINE"].unit != "keV":
                raise ValueError("line energies must be keV")
            line_energy = np.asarray(hdus["LINES"].data["E_LINE"], dtype=np.float64)
            labels = tuple(
                value.strip() for value in hdus["LINES"].data["LABEL"].astype(str)
            )
            if hdus["LFLUX"].header.get("BUNIT") != "ph cm-2 s-1":
                raise ValueError("line flux must be integrated photon flux")
            line_flux = np.asarray(hdus["LFLUX"].data, dtype=np.float64)
        else:
            line_energy = np.array([], dtype=np.float64)
            labels = ()
            line_flux = np.empty((starts.size, 0), dtype=np.float64)
        source = SourceFluxFile(
            time_edges_s=time_edges,
            continuum_energy_edges_kev=energy_edges,
            continuum_flux=continuum_flux,
            continuum_shape=shapes,
            continuum_photon_index=gamma,
            line_energy_kev=line_energy,
            line_flux=line_flux,
            line_labels=labels,
            mjdref=primary.get("MJDREF"),
            timesys=primary.get("TIMESYS"),
            file_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        )
        if not np.isclose(
            source.continuum_fluence,
            primary.get("FLU_CONT", np.nan),
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError("continuum fluence closure failed")
        if not np.isclose(
            source.line_fluence,
            primary.get("FLU_LINE", np.nan),
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError("line fluence closure failed")
        return source
