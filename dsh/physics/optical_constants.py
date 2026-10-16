"""Pinned Draine (2003) complex indices for the supported 2--10 keV domain."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HC_KEV_UM = 0.0012398419843320026
OPTICAL_DIRECTORY = (
    Path(__file__).resolve().parent.parent / "data/optical_constants/draine2003"
)
OPTICAL_FILES = {
    "silicate": "silD03.dat",
    "graphite_parallel": "graphite_parallel_D03.dat",
    "graphite_perpendicular": "graphite_perpendicular_D03.dat",
}


@dataclass(frozen=True)
class OpticalConstants:
    energy_kev: np.ndarray
    real_index_minus_one: np.ndarray
    imaginary_index: np.ndarray
    source_sha256: str
    source_url: str

    def index_at_energy(self, energy_kev):
        """Piecewise interpolation of m-1; never extrapolate across the domain.

        Interpolate each signed component logarithmically in energy/magnitude
        when its endpoint signs agree, otherwise linearly in log energy. This
        preserves the native edge samples but is not an atomic edge model.
        """
        energy = np.asarray(energy_kev, dtype=np.float64)
        if np.any(~np.isfinite(energy)) or np.any((energy < 2) | (energy > 10)):
            raise ValueError(
                "D03 X-ray provider supports finite energies from 2 to 10 keV"
            )
        upper = np.clip(
            np.searchsorted(self.energy_kev, energy, side="right"),
            1,
            self.energy_kev.size - 1,
        )
        lower = upper - 1
        fraction = np.log(energy / self.energy_kev[lower]) / np.log(
            self.energy_kev[upper] / self.energy_kev[lower]
        )

        def interpolate(values):
            lo, hi = values[lower], values[upper]
            safe_lo = np.maximum(np.abs(lo), np.finfo(float).tiny)
            safe_hi = np.maximum(np.abs(hi), np.finfo(float).tiny)
            log_value = np.sign(lo) * np.exp(
                np.log(safe_lo) + fraction * np.log(safe_hi / safe_lo)
            )
            return np.where(lo * hi > 0, log_value, lo + fraction * (hi - lo))

        return (
            1
            + interpolate(self.real_index_minus_one)
            + 1j * interpolate(self.imaginary_index)
        )


def load_draine_optical_constants(name, directory=OPTICAL_DIRECTORY):
    if name not in OPTICAL_FILES:
        raise ValueError(f"unknown D03 material: {name}")
    directory = Path(directory)
    manifest = json.loads(
        (directory / "source_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported optical manifest")
    filename = OPTICAL_FILES[name]
    raw = (directory / filename).read_bytes()
    record = manifest["files"][filename]
    digest = hashlib.sha256(raw).hexdigest()
    if digest != record["sha256"]:
        raise ValueError("Draine optical-constant checksum mismatch")
    values = np.loadtxt(directory / filename, skiprows=5)
    if values.ndim != 2 or values.shape[1] != 5 or not np.all(np.isfinite(values)):
        raise ValueError("invalid Draine optical-constant columns")
    energy = HC_KEV_UM / values[:, 0]
    if np.any(energy <= 0) or np.any(np.diff(energy) <= 0) or np.any(values[:, 4] < 0):
        raise ValueError(
            "Draine energies must increase and imaginary index must be nonnegative"
        )
    arrays = [
        np.array(a, dtype=np.float64, copy=True)
        for a in (energy, values[:, 3], values[:, 4])
    ]
    for array in arrays:
        array.flags.writeable = False
    return OpticalConstants(*arrays, source_sha256=digest, source_url=record["url"])


def default_wd01_energy_grid():
    """Union of native D03 samples, a 0.1-keV scaffold and regression anchors."""
    nodes = [np.linspace(2, 10, 81), np.array([3.3, 4.9, 6.9])]
    for name in OPTICAL_FILES:
        energy = load_draine_optical_constants(name).energy_kev
        nodes.append(energy[(energy >= 2) & (energy <= 10)])
    return np.unique(np.concatenate(nodes))
