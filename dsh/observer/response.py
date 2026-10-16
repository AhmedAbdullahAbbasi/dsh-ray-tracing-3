"""Reusable elastic energy/sky/delay samples for intrinsic spectral histories.

This is a physical halo response for one static cloud and material model.
It includes the extinction already applied by transport and peel-off scoring.
It has no detector area, redistribution, PSF, or direct unscattered beam.
Keeping exact event energies and delays avoids interpolation across edges or
rounding delays to time-bin centers. All orders from a history stay grouped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io.atomic import atomic_savez_compressed
from ..sources.histories import _edges, integrated_spectral_density, spectral_density
from .scoring import PC_LIGHT_TRAVEL_TIME_S

_ARRAYS = (
    "energy_kev",
    "delay_s",
    "sky_x_arcsec",
    "sky_y_arcsec",
    "scattering_order",
    "history_index",
    "weight_kev",
)
_PROVENANCE = ("material_sha256", "cloud_sha256", "launch_sha256", "runtime_sha256")


def array_fingerprint(arrays):
    """Content hash including array names, shapes, dtypes, and exact bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise ValueError("object arrays cannot be fingerprinted")
        descriptor = json.dumps([name, array.dtype.str, list(array.shape)])
        digest.update(descriptor.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SpectralResponse:
    energy_kev: np.ndarray
    delay_s: np.ndarray
    sky_x_arcsec: np.ndarray
    sky_y_arcsec: np.ndarray
    scattering_order: np.ndarray
    history_index: np.ndarray
    weight_kev: np.ndarray
    n_histories: int
    energy_support_kev: tuple[float, float]
    provenance: dict

    def __post_init__(self):
        if not isinstance(self.n_histories, int) or self.n_histories < 2:
            raise ValueError("response requires at least two launched histories")
        support = _edges(self.energy_support_kev, "energy support", positive=True)
        if len(support) != 2:
            raise ValueError("energy support must be a pair")
        size = np.asarray(self.energy_kev).size
        for name in _ARRAYS:
            original = np.asarray(getattr(self, name))
            if (
                original.ndim != 1
                or original.size != size
                or not np.all(np.isfinite(original))
            ):
                raise ValueError("response arrays must be matching finite vectors")
            if (
                name in ("history_index", "scattering_order")
                and original.dtype.kind not in "iu"
            ):
                raise ValueError("response history indices and orders must be integers")
            array = original.astype(
                np.int64
                if name in ("history_index", "scattering_order")
                else np.float64,
                copy=True,
            )
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        if (
            np.any(self.delay_s < 0)
            or np.any(self.weight_kev < 0)
            or np.any(self.energy_kev < support[0])
            or np.any(self.energy_kev > support[-1])
            or np.any(self.scattering_order < 1)
            or np.any(self.history_index < 0)
            or np.any(self.history_index >= self.n_histories)
            or np.any(np.diff(self.history_index) < 0)
        ):
            raise ValueError(
                "response has invalid support, weights, delays, orders, or history indices"
            )
        provenance = json.loads(json.dumps(self.provenance, allow_nan=False))
        if any(
            not isinstance(provenance.get(key), str)
            or len(provenance[key]) != 64
            or any(c not in "0123456789abcdef" for c in provenance[key])
            for key in _PROVENANCE
        ):
            raise ValueError(
                "response requires material, cloud, launch, and runtime SHA256 provenance"
            )
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "energy_support_kev", tuple(map(float, support)))


def response_from_events(events, packets, proposal, transport_status, *, provenance):
    """Remove source/proposal weights from completed production histories.

    The proposal is a continuous source with one time interval and positive
    flux in every energy band. Its normalized spectral density is q(E).
    For N launched histories, each saved sample is G/(N*q(E)) [keV], where
    G is the scorer's dimensionless angular/escape contribution. ``packets``
    must be exactly those used to launch these histories. Zero-score histories
    are included in N; truncated or numerically invalid histories are rejected.
    """
    energy = np.asarray(packets.energy_kev, dtype=float)
    weights = np.asarray(packets.weight_observer_fluence, dtype=float)
    n = energy.size
    status = np.asarray(transport_status)
    if status.shape != (n,) or not np.all(np.isin(status, [1, 2, 3])):
        raise ValueError("response requires completed, uncapped, valid histories")
    if (
        proposal.band_flux.shape[0] != 1
        or np.any(np.asarray(proposal.band_flux) <= 0)
        or weights.shape != (n,)
        or np.any(weights <= 0)
        or not np.all(np.isfinite(weights))
    ):
        raise ValueError(
            "response proposal requires one interval and positive band/packet weights"
        )
    q = spectral_density(proposal, energy)[0] / np.asarray(proposal.band_flux).sum()
    if np.any(q <= 0) or not np.all(np.isfinite(q)):
        raise ValueError("proposal has no support at a sampled energy")
    valid = np.asarray(events.valid, dtype=bool)
    if valid.ndim != 2 or valid.shape[0] != n:
        raise ValueError("event rows must correspond to launched histories")
    rows = np.broadcast_to(np.arange(n)[:, None], valid.shape)[valid]
    event_energy = np.asarray(events.energy_kev)[valid]
    if not np.allclose(event_energy, energy[rows], rtol=2e-6, atol=0):
        raise ValueError("response requires elastic events at the launched energies")
    return SpectralResponse(
        energy_kev=event_energy,
        delay_s=np.asarray(events.excess_path_length_pc)[valid].astype(float)
        * PC_LIGHT_TRAVEL_TIME_S,
        sky_x_arcsec=np.asarray(events.sky_x_arcsec)[valid],
        sky_y_arcsec=np.asarray(events.sky_y_arcsec)[valid],
        scattering_order=np.asarray(events.scattering_order)[valid],
        history_index=rows,
        weight_kev=np.asarray(events.weight_observer_fluence)[valid]
        / (weights[rows] * n * q[rows]),
        n_histories=n,
        energy_support_kev=(
            float(proposal.energy_edges_kev[0]),
            float(proposal.energy_edges_kev[-1]),
        ),
        provenance=provenance,
    )


@dataclass(frozen=True)
class FoldedSpectralResponse:
    """Fluence [ph cm^-2] and MC uncertainty, axes (time, energy, radius).

    Covariance uses that C-order flattening. It describes finite response
    histories, including zero scorers and correlations between all events of
    each photon. It excludes source fit errors and detector counting noise.
    """

    fluence: np.ndarray
    standard_error: np.ndarray
    covariance: np.ndarray | None
    n_histories: int


def fold_spectral_response(
    response,
    source,
    arrival_time_edges_s,
    energy_edges_kev,
    radial_edges_arcsec,
    *,
    scattering_orders=None,
    covariance=True,
    history_block_size=2048,
):
    """Integrate a history exactly in time at each response sample's energy.

    Energy/sky selection uses half-open bins, including the last upper edge.
    Every source energy must be covered by the proposal. A finite arrival or
    radial window intentionally loses fluence; nothing is renormalized.
    """
    times = _edges(arrival_time_edges_s, "arrival_time_edges_s")
    energies = _edges(energy_edges_kev, "energy_edges_kev", positive=True)
    radii = _edges(radial_edges_arcsec, "radial_edges_arcsec")
    if (
        radii[0] < 0
        or not isinstance(history_block_size, int)
        or history_block_size <= 0
    ):
        raise ValueError("radii must be nonnegative and history_block_size positive")
    if source.energy_edges_kev is None or source.photon_index is None:
        raise ValueError("folding requires a continuous spectral history")
    bounds = np.asarray(source.energy_edges_kev)
    if (
        bounds[0] < response.energy_support_kev[0]
        or bounds[-1] > response.energy_support_kev[1]
    ):
        raise ValueError("source energy support extends outside response proposal")
    shape = (len(times) - 1, len(energies) - 1, len(radii) - 1)
    n_bins = int(np.prod(shape))
    # Bound the optional dense covariance before allocating it.
    if covariance and n_bins > 2048:
        raise ValueError(
            "dense covariance limited to 2048 bins; request covariance=False"
        )
    total = np.zeros(n_bins)
    second = np.zeros((n_bins, n_bins)) if covariance else np.zeros(n_bins)
    radius = np.hypot(response.sky_x_arcsec, response.sky_y_arcsec)

    def bin_index(axis, values):
        index = np.searchsorted(axis, values, side="right") - 1
        return np.where(values == axis[-1], len(axis) - 2, index)

    ebin, rbin = bin_index(energies, response.energy_kev), bin_index(radii, radius)
    selected = (ebin >= 0) & (ebin < shape[1]) & (rbin >= 0) & (rbin < shape[2])
    if scattering_orders is not None:
        orders = np.asarray(scattering_orders)
        if orders.ndim != 1 or orders.dtype.kind not in "iu" or np.any(orders < 1):
            raise ValueError("scattering_orders must be positive integers")
        selected &= np.isin(response.scattering_order, orders)
    ids = np.flatnonzero(selected)
    selected_histories = response.history_index[ids]
    for start in range(0, response.n_histories, history_block_size):
        stop = min(start + history_block_size, response.n_histories)
        left, right = np.searchsorted(selected_histories, [start, stop])
        block = ids[left:right]
        if not block.size:
            continue
        grouped = np.zeros((stop - start, n_bins))
        history = response.history_index[block] - start
        spatial = ebin[block] * shape[2] + rbin[block]
        for t in range(shape[0]):
            integrated = integrated_spectral_density(
                source,
                response.energy_kev[block],
                times[t] - response.delay_s[block],
                times[t + 1] - response.delay_s[block],
            )
            contribution = response.weight_kev[block] * integrated
            np.add.at(
                grouped, (history, t * shape[1] * shape[2] + spatial), contribution
            )
        total += grouped.sum(axis=0)
        second += grouped.T @ grouped if covariance else np.sum(grouped**2, axis=0)
    n = response.n_histories
    correction = np.outer(total, total) / n if covariance else total**2 / n
    uncertainty = n / (n - 1) * (second - correction)
    variance = np.diag(uncertainty) if covariance else uncertainty
    return FoldedSpectralResponse(
        total.reshape(shape),
        np.sqrt(np.maximum(variance, 0)).reshape(shape),
        uncertainty if covariance else None,
        n,
    )


def write_spectral_response(path, response):
    """Save exact samples, declared units, provenance, and a content checksum."""
    arrays = {key: getattr(response, key) for key in _ARRAYS}
    metadata = {
        "schema_version": 1,
        "n_histories": response.n_histories,
        "energy_support_kev": response.energy_support_kev,
        "provenance": response.provenance,
        "weight_units": "keV",
        "convention": "elastic halo G/(N*q(E)); extinction included",
        "arrays_sha256": array_fingerprint(arrays),
    }
    text = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    arrays["metadata_json"] = np.asarray(text)
    arrays["metadata_sha256"] = np.asarray(hashlib.sha256(text.encode()).hexdigest())
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("response path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez_compressed(path, **arrays)
    return path


def load_spectral_response(path):
    """Load without pickle, checking checksum, units, schema, and invariants."""
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(_ARRAYS) | {"metadata_json", "metadata_sha256"}:
            raise ValueError("unexpected response fields")
        text = str(data["metadata_json"])
        if hashlib.sha256(text.encode()).hexdigest() != str(data["metadata_sha256"]):
            raise ValueError("response metadata checksum mismatch")
        metadata = json.loads(text)
        arrays = {key: data[key] for key in _ARRAYS}
        if (
            metadata["schema_version"] != 1
            or metadata["weight_units"] != "keV"
            or metadata["convention"] != "elastic halo G/(N*q(E)); extinction included"
        ):
            raise ValueError("unsupported response schema, units, or convention")
        if array_fingerprint(arrays) != metadata["arrays_sha256"]:
            raise ValueError("response array checksum mismatch")
        return SpectralResponse(
            **arrays,
            n_histories=metadata["n_histories"],
            energy_support_kev=tuple(metadata["energy_support_kev"]),
            provenance=metadata["provenance"],
        )
