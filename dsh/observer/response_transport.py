"""Collect physical response samples using the unchanged production transport."""

from __future__ import annotations

import hashlib
from pathlib import Path

import jax
import numpy as np
from jax import random

from ..sources.launch import sample_source_launches
from ..sources.models import sample_tabulated_band_source
from ..transport.kernel import transport_photon_batch
from .response import SpectralResponse, array_fingerprint, response_from_events
from .scoring import score_peeloff_events


def response_provenance(cloud, physics, launch_geometry):
    """Fingerprint actual arrays and the runtime files used to collect samples."""
    root = Path(__file__).resolve().parents[1]
    files = (
        "io/atomic.py",
        "geometry/clouds.py",
        "geometry/coordinates.py",
        "geometry/rays.py",
        "physics/dust.py",
        "sources/models.py",
        "sources/histories.py",
        "sources/launch.py",
        "sources/radial_launch.py",
        "transport/kernel.py",
        "transport/directions.py",
        "observer/scoring.py",
        "observer/response.py",
        "observer/response_transport.py",
    )
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files
    }
    digest = hashlib.sha256()
    for name, value in sorted(hashes.items()):
        digest.update(f"{name}:{value}\n".encode())

    def arrays(value):
        return {
            name: np.asarray(getattr(value, name))
            for name in value._fields
            if getattr(value, name) is not None
        }

    return {
        "cloud_sha256": array_fingerprint(arrays(cloud)),
        "material_sha256": array_fingerprint(arrays(physics)),
        "launch_sha256": array_fingerprint(arrays(launch_geometry)),
        "runtime_sha256": digest.hexdigest(),
        "runtime_files_sha256": hashes,
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "jax_backend": jax.default_backend(),
    }


def collect_spectral_response(
    proposal,
    cloud,
    physics,
    launch_geometry,
    *,
    n_packets,
    chunk_size=8192,
    max_interactions=32,
    seed=20261016,
    progress=None,
):
    """Sample independent proposal histories; reject any incomplete transport.

    ``proposal`` has one interval, continuous energy, and positive flux in every
    band. Absolute proposal fluence cancels. The angular importance correction
    and outgoing extinction remain exactly those of the production scorer.
    """
    if n_packets < 2 or chunk_size < 2 or max_interactions < 1:
        raise ValueError(
            "require at least two histories/chunk and positive max_interactions"
        )
    if (
        proposal.energy_edges_kev is None
        or proposal.band_flux.shape[0] != 1
        or np.any(np.asarray(proposal.band_flux) <= 0)
    ):
        raise ValueError(
            "proposal requires one interval, continuous energy, and positive bands"
        )
    bounds = np.asarray(proposal.energy_edges_kev)
    if bounds[0] < float(physics.energy_kev[0]) or bounds[-1] > float(
        physics.energy_kev[-1]
    ):
        raise ValueError("proposal lies outside material energy support")
    provenance = response_provenance(cloud, physics, launch_geometry)
    provenance.update(
        {
            "seed": seed,
            "chunk_size": chunk_size,
            "max_interactions": max_interactions,
            "proposal_energy_edges_kev": bounds.tolist(),
            "proposal_band_flux": np.asarray(proposal.band_flux).tolist(),
            "proposal_photon_index": np.asarray(proposal.photon_index).tolist(),
        }
    )

    def batch(key, n):
        source_key, launch_key, transport_key = random.split(key, 3)
        packets = sample_tabulated_band_source(source_key, proposal, n)
        launched = sample_source_launches(launch_key, packets, launch_geometry)
        transported = transport_photon_batch(
            transport_key,
            launched.position_pc,
            launched.momentum_kev,
            cloud,
            physics,
            max_interactions=max_interactions,
        )
        return (
            packets,
            transported.status,
            score_peeloff_events(launched, transported, cloud, physics),
        )

    compiled = jax.jit(batch, static_argnames=("n",))
    counts = [
        min(chunk_size, n_packets - start) for start in range(0, n_packets, chunk_size)
    ]
    if counts[-1] == 1:
        counts[-2:] = [counts[-2] + 1]
    parts = []
    statuses = np.zeros(7, dtype=np.int64)
    offset = 0
    for index, count in enumerate(counts):
        packets, status, events = compiled(
            random.fold_in(random.PRNGKey(seed), index), n=count
        )
        statuses += np.bincount(np.asarray(status), minlength=7)
        part = response_from_events(
            events, packets, proposal, status, provenance=provenance
        )
        # response_from_events uses this batch's count. Convert to the global
        # estimator before concatenating; zero-score histories still count.
        parts.append((part, offset, count / n_packets))
        offset += count
        if progress is not None:
            progress(offset, n_packets)
    fields = {}
    for name in (
        "energy_kev",
        "delay_s",
        "sky_x_arcsec",
        "sky_y_arcsec",
        "scattering_order",
        "history_index",
        "weight_kev",
    ):
        fields[name] = np.concatenate(
            [
                getattr(part, name) + start
                if name == "history_index"
                else getattr(part, name) * scale
                if name == "weight_kev"
                else getattr(part, name)
                for part, start, scale in parts
            ]
        )
    provenance["transport_status_counts"] = statuses.tolist()
    return SpectralResponse(
        **fields,
        n_histories=n_packets,
        energy_support_kev=(float(bounds[0]), float(bounds[-1])),
        provenance=provenance,
    )
