"""Compile canonical source fluxes into one JAX-ready photon proposal."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from dsh.contracts import FLAT, LINE, POWERLAW, SourceCells
from dsh.contracts import SourcePackets as SourcePackets
from dsh.sources.format import SourceFluxFile

from ..core.sampling import sample_source_cells as sample_source_cells
from .models import _sample_powerlaw_energy as _sample_powerlaw_energy


def build_source_cells(
    source: SourceFluxFile, *, components: str = "both"
) -> SourceCells:
    """Select continuum/lines and preserve the selected absolute fluence."""

    if components not in {"both", "continuum", "lines"}:
        raise ValueError("components must be both, continuum or lines")
    n_cont = source.continuum_flux.shape[1]
    n_line = source.line_energy_kev.size
    if components == "continuum" and not n_cont:
        raise ValueError("source file has no continuum")
    if components == "lines" and not n_line:
        raise ValueError("source file has no lines")
    durations = np.diff(source.time_edges_s)
    records = []
    for t, duration in enumerate(durations):
        if components != "lines":
            for b in range(n_cont):
                fluence = float(source.continuum_flux[t, b] * duration)
                if fluence > 0.0:
                    kind = FLAT if source.continuum_shape[b] == "FLAT" else POWERLAW
                    records.append(
                        (
                            t,
                            b,
                            kind,
                            source.continuum_energy_edges_kev[b],
                            source.continuum_energy_edges_kev[b + 1],
                            source.continuum_photon_index[t, b],
                            fluence,
                        )
                    )
        if components != "continuum":
            for b in range(n_line):
                fluence = float(source.line_flux[t, b] * duration)
                if fluence > 0.0:
                    energy = source.line_energy_kev[b]
                    records.append((t, n_cont + b, LINE, energy, energy, 0.0, fluence))
    if not records:
        raise ValueError("selected source components have zero total fluence")
    values = np.asarray(records, dtype=np.float64)
    fluence = values[:, 6]
    total = fluence.sum(dtype=np.float64)
    cdf = np.cumsum(fluence, dtype=np.float64) / total
    cdf[-1] = 1.0
    return SourceCells(
        time_edges_s=jnp.asarray(source.time_edges_s),
        start_s=jnp.asarray(source.time_edges_s[values[:, 0].astype(np.int32)]),
        stop_s=jnp.asarray(source.time_edges_s[values[:, 0].astype(np.int32) + 1]),
        energy_low_kev=jnp.asarray(values[:, 3]),
        energy_high_kev=jnp.asarray(values[:, 4]),
        photon_index=jnp.asarray(values[:, 5]),
        kind=jnp.asarray(values[:, 2], dtype=jnp.int32),
        time_index=jnp.asarray(values[:, 0], dtype=jnp.int32),
        spectral_bin_index=jnp.asarray(values[:, 1], dtype=jnp.int32),
        cell_fluence=jnp.asarray(fluence),
        flat_cdf=jnp.asarray(cdf),
        total_fluence=jnp.asarray(total),
    )
