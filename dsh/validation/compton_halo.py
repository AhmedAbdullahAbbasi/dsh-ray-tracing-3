"""Paired elastic/inelastic halo moments for the fixed RefleX camera experiment.

The negligible-opacity control uses the SAME six-stream transport as the
enabled model. Its 1e-35 cm2/H opacity gives only 1e-13 nominal optical depth.
Strictly positive opacity respects the Compton table's log interpolation.
Scores are unscaled per history; divide sums by N only during analysis.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CHANNELS = (
    "off_dust",
    "on_dust_only",
    "on_compton_only",
    "on_mixed",
    "on_dust_selected",
)


def file_sha(path):
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix in (".py", ".json", ".par", ".log"):
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def mean_covariance(total, cross, histories):
    """Covariance of the sample MEAN, including zero-score paired histories."""
    n = int(histories)
    if n < 2:
        raise ValueError("at least two histories are required")
    total, cross = np.asarray(total, dtype=float), np.asarray(cross, dtype=float)
    if cross.shape != (total.size, total.size):
        raise ValueError("cross-moment shape differs from the sum vector")
    return (cross - np.outer(total, total) / n) / (n * (n - 1))


def ratio_and_error(numerator, denominator, variance_num, variance_den, covariance):
    if denominator <= 0:
        raise ValueError("ratio denominator must be positive")
    ratio = numerator / denominator
    variance = (
        variance_num + ratio**2 * variance_den - 2 * ratio * covariance
    ) / denominator**2
    return float(ratio), float(np.sqrt(max(variance, 0.0)))


def histogram_indices(edges, values):
    index = np.searchsorted(edges, values, side="right") - 1
    index = np.where(values == edges[-1], len(edges) - 2, index)
    return index, (index >= 0) & (index < len(edges) - 1)


def verify_reference(repo, manifest_path):
    """Verify raw bytes, matched active parameters and camera component semantics."""
    from astropy.io import fits

    repo = Path(repo)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for relative, expected in manifest["files_sha256"].items():
        if file_sha(repo / relative) != expected:
            raise ValueError("reference evidence changed: " + relative)

    def active(path):
        return [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    on, off = (
        active(repo / manifest["on_params_file"]),
        active(repo / manifest["off_params_file"]),
    )
    changes = [(a, b) for a, b in zip(off, on, strict=True) if a != b]
    if changes != [("PHYSICS COMPTON OFF", "PHYSICS COMPTON ON")]:
        raise ValueError("reference parameters differ beyond the Compton switch")
    for setting in (
        "NPHOTS 100000000",
        "RANDOM 2026",
        "EMSPEC MONO 4900",
        "H2FRACTION 0.2",
        "DUST 0.691",
        "IMAGE DUSTSCAT > 0",
    ):
        if setting not in on:
            raise ValueError("reference setting missing: " + setting)
    images = {}
    for name in ("off_dust", "on_dust", "off_all", "on_all"):
        values = np.asarray(fits.getdata(repo / manifest[name + "_file"]), dtype=float)[
            ::-1
        ].copy()
        if (
            values.shape != (60, 60)
            or np.any(values < 0)
            or not np.array_equal(values, np.rint(values))
            or int(values.sum()) != manifest["expected_counts"][name]
        ):
            raise ValueError("raw reference image/counts differ: " + name)
        images[name] = values
    core = np.zeros((60, 60), dtype=bool)
    core[29:31, 29:31] = True
    for condition in ("off", "on"):
        remainder = images[condition + "_all"] - images[condition + "_dust"]
        if np.any(remainder < 0) or np.any(remainder[~core] != 0):
            raise ValueError(
                "reference has an unexpected non-dust component outside core"
            )
    return manifest, images


class HaloCollector:
    """Actual accepted transport/scoring, plus calibrated finite camera-aperture blur."""

    def __init__(self, config, repo):
        import jax
        import jax.numpy as jnp
        from jax import random

        from dsh.observer.inelastic import score_inelastic_peeloff
        from dsh.physics.cold_compton import build_cold_compton_table
        from dsh.physics.compton import to_compton_physics
        from dsh.physics.wd01_ism import load_wd01_ism
        from dsh.sources.launch import sample_source_launches
        from dsh.sources.models import SourcePackets
        from dsh.sources.radial_launch import build_radial_cloud_launch_geometry
        from dsh.transport.inelastic import transport_inelastic_batch
        from reflex_adapter.compare_reflex import (
            image_edges,
            make_cloud,
            radial_pixel_map,
        )

        self.config, self.repo = dict(config), Path(repo)
        definition, scattering, _, absorption, physics = load_wd01_ism(
            self.repo / config["material_directory"]
        )
        energy = np.unique(np.r_[np.geomspace(2, 10, 129), config["energy_kev"]])
        built = build_cold_compton_table(
            definition, energy, grain_electrons=config["grain_electrons"]
        )
        self.compton = to_compton_physics(built.table)
        self.off_control = self.compton._replace(
            cross_section_cm2_per_h=jnp.full_like(
                self.compton.cross_section_cm2_per_h, 1e-35
            )
        )
        self.physics = physics
        self.cloud = make_cloud(config)
        self.launch = build_radial_cloud_launch_geometry(self.cloud, physics)
        self.edges = image_edges(config)
        self.radial_map = radial_pixel_map(config)
        self.n_side = len(self.edges) - 1
        self.n_radius = len(config["annulus_edges_arcsec"]) - 1
        self.energy_edges = np.array(
            [2, 3, 4, 4.5, 4.7, 4.8, 4.85, 4.89, config["energy_kev"] + 1e-5, 5.0]
        )
        self.time_edges = np.array([0, 0.1, 1, 10, 100, 1000, 1e5, 1e9]) * 86400
        self.material = {
            "definition_sha256": definition.sha256,
            "definition": definition.as_dict(),
            "scattering": scattering.metadata,
            "absorption": absorption.metadata,
            "compton": built.table.metadata,
            "reflex_equivalent_material": False,
        }
        n, guard = config["chunk_size"], config["max_interactions"]
        pk = SourcePackets(
            jnp.full(n, config["energy_kev"]),
            jnp.zeros(n),
            jnp.ones(n),
            jnp.zeros(n, dtype=jnp.int32),
            jnp.zeros(n, dtype=jnp.int32),
        )

        @jax.jit
        def batch(key, compton):
            launch_key, transport_key = random.split(key)
            launched = sample_source_launches(launch_key, pk, self.launch)
            history = transport_inelastic_batch(
                transport_key,
                launched.position_pc,
                launched.momentum_kev,
                self.cloud,
                physics,
                compton,
                energy_cutoff_kev=config["energy_cutoff_kev"],
                max_interactions=guard,
            )
            scored = score_inelastic_peeloff(
                launched, history, self.cloud, physics, compton
            )
            ev = scored.events
            return (
                ev.valid,
                ev.sky_x_arcsec,
                ev.sky_y_arcsec,
                ev.weight_observer_fluence,
                scored.component,
                ev.energy_kev,
                ev.arrival_time_s,
                jnp.linalg.norm(history.interactions.position_pc, axis=-1),
                history.status,
                history.n_compton,
                history.recoil_energy_kev,
                history.absorbed_energy_kev,
                jnp.abs(
                    history.source_energy_kev
                    - history.deposited_energy_kev
                    - history.momentum_kev[:, 0]
                ),
                scored.unsupported_escape,
                scored.unsupported_phase,
            )

        self.batch = batch

    def empty(self):
        return {
            "histories": np.asarray(0),
            "sum": np.zeros(len(CHANNELS) * self.n_radius),
            "cross": np.zeros((len(CHANNELS) * self.n_radius,) * 2),
            "image": np.zeros((len(CHANNELS), self.n_side, self.n_side)),
            "image_square": np.zeros((len(CHANNELS), self.n_side, self.n_side)),
            "energy_sum": np.zeros((len(CHANNELS), len(self.energy_edges) - 1)),
            "energy_square": np.zeros((len(CHANNELS), len(self.energy_edges) - 1)),
            "time_sum": np.zeros((len(CHANNELS), len(self.time_edges) - 1)),
            "time_square": np.zeros((len(CHANNELS), len(self.time_edges) - 1)),
            "statuses": np.zeros((2, 9), dtype=np.int64),
            "analog_compton_count": np.zeros(2, dtype=np.int64),
            "recoil_sum_kev": np.zeros(2),
            "absorption_sum_kev": np.zeros(2),
            "max_energy_error_kev": np.asarray(0.0),
            "unsupported_observer_count": np.asarray(0),
            "outside_image_sum": np.zeros(len(CHANNELS)),
        }

    def collect(self, key, size, smear_seed):
        import jax

        from reflex_adapter.compare_reflex import ARCSEC

        off = jax.device_get(self.batch(key, self.off_control))
        on = jax.device_get(self.batch(key, self.compton))
        n, guard = off[0].shape
        rng = np.random.default_rng(smear_seed)
        azimuth = rng.uniform(0, 2 * np.pi, (n, guard))
        radius = (
            np.sqrt(rng.uniform(size=(n, guard)))
            * self.config["camera_aperture_radius_pc"]
            / ARCSEC
        )
        state = self.empty()
        state["histories"] = np.asarray(size)
        grouped = np.zeros((size, len(CHANNELS), self.n_radius))
        for condition, arrays in enumerate((off, on)):
            (
                valid,
                x,
                y,
                weight,
                component,
                energy,
                time,
                distance,
                status,
                nc,
                recoil,
                absorbed,
                error,
                unsupported,
                missing,
            ) = [np.asarray(value)[:size] for value in arrays]
            if any(
                not np.isfinite(value[valid]).all()
                for value in (weight, x, y, energy, time)
            ) or np.any(weight[valid] < 0):
                raise ValueError("invalid observer score")
            state["statuses"][condition] = np.bincount(status.astype(int), minlength=9)
            state["analog_compton_count"][condition] = nc.sum()
            state["recoil_sum_kev"][condition] = recoil.sum()
            state["absorption_sum_kev"][condition] = absorbed.sum()
            state["max_energy_error_kev"] = np.maximum(
                state["max_energy_error_kev"], error.max()
            )
            state["unsupported_observer_count"] += unsupported.sum() + missing.sum()
            dx = radius[:size] * np.cos(azimuth[:size]) / np.maximum(distance, 1.0)
            dy = radius[:size] * np.sin(azimuth[:size]) / np.maximum(distance, 1.0)
            ix, inside_x = histogram_indices(self.edges, x + dx)
            iy, inside_y = histogram_indices(self.edges, y + dy)
            inside = inside_x & inside_y
            channels = (
                ((0, valid),)
                if condition == 0
                else (
                    (1, valid & (component == 1)),
                    (2, valid & (component == 2)),
                    (3, valid & (component == 3)),
                    (4, valid & (component != 2)),
                )
            )
            for channel, selected in channels:
                state["outside_image_sum"][channel] += weight[selected & ~inside].sum()
                histories, slots = np.nonzero(selected & inside)
                weights = weight[histories, slots].astype(float)
                xp, yp = ix[histories, slots], iy[histories, slots]
                np.add.at(state["image"][channel], (yp, xp), weights)
                pixel_key, inverse = np.unique(
                    histories * self.n_side**2 + yp * self.n_side + xp,
                    return_inverse=True,
                )
                combined = np.bincount(inverse, weights=weights)
                np.add.at(
                    state["image_square"][channel].ravel(),
                    pixel_key % self.n_side**2,
                    combined**2,
                )
                annuli = self.radial_map[yp, xp]
                if np.any(annuli < 0) or np.any(annuli >= self.n_radius):
                    raise ValueError("annuli do not cover image")
                np.add.at(grouped[:, channel], (histories, annuli), weights)
                for axis, values, prefix in (
                    (self.energy_edges, energy, "energy"),
                    (self.time_edges, time, "time"),
                ):
                    index, covered = histogram_indices(axis, values[histories, slots])
                    if not np.all(covered):
                        raise ValueError(
                            "diagnostic axis loses scored photons: " + prefix
                        )
                    h = np.zeros((size, len(axis) - 1))
                    np.add.at(h, (histories, index), weights)
                    state[prefix + "_sum"][channel] += h.sum(axis=0)
                    state[prefix + "_square"][channel] += (h * h).sum(axis=0)
        flat = grouped.reshape(size, -1)
        state["sum"] = flat.sum(axis=0)
        state["cross"] = flat.T @ flat
        return state


def add_moments(left, right):
    if left.keys() != right.keys():
        raise ValueError("moment fields differ")
    return {
        name: np.maximum(left[name], right[name])
        if name == "max_energy_error_kev"
        else left[name] + right[name]
        for name in left
    }


def save_checkpoint(path, state, seed, fingerprint):
    from ..io.atomic import atomic_savez_compressed

    atomic_savez_compressed(
        path,
        **state,
        _seed=np.asarray(seed),
        _collection_fingerprint=np.asarray(fingerprint),
    )


def load_checkpoint(path, seed, fingerprint, expected_histories=None):
    with np.load(path, allow_pickle=False) as data:
        if (
            int(data["_seed"]) != seed
            or str(data["_collection_fingerprint"]) != fingerprint
        ):
            raise ValueError("checkpoint seed or collection fingerprint differs")
        state = {name: data[name] for name in data.files if not name.startswith("_")}
    if expected_histories is not None and int(state["histories"]) != expected_histories:
        raise ValueError("checkpoint history count differs")
    return state


def merge_seed_estimates(states):
    """Equal-weight seed means; independent-seed covariance, no double counting."""
    counts = [int(state["histories"]) for state in states]
    if len(set(counts)) != 1:
        raise ValueError("independent seeds must have equal history counts")
    n, k = counts[0], len(states)
    total = sum(state["sum"] for state in states)
    cross = sum(state["cross"] for state in states)
    return total / (n * k), mean_covariance(total, cross, n * k)
