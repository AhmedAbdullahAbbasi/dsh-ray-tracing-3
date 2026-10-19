"""Translate maintained historical interfaces into one validated run plan.

This is an explicit compatibility boundary. Normal TOML loading never imports
or executes Python settings, infers a cadence, or changes material selection.
"""

from __future__ import annotations

import json
import math
import re
import tempfile
from dataclasses import replace
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
DAY_S = 86400.0


def legacy_arrival_edges(arrival_days, time_bin_days, explicit_edges_days=None):
    """Retain the historical equal-width linspace convention (in seconds)."""
    import numpy as np

    if explicit_edges_days is not None:
        edges = np.asarray(explicit_edges_days, dtype=float)
        if (
            edges.ndim != 1
            or len(edges) < 2
            or not np.all(np.isfinite(edges))
            or edges[0] != 0
            or not np.all(np.diff(edges) > 0)
        ):
            raise ValueError(
                "arrival-time-edges-days must start at zero and be finite, strictly increasing values"
            )
        return edges * DAY_S
    if not math.isfinite(arrival_days) or arrival_days <= 0:
        raise ValueError("arrival_days must be finite and positive")
    if not math.isfinite(time_bin_days) or time_bin_days <= 0:
        raise ValueError("time_bin_days must be finite and positive")
    count = arrival_days / time_bin_days
    if not math.isfinite(count) or count > 1_000_000:
        raise ValueError("observer.arrival: more than one million bins")
    return np.linspace(0, arrival_days * DAY_S, math.ceil(count) + 1)


def _portable(value):
    if isinstance(value, Path):
        return value.resolve().as_posix()
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _toml(data):
    """Serialize only the small scalar/list/table subset owned by this adapter."""
    lines = []

    def table(values, prefix=""):
        if prefix:
            lines.append("\n[" + prefix + "]")
        for key, value in values.items():
            if not isinstance(value, dict) and value is not None:
                lines.append(
                    key + " = " + json.dumps(_portable(value), allow_nan=False)
                )
        for key, value in values.items():
            if isinstance(value, dict):
                table(value, prefix + "." + key if prefix else key)

    table(data)
    return "\n".join(lines) + "\n"


def material_choice(choice, directory=None):
    if choice == "declared":
        if directory is None:
            raise ValueError("--materials declared requires --material-directory")
        return {"kind": "declared_bundle", "directory": Path(directory)}
    if directory is not None:
        raise ValueError("--material-directory requires --materials declared")
    if choice == "2-10":
        return {
            "kind": "table_pair",
            "scattering_file": DATA / "scattering/rg_drude_2_10_v2.npz",
            "absorption_file": DATA / "absorption/tbabs_wilm_vern_2_10_v2.npz",
            "grid_file": DATA / "materials/material_grid_2_10_v2.json",
        }
    return {
        "kind": "table_pair",
        "scattering_file": DATA / "newdust/mrn_rg_drude_v1.npz",
        "absorption_file": DATA / "absorption/tbabs_wilm_vern_v1.npz",
    }


def _base(name, output, device, packets, chunks, cap, seed, sampler, distance, arrival):
    name = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:80].strip("_") or "legacy_run"
    return {
        "config_version": 1,
        "run": {"name": name},
        "scene": {"kind": "example_four_cloud", "source_distance_kpc": distance},
        "execution": {
            "device": device,
            "packets": packets,
            "chunk_size": chunks,
            "max_interactions": cap,
            "seed": seed,
            "launch_sampler": sampler,
        },
        "observer": {
            "arrival": {
                "kind": "explicit",
                "edges_days": [float(x / DAY_S) for x in arrival],
            }
        },
        "output": {"directory": output, "overwrite": True},
    }


def _decay(peak, baseline, tau, start, duration, step):
    from .examples import post_peak_time_edges_s

    return {
        "kind": "exponential_decay",
        "time_edges_days": (
            post_peak_time_edges_s(start, duration, step) / DAY_S
        ).tolist(),
        "effective_energy_kev": [3.3, 4.9, 6.9],
        "peak_band_flux": peak,
        "baseline_band_flux": baseline,
        "decay_time_days": tau,
    }


def cli_translation(args, device):
    """Return explicit settings, output layout and metadata for old CLI flags."""
    import numpy as np

    output = args.output.resolve()
    data = _base(
        output.stem,
        output.parent / (output.stem + ".run"),
        device,
        args.packets,
        args.chunk_size,
        args.max_interactions,
        args.seed,
        args.launch_sampler,
        args.source_distance_kpc,
        legacy_arrival_edges(
            args.arrival_days, args.time_bin_days, args.arrival_time_edges_days
        ),
    )
    if args.cloud_fits:
        data["scene"].update(kind="fits", cloud_file=args.cloud_fits)
    data["materials"] = material_choice(args.materials, args.material_directory)
    hard = args.source_spectrum == "hard-state-powerlaw"
    if args.source_history:
        if args.source_model != "constant-flare":
            raise ValueError("--source-history supplies its own time history")
        if args.materials not in {"2-10", "declared"}:
            raise ValueError("--source-history requires --materials 2-10 or declared")
        from .sources.histories import load_spectral_history

        source, _ = load_spectral_history(args.source_history)
        data["source"] = {"kind": "spectral_history", "file": args.source_history}
        energies = np.asarray(source.energy_edges_kev).tolist()
    elif hard:
        if (
            args.materials not in {"2-10", "declared"}
            or args.source_model != "constant-flare"
        ):
            raise ValueError(
                "hard-state-powerlaw requires --materials 2-10 or declared and constant-flare"
            )
        energies = [2.0, 4.0, 6.0, 10.0]
        data["source"] = {
            "kind": "powerlaw",
            "time_edges_days": [0.0, 1 / 24],
            "energy_edges_kev": energies,
            "photon_flux": [args.total_2_10_photon_flux],
            "photon_index": args.photon_index,
        }
    else:
        energies = [2.5, 4.1, 5.9, 7.9]
        data["source"] = (
            _decay(
                args.peak_band_fluxes,
                args.baseline_band_fluxes,
                args.decay_time_days,
                args.decay_start_days,
                args.decay_duration_days,
                args.source_time_bin_days,
            )
            if args.source_model == "exponential-decay"
            else {
                "kind": "representative",
                "time_edges_days": [0.0, 1 / 24],
                "effective_energy_kev": [3.3, 4.9, 6.9],
                "band_flux": [args.peak_band_fluxes],
            }
        )
    data["observer"]["energy_edges_kev"] = energies
    name = data["run"]["name"]
    mapping = {
        name + ".npz": output,
        name + ".fits": (args.fits_output or output.with_suffix(".fits")).resolve(),
    }
    metadata = {
        "legacy_interface": "dsh-v1",
        "legacy_arguments": _portable(vars(args)),
        "material_tables": args.materials,
        "source_model": "spectral-history"
        if args.source_history
        else args.source_model,
        "source_spectrum": "spectral-history"
        if args.source_history
        else args.source_spectrum,
        "source_photon_index": args.photon_index if hard else None,
    }
    return data, mapping, metadata


def root_translation(settings):
    """Resolve old globals; record every formerly implicit timing/model choice."""
    import numpy as np

    from .io.provenance import hash_files, verify_inputs

    selected = {key: value for key, value in vars(settings).items() if key.isupper()}
    extra = (Path(settings.__file__).resolve(),)
    expected = hash_files(extra)
    output = Path(settings.OUTPUT_PNG).resolve()
    arrival = legacy_arrival_edges(settings.ARRIVAL_DAYS, settings.TIME_BIN_DAYS)
    data = _base(
        output.stem,
        output.parent / (output.stem + ".run"),
        settings.DEVICE,
        settings.PACKETS,
        settings.CHUNK_SIZE,
        settings.MAX_INTERACTIONS,
        settings.SEED,
        getattr(settings, "LAUNCH_SAMPLER", "radial"),
        settings.SOURCE_DISTANCE_KPC,
        arrival,
    )
    if settings.CLOUD_FITS_PATH is not None:
        data["scene"].update(kind="fits", cloud_file=Path(settings.CLOUD_FITS_PATH))
    else:
        data["scene"].update(
            kind="uniform_disc",
            **{
                key.lower(): getattr(settings, key)
                for key in (
                    "GRID_X_ARCSEC",
                    "GRID_Y_ARCSEC",
                    "GRID_Z_KPC",
                    "DISC_CENTER_X_ARCSEC",
                    "DISC_CENTER_Y_ARCSEC",
                    "DISC_CENTER_Z_KPC",
                    "DISC_RADIUS_ARCSEC",
                    "DISC_THICKNESS_KPC",
                    "DISC_N_H_CM3",
                )
            },
        )
    continuous = settings.ENERGY_EDGES_KEV is not None
    material = getattr(settings, "MATERIAL_MODEL", "default")
    if material not in {"default", "reflex"}:
        raise ValueError("MATERIAL_MODEL must be default or reflex")
    if material == "reflex":
        if continuous:
            raise ValueError("The legacy RefleX hook requires ENERGY_EDGES_KEV = None")
        directory = Path(
            getattr(settings, "REFLEX_MATERIAL_DIRECTORY", "reflex_adapter/materials")
        )
        data["materials"] = {
            "kind": "table_pair",
            "scattering_file": directory / "reflex_scattering_v1.npz",
            "absorption_file": directory / "reflex_absorption_v1.npz",
            "absorption_enabled": getattr(settings, "REFLEX_ABSORPTION_ENABLED", True),
        }
    else:
        data["materials"] = material_choice("2-10" if continuous else "v1")
    mode = settings.SOURCE_MODEL
    if continuous and mode != "flux-file":
        raise ValueError("ENERGY_EDGES_KEV requires SOURCE_MODEL = flux-file")
    metadata = {
        "legacy_interface": "run_simulation.py",
        "legacy_settings": _portable(selected),
        "legacy_timing_policy": "equal-width linspace arrival bins; explicit converted source intervals",
    }
    energies = list(settings.ENERGY_EDGES_KEV) if continuous else [2.5, 4.1, 5.9, 7.9]
    if mode == "flux-file":
        file = Path(settings.FLUX_FILE).resolve()
        expected.update(hash_files((file,)))
        table = np.loadtxt(file, ndmin=2)
        if table.shape[0] < 2 or table.shape[1] < 2:
            raise ValueError(
                "FLUX_FILE: legacy cadence inference needs at least two data rows"
            )
        times = table[:, 0]
        gaps = np.diff(times)
        if not np.all(np.isfinite(times)) or not np.all(gaps > 0):
            raise ValueError(
                "FLUX_FILE: MJD times must be finite, distinct and increasing"
            )
        data["source"] = {
            "kind": "flux_table",
            "file": file,
            "time_column": 1,
            "flux_columns": [int(c) + 1 for c in settings.FLUX_COLUMNS]
            if continuous
            else [2, 3, 4],
            "column_numbering": "one_based_full_table",
            "time_format": "mjd",
            "time_scale": getattr(settings, "FLUX_TIME_SCALE", "unspecified"),
            "epoch_mjd": float(times[0]),
            "interval_policy": "hold_until_next",
            "last_interval_days": float(gaps[-1]),
            "max_interval_days": float(max(gaps)),
            "missing_data": "error",
            "flux_convention": "unabsorbed observer-equivalent",
            "flux_unit": "ph cm^-2 s^-1",
            "flux_scale": settings.FLUX_SCALE,
            "spectral_mode": "band_powerlaw" if continuous else "representative",
        }
        data["source"].update(
            {"energy_edges_kev": energies, "photon_index": settings.PHOTON_INDEX}
            if continuous
            else {"effective_energy_kev": [3.3, 4.9, 6.9]}
        )
        metadata["legacy_flux_policy"] = (
            "first MJD is epoch; last measured cadence is final duration; maximum observed cadence is explicit gap limit; time scale unspecified unless FLUX_TIME_SCALE is declared"
        )
    elif mode == "custom":
        data["source"] = {
            "kind": "representative",
            "time_edges_days": settings.CUSTOM_TIME_EDGES_DAYS,
            "effective_energy_kev": [3.3, 4.9, 6.9],
            "band_flux": settings.CUSTOM_BAND_FLUXES,
        }
    elif mode == "exponential-decay":
        data["source"] = _decay(
            settings.PEAK_BAND_FLUXES,
            settings.BASELINE_BAND_FLUXES,
            settings.DECAY_TIME_DAYS,
            settings.DECAY_START_DAYS,
            settings.DECAY_DURATION_DAYS,
            settings.SOURCE_TIME_BIN_DAYS,
        )
    elif mode == "constant-flare":
        data["source"] = {
            "kind": "representative",
            "time_edges_days": [0.0, 1 / 24],
            "effective_energy_kev": [3.3, 4.9, 6.9],
            "band_flux": [settings.PEAK_BAND_FLUXES],
        }
    else:
        raise ValueError(
            "SOURCE_MODEL must be constant-flare, exponential-decay, custom or flux-file"
        )
    data["observer"]["energy_edges_kev"] = energies
    plots = {
        "enabled": True,
        "scales": settings.IMAGE_SCALES,
        "quantity": "fluence" if settings.IMAGE_EPOCH_DAYS is None else "intensity",
        "density_per_kev": continuous,
        "save_count_plots": True,
        "save_scored_event_map": True,
        "dpi": settings.IMAGE_DPI,
    }
    if settings.IMAGE_EPOCH_DAYS is not None:
        lo, hi = settings.IMAGE_EPOCH_DAYS
        days = arrival / DAY_S
        first = int(np.searchsorted(days, lo - 1e-9))
        end = int(np.searchsorted(days, hi + 1e-9, side="right")) - 1
        if end <= first:
            raise ValueError(
                "IMAGE_EPOCH_DAYS must contain at least one whole arrival bin"
            )
        plots["epoch_days"] = [float(days[first]), float(days[end])]
    if settings.IMAGE_ENERGY_KEV is not None:
        target = settings.IMAGE_ENERGY_KEV
        if continuous:
            # Preserve the old nearest-band choice while the shared history
            # adapter uses geometric labels for continuous spectral cells.
            from .sources.models import build_powerlaw_band_source

            labels = np.asarray(
                build_powerlaw_band_source(
                    [0.0, 1.0], [1.0], energies, settings.PHOTON_INDEX
                ).effective_energy_kev
            )
            index = int(np.argmin(np.abs(labels - target)))
            target = float(np.sqrt(energies[index] * energies[index + 1]))
        plots.update(energy_selection="nearest", energy_kev=target)
    if settings.IMAGE_FOV_ARCSEC is not None:
        plots.update(crop="centered", fov_arcsec=settings.IMAGE_FOV_ARCSEC)
    data["plots"] = plots
    data["logging"] = {"progress_every_chunks": settings.PROGRESS_EVERY_CHUNKS}
    mapping = {
        "run.log": Path(settings.PROGRESS_LOG).resolve(),
        "scored_mc_events.npy": Path(settings.OUTPUT_COUNT_NPY).resolve(),
    }
    for scale in settings.IMAGE_SCALES:
        for stem, original in (
            ("intensity_", settings.OUTPUT_PNG),
            ("scored_events_", settings.OUTPUT_COUNT_PNG),
        ):
            original = Path(original).resolve()
            mapping[stem + scale + ".png"] = original.with_name(
                original.stem + "_" + scale + original.suffix
            )
    verify_inputs(expected)
    return data, mapping, metadata, extra, expected


def _run(data, mapping, metadata, *, check=False, extra_inputs=(), expected=None):
    from .build import build_run_plan
    from .config import load_run_config
    from .io.provenance import verify_inputs
    from .run import publish_run

    with tempfile.TemporaryDirectory(prefix="dsh compatibility ") as temporary:
        path = Path(temporary) / "effective.toml"
        provenance = json.dumps(_portable(metadata), allow_nan=False)
        path.write_text(
            "# Explicit translation of historical settings.\n# "
            + provenance
            + "\n"
            + _toml(data),
            encoding="utf-8",
        )
        config = load_run_config(path)
        plan = build_run_plan(
            config, publication_paths=mapping, extra_inputs=extra_inputs
        )
        if expected:
            verify_inputs(expected)
        plan = replace(plan, metadata={**plan.metadata, **metadata})
        print(json.dumps(plan.summary, indent=2, allow_nan=False), flush=True)
        if check:
            return 0
        return publish_run(plan)


def run_cli(args, device):
    data, mapping, metadata = cli_translation(args, device)
    return _run(data, mapping, metadata, check=args.check)


def run_root(settings, *, check=False):
    data, mapping, metadata, extra, expected = root_translation(settings)
    return _run(
        data, mapping, metadata, check=check, extra_inputs=extra, expected=expected
    )
