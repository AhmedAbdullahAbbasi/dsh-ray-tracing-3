"""Shared staged publication and completion records for configured workflows."""

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from ..config.load import validate_destinations
from .provenance import hash_files, verify_inputs


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def publish_file(source, destination, overwrite):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if overwrite:
            os.replace(source, destination)
        else:
            os.link(source, destination)
            source.unlink()
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        with tempfile.TemporaryDirectory(
            prefix=".dsh-publish-", dir=destination.parent
        ) as temporary:
            local = Path(temporary) / destination.name
            shutil.copyfile(source, local)
            publish_file(local, destination, overwrite)
        source.unlink()


@contextmanager
def staged_run(config, input_hashes, *, mapping=None, extra_inputs=()):
    validate_destinations(config, publication_paths=mapping, extra_inputs=extra_inputs)
    verify_inputs(input_hashes)
    directory = config.output.directory
    directory.mkdir(parents=True, exist_ok=True)
    if config.output.overwrite:
        (directory / "complete.json").unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dsh-run-", dir=directory) as temporary:
        yield Path(temporary)


def write_run_records(
    stage,
    config,
    input_hashes,
    metadata,
    report,
    paths,
    *,
    log_path,
    resolved_extra=None,
):
    """Preserve the order and schemas of each workflow's product inventory."""
    paths = list(paths)
    original = stage / "original.toml"
    original.write_bytes(config.config_bytes)
    resolved = stage / "resolved.json"
    write_json(
        resolved,
        {
            "manifest_version": 1,
            "configuration": config.to_dict(),
            "input_sha256": input_hashes,
            "runtime": metadata,
            **(resolved_extra or {}),
        },
    )
    paths += [original, resolved]
    if log_path not in paths:
        paths.append(log_path)
    report["output_inventory"] = [path.name for path in paths] + ["report.json"]
    report_path = stage / "report.json"
    write_json(report_path, report)
    return paths + [report_path]


def publish_outputs(
    stage,
    config,
    input_hashes,
    paths,
    *,
    complete,
    mapping=None,
    extra_inputs=(),
    publisher=None,
):
    """Recheck inputs/destinations and publish the completion marker last.

    Individual files are atomic. A failed group may leave some products but
    cannot acquire a completion marker from this invocation.
    """
    verify_inputs(input_hashes)
    validate_destinations(config, publication_paths=mapping, extra_inputs=extra_inputs)
    inventory = {Path(path).name: digest for path, digest in hash_files(paths).items()}
    publish = publisher or publish_file
    directory = config.output.directory
    for path in paths:
        destination = Path((mapping or {}).get(path.name, directory / path.name))
        publish(path, destination, config.output.overwrite)
    if complete:
        marker = stage / "complete.json"
        record = {
            "completion_manifest_version": 1,
            "run": config.name,
            "config_sha256": config.config_sha256,
            "output_sha256": inventory,
        }
        if mapping is not None:
            record["output_paths"] = {
                name: str(mapping[name]) for name in inventory if name in mapping
            }
        write_json(marker, record)
        publish(marker, directory / marker.name, config.output.overwrite)
