"""Audit the exported tables against RefleX's analytic angle distribution.

Run from this directory. With --repository, also run the real DSH host
loaders and DustPhysicsTable construction in an environment with JAX.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path)
    args = parser.parse_args()
    root = Path(__file__).parent
    with np.load(root / "materials/reflex_scattering_v1.npz", allow_pickle=False) as data:
        energy = data["energy_kev"]
        angle = data["scattering_angle_rad"]
        differential = data["differential_cross_section_cm2_per_sr_per_h"]
        sigma = data["scattering_cross_section_cm2_per_h"]
        cdf = data["scattering_angle_cdf"]
    meta = json.loads((root / "materials/reflex_scattering_v1.json").read_text())
    for prefix in ("scattering", "absorption"):
        p = root / "materials" / f"reflex_{prefix}_v1.npz"
        m = json.loads(p.with_suffix(".json").read_text())
        assert hashlib.sha256(p.read_bytes()).hexdigest() == m["table_sha256"]
    assert np.all(np.isfinite(differential)) and np.all(differential >= 0)
    assert angle[0] == 0 and angle[-1] == np.pi
    integrand = 2 * np.pi * np.sin(angle)[None, :] * differential
    integrated = np.trapezoid(integrand, angle, axis=1)
    closure = np.max(np.abs(integrated / sigma - 1))
    assert closure < 5e-12
    scale = np.array(meta["angular_scale_rad"])
    # Independent closed-form probability mass, rather than reintegrating
    # the implementation's differential table to validate its own shape.
    exact = angle[None, :]**2 / (angle[None, :]**2 + scale[:, None]**2)
    exact /= exact[:, -1:]
    cdf_error = np.max(np.abs(cdf - exact))
    assert cdf_error < 2e-6
    assert np.all(np.diff(cdf, axis=1) >= 0)
    assert np.all(cdf[:, 0] == 0) and np.all(cdf[:, -1] == 1)
    tail = np.max(scale**2 / (np.pi**2 + scale**2))
    assert tail < 1e-7
    with np.load(root / "materials/reflex_absorption_v1.npz", allow_pickle=False) as data:
        assert np.array_equal(energy, data["energy_kev"])
        assert np.all(data["absorption_cross_section_cm2_per_h"] > 0)
    result = dict(solid_angle_closure_relative_error=float(closure),
                  max_cdf_absolute_error=float(cdf_error), excluded_tail_probability=float(tail),
                  production_jax_loader_checked=False)
    if args.repository:
        sys.path.insert(0, str(args.repository.resolve()))
        from reflex_materials import load_reflex_materials
        scattering, absorption, physics = load_reflex_materials(root / "materials")
        _, _, zero_absorption = load_reflex_materials(root / "materials", absorption_enabled=False)
        assert np.array_equal(scattering.energy_kev, absorption.energy_kev)
        assert np.all(np.asarray(zero_absorption.absorption_cross_section_cm2_per_h) == 0)
        assert np.all(np.asarray(physics.scattering_cross_section_cm2_per_h) > 0)
        result["production_jax_loader_checked"] = True
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
