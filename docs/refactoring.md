# Input and structural refactor

The input and package restructuring is complete on `refactor/inputs-and-config`.
Source flux and material physics are validated file inputs. The numerical
engine consumes fixed-shape arrays through shared contracts. This checkpoint
preserves the physical calculation, isotropic emission, existing rectangular
launch proposal, PRNG splitting, weights, and output schemas.

## Responsibilities and dependencies

| Layer | Canonical interface | Inputs and outputs |
| --- | --- | --- |
| Configuration | `dsh.config.load.load_run_config` | TOML to immutable `ResolvedRun`; resolves file paths |
| Source | `dsh.sources.format.load_source_fits`, `dsh.sources.cells.build_source_cells` | Validated FITS flux and metadata to numerical `SourceCells` |
| Material | `dsh.materials.load_material_inputs` | Independent scattering/absorption NPZ+JSON to host provenance and numerical `Material` |
| Build | `dsh.build.build_run_plan` | Validated source, material and scene to `RunPlan` |
| Numerical execution | `dsh.run.run_plan`, `dsh.core.pipeline` | Numerical plan to binned products and diagnostics |
| Configured execution | `dsh.run.run_configured_simulation` | Configuration to schema-7 NPZ/FITS and provenance reports |
| Product inspection | `dsh.products.audit`, `dsh.products.snapshots` | Saved products to integrity reports and monochromatic snapshots |

`dsh.contracts` owns source packets, cloud/ray geometry, material arrays,
interaction histories, observer events/products, statuses, axis conventions,
and the host-side `RunPlan`. It retains the validated NamedTuple field order
and defaults. `Material` is an alias of the existing `DustPhysicsTable` type.
Paths, model metadata, digests, and file-format objects stay outside the
numerical plan and JIT calls.

Core imports only `dsh.contracts`, other core modules, JAX, NumPy, and small
standard-library numerical helpers. Numeric builders can validate array
shapes on the host; core performs no file I/O. Production modules do not
import validation. Stage 9F reference quadrature, phase interpolation and
host column/scoring implementations stay independent of the production ray
integrator and scorer. Package-boundary tests enforce these dependencies.

The two existing interpolation implementations now live in `dsh.core.interp`
under separate names. Their evaluation order and zero handling are retained;
this step does not merge algorithms that could alter rounding.

## Python and command interfaces

```python
from dsh.build import build_run_plan
from dsh.config import load_run_config
from dsh.run import run_plan

config = load_run_config("configs/file_input_smoke.toml")
plan = build_run_plan(config)
result = run_plan(plan)  # Numerical products; writes no files.
```

To save provenance and configured products, use
`run_configured_simulation(config)` or the command interface:

```powershell
python -m dsh.command check configs/file_input_smoke.toml
python -m dsh.command run configs/file_input_smoke.toml
python -m dsh.command audit configs/file_input_smoke.toml
python -m dsh.command snapshot outputs/realistic_line_5p35/full.toml
```

An installed package exposes the same commands as `dsh`. The snapshot command
currently supports one monochromatic source cell, one energy bin, a FITS cloud,
and windows that each match one stored arrival bin. It preserves history-based
pixel errors and rejects unsupported windows.

## Adding inputs and models

| Desired change | Work required |
| --- | --- |
| Different flare or series of spectra | Write source FITS schema 1 and select it in TOML; existing continuum, lines, or mixed cells are supported |
| Different elastic dust model | Produce intrinsic scattering NPZ+JSON on a supported energy/angle grid; select it in TOML |
| TBabs to TBnew or another photoelectric model | Produce absorption NPZ+JSON on the scattering energy grid; change the absorption path |
| New offline dust calculation | Add a recipe under `dsh.materials.recipes` that emits the standard component files |
| New external spectrum format | Add a converter to source FITS; the numerical engine keeps its existing source contract |

Model labels are provenance rather than switches in the engine. The generic
scattering loader retains the earlier schema and validation implementation;
it does not require the model label to be NewDust. Scattering and absorption
currently require exactly matched energy nodes, per-hydrogen cross sections,
and full physical scattering angles. These are mathematical contract limits.
Nonelastic interactions or spatially varying material mixtures would need a
new contract and validation beyond this refactor.

## Compatibility and regression evidence

The earlier geometry, transport, observer, physics, source-file, and pipeline
module paths remain compatibility aliases. Their canonical module identity
is shared so existing mock assignments still reach the implementation.
Source-model builders reexport the moved samplers. The former combined
`dsh.config.run` interface still exports `build_run` and the configured runner;
the six-item `build_run` return value is preserved. Legacy snapshot scripts
and `dsh-v1` remain available. There is one implementation per moved function.

For the baseline at `40a75c1`, all 155 existing unit tests passed before the
moves. On the same CPU environment (Python 3.12.14, JAX 0.11.2, NumPy 2.5.3),
263 captured array shapes, dtypes, and SHA-256 values matched exactly after
the moves. Coverage included material/cloud/bin arrays, JIT source proposals,
launch packets, and chunked products/diagnostics for mixed, line-only,
continuum-only, legacy representative-energy, and continuous power-law sources.
Seeds were 741 for source sampling, 743 for launches, and 745 for the pipeline;
simulations used 1,024 packets, chunks of 128, and a cap of 32. This proves
preservation for those seeded cases and that environment; it is not a promise
of bitwise results across JAX versions, hardware, or chunk sizes.

Pinned source/material files retain their exact bytes. Numerical field order,
transport status meanings, schema-6/schema-7 output contracts, photon-history
moments, and independent reference algorithms are preserved.

On 2026-09-30, 170 moved or independent-reference function bodies also matched
their baseline parsed syntax trees. The full 161-test suite, Ruff lint, and
Ruff formatting checks passed. The new tests enforce package dependencies,
configuration-only imports, and compatibility module identity; the line-run
workflow also tests the snapshot command and rejection of an unaligned window.
Controlled post-move comparisons passed:

- File-loaded heterogeneous 5.35-keV line: all seven reference gates, three
  30,000-packet seeds (912, 319, 141), plus the 8,192-packet configured
  full-cloud numerical/product audit.
- Absorbed first-order annuli at 5.35 keV: all gates, three 100,000-packet
  seeds (912, 319, 141), chunks of 256 and cap 32.

These reference runs test their controlled scenes and launch support. They
do not establish convergence of realistic one-day images.

## Remaining work outside this refactor

- XSPEC and observed-spectrum conversion helpers, including explicit time and flux conventions.
- A more efficient launch proposal with correct importance weights and separate statistical validation.
- Precision and seed/packet convergence for realistic heterogeneous time-resolved images.
- Continuous-spectrum spatial validation, external matched-material benchmarks, and observational comparisons.

Compatibility paths can be removed in a later deliberate API cleanup. Keep
the prior branch/checkpoint until local tests and the representative workflow
pass on the new branch; no previous branch is retired by this checkpoint.
