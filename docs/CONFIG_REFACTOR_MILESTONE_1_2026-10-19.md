# Configuration refactor: milestone 1 implementation and validation

19 October 2026. Branch: `refactor/simulation-config`.

The implementation begins at documentation checkpoint `80f42c5`, preserving
the numerical checkpoint `1cea953` and preceding `eeb96ea` comparison commit.
The delivery's `package.json` records its exact final tip and bundle checksum.

## Delivered behavior

- Strict version-1 TOML, frozen host settings and paths relative to the file.
  Parser imports do not load JAX, NumPy, Astropy or Matplotlib.
- Installed `dsh check FILE` and `dsh run FILE`, with early CPU/CUDA selection.
- Existing native FITS and four-cloud example scene adapters.
- Explicit legacy material pairs and complete declared bundles. Existing
  checksum, phase, grid, chemistry and stored-total checks remain active.
- Representative, scalar power-law, per-band power-law, exponential-decay,
  intrinsic-history JSON and explicit MJD photon-flux-table source adapters.
- Independent observer energy/arrival bins and radial/rectangular proposals.
- Unchanged numerical pipeline, seed splitting, packet normalization and
  short-batch accounting, with a no-I/O numerical plan executor.
- Production quality gating, diagnostically labeled incomplete exports,
  physical NPZ/FITS, raw moments and separate window-loss reporting.
- Original/resolved configuration, input and runtime fingerprints, staged
  per-file publication and a final inventory-hashing completion marker.
- Optional area-correct intensity plots and scored Monte Carlo event maps.
- Windows Desktop setup script: clone the published prior branch into a new
  directory, import the exact bundle, create a virtual environment, install,
  test, check both material examples and optionally push without force.

See `CONFIGURED_SIMULATIONS.md` for the active schema and commands. The
original handoff still describes later migration slices; not every proposed
section in that handoff is implemented. Unsupported fields fail explicitly.

## Validation evidence

| Check | Result |
| --- | --- |
| Unmodified baseline main suite | 230 passed, 395.208 s |
| Integrated main suite | 251 passed, 422.477 s, no reported skips |
| Final focused configured-run suite | 21 passed, 13.431 s |
| Separate RefleX material tests | 4 passed |
| Separate NewDust azimuth discovery | 3 passed |
| Protected RefleX manifest | 63/63 match |
| WD01–RefleX manifest | 25/25 match |
| Multi-material manifest | 58/58 match |
| Seeded plan/existing-pipeline parity | Exact source/product/diagnostic arrays, 37 packets, chunks of 16 including a tail of 5 |
| Maintained pair/declared adapters | Exact arrays against existing V2/WD01 loaders |
| Installed wheel outside source checkout | Legacy and WD01 64-packet runs complete, NPZ/FITS and completion inventories written |
| Actual configured CLI incomplete-run gate | 128 packets, cap 1: 15 capped histories, exit 3, no production NPZ or completion marker |
| Flux-table contract | Whole-table column selection, independently integrated bands, epoch subtraction, one-row duration and gap rejection verified |
| Output safety | Config/input alias protection, exclusive outputs, interrupted staged export, hashes and NPZ/FITS identity verified |
| Plot checks | Nonuniform-pixel intensity invariant; positive/zero-image linear/log exports and scored-event maps; positive-image layout inspected |
| TOML compatibility | Native parser and forced tomli import route pass; 70 files pass Python 3.10 grammar parsing |
| New/changed Python lint and formatting | Pass with Ruff 0.17.1 |
| Whole documented lint scope | Same 4 pre-existing findings; protected historical source was not reformatted |

The final focused run follows a small resolved-manifest change that omits
inactive source-mode defaults, so a JSON history never appears to inherit a
competing scalar photon index from an unused configuration field. All
effective defaults for the selected mode are still expanded. Numerical
samplers and transport were not changed by that adjustment.

The environment was Linux CPU, Python 3.12.14, JAX/JAXlib 0.11.2, NumPy
2.3.5, Astropy 8.0.1, with optional material dependencies installed. Conditional
TOML loading and grammar checks are not a Python 3.10 runtime matrix. Windows,
PowerShell and GPU execution were not available for direct testing; the setup
script runs focused tests and both checks in the user's actual Windows
environment. The exact bundle and its import are verified independently.

No expensive scientific benchmark was recollected. Existing benchmark
archives retain their original provenance and hashes. Small simulations check
orchestration and completion, not scientific convergence.

## Remaining staged work

Root-runner migration, disc/FRED/variable-source configuration, maintained
RefleX-export integration, Slurm resources/submission, configured response
collection/folding, partial energy windows, arbitrary image band/crop choices,
source-pixel placement, cleanup and profiling remain later slices. The legacy
interfaces retain their existing behavior, including their old warning-only
incomplete-run handling; the new production gate applies to the configured
runner introduced here.

This is the first useful configuration milestone, not a claim that the entire
architecture migration or scientific validation agenda is complete.
