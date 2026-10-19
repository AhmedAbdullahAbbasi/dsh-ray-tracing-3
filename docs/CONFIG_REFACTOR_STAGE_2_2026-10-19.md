# Configuration usability and coverage: Stage 2

19 October 2026. Branch `refactor/simulation-config`; based on the Windows-tested
Stage 1 commit `a176ea4`. The delivery manifest records the exact Stage 2 tip.

## Delivered behavior

- Seven runnable TOMLs cover all six configured source modes, both scene modes
  and both material-selection modes. The five new examples include a tiny
  native FITS column cube, an intrinsic spectral-history JSON and an explicit
  MJD flux table. All run settings remain in the selected TOML.
- Likely key typos show spelling hints. Irrelevant source fields identify the
  selected mode. Existing-output errors explain the explicit ways to repeat a run.
- Host-only checks reject zero-flux inline sources, decay peaks below their
  baselines, incompatible source/observer support, inconsistent flux interval
  policy, invalid plot epochs and overflowing/excessive uniform-bin requests.
  Missing or invalid output parents fail during checking, without creating files.
- File-adapter errors carry their owning TOML section or file. Empty flux
  tables, bad selected fluxes, malformed histories, missing FITS headers and
  invalid cloud/material inputs return CLI exit 2 without a Python traceback.
- Independent tests check integrated fluence for every source mode, native
  FITS column/axis closure, and separate NPZ-only and FITS-only publication.
- The portable acceptance runner executes checks and simulations from a path
  containing spaces, verifies products with the existing independent inspector,
  and saves a compact `stage2_report.zip`, including logs on failure.

The configuration version and physical product schemas retain their previous
meanings. No transport, scattering, absorption, material producer, source
sampler, observer estimator or normalization algorithm was changed. The
numerical plan's adapter boundary now supplies field context for input errors.

## Validation

| Check | Result |
| --- | --- |
| Main regression suite | 276 tests passed, 246.246 s; no reported skips |
| Final focused configuration/product suite | 46 passed, 8.040 s |
| Separate RefleX material tests | 4 passed |
| Separate NewDust analysis tests | 3 passed |
| End-to-end examples | 7/7 passed; 64 packets each, CPU |
| Read-only `check` | All examples pass without transport, publication or output-directory creation |
| All source modes | Independently integrated source fluence agrees within float32 tolerance |
| Native FITS example | Preserved column, axes and density-times-width closure |
| Installed wheel outside checkout | Explicit MJD-table example completed with packaged WD01 materials; independent output inspection passed |
| Protected manifests | RefleX 63/63, WD01–RefleX 25/25, multi-material 58/58 unchanged |
| Changed Python lint/format | Ruff 0.17.1 passes |
| Whole documented lint scope | Same four pre-existing findings, in unchanged historical files |
| Python 3.10 compatibility | 71 Python files pass grammar parsing; native and forced tomli parser routes pass |

Stage 2 validation used Linux CPU, Python 3.12.14, NumPy 2.3.5, JAX 0.11.2,
Astropy 8.0.1 and Matplotlib. The Stage 1 report from the user's Windows machine
passed on Python 3.14.4/JAX 0.11.2; Stage 2's new Windows acceptance run remains
to be recorded. Grammar/import-route checks are not a Python 3.10 runtime test.
PowerShell is unavailable in this environment; the packaged updater retains
the previously exercised bundle/hash/fast-forward/push pattern.

The native FITS fixture is deliberately small and low-column. A 64-packet run
can produce an empty halo image; numerical completion and column closure are
still meaningful. These are integration examples, not converged scientific
simulations or a new material-comparison campaign.

## Apply and verify on Windows

Download the delivered `run_configuration_stage2.ps1`, then run from the
existing repository prompt:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\Downloads\run_configuration_stage2.ps1" -Push
```

The self-contained script defaults to the current directory, verifies its
embedded package, requires the expected branch and a clean tracked worktree,
and fast-forwards from the Stage 1 checkpoint. It runs the focused tests and
all seven examples with the checkout's `.venv` Python. On success, `-Push`
publishes to the expected GitHub repository and verifies the exact remote tip.
Omit `-Push` for local validation only. The printed report path is under
`outputs/configuration-stage2/<session>/stage2_report.zip`; existing run outputs
are not reused. The updater can rerun at the delivered tip with a fresh session.

`docs/CONFIGURED_SIMULATIONS.md` documents the examples and active schema;
`configs/data/README.md` records each synthetic input contract. The next planned
slice is Stage 3, unifying the maintained legacy/root runners with the shared
configuration/build/run path, including their remaining source/scene features.
