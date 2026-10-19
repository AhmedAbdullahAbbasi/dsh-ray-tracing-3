# DSH simulation-configuration refactor: implementation handoff

Date: 19 October 2026 (Europe/Istanbul)

New branch: `refactor/simulation-config`

Starting checkpoint: `1cea953cdd2fe8c3161459ed5ea5a55dd3e3926d`

Repository: <https://github.com/AhmedAbdullahAbbasi/dsh-ray-tracing-3>

## 1. Purpose, user requirement, and implementation status

The user wants **one configuration file per simulation**. Material files, source flux/spectral files, the cloud FITS cube, geometry, numerical settings, observer bins, outputs, and plotting choices must be selected there. Running a different simulation should require editing or selecting that file, without editing Python modules, notebooks, or a shell script and without assembling a long set of command-line physics options.

This handoff documents the current architecture, a concrete proposed configuration contract, changes needed, implementation order, and acceptance tests. **The new parser and configured runner are not implemented in this handoff commit.** Existing simulations still use their current interfaces. TOML snippets and new commands below are proposals, not commands already supported at this checkpoint.

The new local branch was created directly from the audited latest saved checkpoint, preserving all current material, history, response, and RefleX comparison work. It does not branch from the older input/config refactor. Publishing the new branch was blocked by automatic approval review because remote publication was not explicitly authorized; no workaround was used. Local branch creation and this report are unaffected.

The remote `development/post-reflex-improvements` was at `a505fc325f238b95251c08ac51d197e49169a942` during the audit. The saved checkpoint adds `eeb96ea` and `1cea953`, two later comparison commits. Preserve those commits when transferring this work to another checkout.

### Definition of “one file”

The run configuration is the single authority for **run-specific choices**. Large flux arrays, cloud data, and versioned material tables remain referenced input files. Their units, epochs, column conventions, and material chemistry remain authoritative data contracts; the configuration must not silently contradict or rewrite them. For example, selecting a different material bundle belongs in the simulation file; changing the grain chemistry requires a newly generated/audited bundle, then selecting it there.

No hidden material/source fallback, unrecorded environment override, or source-code default should change a configured experiment. Documented parser defaults are acceptable if they are expanded into the saved resolved configuration. The intended normal command selects a file only:

```powershell
dsh check configs/4u1630.toml
dsh run configs/4u1630.toml
```

These commands are proposed. Keep the existing `dsh-v1` command available during migration. Do not add a second set of `--photon-index`, `--material`, `--packets`, etc. overrides to the new configured interface. An alternate run should have an alternate configuration file.

## 2. Baseline evidence and protected behavior

The preceding audit on the same numerical source checkpoint established:

| Check | Result |
| --- | --- |
| Main suite under `tests/` | 230 tests passed, no reported skips with physics extras installed |
| Separate RefleX material tests | 4 passed |
| Separate NewDust azimuth tests | 3 passed using documented discovery |
| Frozen RefleX manifest | 63/63 entries match |
| WD01–RefleX comparison manifest | 25/25 entries match |
| Multi-material comparison manifest | 58/58 entries match |
| Installed-wheel material/CLI smoke | Passed outside source checkout |
| Response collect/save/load/fold smoke | Passed with 129 histories and a short final batch |
| Main-suite `dsh` coverage | 84.58% statements; 66.11% branches |

These checks were performed for the preceding audit, not repeated as a fresh physics-validation campaign for this documentation-only change. The audit environment was Linux CPU, Python 3.12.14, JAX/JAXlib 0.11.2, NumPy 2.3.5, Astropy 8.0.1, SciPy 1.17.0, miepython 3.3.0. No GPU/Windows/XSPEC rerun or new 24-million-history comparison is claimed.

Retain the frozen matched-RefleX checkpoint `8a5073cab8020f924ca6f4d30548413ffe3bee5f` and the exact protected adapter/evidence files. More comprehensive WD01 material does not make V1/RG–Drude/TBabs or rectangular-launch baselines obsolete: they are regression controls.

### Current physics that the usability work must preserve

- Hydrogen-column increments per cloud voxel, with radial density conversion and exact native-frustum ray integration.
- Isotropic source normalization with importance-sampled launch directions and exact PDF correction; radial and rectangular proposals remain available.
- Elastic dust scattering, competing analog absorption, multiple scattering, explicit interaction-limit/invalid states, and peel-off observer scoring.
- Outgoing event-to-observer extinction on virtual rays. Earlier observer scores remain valid when the continued analog history is later absorbed.
- Intrinsic cross sections per H nucleus, normalized phase functions/CDFs, declared material budgets, disjoint gas/H2/grain absorption, and one-sided edge conventions.
- Time-variable intrinsic spectra and physical energy/sky/delay responses; all events from a photon remain grouped for Monte Carlo moments/covariance, including zero-score histories.
- Physical ideal-observer fluence, not instrument counts. Effective area, PSF, RMF, background and counting noise remain separate future features.

The bundled complete WD01 recipe is an experimental WD01-shaped/D03-optics graphite–silicate sphere material under a Wilms C50/Si90 solid inventory, not nominal published WD01 and not identical RefleX physics. It combines residual neutral TBnew/Verner gas, H2, and Mie grain absorption. Preserve these distinctions in configuration summaries and output metadata.

## 3. Current architecture

The numerical calculation is already separated more cleanly than the user-facing execution paths. The pipeline is source sampling → launch → transport → peel-off scoring → binning. Files and model selection are handled outside that pipeline, but in several different runners.

| Module or area | Current responsibility | Consequence for this refactor |
| --- | --- | --- |
| `dsh.cli` | Argument parsing, material/source/cloud construction, observer setup, execution, metadata, NPZ/FITS export | Extract orchestration; retain old argument compatibility |
| Root `sim_config.py` | Executable Python globals for cloud, flux-file, numerical, image and device choices | Familiar one-file idea, but incomplete feature coverage and not a validated declarative contract |
| Root `run_simulation.py` | Reads those globals, sets JAX/Matplotlib environment, constructs disc/FITS scene and flux-file/custom sources, runs and renders | Active functionality must migrate into shared adapters, not be deleted |
| `submit_simulation.sh` | Slurm directives, site-specific interpreter/account/resources, reads device from Python configuration | Accept the same run file; remove duplicated simulation choices from shell code |
| `dsh.pipeline` | JIT-compatible single batches and host-coordinated chunk accumulation | Keep numerical interfaces and PRNG/chunk behavior initially |
| `dsh.sources.models` | Tabulated and power-law source builders/samplers; variable light-curve models | Reuse, consolidate precision validation, preserve source fluence |
| `dsh.sources.histories` | Intrinsic versioned JSON history, cell slopes/fluxes, exact time integration | File-selected source adapter; do not require conversion to a new format first |
| `dsh.sources.launch`, `radial_launch` | Proposal construction and direction sampling | Explicit configured sampler, preserved weights |
| `dsh.io.cloud_fits` | Primary FITS cube, linear local angular/distance axes and units | Keep native cube contract; reject unsupported celestial/rotated WCS explicitly |
| `dsh.geometry` | Validated cloud arrays, coordinates, exact ray columns | Keep independent of file paths/configuration |
| `dsh.physics.newdust`, `absorption`, `materials` | Generic table loading, closure/checksum checks, pair assembly and declared composition | Build a common host material-input adapter without changing runtime arrays |
| `dsh.physics.wd01_ism` | Load/build complete budgeted gas/H2/grain material directories | Preserve all component and stored-total consistency checks |
| Other `dsh.physics` producers | Grain populations, optical constants, Mie/ADT and gas basis generation | Offline producer work, not automatically invoked by a simulation run |
| `dsh.transport` | Photon histories and terminal states | Preserve numerical engine; centralize host acceptance outside kernel |
| `dsh.observer` | Scoring/binning and physical response collection/folding | Shared run inputs; later expose response workflows through the same configuration system |
| `dsh.io.npz_output`, `fits_output`, `atomic` | Physical products, embedded arrays/provenance, NPZ atomic writes | Share metadata and publication policy; protect schemas |
| `dsh.validation` and `tests/` | Independent references and regression tests | Keep scientific oracles independent of production helpers |
| Comparison/generation/plotting scripts | Experiment-specific setup, collection, analysis, provenance and views | Thin adapters over maintained support; frozen scripts remain immutable evidence |

### Parallel execution paths and feature differences

| Feature | CLI / `dsh-v1` | Root config/cluster runner |
| --- | --- | --- |
| Synthetic four-cloud example | Yes | Disc example instead |
| Native cloud FITS | Yes | Yes |
| Representative flare / exponential decay | Yes | Yes |
| Explicit custom tabulated source | No direct CLI input | Yes |
| MJD flux text file | No direct CLI adapter | Yes |
| Intrinsic spectral-history JSON | Yes | No |
| Declared complete WD01 bundle | Yes | No |
| Frozen V1 and 2–10 keV legacy materials | Yes | Yes, selected partly by source-energy mode |
| RefleX-exported material | Separate benchmark tools | Optional root hook |
| Physical NPZ/FITS export | Yes | No: PNGs and scored-event NPY map |
| Configured image selection/scales | Limited separate tools | Yes |
| Shared nonzero exit for incomplete transport | No | No |

Material selection must become independent of source file type and whether the source uses continuous energies. Changing energy bands must not silently select a different dust/absorption prescription.

### Important present limitations

1. Root configuration defaults request one billion packets on GPU with interaction cap four and a cloud file absent from the checkout. Do not run those defaults during development tests.
2. The root runner sets JAX platform/backend-related state at import time. The CLI imports JAX before parsing arguments. A future device choice from a file therefore needs a bootstrap boundary.
3. Cloud FITS expects NumPy order `(distance, sky_y, sky_x)`, angular offsets in arcsec, radial distance in kpc, and voxel `delta_NH` in cm^-2. It rejects celestial/rotated WCS and does not automatically recenter on arbitrary source-position header keywords.
4. The numerical source is on the line-of-sight axis at `(D,0,0)` in Cartesian `(los,sky_x,sky_y)`. A generic source-position field is not already supported.
5. Source bands, material interpolation nodes, and observer energy bins are different axes and must remain separately defined.
6. Main simulation NPZ schemas 6/7/8 encode different capabilities. Response archives have their own schema 1. Do not conflate these with the new configuration version.

## 4. Existing older refactor: reuse selectively

Remote branch `refactor/inputs-and-config` at `14f750967ca35160d167acd2a681715db88c564c` contains a previous TOML parser, immutable `ResolvedRun`, build/run-plan separation, core contracts, material/source adapters, product inspection, compatibility paths, and package-boundary tests.

It shares ancestor `e701700efaa4e3c2f9cd1c856f40acf37ed83d3e` with the current line; the audited line has ten unique commits and that branch has six. It predates the present radial proposal, WD01 declared composition, edges, histories/response, and recent comparisons. Its source-FITS format and configured schema-7 products are not automatically equivalent to current JSON histories or current edge-aware schema 7.

Use its architecture and tests as reference, not as replacement numerical code. Do not merge it wholesale or remove current features to fit its narrower configuration object. Inspect without changing branches:

```bash
git show 14f750967ca35160d167acd2a681715db88c564c:docs/refactoring.md
git show 14f750967ca35160d167acd2a681715db88c564c:dsh/config/load.py
git show 14f750967ca35160d167acd2a681715db88c564c:dsh/build.py
git show 14f750967ca35160d167acd2a681715db88c564c:tests/test_package_boundaries.py
```

The current new branch deliberately starts at `1cea953`.

## 5. Proposed configuration contract

### Format and policy

Use **TOML**, with `config_version = 1`, comments, explicit units in field names where appropriate, and strictly validated sections. It is human-editable without executing arbitrary Python. Keep Python >=3.10 support by adding a conditional `tomli` dependency for Python <3.11; use `tomllib` on newer Python. Test both import routes. Do not change the minimum Python version incidentally.

Resolve relative paths against the configuration file's directory, never the shell working directory. Support native absolute paths and user-home expansion deliberately. Do not interpolate environment variables or allow arbitrary Python expressions. On Windows, recommend forward-slash paths or TOML literal strings, because quoted backslashes otherwise have escape meanings. Paths with spaces must work.

Use one explicit source kind, scene kind and material kind per run; reject irrelevant fields instead of silently ignoring them. Reject unknown keys and unsupported versions, duplicate/conflicting definitions, invalid scalar types (including booleans supplied as integers), nonfinite values, missing files, and invalid domains with errors naming the field/file. Unsupported future workflow sections must fail clearly rather than be ignored.

### Sections and ownership

| Section | Run-specific choices | Validation / authoritative metadata |
| --- | --- | --- |
| `run` | Name and workflow (`simulate` initially) | Safe output identifier; known workflow |
| `scene` | Cloud file or explicit example kind; source distance; coordinate policy | Native axis units and physical bounds; no assumed recentering |
| `source` | File or inline model, columns, time convention, source bands, slopes, flux scale and interval policy | Intrinsic photon-flux convention, epoch/time scale, gap policy, precision |
| `materials` | Complete bundle directory or explicit scattering/absorption files | Existing checksums, matching grids, phase closure, edge and chemical coverage |
| `execution` | Packets, chunk size, interaction cap, seed, CPU/GPU request, launch sampler | Positive integer counts, supported device, effective dtype/backend recorded |
| `observer` | Sky grid/FOV, energy output edges, arrival bins | Separate from source/material axes; finite ordered bins and coverage |
| `quality` | Incomplete-run policy and diagnostic export | Production success requires clean numerical terminal states |
| `output` | Output directory, formats, overwrite policy | Protect every input and sidecar from overwrite; save physical products |
| `plots` | Enable, scales, energy selection, epoch windows, crop, DPI, event-map selection | Supported explicit selections, pixel-area normalization, units |
| `logging` | Progress cadence and log filename | Under resolved output directory; no import-time logging side effect |
| `scheduler` | Optional Slurm account/partition, CPU/memory/time/GPU resources, optional notifications | Same file controls per-run resources; no duplicate physics settings in shell |
| Workflow-specific response fields | Proposal or response file, fold bins and covariance selection | Later staged integration; only valid in matching response workflow |

There is no requirement to expose every internal numerical constant as a setting. In particular, no arbitrary absorption/scattering scale factor should be introduced as part of the refactor. Model identities come from validated inputs; labels are descriptive, not switches that override their physics.

### Proposed example: one 4U 1630–47 run

The following is a **design example**, syntactically valid TOML, intended to reside at `configs/4u1630.toml`. It is not an implemented schema or a claim that the cloud file is available here. The MJD time scale is illustrative and must be verified from the flux-data provenance before a science run. A text table whose metadata are unknown must be rejected until the user supplies them correctly.

```toml
config_version = 1

[run]
name = "4u1630_e2_wd01"
workflow = "simulate"

[scene]
kind = "fits"
cloud_file = "../clouds/4u1630_NH_111111111100001_DSH_11p5kpc.fits"
source_distance_kpc = 11.5
coordinates = "source_centered_offsets"

[source]
kind = "flux_table"
file = "../fluxes/1630_allflux.txt"
time_column = 1
flux_columns = [3]
column_numbering = "one_based_full_table"
time_format = "mjd"
time_scale = "UTC" # Illustrative: verify from the data, do not assume.
epoch_mjd = 57625.0
interval_policy = "hold_until_next"
last_interval_days = 1.0
max_interval_days = 1.0
missing_data = "error"
flux_convention = "unabsorbed observer-equivalent"
flux_unit = "ph cm^-2 s^-1"
flux_scale = 1.0
spectral_mode = "band_powerlaw"
energy_edges_kev = [2.25, 3.15]
photon_index = 2.0

[materials]
kind = "declared_bundle"
directory = "../dsh/data/materials/wd01_wilm_c50_si90_v1"

[execution]
packets = 1000000
chunk_size = 8192
max_interactions = 32
seed = 2026
device = "cpu"
launch_sampler = "radial"

[observer]
sky_grid = "cloud"
energy_edges_kev = [2.25, 3.15]

[observer.arrival]
kind = "uniform"
start_days = 0.0
stop_days = 200.0
step_days = 1.0
last_bin = "shorten"

[quality]
allow_incomplete = false
save_failure_diagnostics = true

[output]
directory = "../outputs/4u1630_e2_wd01"
formats = ["npz", "fits"]
overwrite = false

[plots]
enabled = true
scales = ["log", "linear"]
energy_selection = "all"
epoch_days = [100.0, 101.0]
crop = "full"
save_scored_event_map = true
dpi = 200

[logging]
progress_every_chunks = 5
file = "run.log"
```

The whole-table one-based column convention is deliberate: in the current text file, MJD is column 1 and old `FLUX_COLUMNS=(2,)` means the **second flux column after removing MJD**, i.e. whole-table column 3. The migration adapter must make this conversion explicit; changing numbering silently would select the wrong source flux.

The numbers above illustrate an initial configured run, not proof that one million packets converges or cap 32 is sufficient. The quality report decides numerical completion, while sampling/convergence remains a separate scientific assessment. Source flux data must actually represent integrated intrinsic photon flux in the declared band; specifying a unit in TOML cannot convert an instrument rate or energy flux without a validated conversion.

#### Alternative material selection

For the legacy 2–10 keV pair, replace the material section only:

```toml
[materials]
kind = "table_pair"
scattering_file = "../dsh/data/scattering/rg_drude_2_10_v2.npz"
absorption_file = "../dsh/data/absorption/tbabs_wilm_vern_2_10_v2.npz"
grid_file = "../dsh/data/materials/material_grid_2_10_v2.json"
```

This explicitly selects files, rather than selecting a source-energy mode that happens to choose them. The frozen V1 pair uses the same kind with its corresponding files, subject to its supported energy domain. Adjacent JSON sidecars remain required and are included in the input manifest.

`declared_bundle` selects all validated component files in the named directory: material definition, scattering, atomic gas, H2, grains and stored total absorption with their metadata. Keep the stored-total consistency check. Do not treat the bundle as an arbitrary scattering/absorption pair or allow adding full TBabs opacity to its existing grain term.

A RefleX-export adapter should select exported files/bundle through this same section and preserve the fixed reference's model metadata. It must not pretend WD01 is the exact RefleX model. Any no-absorption control is an explicit diagnostic capability, recorded in provenance, not a hidden default. Keep frozen adapter source unchanged while implementing a maintained adapter.

#### Alternative intrinsic history selection

For an existing versioned spectral-history JSON, replace the source section with:

```toml
[source]
kind = "spectral_history"
file = "../inputs/source_history.json"
```

Epoch, time scale, energy edges, cell slopes, flux convention and units are taken from that validated history file and shown in the resolved summary. Do not duplicate them as competing TOML overrides. Observer output bins can differ from source bands, with explicit coverage checks and documented window losses.

Retain adapters for representative-energy flare, exponential decay, inline tabulated/custom source, continuous power law, and the current variable-source API. Preserve independently specified per-band fluxes: an adapter accepting integrated band fluxes must not renormalize them as if they were one broad-band total. Porting the older source-FITS/line capabilities is separate compatibility work, not a prerequisite for using current JSON histories.

### Required bin, time and position semantics

- Make source-emission times, observer-arrival times, and excess scattering delays explicit. Preserve the shared epoch; MJD conversion must subtract epoch on the host before JAX float32 conversion, rather than placing a large absolute timestamp in float32.
- For a flux table, require explicit interpretation of each row, the last row's duration and gap handling. Current code extends the last interval using the previous cadence; record an explicit duration when migrating. Do not interpolate across missing data automatically.
- For uniform arrival bins, the proposed `last_bin="shorten"` uses the requested step and a final shorter bin. The current CLI uses `linspace` after `ceil`, which redistributes all widths for a non-divisible span. Preserve the old algorithm in the legacy adapter or identify a documented intentional change; do not change it invisibly during extraction.
- Explicit edges are an alternative to uniform bin parameters, never an additional conflicting specification. Keep current full-window simulation checks initially. Deliberate partial arrival/energy windows must have explicit policy and recorded lost fluence; they are not numerical failures or automatic renormalization requests.
- Source/material energy support is checked before transport. Output-energy selection is separate; avoid silently snapping a requested image energy to the nearest unrelated band.
- Initial FITS mode requires already source-centered local offset axes. If future configuration supports `source_pixel` or source header coordinates, specify zero/one-based indexing and convert through a tested column-preserving adapter. Do not claim arbitrary source recentering already works. Celestial WCS reprojection is additional work.
- Image epochs either align with stored bins or explicitly reject unsupported partial-bin extraction. Scored-event maps are sampling diagnostics. Surface intensity uses each pixel's own area and actual selected duration/band width.

## 6. Target architecture and required code changes

Keep the existing numerical module paths in the first implementation. A complete `dsh/core` relocation is not necessary to deliver usability and would complicate provenance/cache preservation.

| Proposed owner | Responsibility | Dependency constraint |
| --- | --- | --- |
| `dsh/config/schema.py` | Frozen typed host configuration objects and mode-specific sections | Paths/settings/metadata only; no JAX arrays |
| `dsh/config/load.py` | TOML parsing, key/type/domain validation, defaults and path resolution | No numerical runtime import or arbitrary config execution |
| `dsh/command.py` | `check`, `run`, later `inspect`/response commands; early backend bootstrap | Parse config before importing JAX-dependent builders |
| `dsh/inputs/source.py` | Source-file/inline adapters using existing builders | No transport or plotting |
| `dsh/inputs/material.py` | Common host bundle around current pair/declared/export loaders | Keep chemistry/checksum rules; no new offline material solves |
| `dsh/inputs/scene.py` | FITS/example adapters and source-coordinate policy | Keep physical column/geometry validation |
| `dsh/build.py` | Validated inputs → numerical run plan plus separate input provenance | Select source/launch/bin builders; no output writing |
| `dsh/run.py` | Run plan → existing pipeline → shared quality result → publication | Numerical API callable without file writes; configured wrapper owns I/O |
| `dsh/quality.py` | Terminal-state evaluation and acceptance/report fields | One policy for CLI, cluster and response tools |
| `dsh/io/provenance.py` | Config/input/runtime hashes and resolved manifest | Explicit raw/normalized-text/array hash policies |
| Existing I/O plus maintained plotting helper | Physical serialization and optional derived views | Consume common products/metadata; no model selection |

These are proposed names, not existing modules at `1cea953`. Reuse names/design from the older branch where useful, but adapt contracts to current features. Avoid a large generic plugin framework or dependency-heavy configuration stack.

### Configuration and runtime bootstrap

Configuration loading must work without importing JAX. The executable first parses the file, then configures the requested CPU/GPU platform, then imports numerical modules. An unavailable requested GPU must fail with a useful diagnostic; do not silently fall back to CPU. A programmatic caller with an already initialized incompatible JAX backend should receive an explicit restart/backend error.

Record actual JAX backend/devices/dtypes and dependency versions. Preserve existing runtime precision in the initial refactor; do not add a float64 switch that falsely claims builders honor it. Thread/resources environment values may describe the host/scheduler, but cannot silently override scientific fields. Validate mode-specific optional dependencies before launching a long simulation.

`check` parses and resolves settings, validates selected files and compatibility, reports source fluence/time/energy support, material identity, cloud shape/column checks, observer shape and a qualified memory estimate. It may initialize the selected backend when building current JAX arrays, but must not compile/run transport or launch a production job. Avoid redundant full input loading in `check` followed by `run` within one process; build a plan once and execute that validated plan, with an explicit input-change policy.

### Numerical plan and host metadata

A plan carries existing source/cloud/material/launch/bin arrays and execution settings. Preserve NamedTuple field order/defaults and JAX pytree behavior. Paths, TOML text, material recipe descriptions, hashes, Astropy objects, loggers and output destinations remain host-side metadata, outside JIT arguments. Existing direct numerical APIs must remain usable independently of TOML.

Extract orchestration before modifying sampling or interpolation. Preserve current seed splitting, packet weight `total_source_fluence / total_packets`, short-batch treatment and float64/int64 host accumulation. Do not promise identical output when chunk size, dependency version or hardware changes; record them all.

### Quality, export and saved run record

The earlier audit reproduced the following defects that need focused tests before migration:

| Defect | Evidence | Required new behavior |
| --- | --- | --- |
| Incomplete transport returns success | 1,024 packets, cap 1: 145 capped histories, exit 0, NPZ/FITS published | Shared failure exit/quality status; only explicit diagnostic export may retain incomplete products |
| Scalar/variable power-law precision guard missing | Gamma -100 over 2–10 keV: all 4,096 samples clip to 10 keV; history constructor rejects | Shared constructor validation or stable supported sampling; no silent endpoint collapse |
| Intensity uses first pixel area | Axes `[0,1,3]` yield intensities `[[1,2],[2,4]]` for a unit-intensity scene | Per-pixel areas and edge-aware rendering, or explicit uniform-grid restriction |
| Timing validation differs | Root accepts zero arrival span as `[0]`; CLI rejects | One validated timing contract with explicit legacy compatibility |

Every configured run should save:

1. Original TOML bytes/hash and a versioned resolved JSON manifest containing expanded defaults and absolute resolved paths.
2. Hashes of all selected inputs and required sidecars/components, with material identity/coverage, source epoch/unit convention, cloud coordinate contract, and actual numerical settings.
3. Software Git identity/dirty state, runtime implementation fingerprint, Python/NumPy/JAX/backend/dtype information, and seed/chunk settings.
4. Canonical physical NPZ/FITS products and moments; plots and integer scored-event maps are optional derived products, not the sole preserved result.
5. A run report containing terminal counts, completion/numerical status, explicit window losses, elapsed time, output inventory, and scientific sampling status distinct from mere numerical completion.

Protect inputs and all metadata sidecars from overwrite, including path aliases/symlinks. Do not publish a success marker before all required outputs finish. Preserve explicit overwrite policy and atomic per-file publication; a multi-file run should have a completion manifest rather than pretend several renames are one atomic operation. Record input hashes for the bytes actually loaded; detect changes during loading/publication rather than checking only the initial filename.

Do not reuse old output schema numbers for a new shape/meaning. Keep configuration version, physical-product schema and response schema separate. Extend metadata compatibly if possible; if a product schema bump is necessary, include old-reader fixtures and a migration note. A structural module move will change source-derived fingerprints: retain archive provenance rather than rewriting old hashes to claim old histories were collected by the new code.

### Root/cluster/response compatibility

The root Python configuration is an old front end, not a second source of runtime defaults. Migrate its useful features into the shared adapters, then retain a thin compatibility wrapper during a documented transition. Do not import/execute an old Python config automatically from the new TOML loader. Provide a migration mapping and an explicit one-time converter only if useful.

The cluster launcher takes a configuration path and reads device/resource requirements from it. Put per-run Slurm fields in an optional `scheduler` section, avoiding a second file of physics settings. Site interpreter/environment setup may remain installation configuration. Slurm submission and notification remain explicit actions, not side effects of parsing, checking or importing a file.

Response collection and folding currently have separate argument-heavy scripts. After the basic simulation interface works, route them through the same config/input/provenance layer with explicit workflow variants. Collection uses proposal history + cloud + material; folding uses saved response + source history + arrival/energy/radius bins + covariance choice. Folding must reject irrelevant transport or material overrides. A saved response's physics is fixed; changing the file selection does not recompute it.

## 7. Migration mapping from present inputs

| Current setting or option | New configuration owner / action |
| --- | --- |
| `CLOUD_FITS_PATH`, `--cloud-fits` | `scene.cloud_file` |
| `SOURCE_DISTANCE_KPC`, `--source-distance-kpc` | `scene.source_distance_kpc` |
| `GRID_*`, `DISC_*` | Tagged disc-example scene section, only valid for that scene kind |
| `SOURCE_MODEL`, `--source-model` | `source.kind` with adapter-specific fields |
| `FLUX_FILE`, `FLUX_SCALE` | `source.file`, `source.flux_scale` plus explicit units/convention |
| `FLUX_COLUMNS` | Convert old post-time-column indexing to explicit full-table indexing |
| Last-row cadence inference | Explicit `last_interval_days`/boundary policy |
| `CUSTOM_*`, `PEAK_*`, `BASELINE_*`, `DECAY_*` | Inline source/model section, preserving integrated fluence |
| Source `ENERGY_EDGES_KEV`, `PHOTON_INDEX` | Source spectral section only |
| `--source-history` | `source.kind="spectral_history"`, file |
| `MATERIAL_MODEL`, `--materials`, material paths | Explicit `materials.kind` and files/directory; record actual bundle identity |
| `PACKETS`, `CHUNK_SIZE`, `MAX_INTERACTIONS`, `SEED` | `execution` |
| `DEVICE`, `LAUNCH_SAMPLER` | `execution`, applied before runtime import where necessary |
| `ARRIVAL_DAYS`, `TIME_BIN_DAYS`, explicit arrival edges | Tagged `observer.arrival` specification; preserve/record bin-generation convention |
| Output energy bin choices | `observer.energy_edges_kev`, independent of source bands |
| Output names and image/count paths | `output.directory` and documented derived names, or validated explicit filenames |
| `IMAGE_EPOCH_DAYS`, energy/FOV/scales/DPI | `plots`, with explicit selection semantics |
| `PROGRESS_LOG`, `PROGRESS_EVERY_CHUNKS` | `logging` |
| Slurm per-job directives | Optional `scheduler`, no source/physics duplication |

## 8. Required tests and evidence

Tests must verify contracts and observable behavior, not mirror parser implementation. Keep reference quadrature/ray/phase/absorption implementations independent. Use small deterministic fixtures for orchestration; retain statistical experiments as separately invoked gates.

### Test matrix

| ID | Area | Required assertions |
| --- | --- | --- |
| C01 | TOML/schema | Accept documented examples; reject unsupported versions, malformed/duplicate keys, unknown sections/keys and empty names |
| C02 | Types/domains | Reject booleans as counts, fractional/zero/negative counts, invalid seeds, nonfinite numbers and invalid enum values |
| C03 | Defaults/resolution | Every effective default appears in resolved manifest; identical config resolves identically independent of working directory |
| C04 | Paths | Relative/absolute paths, spaces, home expansion, Windows TOML escaping, missing files, case behavior on target OS; no undocumented environment expansion |
| C05 | Output protection | Reject collisions with source/cloud/material files and sidecars through relative aliases/symlinks; enforce overwrite=false |
| C06 | Mode exclusivity | Reject bundle + pair fields together, competing source formats, explicit edges + uniform-bin fields, and irrelevant workflow sections |
| C07 | Import boundary | Loading config does not import JAX/Astropy/Matplotlib, initialize a device, create outputs or execute Python config |
| C08 | Bootstrap | Subprocess sets CPU/GPU selection before JAX import; unavailable GPU gives clear failure; already initialized API backend handled explicitly |
| I01 | FITS cloud | Preserve delta_NH/column closure, `(z,y,x)` order, positive/negative axis normalization and arcsec/kpc units; reject unsupported WCS and invalid bounds |
| I02 | Source position | Preserve source-centered input without double shift; reject unimplemented pixel/header placement; later test index base/recentering explicitly |
| I03 | Flux columns | Whole-table column 3 equals legacy second flux column; nontrivial column permutations catch off-by-one errors |
| I04 | Flux times | Convert MJD after epoch subtraction, preserve TT/UTC declaration, irregular intervals, final boundary, one-row policy, duplicate/decreasing times, gap/missing-data failure |
| I05 | Flux meaning | Per-band photon flux remains per-band; broad-band total splits exactly once; reject unsupported absorbed/rate/energy-flux interpretation without conversion |
| I06 | Source parity | Representative, decay, custom, band power law, JSON histories and variable source preserve fluence/energy/time support for representative fixtures |
| I07 | Precision | Gamma near 1, extreme slopes, collapsed edges after dtype conversion, underflowed cells/CDF increments and large epochs cannot silently distort sampling |
| I08 | Material pairs | Matching units/grids/checksums, sidecar mismatches, energy support, phase integral/CDF closure, CRLF policy and explicit no extrapolation |
| I09 | Declared material | Element budget, disjoint gas/H2/grain coverage, stored-total consistency and edge declaration; reject full-ISM-plus-grain double counting |
| I10 | RefleX export | Maintained adapter loads identical arrays to frozen provider for normal and explicit diagnostic control; frozen files unchanged |
| I11 | Observer bins | Separate source/material/output grids, inclusive final boundary, uniform-step short final bin vs legacy redistribution, deliberate partial windows with loss reporting |
| P01 | Plan equivalence | Config-derived inputs/arrays match explicit existing builders for representative/history/declared and legacy cases, with identical shapes/dtypes/fields |
| P02 | Seeded numerical parity | Same backend/versions/seed/chunk/cap preserve source draws, launch weights, histories/products/moments for pure extraction |
| P03 | Chunk accounting | Divisible/nondivisible totals, total below chunk, short tails, normalization once, zero scorers and photon grouping preserved |
| Q01 | Incomplete run | Force capped/invalid statuses; new CLI returns nonzero, report says incomplete, no production success marker, diagnostic exports clearly marked |
| Q02 | Window loss | Finite aperture/time/energy losses are recorded independently of invalid transport; no silent rescaling |
| O01 | Product round-trip | NPZ/FITS agree on axes, units, source/material identifiers, total/first/multiple products and raw moments; old-schema fixtures remain readable |
| O02 | Manifest integrity | Original/resolved config, loaded-input/component hashes and runtime identity saved; changed inputs detected; metadata does not pretend stale archives are new |
| O03 | Failure publication | Interrupted write preserves prior complete file; partial multi-file export has no success marker; overwrite/input protection and existing atomic tests pass |
| O04 | Plot correctness | Constant intensity on nonuniform pixels stays constant, edge-aware positions correct, zero image/log handling, epoch/bin alignment, explicit band selection |
| O05 | Labels/output choices | Event maps say scored Monte Carlo events; fluence/intensity units correct; empty/unsupported scales rejected early; physical output exists even when plots disabled |
| R01 | Actual response collection | Small real collector test, zero events, short/single-tail case, capped-history rejection, save/load/fold, zero-history moments |
| R02 | Response reuse | Exact delays/energies and all orders stay grouped; incompatible source support/provenance and irrelevant material overrides rejected |
| U01 | Command UX | From a different directory, check/run take only a config path; check never calls transport/submits jobs; field-specific errors and nonzero exit on failure |
| U02 | Packaging | Build/install wheel; bundled pair and declared material work without repo-relative imports; conditional TOML dependency tested on Python 3.10 |
| U03 | Legacy front ends | `dsh-v1`, wrapper imports and intended root-runner scenarios keep documented behavior; converter/mapping preserves settings |
| U04 | Cluster launcher | Takes config path with spaces, derives device/resources once, dry-run shows job spec, no submission or notification during tests |
| V01 | Scientific ladder | Existing column, competing-risk, multiple-scattering, observer extinction, edge, ring/delay and covariance controls continue passing |
| V02 | Frozen benchmarks | All 63/25/58 manifest entries match; existing statistics/camera normalization unchanged; numerical versus physical-model mismatch remains distinct |

### Proposed test organization

Add focused files such as `tests/test_run_config.py`, `tests/test_run_inputs.py`, `tests/test_run_plan.py`, `tests/test_configured_run.py`, `tests/test_configured_cli.py`, `tests/test_runtime_bootstrap.py`, `tests/test_plot_products.py`, and `tests/test_package_boundaries.py`. Reuse current fixtures instead of constructing many full dense clouds. Extend current source/material/output/response tests where the invariant belongs rather than duplicating the entire physics suite in new files.

For pure moves/extraction, capture seeded array shape/dtype/content snapshots under the same environment before and after. Bitwise equality is useful there, not a cross-hardware promise. For an intentional numerical correction, write a targeted counterexample and use justified tolerances; keep that correction separate from broad structural changes. Monte Carlo image comparisons require predetermined statistical gates, not adjusted tolerances until a plot looks similar.

### Execution tiers

1. Per change: parser/path/type tests, targeted adapter/plan/status tests, relevant numerical unit tests and boundary checks.
2. Per integration checkpoint: complete main suite, separate RefleX/NewDust tests, package installation smoke, current manifest verification and representative configured end-to-end cases with material extras enabled.
3. Before merging migration: Windows/PowerShell path tests and supported minimum/current Python/JAX environments; GPU job separately where available; cluster helper mocked/dry-run first.
4. If numerical/sampling/precision logic changes: relevant statistical ladder cases and targeted RefleX collection comparison, with expensive full recollection only when warranted.

Current baseline commands, valid now:

```powershell
python -m unittest discover -s tests -t . -v
python -m unittest reflex_adapter.test_reflex_materials -v
python -m unittest discover -s benchmarks/newdust -p "test_*.py" -v
ruff check dsh scripts tests
ruff format --check dsh scripts tests
```

In the audit environment Ruff 0.17.1 finds four lint items within `dsh scripts tests` and an active validation-format difference, plus historical/notebook issues in whole-tree checks. Pin the intended tool version and record any existing exceptions; do not promise a clean baseline or autoformat frozen benchmark source. Some Mie tests skip without optional material dependencies. Use an explicit CI environment with producer extras and supported external dependencies rather than interpreting a lightweight install as full validation.

## 9. Implementation order and completion criteria

| Slice | Work | Gate |
| --- | --- | --- |
| 0 — Establish baseline | Preserve `1cea953`/frozen evidence; reconcile old refactor design; capture representative explicit-build arrays and schema inventory | Prior audit results understood, current source clean, no lost feature contract |
| 1 — Schema and parser | Typed TOML loading, relative paths, defaults, strict modes and import boundary; examples with small CPU workloads | C01–C08 and parser-only command tests |
| 2 — Input adapters and plan | Material pair + declared bundle; native FITS/example; JSON history + existing source modes/flux table; observer/launch construction | I01–I11 and P01; no transport algorithm move |
| 3 — Configured execution | Early backend bootstrap, shared quality result, numerical runner and canonical physical exports, resolved manifest/check/run commands | P02–P03, Q01–Q02, O01–O03, U01–U02 |
| 4 — Usability parity | Plots, root-runner migration, optional Slurm section/launcher, documentation and config-only demonstrations | O04–O05, U03–U04; no Python edits needed for a new run |
| 5 — Response and benchmark convergence | Configured response workflows, maintained benchmark adapters/support and explicit archive compatibility | R01–R02, V01–V02; frozen collector unchanged |
| 6 — Cleanup and speed | Retire confirmed unused helpers/duplicate notebooks after compatibility coverage; profile representative workloads | No undocumented feature loss; measured speed gains with scientific invariants retained |

Fix the demonstrated precision/status/intensity defects in small, separately reviewable commits as their owning slice is introduced. Avoid combining configuration migration, source resampling, interpolation changes and GPU optimization into one commit. ZDA04 and detector effects are separate future features.

### User-facing completion criteria

The first useful milestone is complete when the user can select a cloud file, choose either legacy pair or complete WD01 material, choose flux-table or history input, set source distance/bins/seed/packet counts/device, and choose outputs by editing **one TOML file**, then run it through one command. It must:

- Report the resolved source/material/geometry before execution and reject incompatible inputs early.
- Produce reusable physical NPZ/FITS data with provenance, not just PNGs.
- Fail clearly on incomplete numerical transport and distinguish that from intentionally lost window flux or insufficient Monte Carlo precision.
- Reproduce equivalent explicit-builder results under the same environment, retain all current regression controls, and run from a directory outside the source tree.
- Preserve the original simulation file and expanded settings with every run, so a future fit/comparison can identify exactly what was simulated.

## 10. Branch and handoff transfer

This report is stored in the new local branch at `docs/SIMULATION_CONFIG_REFACTOR_HANDOFF_2026-10-19.md`. No runtime/configuration implementation is included yet. Branch publication was blocked as described in section 1, so do not assume the branch exists on the remote.

If recreating the branch in another clone that already contains the saved checkpoint, verify a clean tracked working tree and the starting commit first:

```powershell
git status --short
git show --no-patch --oneline 1cea953cdd2fe8c3161459ed5ea5a55dd3e3926d
git switch -c refactor/simulation-config 1cea953cdd2fe8c3161459ed5ea5a55dd3e3926d
```

Untracked personal data/bookkeeping archives should remain outside commits. Do not run `git add .` or delete such files to make this branch clean. If the checkpoint is absent, recover its two comparison commits from the saved Git bundle before branching; do not silently substitute remote `a505fc3` and lose the latest comparisons. Remote publication is a separate action requiring authorization.

## Appendix — Source evidence map

Paths are relative to the audited repository, whose local checkout is `/workspace/scratch/ce63bade8aaf/audit_repo`. They describe current modules, distinct from the proposed layout in section 6.

| Current file | Relevant evidence |
| --- | --- |
| [dsh/cli.py](sandbox:/workspace/scratch/ce63bade8aaf/audit_repo/dsh/cli.py) | Existing material/history/source restrictions, setup and export, warning-only incomplete policy |
| [sim_config.py](sandbox:/workspace/scratch/ce63bade8aaf/audit_repo/sim_config.py) | Current user-editable Python settings, flux columns, GPU/billion-packet defaults |
| [run_simulation.py](sandbox:/workspace/scratch/ce63bade8aaf/audit_repo/run_simulation.py) | Backend import side effect, flux-file adapter, disc scene, image area/count labels and limited persisted products |
| [submit_simulation.sh](sandbox:/workspace/scratch/ce63bade8aaf/audit_repo/submit_simulation.sh) | Current device-to-Slurm resource logic and site settings |
| [dsh/pipeline.py](sandbox:/workspace/scratch/ce63bade8aaf/audit_repo/dsh/pipeline.py) | Shared numerical coordinator, source normalization, PRNG/chunk behavior and accumulation precision |
| [dsh/io/cloud_fits.py](sandbox:/workspace/scratch/ce63bade8aaf/audit_repo/dsh/io/cloud_fits.py) | Native linear angular/distance FITS contract and WCS restrictions |
| [dsh/geometry/clouds.py](sandbox:/workspace/scratch/ce63bade8aaf/audit_repo/dsh/geometry/clouds.py) | Physical cloud validation and column closure |
| [dsh/sources/histories.py](sandbox:/workspace/scratch/ce63bade8aaf/audit_repo/dsh/sources/histories.py) | Intrinsic history schema, epoch/units, precision guards and exact time integration |
| [dsh/sources/models.py](sandbox:/workspace/scratch/ce63bade8aaf/audit_repo/dsh/sources/models.py) | Existing scalar/variable/tabulated model APIs and power-law sampling |
| [dsh/physics/materials.py](sandbox:/workspace/scratch/ce63bade8aaf/audit_repo/dsh/physics/materials.py) | Matched pair and disjoint declared absorption composition |
| [dsh/physics/wd01_ism.py](sandbox:/workspace/scratch/ce63bade8aaf/audit_repo/dsh/physics/wd01_ism.py) | Complete bundle loader and component/stored-total verification |
| [dsh/io/npz_output.py](sandbox:/workspace/scratch/ce63bade8aaf/audit_repo/dsh/io/npz_output.py) | Physical data/provenance fields and schemas 6/7/8 |
| [dsh/observer/response_transport.py](sandbox:/workspace/scratch/ce63bade8aaf/audit_repo/dsh/observer/response_transport.py) | Actual physical response collector and runtime fingerprint |
| [scripts/generate_spectral_response.py](sandbox:/workspace/scratch/ce63bade8aaf/audit_repo/scripts/generate_spectral_response.py) | Current response input options to migrate |
| [scripts/fold_spectral_response.py](sandbox:/workspace/scratch/ce63bade8aaf/audit_repo/scripts/fold_spectral_response.py) | Current folding inputs/bins/covariance options |
| [pyproject.toml](sandbox:/workspace/scratch/ce63bade8aaf/audit_repo/pyproject.toml) | Python/dependency support, extras, package-data and installed CLI |

For the full dead-code/redundancy disposition and reproduced audit probes, see the separately delivered `DSH_PRE_REFACTOR_AUDIT_2026-10-19.md`. This handoff contains the decisions and requirements needed to implement the requested usability work without relying on that report being open.
