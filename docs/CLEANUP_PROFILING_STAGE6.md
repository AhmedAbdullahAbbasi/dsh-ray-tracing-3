# Stage 6: cleanup and profiling audit

## Scope and evidence

Starting checkpoint: `6ce16e5` on `refactor/simulation-config`. Stages 1–5 had
passed portable Windows checks; actual cluster testing remains deferred.
The initial inventory examined 163 Python files across root entry points,
`dsh`, `scripts` and `tests`. AST top-level definitions plus repository text
references were used to find review candidates; imports, explicit entry points,
compatibility tests and frozen manifests determined their disposition.
This is a conservative static review, not proof of whole-program reachability.
Notebook/external users and dynamic dispatch are outside a text-reference scan.

| Finding | Disposition | Reason |
|---|---|---|
| Simulation and response publication lifecycles | Consolidated in `dsh/io/publication.py` | The same staging, provenance, input recheck, hashing and completion policy had two implementations. |
| `_json`/`_publish` housed in the simulation runner | Moved to public `write_json`/`publish_file` I/O helpers | Response orchestration no longer imports private simulation-runner helpers. The private names had repository callers only at the reviewed migration sites. |
| Adapter error annotation and backend validation | Shared in `dsh/inputs/common.py` | Response construction no longer imports a private helper from the simulation builder; backend checks share the same contract. |
| Imports made redundant by those moves | Removed | Ruff verifies the modified modules. |
| `run_variable_powerlaw_source_to_observer_chunked` | Retained | The scan found no direct repository call, but it is a public numerical API. No absence-of-use claim is made for external callers. |
| Unreferenced `unittest.TestCase` class names | Retained | Test discovery instantiates them; direct textual calls are unnecessary. |
| Historical standalone response and cluster scripts | Retained | They are explicit external entry points and compatibility workflows; cluster migration is not yet site-tested. |
| Archived benchmark code/data and older scientific interfaces | Retained | Reproducibility and compatibility take precedence over speculative deletion. |

No source sampler, transport kernel, material prescription, cross-section
interpolation, response estimator or archive numerical format was changed.
The cleanup removes demonstrated duplication and relocates shared boundaries.
It does not claim that every unused external API has been identified or removed.

## Publication invariants

All configured workflows now use shared staged publication. Input hashes and
output destinations are checked before execution and again before publication.
An overwrite invalidates the previous completion marker before new work starts.
The new completion marker is written only after every selected file publishes.
Cross-device publication still stages a complete file at its destination;
no-overwrite publication retains exclusive file creation. Mapped legacy paths
and their completion inventory remain supported. Diagnostic/incomplete runs
retain their workflow-specific quality policy and cannot obtain a production
success marker.

Coverage includes real NPZ/FITS products, response archive loading/folding,
legacy output paths, injected exporter/publication failures, stale inputs,
overwrite invalidation, no-overwrite races and cross-device publication.
Private test patch sites moved to the shared I/O module; public workflow
commands and product schemas are unchanged.

## Profiling methodology

`python -m dsh.profiling CONFIG --output-directory NEW_DIRECTORY --repeats N`
uses the configured builder and executor. It rejects incomplete numerical runs,
waits for asynchronous JAX arrays, and hashes scientific result arrays and
history diagnostics to require exact seeded replay. Response provenance is
excluded from this numerical fingerprint; configuration, input and runtime
identity are recorded separately. The profile directory cannot replace or sit
inside a production output directory.

Measurements separate parsing, numerical imports, plan construction, first
execution and repeated execution of the same in-memory plan. Execution includes
input-hash rechecks; fingerprinting, exports and plotting are outside the timed
interval. An additional cProfile pass produces `calls.pstats`/`calls.txt` without
contaminating the uninstrumented execution timings. First execution may include
compilation; repeat calls can also compile, especially where a collector creates
a new JIT closure. Repeated measurements are not equivalent to separate fresh
CLI jobs, and first execution is not asserted to be a cache-free compilation.

Acceptance profiles use a 2048-packet representative-energy simulation, a
512-history WD01 response collection, and folding of a saved 64-history response.
All run on CPU with two uninstrumented repetitions plus a separate instrumented
pass. These are integration workloads, not production-scale convergence or
performance-scaling benchmarks. Profile output includes exact parameters,
backend, dependency versions, hashes and runtime identity.

No speedup is attributed to the cleanup: there is no controlled before/after
performance claim. GPU/cluster performance, peak-memory use, export latency and
plot generation are not measured here. Numerical optimizations remain separate
work and need a representative workload plus normalization/parity evidence.

## Observed development CPU profile

These illustrative measurements used Linux CPU, Python 3.12.14 and JAX 0.11.2
in the development environment; other validation work could be active. They
are observations for the recorded integration inputs, not hardware-independent
benchmarks or before/after comparisons.

| Workload | First execution (s) | Repeat median (s; 2 samples) |
|---|---:|---:|
| 2048-packet representative simulation, chunks of 256 | 6.249 | 3.619 |
| 512-history WD01 response collection, chunks of 256 | 2.989 | 2.630 |
| Folding the saved 64-history response | 0.00109 | 0.000754 |

The simulation cProfile pass attributed about 3.37 of 3.40 seconds to NumPy
materialization at the JAX/host boundary. This includes waiting for transport
work; it does not isolate conversion/copy cost or justify removing the
synchronization that enforces bounded memory and honest progress reporting.
The response collection pass spent about 1.34 of 2.75 seconds in compilation,
consistent with creation of a fresh collector JIT closure on each invocation.
Hoisting/caching that compilation is a future performance candidate requiring
separate numerical and memory validation. The tiny fold completed in roughly
a millisecond; its behavior cannot predict a large response/covariance workload.
All first, repeated and instrumented scientific result fingerprints matched.

## Validation and delivery

`python scripts/validate_cleanup_stage6.py` runs publication/profiling tests,
Stage 3 runner acceptance, Stage 5 response acceptance, frozen-manifest checks
and the three profile workloads, then creates `stage6_report.zip`. Windows
manifest checking accepts a separately recorded CRLF-only difference for known
text suffixes; numerical binary evidence must match its raw checksum.
The small profile and smoke workloads do not establish scientific convergence.

Local validation passed 320 grouped regression tests and an updated 11-test
publication/profiling suite (321 distinct tests overall). The six runner cases,
three response cases, three profiles and installed-wheel profiling/publication
checks passed. All 146 protected entries matched their raw checksums locally.

The existing 146 protected RefleX/material benchmark entries must remain intact.
Dated handoffs and frozen reports were not rewritten. See
[the current overview](CONFIG_REFACTOR_STATUS.md) for the maintained workflow and
remaining cluster/production work.
