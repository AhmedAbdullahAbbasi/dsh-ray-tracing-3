# Slurm integration (Stage 4)

The same TOML file controls local and cluster simulations. An optional `[slurm]`
table describes a single-node, single-process allocation. The shared worker
runs `dsh run`; no transport, source weighting or observation algorithms change.
The historical `submit_simulation.sh` remains available with its old settings.

## Cluster setup and use

Install the package with FITS/plot dependencies in your cluster Python environment.
For GPU runs, install a JAX build compatible with that cluster's driver and GPUs.
All configuration, material and cloud paths must be accessible on the allocated
node. Run planning on the cluster login node so resolved paths refer to its
filesystem. Windows planning is useful for validation, but Windows paths do not
become cluster paths automatically.

Start from `configs/slurm/cpu.toml` or `configs/slurm/gpu.toml`, or add this
to your simulation TOML, choosing your own account and partition:

```toml
[slurm]
partition = "compute"
account = "your_account"
# qos = "your_qos"
time = "01:00:00"
memory = "16G"
cpus_per_task = 4
gpus = 0
python = "/path/to/cluster/environment/bin/python"
working_directory = ".."
log_directory = "../logs"
```

Directory values resolve relative to the TOML file. A Python executable name
such as `python3` uses the job's PATH; an absolute executable path is recommended.
Shell activation commands and module loading are not accepted as executable
values. Prepare that environment before submission. Defaults: one hour, 16G,
four CPUs, zero GPUs, `python3`, config directory as working directory, and
`slurm-logs` beside the simulation's output directory. Omitted account,
partition and QoS use scheduler defaults. No mail recipient is configured.

```bash
dsh slurm-plan configs/my_simulation.toml
dsh slurm-submit configs/my_simulation.toml
```

`slurm-plan` prints JSON containing the exact sbatch argument vector and shell
script. It creates no files, invokes no scheduler and imports no numerical
libraries, so GPU configurations can be planned on a CPU-only login node.
It validates schema and input paths; it does not load or scientifically validate
material/FITS contents. That numerical preflight happens on the allocated node.

`slurm-submit` makes the log directory, passes the script to sbatch on stdin,
and prints the accepted job ID and configuration SHA256. Submission is Linux
only; submission from inside an existing Slurm job is rejected. There is no
automatic retry or self-resubmission. Monitor accepted jobs with your cluster's
normal squeue/sacct tools. A successful submission is not a completed simulation.

For a GPU run, set `execution.device = "gpu"` and `slurm.gpus = 1` (or greater).
The launcher requests `--gres=gpu:N`; adapt cluster policy through supported
partition/QoS settings. `exclude = "cn[05-06]"` can exclude unsuitable nodes.
CPU runs require zero GPUs. This launches one process; requesting multiple GPUs
does not add distributed transport or guarantee scaling.

The worker rejects configuration edits made while queued and selects the backend
before importing JAX. Use distinct config/output files for concurrent runs.
Input data and installed software are not snapshotted at submission: the shared
builder hashes and validates the versions present when execution starts.
Runtime provenance records the job ID, partition, CPU count and node list when
provided by Slurm. Log names use `%x_%j`; physical products retain their normal
completion manifest and quality gates. Worker exit codes propagate to Slurm,
including code 3 for numerically incomplete simulations.

## Validation and limits

`python -m unittest tests.test_slurm -v` checks resource selection, CPU/GPU
conflicts, quoting, pure planning, scheduler error handling, queued config edits,
and exit-code propagation without an installed scheduler.
`python scripts/validate_slurm_stage4.py` produces `stage4_report.zip` with
portable checks and a rendered job plan. Linux development also executes a
rendered CPU worker locally to check integration and runs `bash -n`.
Actual scheduler allocation, GPU execution and site-specific resource policy
require a real cluster smoke run; portable validation does not establish them.
