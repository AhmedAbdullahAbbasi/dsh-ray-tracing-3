#!/bin/bash
#
# SLURM submission script for run_simulation.py (parameters live in sim_config.py).
#
# Submit with either:  bash submit_simulation.sh   or   sbatch submit_simulation.sh
# The script reads DEVICE from sim_config.py and makes sure the job has a GPU when
# DEVICE = "gpu": run with bash it submits itself with the GPU request; submitted
# directly with sbatch it resubmits itself with the GPU request and exits.
#

#SBATCH --job-name=dsh_sim
#SBATCH --account=mdbf
#SBATCH --qos=mid_mdbf
#SBATCH --partition=mid_mdbf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/dsh_sim_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=atakans@sabanciuniv.edu

set -euo pipefail

PYTHON=/cta/users/atakans/anaconda3/envs/pyt_env/bin/python

# Added to the request when DEVICE = "gpu". cn05/cn06 carry Tesla K80s, which are
# too old for JAX's CUDA 12 build.
GPU_SBATCH_ARGS=(--gres=gpu:1 --exclude=cn05,cn06)

if [ -n "${SLURM_JOB_ID:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$(readlink -f "$0")")"
fi
mkdir -p logs

DEVICE=$("$PYTHON" -c "import sim_config; print(sim_config.DEVICE)")
case "$DEVICE" in
    cpu) DEVICE_SBATCH_ARGS=() ;;
    gpu) DEVICE_SBATCH_ARGS=("${GPU_SBATCH_ARGS[@]}") ;;
    *) echo "sim_config.DEVICE must be 'cpu' or 'gpu', got '$DEVICE'" >&2; exit 1 ;;
esac

if [ -z "${SLURM_JOB_ID:-}" ]; then
    # Run with bash on the login node: submit with options matching DEVICE.
    echo "Submitting with DEVICE=$DEVICE ${DEVICE_SBATCH_ARGS[*]:-}"
    exec sbatch "${DEVICE_SBATCH_ARGS[@]}" "$@" submit_simulation.sh
fi

if [ "$DEVICE" = gpu ] && [ -z "${SLURM_GPUS_ON_NODE:-}" ]; then
    # Submitted with plain sbatch, so this job has no GPU: resubmit with one.
    if [ -n "${DSH_RESUBMITTED:-}" ]; then
        echo "Resubmitted job $SLURM_JOB_ID still has no GPU; giving up." >&2
        exit 1
    fi
    echo "DEVICE=gpu but job $SLURM_JOB_ID has no GPU; resubmitting with ${GPU_SBATCH_ARGS[*]}"
    sbatch --export=ALL,DSH_RESUBMITTED=1 "${GPU_SBATCH_ARGS[@]}" submit_simulation.sh
    exit 0
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $(hostname), started $(date), DEVICE=$DEVICE"
START=$(date +%s)

"$PYTHON" run_simulation.py

echo "Finished $(date), elapsed $(( $(date +%s) - START )) s"
