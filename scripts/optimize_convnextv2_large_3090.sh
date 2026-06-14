#!/bin/bash
#SBATCH --job-name=face-occ-convnextv2-large-3090-v36-optuna
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=35:30:00   # qos-student MaxWall = 36h; study resumes across jobs via optuna.db

set -e

echo "================================================================================"
echo "OPTUNA HPO - ConvNeXt v2 Large (~198M) - v36 pipeline - single GPU (3090)"
echo "  Antoine's v36 framework (importance reweighting + searchable aug pool)"
echo "  batch 16 (24GB, no grad-checkpointing) | single-GPU = no DDP"
echo "================================================================================"
echo "Node: $(hostname) | Job ID: $SLURM_JOB_ID | GPUs: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "================================================================================"

PROJECT_DIR="${HOME}/face-occ-detector"
VENV_DIR="/tmp/face_occ_venv_${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"

export UV_PYTHON=python3.12
export UV_PYTHON_DOWNLOADS=automatic
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export PATH="$HOME/.local/bin:$PATH"
uv sync --no-dev
source "${VENV_DIR}/bin/activate"

"${VENV_DIR}/bin/python" -c "import timm" 2>/dev/null || "${VENV_DIR}/bin/pip" install timm

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4

ulimit -n 65536 || ulimit -n 8192

mkdir -p scripts/logs

# Optional overrides (for a short validation run alongside the full sweep):
#   CONFIG   : architecture config name (default = full run config)
#   TRIALS   : cap the number of Optuna trials
#   STORAGE  : Optuna sqlite path — give the mini-run its OWN db to avoid two
#              concurrent jobs hammering the same sqlite study (NFS-unsafe).
#   TRACKING : MLflow sqlite path (likewise separate for the mini-run).
CONFIG="${CONFIG:-convnextv2-large-3090}"
TRIALS_ARG=""
[ -n "${TRIALS:-}" ] && TRIALS_ARG="--trials ${TRIALS}"
STORAGE="${STORAGE:-sqlite:///optuna.db}"
TRACKING="${TRACKING:-sqlite:///mlflow.db}"

# v36 entrypoint: optimize.py takes the config via --config or FACE_OCC_ARCH.
export FACE_OCC_ARCH="${CONFIG}"

# Single-GPU: pas de torchrun (évite le bug DDP qui a tué la migration v30 de Cora).
"${VENV_DIR}/bin/python" src/optimize.py --config "${CONFIG}" \
    ${TRIALS_ARG} --storage "${STORAGE}" --tracking-uri "${TRACKING}"

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "================================================================================"
