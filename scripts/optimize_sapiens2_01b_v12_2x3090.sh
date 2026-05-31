#!/bin/bash
#SBATCH --job-name=face-occ-sapiens2-01b-v12-optuna
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=30:00:00

set -e

echo "================================================================================"
echo "OPTUNA HPO - Sapiens2-0.1B (114M) v12 - v11 refinements (2x 3090, BF16)"
echo "  ▶ Refinements vs v11:"
echo "    - K-query HPO ranges restored to v4 (more exploration)"
echo "    - DANN re-added to feature_fairness {none, ot, dann}"
echo "    - min_lr_rate 0.1 -> 0.3 (gentler cosine decay)"
echo "    - logging_steps 200 -> 25 (faster MLflow refresh)"
echo "================================================================================"
echo "Node: \$(hostname) | Job ID: \$SLURM_JOB_ID | GPUs: \$CUDA_VISIBLE_DEVICES"
echo "Started: \$(date)"
echo "================================================================================"

PROJECT_DIR="${HOME}/face-occ-detector"
VENV_DIR="/tmp/face_occ_venv_${SLURM_JOB_ID}"
TORCHRUN="${VENV_DIR}/bin/torchrun"

cd "${PROJECT_DIR}"

export UV_PYTHON=python3.12
export UV_PYTHON_DOWNLOADS=automatic
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
uv sync --no-dev
source "${VENV_DIR}/bin/activate"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Raise file descriptor limit — PyTorch DataLoader workers accumulate shared-memory
# handles across trials. Default ~1024-4096 exhausted after ~10 trials → "Too many
# open files" hang. Use 65536 (max often allowed without root).
ulimit -n 65536 || ulimit -n 8192   # fallback if 65536 forbidden
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=2
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

export FACE_OCC_ARCH=sapiens2-01b-3090-v12

mkdir -p scripts/logs

MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
${TORCHRUN} --nproc_per_node=2 --master_port=$MASTER_PORT src/optimize.py

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "================================================================================"
