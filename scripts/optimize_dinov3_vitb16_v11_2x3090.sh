#!/bin/bash
#SBATCH --job-name=face-occ-vitb16-v11-optuna
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=30:00:00

set -e

echo "================================================================================"
echo "OPTUNA HPO - DINOv3 ViT-B/16 (86M) v11 - Back to v3 fundamentals (2x 3090, BF16)"
echo "  ▶ 15 bins x 0.0333 (re-extracted from PDF via pixel reading)"
echo "  ▶ sample_weight normalized to mean=1 (v3 trick)"
echo "  ▶ aug_share / replication REMOVED (v3/v4 winning recipe used none)"
echo "  ▶ LLRD restored (layer_decay HPO in [0.65, 0.95])"
echo "  ▶ clip lowered 20 -> 10, focal range [0, 1.5]"
echo "================================================================================"
echo "Node: $(hostname) | Job ID: $SLURM_JOB_ID | GPUs: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
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

# Raise file descriptor limit — DataLoader workers accumulate shared-memory handles
# across trials. Default ~1024 exhausted after ~10 trials → "Too many open files" hang.
ulimit -n 65536 || ulimit -n 8192
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=2
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

export FACE_OCC_ARCH=dinov3-vitb16-3090-v11

mkdir -p scripts/logs

MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
${TORCHRUN} --nproc_per_node=2 --master_port=$MASTER_PORT src/optimize.py

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "================================================================================"
