#!/bin/bash
#SBATCH --job-name=face-occ-vitl16-3090-optuna
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=A40
#SBATCH --gres=gpu:rtx3090:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=30:00:00

set -e

echo "================================================================================"
echo "OPTUNA HPO - DINOv3 ViT-H+/16 - Face Occlusion Regression (2x RTX3090 DDP)"
echo "================================================================================"
echo "Node: $(hostname) | Job ID: $SLURM_JOB_ID | GPUs: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "================================================================================"

PROJECT_DIR="${HOME}/face-occ-detector"
VENV_DIR="/tmp/face_occ_venv_${SLURM_JOB_ID}"
TORCHRUN="${VENV_DIR}/bin/torchrun"

cd "${PROJECT_DIR}"

export UV_PYTHON=python3.11
export UV_PYTHON_DOWNLOADS=never
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
cp pyproject.cluster.toml pyproject.toml && uv sync --no-dev
source "${VENV_DIR}/bin/activate"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=8

mkdir -p scripts/logs

export FACE_OCC_ARCH=dinov3-vitl16-3090
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
${TORCHRUN} --nproc_per_node=2 --master_port=$MASTER_PORT src/optimize.py

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "================================================================================"
