#!/bin/bash
#SBATCH --job-name=face-occ-vits16-ensemble
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=30:00:00

set -e

echo "================================================================================"
echo "K-FOLD ENSEMBLE - DINOv3 ViT-S/16 - Face Occlusion (2 GPU DDP)"
echo "================================================================================"
echo "Node: $(hostname) | Job ID: $SLURM_JOB_ID"
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
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4

mkdir -p scripts/logs

MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
${TORCHRUN} --nproc_per_node=2 --master_port=$MASTER_PORT src/ensemble_train.py

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "================================================================================"
