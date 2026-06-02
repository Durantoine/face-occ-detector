#!/bin/bash
#SBATCH --job-name=face-occ-dinov3-vitb16-3090-v19-optuna
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=30:00:00

set -e

echo "================================================================================"
echo "OPTUNA HPO - DINOv3 ViT-B/16 (86M) v19 - single GPU (3090)"
echo "  Pas de torchrun = pas de DDP, pas d'overhead inter-rank"
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
uv sync --no-dev
source "${VENV_DIR}/bin/activate"

"${VENV_DIR}/bin/python" -c "import timm" 2>/dev/null || "${VENV_DIR}/bin/pip" install timm

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4

ulimit -n 65536 || ulimit -n 8192

export FACE_OCC_ARCH=dinov3-vitb16-3090-v19

mkdir -p scripts/logs

# Single-GPU: pas de torchrun
"${VENV_DIR}/bin/python" src/optimize.py

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "================================================================================"
