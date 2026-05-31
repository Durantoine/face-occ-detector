#!/bin/bash
#SBATCH --job-name=face-occ-efficientnet-b0-v15-optuna
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=20:00:00

set -e

echo "================================================================================"
echo "OPTUNA HPO - EfficientNet-B0 (CNN ~30M) v15 - Baseline CNN (2x 3090, BF16)"
echo "  Backbone: timm tf_efficientnet_b0 (ImageNet pretrained)"
echo "  Pools: attention_k_query + mil (no CLS, no MHA)"
echo "  All v15 features: tri-split + IS stratified + Lagrangian + 3 calibrators"
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

# Install timm if missing (CNN backbones)
"${VENV_DIR}/bin/python" -c "import timm" 2>/dev/null || "${VENV_DIR}/bin/pip" install timm

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=2
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

ulimit -n 65536 || ulimit -n 8192

export FACE_OCC_ARCH=efficientnet-b0-3090-v15

mkdir -p scripts/logs

MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
${TORCHRUN} --nproc_per_node=2 --master_port=$MASTER_PORT src/optimize.py

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "================================================================================"
