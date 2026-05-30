#!/bin/bash
#SBATCH --job-name=face-occ-sapiens2-01b-v4-optuna
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=30:00:00

set -e

echo "================================================================================"
echo "OPTUNA HPO - Sapiens2-0.1B (114M) v4 - extended search space (2x RTX 3090, BF16)"
echo "  ▶ pretrained_source ∈ {sapiens_default, ibot:runs:/...}"
echo "  ▶ pooling_type ∈ {cls, gem, attention_k_query, multihead_attention}"
echo "  ▶ balancing_strategy ∈ {A, D, E, F, G (DANN), H (MMD), I (Mixup)}"
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
export PATH="$HOME/.local/bin:$PATH"
uv sync --no-dev
source "${VENV_DIR}/bin/activate"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=8

export FACE_OCC_ARCH=sapiens2-01b-3090-v4

mkdir -p scripts/logs

MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
${TORCHRUN} --nproc_per_node=2 --master_port=$MASTER_PORT src/optimize.py

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "================================================================================"
