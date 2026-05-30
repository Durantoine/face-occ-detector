#!/bin/bash
#SBATCH --job-name=face-occ-sapiens2-01b-v10-optuna
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=30:00:00

set -e

echo "================================================================================"
echo "OPTUNA HPO - Sapiens2-0.1B (114M) v10 - Back to basics (2x 3090, BF16)"
echo "  ▶ Unified target : P_target(g,y) = mix_y(α1) × mix_g(α2)"
echo "  ▶ Axe 1 : axis1_power ∈ [0, 1] — tension vers P_test sur Y marginal"
echo "  ▶ Axe 2 : axis2_power ∈ [0, 1] — tension vers 50/50 F/M intra-Y"
echo "  ▶ aug_share ∈ [0, 0.5] — split loss vs aug replication (K_max=3)"
echo "  ▶ Axe 3 : feature_fairness {none, mmd, dann, both} + mmd_lambda (DANN adv=0.01 fixé)"
echo "  ▶ Focal γ ∈ [0, 2.5]  |  fairness_λ pinned à 1.0"
echo "  ▶ pretrained_source ∈ {sapiens_default, ibot:runs:/b70b78e8.../encoder_*}"
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
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=8
# v10 : NCCL stability fixes after observed AllReduce timeouts
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Match ddp_timeout in HF TrainingArguments (3600s = 1h)
export NCCL_TIMEOUT=3600

export FACE_OCC_ARCH=sapiens2-01b-3090-v10

mkdir -p scripts/logs

MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
${TORCHRUN} --nproc_per_node=2 --master_port=$MASTER_PORT src/optimize.py

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "================================================================================"
