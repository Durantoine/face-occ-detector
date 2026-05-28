#!/bin/bash
#SBATCH --job-name=face-occ-ibot-vith16plus
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=30:00:00

set -e

echo "================================================================================"
echo "iBOT pretraining - DINOv3 ViT-H+/16 @ 224x224 - pretrain_224 (2x RTX 3090 DDP, BF16)"
echo "  ▶ 224x224: BS=2, grad_accum=16 (eff_batch=64), grad_ckpt, teacher_frozen"
echo "  ▶ ~4x slower per step than 112x112 → chain plusieurs jobs"
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

export FACE_OCC_PRETRAIN_ARCH=dinov3_vith16plus
export FACE_OCC_PRETRAIN_SRC="data/pretrain_224"
# 100k steps @224 ≈ 8 chain links de 30h (≈10s/step avec GA=16, BS=2). Stop-early possible :
# scancel quand le sweep Optuna révèle qu'un snapshot intermédiaire suffit.
# eff_batch=64 × 100k = 6.4M faces vues (~1.2 epoch MS1MV3)
# Caveat cosine LR: schedule étalé sur 100k → si stop à 50k, LR final ~0.71 init (vs ~0 si MAX=50k).
# Impact minime sur iBOT-light/frozen-teacher (teacher = anchor).
export FACE_OCC_PRETRAIN_MAX_STEPS=88000
export FACE_OCC_PRETRAIN_OUT="./results/pretrain_vith16plus"
export FACE_OCC_PRETRAIN_TEACHER_FROZEN=1
export FACE_OCC_PRETRAIN_SNAPSHOT_STEPS="25000,50000,75000"
export FACE_OCC_PRETRAIN_IMG_SIZE=224

# === Memory-conservative settings for ViT-H+/16 @ 224x224 on 2x 24GB ===
# Default BS=32 OOM à 224 → divisé par 16. Eff_batch=64 (vs 128 défaut à 112)
export FACE_OCC_PRETRAIN_BS=2
export FACE_OCC_PRETRAIN_GA=16              # eff_batch = 2 * 16 * 2 = 64

mkdir -p scripts/logs results/pretrain_vith16plus

MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
${TORCHRUN} --nproc_per_node=2 --master_port=$MASTER_PORT src/pretrain_ibot.py

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "================================================================================"
