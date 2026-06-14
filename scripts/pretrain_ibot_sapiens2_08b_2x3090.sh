#!/bin/bash
#SBATCH --job-name=face-occ-ibot-sapiens2-08b
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=30:00:00

set -e

echo "================================================================================"
echo "iBOT pretraining - Sapiens2-0.8B (800M) @ 224x224 - pretrain_224 (2x RTX 3090 DDP, BF16)"
echo "  ▶ Largest Sapiens2 fittable on 2x 24GB without FSDP/DeepSpeed"
echo "  ▶ 224x224: BS=1, grad_accum=32 (eff_batch=64), grad_ckpt, teacher_frozen"
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
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=8

export FACE_OCC_PRETRAIN_ARCH=sapiens2_0.8b
export FACE_OCC_PRETRAIN_SRC="data/pretrain_224"
export FACE_OCC_PRETRAIN_OUT="./results/pretrain_sapiens2_08b"
export FACE_OCC_PRETRAIN_IMG_SIZE=224

# === Memory-conservative settings for 0.8B @ 224x224 on 2x 24GB ===
# 224 → 196 patches (vs 49 at 112) → BS divisé par 4 vs 112x112 setting
export FACE_OCC_PRETRAIN_BS=1               # vs 4 at 112x112
export FACE_OCC_PRETRAIN_GA=32              # eff_batch = 1 * 32 * 2 = 64

# === Pretrain hyperparams (conservative, anchored to Meta original) ===
# 100k steps @224 ≈ 8 chain links de 30h (≈10s/step avec GA=32). Stop-early possible :
# scancel quand le sweep Optuna révèle qu'un snapshot intermédiaire suffit.
# eff_batch=64 × 100k = 6.4M faces vues (~1.2 epoch MS1MV3)
# Caveat cosine LR: schedule étalé sur 100k → si stop à 50k, LR final ~0.71 init (vs ~0 si MAX=50k).
# Impact minime sur iBOT-light/frozen-teacher (teacher = anchor).
export FACE_OCC_PRETRAIN_MAX_STEPS=88000
export FACE_OCC_PRETRAIN_LR=1.0e-05
export FACE_OCC_PRETRAIN_MASK_RATIO=0.4
export FACE_OCC_PRETRAIN_EMA_DECAY=0.9995
export FACE_OCC_PRETRAIN_TEACHER_FROZEN=1
export FACE_OCC_PRETRAIN_SNAPSHOT_STEPS="25000,50000,75000"
export FACE_OCC_PRETRAIN_EPOCHS=5

mkdir -p scripts/logs results/pretrain_sapiens2_08b

MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
${TORCHRUN} --nproc_per_node=2 --master_port=$MASTER_PORT src/pretrain_ibot.py

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "  ▶ run_id: cat results/pretrain_sapiens2_08b/mlflow_run_id.txt"
echo "  ▶ Fill into your sapiens2-08b arch yaml (pretrained_source) — no 08b v6 yaml yet on this branch"
echo "================================================================================"
