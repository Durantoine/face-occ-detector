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
echo "iBOT pretraining - Sapiens2-0.8B (800M) - MS1MV3 (2x RTX 3090 DDP, BF16)"
echo "  ▶ Largest Sapiens2 fittable on 2x 24GB without FSDP/DeepSpeed"
echo "  ▶ Conservative: BS=4, grad_accum=8 (eff_batch=64), grad_ckpt, teacher_frozen"
echo "  ▶ 200k steps ≈ 30-40h → recommend chaining: ./scripts/chain_pretrain.sh 2 \$0"
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
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=8

export FACE_OCC_PRETRAIN_ARCH=sapiens2_0.8b
export FACE_OCC_PRETRAIN_SRC="data/pretrain/datasets--gaunernst--ms1mv3-wds/snapshots/cbe71fd17b8d1ed61e40508eba78aec6d4c8df46"
export FACE_OCC_PRETRAIN_OUT="./results/pretrain_sapiens2_08b"

# === Memory-conservative settings for 0.8B on 2x 24GB ===
export FACE_OCC_PRETRAIN_BS=4               # vs 32 for 0.1B
export FACE_OCC_PRETRAIN_GA=8               # eff_batch = 4 * 8 * 2 = 64

# === Pretrain hyperparams (conservative, anchored to Meta original) ===
export FACE_OCC_PRETRAIN_MAX_STEPS=200000   # ~5 epochs MS1MV3 ; ~30-40h → chain 2 jobs
export FACE_OCC_PRETRAIN_LR=1.0e-05
export FACE_OCC_PRETRAIN_MASK_RATIO=0.4
export FACE_OCC_PRETRAIN_EMA_DECAY=0.9995
export FACE_OCC_PRETRAIN_TEACHER_FROZEN=1   # teacher gelé sur Meta Sapiens2-0.8B
export FACE_OCC_PRETRAIN_SNAPSHOT_STEPS="50000,100000,150000"
export FACE_OCC_PRETRAIN_EPOCHS=5

mkdir -p scripts/logs results/pretrain_sapiens2_08b

MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
${TORCHRUN} --nproc_per_node=2 --master_port=$MASTER_PORT src/pretrain_ibot.py

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "  ▶ run_id: cat results/pretrain_sapiens2_08b/mlflow_run_id.txt"
echo "  ▶ Fill into configs/architectures/sapiens2-08b-3090-v3.yaml (pretrained_source)"
echo "================================================================================"
