#!/bin/bash
#SBATCH --job-name=face-occ-ibot-sapiens2-01b
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=30:00:00

set -e

echo "================================================================================"
echo "iBOT pretraining - Sapiens2-0.1B (114M) - MS1MV3 (2x RTX 3090 DDP, BF16)"
echo "  ▶ pixel-space masking (Sapiens2 has no forward_features(masks=...))"
echo "  ▶ conservative hyperparams: ne pas abîmer le backbone Meta Sapiens2"
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

export FACE_OCC_PRETRAIN_ARCH=sapiens2_0.1b
export FACE_OCC_PRETRAIN_SRC="data/pretrain/datasets--gaunernst--ms1mv3-wds/snapshots/cbe71fd17b8d1ed61e40508eba78aec6d4c8df46"
export FACE_OCC_PRETRAIN_MAX_STEPS=150000
export FACE_OCC_PRETRAIN_OUT="./results/pretrain_sapiens2_01b"

export FACE_OCC_PRETRAIN_LR=1.0e-05
export FACE_OCC_PRETRAIN_MASK_RATIO=0.4
export FACE_OCC_PRETRAIN_EMA_DECAY=0.9995
export FACE_OCC_PRETRAIN_TEACHER_FROZEN=1
export FACE_OCC_PRETRAIN_SNAPSHOT_STEPS="50000,100000"
export FACE_OCC_PRETRAIN_EPOCHS=5

mkdir -p scripts/logs results/pretrain_sapiens2_01b

MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
${TORCHRUN} --nproc_per_node=2 --master_port=$MASTER_PORT src/pretrain_ibot.py

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "  ▶ Get the MLflow encoder run_id from results/pretrain_sapiens2_01b/mlflow_run_id.txt"
echo "  ▶ Fill it into configs/architectures/sapiens2-01b-3090-v4.yaml"
echo "    (search_space.pretrained_source.choices, replace __FILL_SAPIENS_01B_IBOT_RUN_ID__)"
echo "================================================================================"
