#!/bin/bash
#SBATCH --job-name=train-one
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=ENSTA-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=24:00:00

set -e

# Configuration from argument
CONFIG="${1:-}"
if [ -z "$CONFIG" ]; then
    echo "ERROR: No config name provided."
    echo "Usage: sbatch scripts/train_one.slurm <config_name>"
    exit 1
fi

echo "================================================================================"
echo "SINGLE TRAINING: ${CONFIG}"
echo "================================================================================"
echo "Node: $(hostname) | Job ID: $SLURM_JOB_ID | GPUs: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "================================================================================"

PROJECT_DIR="${HOME}/face-occ-detector"
VENV_DIR="/tmp/face_occ_venv_train_${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"

export UV_PYTHON=python3.12
export UV_PYTHON_DOWNLOADS=automatic
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export PATH="$HOME/.local/bin:$PATH"

# Default bucket configuration
DEFAULT_BUCKET="gs://mon-face-occ-bucket"
BUCKET_URL="${FACE_OCC_BUCKET_URL:-$DEFAULT_BUCKET}"
BUCKET_URL="${BUCKET_URL%/}"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

# Sync required artifacts (encoders and DB)
# Note: Skipping raw data as it's assumed to be present localy, but syncing encoders is safer.
echo "Syncing required artifacts from bucket..."
GS_CMD="gcloud storage"
if ! command -v gcloud >/dev/null 2>&1; then
    GS_CMD="gsutil"
fi

# Sync iBOT encoders (needed for Dino/Sapiens init)
mkdir -p mlruns/1
if [ "${GS_CMD}" = "gcloud storage" ]; then
    gcloud storage rsync -r "${BUCKET_URL}/mlruns/1" mlruns/1/ || echo "  ! mlruns/1 sync failed"
else
    gsutil -m rsync -r "${BUCKET_URL}/mlruns/1" mlruns/1/ || echo "  ! mlruns/1 sync failed"
fi

# Sync DB for tracking
if [ ! -f mlflow.db ]; then
    ${GS_CMD} cp "${BUCKET_URL}/mlflow.db" mlflow.db || echo "  ! mlflow.db not found"
fi

echo "Setting up environment..."
uv sync --no-dev
source "${VENV_DIR}/bin/activate"

# Fix paths in DB
if [ -f "mlflow.db" ]; then
    python3 scripts/fix_mlflow_paths.py mlflow.db --apply
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16

ulimit -n 65536 || ulimit -n 8192

mkdir -p scripts/logs results

echo "Launching training..."
python src/train.py "${CONFIG}"

echo "================================================================================"
echo "TRAINING COMPLETE - Finished: $(date)"
echo "================================================================================"
