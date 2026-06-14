#!/bin/bash
#SBATCH --job-name=face-occ-sapiens-final-v2-h100-optuna
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=ENSTA-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=24:00:00

set -e

echo "================================================================================"
echo "OPTUNA HPO - Sapiens 0.1B (~100M) - FINAL - single H100"
echo "  Stratified eval (alpha=0.6), group_batch sampler, val-tilted asymmetry,"
echo "  MERGED aug pool (synthetic_all + blend_v3_refined high-occ blends)"
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
export PATH="$HOME/.local/bin:$PATH"

# Default bucket URL (hardcoded for convenience)
DEFAULT_BUCKET="gs://mon-face-occ-bucket"
BUCKET_URL="${FACE_OCC_BUCKET_URL:-$DEFAULT_BUCKET}"
BUCKET_URL="${BUCKET_URL%/}"

echo "================================================================================"
echo "Syncing from bucket ${BUCKET_URL}..."
echo "================================================================================"

# 1. Install Google Cloud SDK (for gsutil)
if ! command -v gcloud >/dev/null 2>&1; then
    echo "Google Cloud SDK not found. Checking ${HOME}/google-cloud-sdk..."
    if [ ! -d "${HOME}/google-cloud-sdk" ]; then
        echo "Installing Google Cloud SDK..."
        curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts > /dev/null
    fi
    export PATH="${HOME}/google-cloud-sdk/bin:${PATH}"
fi

# 2. Install uv
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

# Dataset
echo "  Syncing dataset..."
mkdir -p data/raw
gsutil -m rsync -r "${BUCKET_URL}/data/raw" data/raw/

# Augmentation pool (synthetic_all + blend_v3_refined + all images, consolidated under data/aug)
echo "  Syncing augmentation pool..."
mkdir -p data/aug
gsutil -m rsync -r "${BUCKET_URL}/data/aug" data/aug/

# MLflow runs/models — the iBOT pretrained ENCODERS referenced by pretrained_source live here.
# Without this, any trial picking an "ibot:runs:/..." pretrained_source crashes (artifact not found).
echo "  Syncing mlruns (iBOT encoder models)..."
mkdir -p mlruns/1
gsutil -m rsync -r "${BUCKET_URL}/mlruns/1" mlruns/1/

# MLflow Database (optional sync)
if [ ! -f mlflow.db ]; then
    echo "  Syncing mlflow.db..."
    gsutil cp "${BUCKET_URL}/mlflow.db" mlflow.db || echo "  (mlflow.db not found in bucket, starting fresh)"
fi

echo "================================================================================"
echo "Setting up environment..."
echo "================================================================================"

export UV_PYTHON=python3.12
export UV_PYTHON_DOWNLOADS=automatic
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"

uv sync --no-dev
source "${VENV_DIR}/bin/activate"

# Fix MLflow Database paths
if [ -f "mlflow.db" ]; then
    echo "Checking & fixing MLflow database paths"
    python3 scripts/fix_mlflow_paths.py mlflow.db --apply
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16

ulimit -n 65536 || ulimit -n 8192

mkdir -p scripts/logs results

# Configuration
CONFIG="sapiens-final-v4"
STORAGE="sqlite:///optuna.db"
TRACKING="sqlite:///mlflow.db"

# Run optimization (resumes across jobs via optuna.db)
export FACE_OCC_ARCH="${CONFIG}"
python src/optimize.py "${CONFIG}" \
    --storage "${STORAGE}" --tracking-uri "${TRACKING}"

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "================================================================================"
