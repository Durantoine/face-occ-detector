#!/bin/bash
#SBATCH --job-name=ensemble-challenge
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=ENSTA-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -e

# Configuration
TOPK=1
CONFIGS="sapiens-hc-ens dino-vitb16-hc-ens"
STORAGE="sqlite:///optuna.db"

echo "================================================================================"
echo "ENSEMBLE CHALLENGE"
echo "  Top-K trials: ${TOPK}"
echo "  Configs: ${CONFIGS}"
echo "  Storage: ${STORAGE}"
echo "================================================================================"
echo "Node: $(hostname) | Job ID: $SLURM_JOB_ID | GPUs: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "================================================================================"

PROJECT_DIR="${HOME}/face-occ-detector"
VENV_DIR="/tmp/face_occ_venv_ensemble_${SLURM_JOB_ID}"

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

# Sync required data using robust logic
# 1. Dataset and Augmentations
echo "  Syncing data/raw and data/aug..."
if command -v gcloud >/dev/null 2>&1; then
    gcloud storage rsync -r "${BUCKET_URL}/data/raw" data/raw/
    gcloud storage rsync -r "${BUCKET_URL}/data/aug" data/aug/
else
    gsutil -m rsync -r "${BUCKET_URL}/data/raw" data/raw/
    gsutil -m rsync -r "${BUCKET_URL}/data/aug" data/aug/
fi

# 2. iBOT encoders (mlruns/1) - Robust sync with retries
echo "  Syncing mlruns (iBOT models)..."
mkdir -p mlruns/1
find mlruns/1 -name '*.gstmp' -delete 2>/dev/null || true
for _i in 1 2 3; do
    if command -v gcloud >/dev/null 2>&1; then
        gcloud storage rsync -r "${BUCKET_URL}/mlruns/1" mlruns/1/ && break
    else
        gsutil -m rsync -r "${BUCKET_URL}/mlruns/1" mlruns/1/ && break
    fi
    echo "  mlruns rsync attempt ${_i} failed; cleaning + retrying..."
    find mlruns/1 -name '*.gstmp' -delete 2>/dev/null || true
    sleep 5
done

# 3. DBs
echo "  Syncing DBs..."
if command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp "${BUCKET_URL}/mlflow.db" mlflow.db || echo "  (mlflow.db not found)"
    gcloud storage cp "${BUCKET_URL}/optuna.db" optuna.db || echo "  (optuna.db not found)"
else
    gsutil cp "${BUCKET_URL}/mlflow.db" mlflow.db || echo "  (mlflow.db not found)"
    gsutil cp "${BUCKET_URL}/optuna.db" optuna.db || echo "  (optuna.db not found)"
fi
echo "Setting up environment..."
uv sync --no-dev
source "${VENV_DIR}/bin/activate"

# Fix paths in DB
if [ -f "mlflow.db" ]; then
    echo "Fixing MLflow paths..."
    python3 scripts/fix_mlflow_paths.py mlflow.db --apply
fi

# v40: Copy optuna.db to local /tmp to avoid SQLite commit/lock errors on cluster filesystems
LOCAL_STORAGE="/tmp/optuna_ensemble_${SLURM_JOB_ID}.db"
cp optuna.db "${LOCAL_STORAGE}"
echo "Using local storage: ${LOCAL_STORAGE}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16

ulimit -n 65536 || ulimit -n 8192

mkdir -p scripts/logs challenge_results

echo "Launching ensemble script..."
python scripts/ensemble_challenge.py "${TOPK}" ${CONFIGS} --storage "sqlite:///${LOCAL_STORAGE}"

# Cleanup local DB
rm -f "${LOCAL_STORAGE}"

echo "Syncing results back to bucket ${BUCKET_URL}..."
if command -v gcloud >/dev/null 2>&1; then
    gcloud storage rsync -r challenge_results/ "${BUCKET_URL}/challenge_results/"
    gcloud storage rsync -r results/ "${BUCKET_URL}/results/"
else
    gsutil -m rsync -r challenge_results "${BUCKET_URL}/challenge_results"
    gsutil -m rsync -r results "${BUCKET_URL}/results"
fi

echo "================================================================================"
echo "ENSEMBLE COMPLETE - Finished: $(date)"
echo "================================================================================"
