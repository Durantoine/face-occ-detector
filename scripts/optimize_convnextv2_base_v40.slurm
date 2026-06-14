#!/bin/bash
#SBATCH --job-name=face-occ-convnextv2-base-v40-optuna
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=ENSTA-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -e

echo "================================================================================"
echo "OPTUNA HPO - ConvNeXt v2 Base (~88M) - v40 pipeline - single H100 (80GB)"
echo "  IS DECOUPLED: train_alpha=0.0 (natural training), eval_alpha=0.5 (selection only)"
echo "  isolated dir face-occ-detector-v40 + own optuna.db (study convnextv2-base-v40)"
echo "================================================================================"
echo "Node: $(hostname) | Job ID: $SLURM_JOB_ID | GPUs: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "================================================================================"

# Isolated project dir so this v40 run never touches the running v39 jobs (~/face-occ-detector)
PROJECT_DIR="${HOME}/face-occ-detector-v40"
VENV_DIR="/tmp/face_occ_venv_${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"

export UV_PYTHON=python3.12
export UV_PYTHON_DOWNLOADS=automatic
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export PATH="$HOME/.local/bin:$HOME/google-cloud-sdk/bin:$PATH"

# Default bucket URL (hardcoded for convenience)
DEFAULT_BUCKET="gs://mon-face-occ-bucket"
BUCKET_URL="${FACE_OCC_BUCKET_URL:-$DEFAULT_BUCKET}"
BUCKET_URL="${BUCKET_URL%/}"

echo "================================================================================"
echo "Syncing from bucket ${BUCKET_URL}..."
echo "================================================================================"

if ! command -v gcloud >/dev/null 2>&1; then
    if [ ! -d "${HOME}/google-cloud-sdk" ]; then
        curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts > /dev/null
    fi
    export PATH="${HOME}/google-cloud-sdk/bin:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

# data/ is a symlink to the v39 run's already-downloaded dataset (saves re-downloading 1.6G);
# rsync below is a quick idempotent no-op that also pulls any extra aug CSVs (e.g. blend_v3_refined).
echo "  Syncing dataset..."
mkdir -p data/raw
gsutil -m rsync -r "${BUCKET_URL}/data/raw" data/raw/
echo "  Syncing augmentation pool..."
mkdir -p data/aug
gsutil -m rsync -r "${BUCKET_URL}/data/aug" data/aug/

echo "================================================================================"
echo "Setting up environment..."
echo "================================================================================"

export UV_PYTHON=python3.12
export UV_PYTHON_DOWNLOADS=automatic
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"

uv sync --no-dev
source "${VENV_DIR}/bin/activate"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16

ulimit -n 65536 || ulimit -n 8192

mkdir -p scripts/logs results

# Configuration
CONFIG="convnextv2-base-v40"
STORAGE="sqlite:///optuna.db"
TRACKING="sqlite:///mlflow.db"

export FACE_OCC_ARCH="${CONFIG}"
python src/optimize.py "${CONFIG}" \
    --storage "${STORAGE}" --tracking-uri "${TRACKING}"

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "================================================================================"
