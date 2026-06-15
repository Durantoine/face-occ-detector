#!/bin/bash
#SBATCH --job-name=face-occ-sapiens2-08b-h100-optuna
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=ENSTA-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=168:00:00

set -e

echo "================================================================================"
echo "OPTUNA HPO - Sapiens 0.8B (~800M) - 3.9/v39 recipe + 3.10 - single H100"
echo "  Stratified eval (alpha=0.3), group_batch sampler, val-tilted asymmetry"
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

# Default bucket URL
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

# Dataset + augmentation pool. Both persist in $HOME across jobs -> only sync when the local dir is
# empty (a re-sync of 130k small files is the SETUP HANG that left the job alive with no logs and
# training never starting). gsutil's -m (parallel) pool deadlocks on these GPU nodes -> aug drops -m.
# To force a clean re-sync: `rm -rf data/aug` (or data/raw) first.
echo "  Syncing dataset..."
mkdir -p data/raw data/aug
if [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
    gsutil -m rsync -r "${BUCKET_URL}/data/raw" data/raw/
else
    echo "    data/raw already present -> skip sync ($(ls data/raw | wc -l) entries)"
fi
if [ -z "$(ls -A data/aug 2>/dev/null)" ]; then
    gsutil rsync -r "${BUCKET_URL}/data/aug" data/aug/   # no -m: parallel pool deadlocks on GPU nodes
else
    echo "    data/aug already present -> skip sync ($(ls data/aug | wc -l) entries)"
fi

# MLflow/Optuna DBs live on NODE-LOCAL disk, NOT on the NFS $HOME: sqlite needs POSIX locks that NFS
# implements incorrectly -> "database disk image is malformed" on a FRESH db, even with one writer.
# We do NOT pull the bucket mlflow.db (it was corrupted and re-propagated the corruption every run).

echo "================================================================================"
echo "Setting up environment..."
echo "================================================================================"

# Setup Python environment
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

# Configuration. SPLIT the DBs by corruption risk + monitorability:
#  - optuna.db stays on $HOME (NFS): Optuna writes only ~2x/trial -> low corruption risk, AND the
#    login-node Optuna dashboard can read it LIVE (the sweep-monitoring tool we actually care about).
#  - mlflow.db goes to NODE-LOCAL /tmp: per-step metric INSERTs (~50/epoch) over NFS are what produced
#    "database disk image is malformed". Off NFS -> no corruption. (MLflow UI won't see it live -> use
#    the stdout [Optimizer]/[Challenge]/[VALIDATION] logs instead.)
CONFIG="sapiens2-0.8b-H100"
STORAGE="sqlite:///optuna.db"
TRACKING="sqlite:////tmp/mlflow_sapiens08b_${SLURM_JOB_ID}.db"

# Run optimization
export FACE_OCC_ARCH="${CONFIG}"
python src/optimize.py "${CONFIG}" \
    --storage "${STORAGE}" --tracking-uri "${TRACKING}"

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "================================================================================"
