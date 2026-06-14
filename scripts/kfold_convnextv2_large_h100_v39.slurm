#!/bin/bash
#SBATCH --job-name=face-occ-kfold-v39-h100
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=ENSTA-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# ALL-IN-ONE K-fold cross-validation of the V39 trial #6 submission config (leaderboard 0.00098).
# One sbatch -> trains the K folds SEQUENTIALLY on a single H100, then aggregates:
#   - pooled out-of-fold (OOF) CV score (interim alpha=0.3 AND final alpha=1.0)
#   - two ensemble submissions: UNWEIGHTED (equal weight) and WEIGHTED (weight ~ 1/fold_score)
#
# Each fold writes results/<name>_fold<k>/{best.pt, val_predictions.csv, test_predictions.csv}.
# Final submissions:
#   results/<name>_kfold_submission_unweighted.csv
#   results/<name>_kfold_submission_weighted.csv
#
# Submit:  sbatch scripts/kfold_convnextv2_large_l40s_v39.slurm
# Adjust:  N_FOLDS below; change --partition if L40S is named otherwise on the cluster.

set -e

N_FOLDS=7                           # ~14.3% val per fold, matching the original val_ratio=0.15
BASE_SEED=129                       # per-fold seed = BASE_SEED + fold (diverse init; partition fixed by kfold_seed)
CONFIG="configs/architectures/config_v39_trial6_resolved.yaml"

echo "================================================================================"
echo "ALL-IN-ONE K-FOLD V39 trial#6 - ConvNeXtV2-Large - single H100 - ${N_FOLDS} folds"
echo "Node: $(hostname) | Job ID: ${SLURM_JOB_ID} | GPUs: ${CUDA_VISIBLE_DEVICES}"
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

echo "Syncing from bucket ${BUCKET_URL}..."

# 1. Google Cloud SDK (for gsutil)
if ! command -v gcloud >/dev/null 2>&1; then
    if [ ! -d "${HOME}/google-cloud-sdk" ]; then
        echo "Installing Google Cloud SDK..."
        curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts > /dev/null
    fi
    export PATH="${HOME}/google-cloud-sdk/bin:${PATH}"
fi

# 2. uv
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

# Dataset + augmentation pool. Both persist in $HOME across jobs, so we only sync when the local
# dir is empty. NOTE: gsutil's -m (parallel) pool can deadlock on this cluster's GPU nodes -> the
# aug sync drops -m. If you ever need to FORCE a re-sync, just `rm -rf data/aug` (or data/raw) first.
mkdir -p data/raw data/aug
if [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
    gsutil -m rsync -r "${BUCKET_URL}/data/raw" data/raw/
else
    echo "  data/raw already present -> skip sync ($(ls data/raw | wc -l) entries)"
fi
if [ -z "$(ls -A data/aug 2>/dev/null)" ]; then
    gsutil rsync -r "${BUCKET_URL}/data/aug" data/aug/   # no -m: parallel pool deadlocks on GPU nodes
else
    echo "  data/aug already present -> skip sync ($(ls data/aug | wc -l) entries)"
fi

echo "Setting up environment..."
uv sync --no-dev
source "${VENV_DIR}/bin/activate"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16

ulimit -n 65536 || ulimit -n 8192

mkdir -p scripts/logs results

# Sequential training: one shared MLflow DB is safe (no concurrent writers); each fold logs under a
# distinct experiment (cfg.name suffixed with _fold<k>).
TRACKING="sqlite:///mlflow_kfold.db"

for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    SEED=$((BASE_SEED + FOLD))
    echo "================================================================================"
    echo "FOLD ${FOLD}/${N_FOLDS} (seed ${SEED}) - $(date)"
    echo "================================================================================"
    python src/train.py "${CONFIG}" \
        --fold "${FOLD}" --n-folds "${N_FOLDS}" --seed "${SEED}" \
        --tracking-uri "${TRACKING}"
done

echo "================================================================================"
echo "ALL ${N_FOLDS} FOLDS DONE - aggregating OOF CV score + ensemble submissions - $(date)"
echo "================================================================================"
python scripts/kfold_oof.py "${CONFIG}" --n-folds "${N_FOLDS}"

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "  results/convnextv2-large-H100-v39_trial6_kfold_submission_unweighted.csv"
echo "  results/convnextv2-large-H100-v39_trial6_kfold_submission_weighted.csv"
echo "================================================================================"
