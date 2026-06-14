#!/bin/bash
#SBATCH --job-name=face-occ-dino-vitb16-h100-optuna
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=ENSTA-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=24:00:00

set -e

echo "================================================================================"
echo "OPTUNA HPO - DINOv3 ViT-B/16 (~86M) - v40 - single H100"
echo "  train_alpha=0 (natural training) + eval_alpha=0.5 selection (decoupled IS),"
echo "  group_batch sampler, val-tilted asymmetry, merged aug pool, per-alpha logging."
echo "  batch_size=256 (dinov3_vitb16 + compile OOMs at 512 on 80GB)."
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

DEFAULT_BUCKET="gs://mon-face-occ-bucket"
BUCKET_URL="${FACE_OCC_BUCKET_URL:-$DEFAULT_BUCKET}"
BUCKET_URL="${BUCKET_URL%/}"

echo "Syncing from bucket ${BUCKET_URL}..."

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

echo "  Syncing dataset..."
mkdir -p data/raw
gsutil -m rsync -r "${BUCKET_URL}/data/raw" data/raw/

echo "  Syncing augmentation pool..."
mkdir -p data/aug
gsutil -m rsync -r "${BUCKET_URL}/data/aug" data/aug/

# iBOT pretrained ENCODERS referenced by pretrained_source live here (required for ibot:runs:/...).
echo "  Syncing mlruns (iBOT encoder models)..."
mkdir -p mlruns/1
# Robust sync: clean stale gsutil temp files, retry the parallel rsync (transient .gstmp races over
# large model.pth abort it), then serial-cp any iBOT encoder referenced by the configs that is still
# missing/empty (a dropped encoder -> "No artifacts found" at train time -> failed trials).
find mlruns/1 -name '*.gstmp' -delete 2>/dev/null || true
for _i in 1 2 3; do
    gsutil -m rsync -r "${BUCKET_URL}/mlruns/1" mlruns/1/ && break
    echo "  mlruns rsync attempt ${_i} failed; cleaning .gstmp + retrying..."
    find mlruns/1 -name '*.gstmp' -delete 2>/dev/null || true
    sleep 5
done
for _m in m-c03a1876c9d441e19cffdd5c73ac5e6d m-8b6b0dc922664f30a1f01765be91bb09 \
          m-e8334758651b46bc923bbe60d74585a9 m-9483fa8a2bc8428c8b87b4b501cbe957; do
    _f="mlruns/1/models/${_m}/artifacts/data/model.pth"
    if [ ! -s "${_f}" ]; then
        echo "  serial re-fetch of encoder ${_m}..."
        mkdir -p "$(dirname "${_f}")"
        gsutil cp "${BUCKET_URL}/mlruns/1/models/${_m}/artifacts/data/model.pth" "${_f}" || echo "  ⚠ failed ${_m}"
    fi
done

if [ ! -f mlflow.db ]; then
    echo "  Syncing mlflow.db..."
    gsutil cp "${BUCKET_URL}/mlflow.db" mlflow.db || echo "  (mlflow.db not found in bucket, starting fresh)"
fi

echo "Setting up environment..."
uv sync --no-dev
source "${VENV_DIR}/bin/activate"

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

CONFIG="dino-vitb16-hc"
STORAGE="sqlite:///optuna.db"
TRACKING="sqlite:///mlflow.db"

export FACE_OCC_ARCH="${CONFIG}"
python src/optimize.py "${CONFIG}" \
    --storage "${STORAGE}" --tracking-uri "${TRACKING}"

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "================================================================================"
