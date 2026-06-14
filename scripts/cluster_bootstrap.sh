#!/bin/bash
# Cluster bootstrap script for face-occ-detector
# 
# Usage:
#   export FACE_OCC_BUCKET_URL="gs://your-bucket/path"
#   bash scripts/cluster_bootstrap.sh <arch_name> [--optimize]
#
# Arguments:
#   arch_name: Name of the architecture config (e.g., sapiens2-0.1b-A100)
#   --optimize: If present, runs src/optimize.py instead of src/train.py

set -euo pipefail

ARCH_NAME="sapiens2-0.1b-H100"
MODE="train"

# Handle arguments: supports both `script <arch> [--optimize]` and `script --optimize`
if [ $# -gt 0 ]; then
    if [[ "$1" == --* ]]; then
        # First arg is a flag, keep default ARCH_NAME
        for arg in "$@"; do
            if [ "$arg" == "--optimize" ]; then MODE="optimize"; fi
        done
    else
        # First arg is architecture name
        ARCH_NAME="$1"
        if [ $# -gt 1 ]; then
            for arg in "${@:2}"; do
                if [ "$arg" == "--optimize" ]; then MODE="optimize"; fi
            done
        fi
    fi
fi

# Default bucket URL (hardcoded for convenience)
DEFAULT_BUCKET="gs://mon-face-occ-bucket"
BUCKET_URL="${FACE_OCC_BUCKET_URL:-$DEFAULT_BUCKET}"
# Strip trailing slash if present
BUCKET_URL="${BUCKET_URL%/}"
PROJECT_DIR="${HOME}/face-occ-detector"
VENV_DIR="${PROJECT_DIR}/.venv"

# ---------------------------------------------------------------------------
say() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }
warn() { printf '\n\033[1;33m⚠ %s\033[0m\n' "$1"; }
die() { printf '\n\033[1;31mFATAL: %s\033[0m\n' "$1" >&2; exit 1; }
# ---------------------------------------------------------------------------

say "Starting bootstrap for ${ARCH_NAME} in mode ${MODE}"

# 0. GCP SDK Check & Authentication
if ! command -v gcloud >/dev/null 2>&1; then
    say "Google Cloud SDK not found. Installing..."
    if [ ! -d "${HOME}/google-cloud-sdk" ]; then
        curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts > /dev/null
    fi
    export PATH="${HOME}/google-cloud-sdk/bin:${PATH}"
    # Also add to bashrc for future sessions
    if ! grep -q "google-cloud-sdk" ~/.bashrc; then
        echo 'export PATH="${HOME}/google-cloud-sdk/bin:${PATH}"' >> ~/.bashrc
    fi
fi

if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    say "Authentication required for GCP"
    gcloud auth login
fi

# 1. Bucket Sync
if [ -n "${BUCKET_URL}" ]; then
    say "Syncing from bucket ${BUCKET_URL}"
    
    # Dataset
    echo "  Syncing dataset..."
    mkdir -p data/raw
    gsutil -m rsync -r "${BUCKET_URL}/data/raw" data/raw/

    # Augmentation pool (occluder images + synthetic_all.csv) — needed when aug_proportion>0
    echo "  Syncing augmentation pool..."
    mkdir -p data/aug
    gsutil -m rsync -r "${BUCKET_URL}/data/aug" data/aug/

    # MLflow artifacts (Pretrains only - Experiment 1)
    echo "  Syncing mlruns experiment 1 (pretrains)..."
    mkdir -p mlruns/1
    gsutil -m rsync -r "${BUCKET_URL}/mlruns/1" mlruns/1/

    # MLflow Database — pull ONLY if absent locally, so re-bootstrapping never clobbers local runs
    # not yet pushed (use scripts/sync_to_bucket.sh to push local->bucket; rm mlflow.db to force a pull).
    if [ ! -f mlflow.db ]; then
        echo "  Syncing mlflow.db..."
        gsutil cp "${BUCKET_URL}/mlflow.db" mlflow.db
    else
        echo "  mlflow.db present locally — keeping it (re-bootstrap no longer overwrites your runs)"
    fi

    # DINOv3 backbone weights (for the optional dinov3 backbone; gitignored, ~83MB each)
    echo "  Syncing DINOv3 weights..."
    mkdir -p src/models/weights
    gsutil -m rsync -r "${BUCKET_URL}/src/models/weights" src/models/weights/ || warn "DINOv3 weights sync failed (only needed for a dinov3 backbone)"
else
    warn "FACE_OCC_BUCKET_URL not set, assuming data is already present."
fi

# 2. Setup Environment
say "Setting up virtual environment"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
uv sync --no-dev
source "${VENV_DIR}/bin/activate"

# 3. Fix MLflow Database
if [ -f "mlflow.db" ]; then
    say "Checking & fixing MLflow database paths"
    # Basic integrity check
    python3 -c "import sqlite3; conn = sqlite3.connect('mlflow.db'); conn.execute('PRAGMA integrity_check').fetchone(); conn.close()" || die "mlflow.db is corrupt. Please delete it from the bucket or locally."
    python3 scripts/fix_mlflow_paths.py mlflow.db --apply
fi

# 4. Launch Task
say "Launching ${MODE} for ${ARCH_NAME}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# H100 / A100 Optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Raise the open-files limit: long Optuna sweeps leak FDs (mlflow + DataLoaders) and hit the default
# ~1024 after ~80 trials -> "[Errno 24] Too many open files" which fails trials (they return inf).
ulimit -n 65536 || ulimit -n 8192 || true

if [ "${MODE}" == "optimize" ]; then
    python3 src/optimize.py "${ARCH_NAME}"
else
    python3 src/train.py "${ARCH_NAME}"
fi
