#!/bin/bash
# Sync cluster progress back to the bucket
# 
# Usage:
#   bash scripts/sync_to_bucket.sh

set -euo pipefail

# Default bucket URL (hardcoded for convenience)
DEFAULT_BUCKET="gs://mon-face-occ-bucket"
BUCKET_URL="${FACE_OCC_BUCKET_URL:-$DEFAULT_BUCKET}"
BUCKET_URL="${BUCKET_URL%/}"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }

say "Syncing progress to bucket ${BUCKET_URL}"

# 1. Sync Databases (from /tmp if using Fast-Disk optimization)
if [ -L "mlflow.db" ]; then
    REAL_DB=$(readlink -f mlflow.db)
    echo "  Syncing mlflow.db (from fast-disk: ${REAL_DB})..."
    gsutil cp "${REAL_DB}" "${BUCKET_URL}/mlflow.db"
elif [ -f "mlflow.db" ]; then
    echo "  Syncing mlflow.db..."
    gsutil cp mlflow.db "${BUCKET_URL}/mlflow.db"
fi

if [ -L "optuna.db" ]; then
    REAL_DB=$(readlink -f optuna.db)
    echo "  Syncing optuna.db (from fast-disk: ${REAL_DB})..."
    gsutil cp "${REAL_DB}" "${BUCKET_URL}/optuna.db"
elif [ -f "optuna.db" ]; then
    echo "  Syncing optuna.db..."
    gsutil cp optuna.db "${BUCKET_URL}/optuna.db"
fi

# 2. Sync MLflow Runs (incremental)
if [ -d "mlruns" ]; then
    echo "  Syncing mlruns artifacts..."
    gsutil -m rsync -r mlruns "${BUCKET_URL}/mlruns"
fi

# 3. Sync Results (incremental)
if [ -d "results" ]; then
    echo "  Syncing results..."
    gsutil -m rsync -r results "${BUCKET_URL}/results"
fi

say "Done! Everything is safe in the bucket."
