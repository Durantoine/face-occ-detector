#!/bin/bash
# Selective sync of Sapiens and Dino studies from bucket to cluster.
# Syncs only the DBs, the specific experiment trials in mlruns, and the results.
#
# Usage:
#   bash scripts/sync_studies_selective.sh

set -euo pipefail

# Default bucket URL
DEFAULT_BUCKET="gs://mon-face-occ-bucket"
BUCKET_URL="${FACE_OCC_BUCKET_URL:-$DEFAULT_BUCKET}"
BUCKET_URL="${BUCKET_URL%/}"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }

# Detect sync command
if ! command -v gcloud >/dev/null 2>&1 && ! command -v gsutil >/dev/null 2>&1; then
    if [ -d "${HOME}/google-cloud-sdk" ]; then
        export PATH="${HOME}/google-cloud-sdk/bin:${PATH}"
    fi
fi

GS_CMD="gcloud storage"
if ! command -v gcloud >/dev/null 2>&1; then
    if command -v gsutil >/dev/null 2>&1; then
        GS_CMD="gsutil"
    else
        echo "ERROR: Neither gcloud nor gsutil found in PATH or ${HOME}/google-cloud-sdk/bin"
        exit 1
    fi
fi

say "Starting selective sync for Sapiens-HC and Dino-HC from ${BUCKET_URL}"

# 1. Sync Databases (needed to identify mlruns folders)
say "1. Syncing Databases..."
if [ "${GS_CMD}" = "gcloud storage" ]; then
    gcloud storage cp "${BUCKET_URL}/mlflow.db" mlflow.db
    gcloud storage cp "${BUCKET_URL}/optuna.db" optuna.db
else
    gsutil cp "${BUCKET_URL}/mlflow.db" mlflow.db
    gsutil cp "${BUCKET_URL}/optuna.db" optuna.db
fi

# 2. Identify Experiment IDs and Sync mlruns folders
say "2. Identifying and syncing mlruns experiment folders..."
# We need to know which experiment IDs correspond to our studies to avoid syncing everything.
# Studies are usually named "optuna-sapiens-hc" and "optuna-dino-vitb16-hc".
# Also, we always need experiment '1' because it contains the iBOT encoders.

EXPS=("optuna-sapiens-hc" "optuna-dino-vitb16-hc")
IDS=("1") # Always include '1' for encoders

for name in "${EXPS[@]}"; do
    ID=$(sqlite3 mlflow.db "SELECT experiment_id FROM experiments WHERE name = '${name}'" 2>/dev/null || echo "")
    if [ -n "$ID" ]; then
        echo "  Found ${name} -> ID ${ID}"
        IDS+=("$ID")
    else
        echo "  ! Could not find experiment ID for ${name} in mlflow.db"
    fi
done

for id in $(echo "${IDS[@]}" | tr ' ' '\n' | sort -u); do
    echo "  Syncing mlruns/${id}..."
    mkdir -p "mlruns/${id}"
    if [ "${GS_CMD}" = "gcloud storage" ]; then
        gcloud storage rsync -r "${BUCKET_URL}/mlruns/${id}" "mlruns/${id}"
    else
        gsutil -m rsync -r "${BUCKET_URL}/mlruns/${id}" "mlruns/${id}"
    fi
done

# 3. Sync Results folders (selective)
say "3. Syncing specific results folders..."
mkdir -p results
# Use gsutil/gcloud ls to find all folders starting with our arch names (including trials)
ARCHS=("sapiens-hc" "dino-vitb16-hc")

for arch in "${ARCHS[@]}"; do
    echo "  Searching results for ${arch}* in bucket..."
    # List folders in results/ that match the architecture
    if [ "${GS_CMD}" = "gcloud storage" ]; then
        FOLDERS=$(${GS_CMD} ls "${BUCKET_URL}/results/" | grep "${arch}" || true)
    else
        FOLDERS=$(gsutil ls "${BUCKET_URL}/results/" | grep "${arch}" || true)
    fi

    for folder_url in ${FOLDERS}; do
        folder_name=$(basename "${folder_url}")
        echo "    Syncing results/${folder_name}..."
        mkdir -p "results/${folder_name}"
        if [ "${GS_CMD}" = "gcloud storage" ]; then
            gcloud storage rsync -r "${folder_url}" "results/${folder_name}"
        else
            gsutil -m rsync -r "${folder_url}" "results/${folder_name}"
        fi
    done
done

say "Selective sync complete! Ready for ensemble."
