#!/bin/bash
# Bootstrap script for Google Cloud VM (A100) — face-occ HPO sweep.
#
# Usage on a fresh GCP VM (Ubuntu 22.04+ with CUDA driver pre-installed) :
#   1. SSH into the VM
#   2. git clone <your-repo-url> face-occ-detector && cd face-occ-detector
#   3. bash scripts/gcp_bootstrap_and_run.sh <FACE_OCC_ARCH>
#
# Example : bash scripts/gcp_bootstrap_and_run.sh sapiens2-01b-scout-v22
#
# What it does (cost-optimized for hourly billing):
#   1. Install uv (fast Python pkg manager) — ~30s
#   2. Sync project deps (cached after first run) — ~2-5min first time
#   3. Download dataset (data/raw/) from GCS bucket if env var set, else assume present
#   4. Optional sync of DINOv3 backbone weights from GCS (not on HF)
#   5. Optional sync of cluster's mlflow.db (= iBOT pretrain runs referenced in yamls)
#   6. Pre-warm HuggingFace cache for Sapiens
#   7. Launch Optuna HPO (single A100, no DDP) → outputs to ./mlflow.db + ./optuna.db
#
# === Required GCS buckets (set env vars before running) ===
#   FACE_OCC_DATA_BUCKET    : dataset dir  (e.g. gs://my-bucket/data/raw)
#   FACE_OCC_DINOV3_BUCKET  : DINOv3 .pth weights dir (optional, only if arch uses DINOv3)
#   FACE_OCC_MLFLOW_BUCKET  : cluster's mlflow.db FILE (optional, only if pretrained_source=ibot:...)
#   FACE_OCC_MLRUNS_BUCKET  : cluster's mlruns/ DIR (paired with MLFLOW_BUCKET, contains the encoder artifacts)
#
# === Pre-upload from local Mac (one-time, after downloading from cluster) ===
#   BUCKET=gs://face-occ-durand-2026
#   gsutil -m rsync -r data/raw                      $BUCKET/data/raw
#   gsutil       cp   mlflow.db                       $BUCKET/mlflow.db
#   gsutil -m rsync -r results/pretrain_sapiens2_01b  $BUCKET/mlruns/1/b70b78e861d54a3aae7659c95f9146ad/artifacts
#   # Add same for c1e1ba1a747747688e3691daa4f7a31a (dinov3 ibot) if needed.
#
# === On the VM, before running this script ===
#   export FACE_OCC_DATA_BUCKET=gs://face-occ-durand-2026/data/raw
#   export FACE_OCC_MLFLOW_BUCKET=gs://face-occ-durand-2026/mlflow.db
#   export FACE_OCC_MLRUNS_BUCKET=gs://face-occ-durand-2026/mlruns
#
# Scout sapiens 0.1B uses pretrained_source ∈ {sapiens_default, ibot:runs:/b70b...}, donc
# tu DOIS uploader mlflow.db + mlruns/ pour que l'option iBOT marche.

set -euo pipefail

FACE_OCC_ARCH="${1:-sapiens2-01b-v23}"
PROJECT_DIR="${HOME}/face-occ-detector"
VENV_DIR="${PROJECT_DIR}/.venv"
DATA_BUCKET="${FACE_OCC_DATA_BUCKET:-}"          # gs://bucket/face-occ/raw
DINOV3_BUCKET="${FACE_OCC_DINOV3_BUCKET:-}"      # gs://bucket/face-occ/dinov3_weights
MLFLOW_BUCKET="${FACE_OCC_MLFLOW_BUCKET:-}"      # gs://bucket/face-occ/cluster_mlflow.db
LOG_DIR="${PROJECT_DIR}/scripts/logs"

echo "================================================================================"
echo "GCP VM bootstrap — face-occ HPO"
echo "  Arch     : ${FACE_OCC_ARCH}"
echo "  Project  : ${PROJECT_DIR}"
echo "  GPUs     : $(nvidia-smi --query-gpu=name --format=csv,noheader || echo 'no nvidia-smi')"
echo "  Started  : $(date -Iseconds)"
echo "================================================================================"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"

# === 1. Install uv if missing ===
if ! command -v uv >/dev/null 2>&1; then
    echo "[1/5] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
uv --version

# === 2. Sync Python deps (uv handles caching) ===
echo "[2/5] uv sync (will be cached after first run)..."
export UV_PYTHON=python3.12
export UV_PYTHON_DOWNLOADS=automatic
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
uv sync --no-dev
source "${VENV_DIR}/bin/activate"
"${VENV_DIR}/bin/python" -c "import torch; print(f'torch {torch.__version__}, CUDA={torch.cuda.is_available()}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# === 3. Dataset ===
if [ ! -f "data/raw/train.csv" ]; then
    if [ -n "${DATA_BUCKET}" ]; then
        echo "[3/7] Downloading dataset from ${DATA_BUCKET}..."
        mkdir -p data/raw
        gsutil -m rsync -r "${DATA_BUCKET}" data/raw/
        echo "  ✓ $(wc -l < data/raw/train.csv) rows in train.csv"
    else
        echo "[3/7] ERR: data/raw/train.csv missing and FACE_OCC_DATA_BUCKET not set"
        echo "       export FACE_OCC_DATA_BUCKET=gs://your-bucket/face-occ/raw"
        exit 1
    fi
else
    echo "[3/7] Dataset already present: $(wc -l < data/raw/train.csv) rows in train.csv"
fi

# === 4. DINOv3 weights — non-HF, need to be uploaded from cluster ===
DINOV3_LOCAL_DIR="${PROJECT_DIR}/src/models/weights"
if [ -n "${DINOV3_BUCKET}" ] && [[ "${FACE_OCC_ARCH}" == *dinov3* ]]; then
    if [ ! -d "${DINOV3_LOCAL_DIR}" ] || [ -z "$(ls -A "${DINOV3_LOCAL_DIR}" 2>/dev/null)" ]; then
        echo "[4/7] Downloading DINOv3 weights from ${DINOV3_BUCKET}..."
        mkdir -p "${DINOV3_LOCAL_DIR}"
        gsutil -m rsync -r "${DINOV3_BUCKET}" "${DINOV3_LOCAL_DIR}/"
        echo "  ✓ DINOv3 weights : $(ls -1 "${DINOV3_LOCAL_DIR}" | wc -l) files"
    else
        echo "[4/7] DINOv3 weights already present"
    fi
elif [[ "${FACE_OCC_ARCH}" == *dinov3* ]]; then
    echo "[4/7] WARNING: arch uses DINOv3 but FACE_OCC_DINOV3_BUCKET not set."
    echo "       If src/models/weights/ is empty → training will fail."
else
    echo "[4/7] Skip DINOv3 (arch=${FACE_OCC_ARCH} doesn't use it)"
fi

# === 5. mlflow.db from cluster + mlruns artifacts (iBOT pretrain runs) ===
# Required when yaml has pretrained_source like "ibot:runs:/abc123.../encoder".
# Bucket layout expected:
#   ${MLFLOW_BUCKET}                  → the mlflow.db file (file path, not dir)
#   ${MLRUNS_BUCKET}                  → the mlruns/ directory (rsync target)
# After download we REWRITE artifact_uri in mlflow.db to local absolute path,
# because cluster paths like /home/infres/adurand-25/... don't exist on VM.
if [ -n "${MLFLOW_BUCKET}" ]; then
    if [ ! -f "mlflow.db" ]; then
        echo "[5/7] Downloading cluster's mlflow.db..."
        gsutil cp "${MLFLOW_BUCKET}" mlflow.db
        echo "  ✓ mlflow.db : $(du -h mlflow.db | cut -f1)"
    fi
    # Download artifacts dir (mlruns/) if bucket var set
    MLRUNS_BUCKET="${FACE_OCC_MLRUNS_BUCKET:-}"
    if [ -n "${MLRUNS_BUCKET}" ] && [ ! -d "mlruns" ]; then
        echo "  Downloading mlruns/ artifacts from ${MLRUNS_BUCKET}..."
        mkdir -p mlruns
        gsutil -m rsync -r "${MLRUNS_BUCKET}" mlruns/
        echo "  ✓ mlruns/ : $(du -sh mlruns | cut -f1), $(find mlruns -name 'encoder' -type d 2>/dev/null | wc -l) encoder dirs"
    fi
    # Rewrite artifact_uri in mlflow.db: cluster paths → local VM path
    "${VENV_DIR}/bin/python" - <<PYEOF || true
import sqlite3, os
db = sqlite3.connect('mlflow.db')
c = db.cursor()
new_root = os.path.abspath('mlruns')
# Find cluster prefix (everything before /mlruns/...) — assume all runs share it
rows = c.execute("SELECT DISTINCT artifact_uri FROM runs WHERE artifact_uri LIKE '%/mlruns/%'").fetchall()
prefixes = set()
for (uri,) in rows:
    idx = uri.find('/mlruns/')
    if idx > 0:
        prefixes.add(uri[:idx])
print(f"  Found {len(prefixes)} distinct artifact path prefix(es): {prefixes}")
n_total = 0
for prefix in prefixes:
    res = c.execute(
        "UPDATE runs SET artifact_uri = REPLACE(artifact_uri, ?, ?)",
        (prefix, os.path.dirname(new_root)),  # parent of mlruns/
    )
    n_total += res.rowcount
db.commit()
print(f"  ✓ Rewrote artifact_uri on {n_total} runs (→ {os.path.dirname(new_root)}/mlruns/...)")
# Sanity check: list iBOT runs available
rows = c.execute("SELECT name, run_uuid, artifact_uri FROM runs r JOIN experiments e USING(experiment_id) WHERE e.name='face-occ-pretrain' AND r.status='FINISHED'").fetchall()
print(f"  iBOT pretrain runs available: {len(rows)}")
for n, rid, uri in rows[:5]:
    ok = "✓" if os.path.exists(uri) else "✗ MISSING"
    print(f"    {ok}  {n}  →  ibot:runs:/{rid}/encoder  ({uri})")
db.close()
PYEOF
else
    echo "[5/7] FACE_OCC_MLFLOW_BUCKET not set → fresh local mlflow.db will be created."
    echo "      Any pretrained_source=ibot:runs:/... in yaml will fail (run not found)."
fi

# === 6. Pre-warm HuggingFace cache for Sapiens ===
if [[ "${FACE_OCC_ARCH}" == *sapiens* ]]; then
    echo "[6/7] Pre-warming HF cache (sapiens2 weights)..."
    # Detect which variant from arch name (0.1b vs 0.8b)
    if [[ "${FACE_OCC_ARCH}" == *01b* ]]; then SAP_VAR="0.1b";
    elif [[ "${FACE_OCC_ARCH}" == *08b* ]]; then SAP_VAR="0.8b";
    else SAP_VAR=""; fi
    if [ -n "${SAP_VAR}" ]; then
        "${VENV_DIR}/bin/python" - <<PYEOF || echo "  (warning: pre-warm failed, will retry at first train step)"
from huggingface_hub import hf_hub_download
fname = "sapiens2_${SAP_VAR}_pretrain.safetensors"
hf_hub_download(repo_id="facebook/sapiens2-pretrain-${SAP_VAR}", filename=fname)
print(f"  ✓ sapiens2_${SAP_VAR} cached: {fname}")
PYEOF
    fi
else
    echo "[6/7] Skip Sapiens HF pre-warm (arch=${FACE_OCC_ARCH} doesn't use it)"
fi

# === 7. Tune environment + launch ===
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8
ulimit -n 65536 || ulimit -n 8192

export FACE_OCC_ARCH="${FACE_OCC_ARCH}"
JOB_LOG="${LOG_DIR}/gcp_${FACE_OCC_ARCH}_$(date +%Y%m%d_%H%M%S).log"

echo "[7/7] Launching Optuna sweep — log → ${JOB_LOG}"
echo "================================================================================"

# Run in foreground (so SSH session sees output). Use `nohup` + `&` if you want detached.
"${VENV_DIR}/bin/python" src/optimize.py "${FACE_OCC_ARCH}" 2>&1 | tee "${JOB_LOG}"

echo "================================================================================"
echo "COMPLETE — $(date -Iseconds)"
echo "  Stop the VM ASAP to limit billing: gcloud compute instances stop <vm-name>"
echo "  Results : mlflow.db + optuna.db + scripts/logs/"
echo "================================================================================"
