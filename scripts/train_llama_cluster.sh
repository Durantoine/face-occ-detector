#!/bin/bash
# Llama 3.2 Vision 11B fine-tuning — SLURM script for ENSTA cluster
#
# Usage:
#   sbatch scripts/train_llama_cluster.sh
#   sbatch scripts/train_llama_cluster.sh --sweep
#   sbatch scripts/train_llama_cluster.sh --config configs/architectures/llama-11b-l40s-v1.yaml

#SBATCH --job-name=llama-finetune
#SBATCH --partition=ENSTA-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=scripts/logs/llama_finetune_%j.out
#SBATCH --error=scripts/logs/llama_finetune_%j.err

set -euo pipefail

PROJECT_DIR="${HOME}/face-occ-detector"
VENV_DIR="/tmp/face_occ_venv_llama_${SLURM_JOB_ID}"
GSUTIL="/home/telecom-paris/tp-adurand-25/google-cloud-sdk/bin/gsutil"
BUCKET="gs://mon-face-occ-bucket"
CONFIG="configs/architectures/llama-11b-l40s-v1.yaml"
SWEEP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --sweep)  SWEEP=true;  shift ;;
        *)        shift ;;
    esac
done

cd "${PROJECT_DIR}"
mkdir -p scripts/logs results/llama_finetune outputs/lora_adapters

echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)"
echo "Config: $CONFIG  Sweep: $SWEEP"
echo "Started: $(date)"
echo "========================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Données depuis le bucket ──────────────────────────────────────────────────
echo "Téléchargement des données depuis ${BUCKET} ..."
mkdir -p data/raw

if [ ! -f "data/raw/train.csv" ]; then
    ${GSUTIL} cp "${BUCKET}/data/raw/train.csv" data/raw/train.csv
fi
if [ ! -f "data/raw/test_students.csv" ]; then
    ${GSUTIL} cp "${BUCKET}/data/raw/test_students.csv" data/raw/test_students.csv
fi

for DB in database1 database2 database3; do
    if [ ! -d "data/raw/${DB}" ]; then
        echo "  Downloading ${DB} images ..."
        mkdir -p "data/raw/${DB}"
        ${GSUTIL} -m rsync -r "${BUCKET}/data/raw/${DB}" "data/raw/${DB}/"
    else
        echo "  ${DB} already present, skipping."
    fi
done
echo "Données prêtes."

[ ! -f "data/train.csv" ] && ln -sf raw/train.csv data/train.csv || true

# ── Python env ────────────────────────────────────────────────────────────────
PYTHON=$(command -v python3.12 || command -v python3)
echo "Python: $PYTHON ($($PYTHON --version))"

if [ ! -d "${VENV_DIR}" ]; then
    $PYTHON -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
pip install -q --upgrade pip

# Torch en premier (cu126 exclusif — driver CUDA 12.6 sur ce cluster)
pip install -q torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126

# Autres dépendances
pip install -q \
    "transformers>=4.45.0" \
    accelerate \
    "peft>=0.13.0" \
    mlflow optuna pillow pandas numpy tqdm pyyaml

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ── Nettoyage bases de données si schéma corrompu ─────────────────────────────
if [ -f "mlflow.db" ]; then
    python3 -c "
import sqlite3
try:
    conn = sqlite3.connect('mlflow.db')
    conn.execute('SELECT step FROM metrics LIMIT 1')
    conn.close()
    print('mlflow.db OK')
except Exception as e:
    conn.close()
    import os; os.remove('mlflow.db')
    print(f'mlflow.db supprimé (schéma corrompu: {e})')
"
fi

# ── Lancement ─────────────────────────────────────────────────────────────────
SWEEP_FLAG=""
$SWEEP && SWEEP_FLAG="--sweep"

echo "Launching Llama 3.2 Vision 11B training..."
accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    src/llama_finetune.py \
    --config "${CONFIG}" \
    --mlflow-uri "sqlite:///mlflow_llama.db" \
    ${SWEEP_FLAG}

echo "========================================"
echo "Terminé: $(date)"
echo "LoRA adapter: ${PROJECT_DIR}/outputs/lora_adapters"
echo "========================================"
