#!/bin/bash
# Qwen fine-tuning — SLURM script for ENSTA cluster
#
# Usage:
#   sbatch scripts/train_qwen_cluster.sh --sweep
#   sbatch scripts/train_qwen_cluster.sh --config configs/architectures/qwen-7b-l40s-v1.yaml

#SBATCH --job-name=qwen-finetune
#SBATCH --partition=ENSTA-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=scripts/logs/qwen_finetune_%j.out
#SBATCH --error=scripts/logs/qwen_finetune_%j.err

set -euo pipefail

PROJECT_DIR="${HOME}/face-occ-detector"
VENV_DIR="/tmp/face_occ_venv_${SLURM_JOB_ID}"
GSUTIL="/home/telecom-paris/tp-adurand-25/google-cloud-sdk/bin/gsutil"
BUCKET="gs://mon-face-occ-bucket"
CONFIG="configs/architectures/qwen-7b-l40s-v1.yaml"
SWEEP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --sweep)  SWEEP=true;  shift ;;
        *)        shift ;;
    esac
done

cd "${PROJECT_DIR}"
mkdir -p scripts/logs results/qwen_finetune outputs/lora_adapters

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

# ── Symlinks pour compatibilité avec le code ──────────────────────────────────
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
    "transformers>=4.52.0" \
    accelerate \
    "peft>=0.13.0" \
    "trl>=0.12.0" \
    "bitsandbytes>=0.44.0" \
    mlflow optuna pillow pandas numpy tqdm qwen-vl-utils pyyaml

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# ── DeepSpeed config (ZeRO-2 pour 7B) ────────────────────────────────────────
cat > /tmp/ds_config_${SLURM_JOB_ID}.json << EOF
{
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "offload_optimizer": {"device": "none"}
  },
  "bf16": {"enabled": true},
  "gradient_clipping": 1.0,
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
EOF

# ── Lancement ─────────────────────────────────────────────────────────────────
SWEEP_FLAG=""
$SWEEP && SWEEP_FLAG="--sweep"

# Supprimer mlflow.db si schéma incompatible (évite "duplicate column name: step")
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

echo "Launching training..."
accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    src/qwen_finetune.py \
    --config "${CONFIG}" \
    --mlflow-uri "sqlite:///mlflow.db" \
    ${SWEEP_FLAG}

echo "========================================"
echo "Terminé: $(date)"
echo "LoRA adapter: ${PROJECT_DIR}/outputs/lora_adapters"
echo "========================================"
