#!/bin/bash
#SBATCH --job-name=qwen-predict-test
#SBATCH --partition=ecole-h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=scripts/logs/qwen_predict_%j.out
#SBATCH --error=scripts/logs/qwen_predict_%j.err

set -euo pipefail

PROJECT_DIR="${HOME}/face-occ-detector"
VENV_DIR="/tmp/face_occ_venv_${SLURM_JOB_ID}"
GSUTIL="/home/telecom-paris/tp-adurand-25/google-cloud-sdk/bin/gsutil"
BUCKET="gs://mon-face-occ-bucket"

cd "${PROJECT_DIR}"
mkdir -p scripts/logs results/qwen_test_inference

# ── Téléchargement des données depuis le bucket public ────────────────────────
echo "Téléchargement des données depuis ${BUCKET} ..."
mkdir -p data/raw

if [ ! -f "data/raw/test_students.csv" ]; then
    echo "  Downloading test_students.csv ..."
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

echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)"
echo "Started: $(date)"
echo "========================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Setup env avec pip (uv non disponible sur ce cluster)
PYTHON=$(command -v python3.12 || command -v python3 || command -v python)
echo "Python: $PYTHON ($($PYTHON --version))"

if [ ! -d "${VENV_DIR}" ]; then
    echo "Création venv dans ${VENV_DIR} ..."
    $PYTHON -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

echo "Installation des dépendances ..."
pip install -q --upgrade pip

# PyPI packages
pip install -q \
    "transformers>=4.52.0" \
    accelerate \
    "peft>=0.13.0" \
    pillow pandas numpy tqdm qwen-vl-utils

# Torch (index séparé pour CUDA 12.8)
pip install -q torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu128

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Lance les 2 GPU en parallèle (chacun traite la moitié du test set)
echo "Lancement GPU 0 (images 0–14989) ..."
CUDA_VISIBLE_DEVICES=0 python src/predict_qwen_test.py \
    --rank 0 --world-size 2 &
PID0=$!

echo "Lancement GPU 1 (images 14990–29979) ..."
CUDA_VISIBLE_DEVICES=1 python src/predict_qwen_test.py \
    --rank 1 --world-size 2 &
PID1=$!

# Attendre les deux
wait $PID0 && echo "GPU 0 terminé." || echo "GPU 0 ERREUR."
wait $PID1 && echo "GPU 1 terminé." || echo "GPU 1 ERREUR."

# Merge des résultats → test_predictions.csv
echo "Merge des prédictions ..."
python src/predict_qwen_test.py --merge

echo "========================================"
echo "Terminé: $(date)"
echo "Fichier de soumission: ${PROJECT_DIR}/test_predictions.csv"
echo "========================================"
