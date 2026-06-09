#!/bin/bash
#SBATCH --job-name=face-occ-inpaint-occluders
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -e

echo "================================================================================"
echo "SYNTHETIC OCCLUDERS - diffusion inpainting (hair / headwear / scarf / hand)"
echo "  Photorealistic occluders, controlled mask => exact FaceOcclusion label"
echo "  Area biased high (0.30-0.70) to fill the rare high-occlusion tail"
echo "  1x3090 (24GB): SDXL@1024 with VAE slicing/tiling, batch 2"
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
uv sync --no-dev
source "${VENV_DIR}/bin/activate"

# diffusers is required by the inpainting generator (added to pyproject; this
# line is a belt-and-braces fallback in case the lockfile wasn't re-synced).
"${VENV_DIR}/bin/python" -c "import diffusers" 2>/dev/null || "${VENV_DIR}/bin/pip" install "diffusers>=0.27.0"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- knobs (override via --export=ALL,VAR=...) ------------------------------
# On this cluster the images are flat under data/raw/database{1,2,3}, so the
# image dir (the folder that CONTAINS database3/) is data/raw.
IMAGE_DIR="${IMAGE_DIR:-data/raw}"
TRAIN_CSV="${TRAIN_CSV:-data/raw/train.csv}"
N_SAMPLES="${N_SAMPLES:-13000}"
TYPES="${TYPES:-hair headwear scarf hand}"
MODEL="${MODEL:-diffusers/stable-diffusion-xl-1.0-inpainting-0.1}"
WORK="${WORK:-1024}"
STEPS="${STEPS:-30}"
BATCH="${BATCH:-2}"            # 3090 24GB: 2 fits SDXL@1024 with VAE slicing/tiling
OUTDIR="${OUTDIR:-data/synthetic/occluder_inpaint}"
#
# SMOKE TEST FIRST (validate quality, esp. hands), e.g.:
#   sbatch --export=ALL,N_SAMPLES=48,OUTDIR=data/synthetic/_smoke \
#          scripts/generate_inpaint_occluders_1x3090.sh
#
# If SDXL@1024 OOMs, drop to WORK=768 BATCH=2, or use the fast 512 model:
#   --export=ALL,MODEL=Lykon/dreamshaper-8-inpainting,WORK=512,BATCH=8
# ---------------------------------------------------------------------------

mkdir -p scripts/logs

"${VENV_DIR}/bin/python" scripts/generate_inpaint_occluders.py \
    --train-csv "${TRAIN_CSV}" \
    --image-dir "${IMAGE_DIR}" \
    --output-dir "${OUTDIR}" \
    --n-samples "${N_SAMPLES}" \
    --types ${TYPES} \
    --model "${MODEL}" \
    --work-size "${WORK}" \
    --area-min 0.30 --area-max 0.70 \
    --batch-size "${BATCH}" --steps "${STEPS}"

echo "================================================================================"
echo "COMPLETE - Finished: $(date)"
echo "================================================================================"
