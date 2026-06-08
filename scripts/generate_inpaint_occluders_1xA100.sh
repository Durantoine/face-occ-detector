#!/bin/bash
#SBATCH --job-name=face-occ-inpaint-occluders
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -e

echo "================================================================================"
echo "SYNTHETIC OCCLUDERS - diffusion inpainting (hair / hand / headwear)"
echo "  Photorealistic occluders, controlled mask => exact FaceOcclusion label"
echo "  Area biased high (0.30-0.70) to fill the rare high-occlusion tail"
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

# --- knobs (override via env when sbatch'ing) -------------------------------
# Adjust the image dir if you extracted crops.zip elsewhere on the cluster.
IMAGE_DIR="${IMAGE_DIR:-data/raw/Crop_224_5fp_100K}"
TRAIN_CSV="${TRAIN_CSV:-data/raw/train.csv}"
N_SAMPLES="${N_SAMPLES:-13000}"
TYPES="${TYPES:-hair headwear scarf hand}"     # hands kept; SDXL + quality-guard make them viable (validate in smoke test)
MODEL="${MODEL:-diffusers/stable-diffusion-xl-1.0-inpainting-0.1}"  # best quality; needs --work-size 1024
WORK="${WORK:-1024}"
STEPS="${STEPS:-30}"
BATCH="${BATCH:-4}"                            # SDXL@1024 is memory-heavy; 4 fits an A100
OUTDIR="${OUTDIR:-data/synthetic/occluder_inpaint}"
#
# SMOKE TEST FIRST: validate quality on the cluster before the full run, e.g.
#   sbatch --export=ALL,N_SAMPLES=48,TYPES='hair headwear scarf hand',OUTDIR=data/synthetic/_smoke \
#          scripts/generate_inpaint_occluders_1xA100.sh
# Inspect data/synthetic/_smoke/, then launch the full run with the defaults.
#
# Faster/cheaper alternative model (512px, ~3x faster, slightly lower quality):
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
