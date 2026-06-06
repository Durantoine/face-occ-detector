#!/bin/bash
#SBATCH --job-name=qwen_inf_%a
#SBATCH --output=logs/inference_%a_%j.log
#SBATCH --error=logs/inference_%a_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --array=0-1

source .venv/bin/activate
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache

mkdir -p logs results

uv run python src/qwen_inference.py \
    config/qwen_inference.yaml \
    --chunk-id ${SLURM_ARRAY_TASK_ID} \
    --n-chunks 2
