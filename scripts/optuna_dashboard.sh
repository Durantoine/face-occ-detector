#!/bin/bash
#SBATCH --job-name=optuna-dashboard
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00

set -e

PORT="${OPTUNA_PORT:-8080}"
PROJECT_DIR="${HOME}/face-occ-detector"
cd "${PROJECT_DIR}"

echo "================================================================================"
echo "Optuna Dashboard — node $(hostname), port $PORT"
echo "Job ID: $SLURM_JOB_ID | Started: $(date)"
echo ""
echo "To access from your laptop:"
echo "  ssh -L ${PORT}:$(hostname):${PORT} adurand-25@gpu-gw"
echo "  open http://localhost:${PORT}"
echo "================================================================================"

mkdir -p scripts/logs
export OMP_NUM_THREADS=1

uvx --python 3.12 --from optuna-dashboard optuna-dashboard \
    sqlite:///optuna.db \
    --host 0.0.0.0 \
    --port "${PORT}"
