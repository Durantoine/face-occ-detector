#!/bin/bash
#SBATCH --job-name=mlflow-ui
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00

set -e

PORT="${MLFLOW_PORT:-5000}"
PROJECT_DIR="${HOME}/face-occ-detector"
cd "${PROJECT_DIR}"

echo "================================================================================"
echo "MLflow UI — node $(hostname), port $PORT"
echo "Job ID: $SLURM_JOB_ID | Started: $(date)"
echo ""
echo "To access from your laptop:"
echo "  ssh -L ${PORT}:$(hostname):${PORT} adurand-25@gpu-gw"
echo "  open http://localhost:${PORT}"
echo "================================================================================"

mkdir -p $HOME/tmp scripts/logs
export OMP_NUM_THREADS=1
export TMPDIR=$HOME/tmp
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false
export MLFLOW_DEPLOYMENTS_TARGET=''

uvx --python 3.12 --from 'mlflow<3' mlflow ui \
    --backend-store-uri sqlite:///mlflow.db \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --workers 1
