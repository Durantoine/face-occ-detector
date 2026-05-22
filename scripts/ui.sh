#!/bin/bash
#SBATCH --job-name=ui-mlflow-optuna
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00

set -e

MLFLOW_PORT="${MLFLOW_PORT:-5000}"
OPTUNA_PORT="${OPTUNA_PORT:-8080}"
PROJECT_DIR="${HOME}/face-occ-detector"
cd "${PROJECT_DIR}"

echo "================================================================================"
echo "Combined UIs (MLflow + Optuna) — node $(hostname)"
echo "  MLflow port: $MLFLOW_PORT  (sqlite:///mlflow.db)"
echo "  Optuna port: $OPTUNA_PORT  (sqlite:///optuna.db)"
echo "Job ID: $SLURM_JOB_ID | Started: $(date)"
echo ""
echo "Access from laptop:"
echo "  ssh -N -L ${MLFLOW_PORT}:$(hostname):${MLFLOW_PORT} -L ${OPTUNA_PORT}:$(hostname):${OPTUNA_PORT} adurand-25@gpu-gw"
echo "================================================================================"

mkdir -p $HOME/tmp scripts/logs
export OMP_NUM_THREADS=1
export TMPDIR=$HOME/tmp
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false
export MLFLOW_DEPLOYMENTS_TARGET=''

# Start both UIs in background, log to per-process files
ML_LOG="scripts/logs/ui-mlflow_${SLURM_JOB_ID}.log"
OP_LOG="scripts/logs/ui-optuna_${SLURM_JOB_ID}.log"

uvx --python 3.12 --from mlflow mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --host 0.0.0.0 --port "${MLFLOW_PORT}" --workers 1 \
    > "$ML_LOG" 2>&1 &
ML_PID=$!
echo "  → MLflow started (pid=$ML_PID, log=$ML_LOG)"

uvx --python 3.12 --from optuna-dashboard optuna-dashboard \
    sqlite:///optuna.db \
    --host 0.0.0.0 --port "${OPTUNA_PORT}" \
    > "$OP_LOG" 2>&1 &
OP_PID=$!
echo "  → Optuna started (pid=$OP_PID, log=$OP_LOG)"

# Clean shutdown on SLURM cancel
trap "echo 'Shutting down...'; kill $ML_PID $OP_PID 2>/dev/null; wait" EXIT TERM INT

# Wait for either to exit (or for SIGTERM via scancel)
wait -n
echo "One of the UI processes exited. Killing the other..."
kill $ML_PID $OP_PID 2>/dev/null || true
wait
