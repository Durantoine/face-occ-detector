#!/bin/bash
# Launch MLflow UI + Optuna Dashboard as SLURM jobs, wait for them to start,
# then print the ssh-tunnel command to copy/paste on your laptop.
#
# Usage: ./scripts/launch_ui.sh

set -e

MLFLOW_PORT="${MLFLOW_PORT:-5000}"
OPTUNA_PORT="${OPTUNA_PORT:-8080}"
GATEWAY="${GATEWAY:-adurand-25@gpu-gw}"

cd "$(dirname "$0")/.."

ML_JOB=$(sbatch --parsable scripts/mlflow_ui.sh)
OP_JOB=$(sbatch --parsable scripts/optuna_dashboard.sh)
echo "Submitted: MLflow job=${ML_JOB}, Optuna job=${OP_JOB}"
echo "Waiting for both jobs to be RUNNING..."

extract_node() {
    squeue -j "$1" -h -o '%R' 2>/dev/null | head -1
}

for i in {1..60}; do
    ML_NODE=$(extract_node "$ML_JOB")
    OP_NODE=$(extract_node "$OP_JOB")
    if [[ -n "$ML_NODE" && "$ML_NODE" != "(Resources)" && "$ML_NODE" != "(Priority)" \
       && -n "$OP_NODE" && "$OP_NODE" != "(Resources)" && "$OP_NODE" != "(Priority)" ]]; then
        break
    fi
    echo "  ... waiting (mlflow=${ML_NODE:-pending}, optuna=${OP_NODE:-pending})"
    sleep 5
done

if [[ "$ML_NODE" == "(Resources)" || "$ML_NODE" == "(Priority)" || -z "$ML_NODE" ]]; then
    echo "TIMEOUT: jobs still pending after 5 min. Check 'squeue --me'." >&2
    exit 1
fi

cat <<EOF

================================================================================
UIs running:
  MLflow  → ${ML_NODE}:${MLFLOW_PORT}     (job ${ML_JOB})
  Optuna  → ${OP_NODE}:${OPTUNA_PORT}     (job ${OP_JOB})

Copy-paste this on your LAPTOP:

  ssh -N -L ${MLFLOW_PORT}:${ML_NODE}:${MLFLOW_PORT} -L ${OPTUNA_PORT}:${OP_NODE}:${OPTUNA_PORT} ${GATEWAY}

Then open in your browser:
  http://localhost:${MLFLOW_PORT}     (MLflow)
  http://localhost:${OPTUNA_PORT}     (Optuna dashboard)

Stop the UIs:
  scancel ${ML_JOB} ${OP_JOB}
================================================================================
EOF
