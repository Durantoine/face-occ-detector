#!/bin/bash
# Launch the combined MLflow + Optuna UI as a single SLURM job, wait for it to start,
# then print the ssh-tunnel command to copy/paste on your laptop.
#
# Usage: ./scripts/launch_ui.sh

set -e

MLFLOW_PORT="${MLFLOW_PORT:-5000}"
OPTUNA_PORT="${OPTUNA_PORT:-8080}"
QUAL_PORT="${QUAL_PORT:-8501}"
GATEWAY="${GATEWAY:-adurand-25@gpu-gw}"

cd "$(dirname "$0")/.."

UI_JOB=$(sbatch --parsable scripts/ui.sh)
echo "Submitted: UI job=${UI_JOB}"
echo "Waiting for job to be RUNNING..."

extract_node() {
    squeue -j "$1" -h -t RUNNING -o '%N' 2>/dev/null | head -1
}

for i in {1..60}; do
    UI_NODE=$(extract_node "$UI_JOB")
    if [[ -n "$UI_NODE" ]]; then
        break
    fi
    echo "  ... waiting (state: $(squeue -j $UI_JOB -h -o '%T' 2>/dev/null || echo gone))"
    sleep 5
done

if [[ -z "$UI_NODE" ]]; then
    echo "TIMEOUT: job still pending after 5 min. Check 'squeue --me'." >&2
    exit 1
fi

cat <<EOF

================================================================================
UIs running on ${UI_NODE} (job ${UI_JOB})

  ▶ MLflow        : sqlite:///mlflow.db
  ▶ Optuna        : sqlite:///optuna.db
  ▶ Analytics     : trials comparison (live) + qualitative best/worst-K viewer

Copy-paste this SSH tunnel on your LAPTOP:

  ssh -N \\
      -L ${MLFLOW_PORT}:${UI_NODE}:${MLFLOW_PORT} \\
      -L ${OPTUNA_PORT}:${UI_NODE}:${OPTUNA_PORT} \\
      -L ${QUAL_PORT}:${UI_NODE}:${QUAL_PORT} \\
      ${GATEWAY}

Then open in your browser:
  http://localhost:${MLFLOW_PORT}     (MLflow)
  http://localhost:${OPTUNA_PORT}     (Optuna dashboard)
  http://localhost:${QUAL_PORT}     (Analytics — trials comparison + qualitative viewer)

Stop the UIs:
  scancel ${UI_JOB}
================================================================================
EOF
