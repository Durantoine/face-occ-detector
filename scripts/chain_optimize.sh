#!/bin/bash
# Chain N successive SLURM submissions of an Optuna optimize script. Each link picks up
# where the previous left off via Optuna's `load_if_exists=True` on the SQLite study
# (sqlite:///optuna.db). Counts trials toward the same `n_trials` budget set in the yaml.
#
# Usage:
#   ./scripts/chain_optimize.sh                                              # 3 links of vitb16 v8 (default)
#   ./scripts/chain_optimize.sh 3 scripts/optimize_sapiens2_01b_v8_2x3090.sh
#
# Recommended for the small configs (vitb16, sapiens 0.1b) with n_trials=200 :
#   ./scripts/chain_optimize.sh 3 scripts/optimize_dinov3_vitb16_v8_2x3090.sh
#   ./scripts/chain_optimize.sh 3 scripts/optimize_sapiens2_01b_v8_2x3090.sh

set -e

N="${1:-3}"
SCRIPT="${2:-scripts/optimize_dinov3_vitb16_v8_2x3090.sh}"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: script not found: $SCRIPT"
    exit 1
fi

echo "Chaining $N optimize submissions of: $SCRIPT"
echo ""

PREV_JOB=""
for i in $(seq 1 "$N"); do
    if [ -z "$PREV_JOB" ]; then
        JOB=$(sbatch --parsable "$SCRIPT")
        echo "  [$i/$N] $JOB  (no dependency)"
    else
        JOB=$(sbatch --parsable --dependency=afterany:"$PREV_JOB" "$SCRIPT")
        echo "  [$i/$N] $JOB  (depends on $PREV_JOB)"
    fi
    PREV_JOB="$JOB"
done

JOB_NAME=$(grep -m1 -E '^#SBATCH --job-name=' "$SCRIPT" | sed 's/.*=//')

echo ""
echo "Chain submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f scripts/logs/${JOB_NAME}_<jobid>.out"
echo ""
echo "Cancel the whole chain with:  scancel \$(squeue -u \$USER -h -o '%i' | tr '\\n' ' ')"
