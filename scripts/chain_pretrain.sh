#!/bin/bash
# Chain N successive SLURM submissions of the same script — each picks up where the previous left off
# via the pretrain checkpoint resumability.
#
# Usage:
#   ./scripts/chain_pretrain.sh                                            # 4 links of vith16plus
#   ./scripts/chain_pretrain.sh 5                                          # 5 links
#   ./scripts/chain_pretrain.sh 4 scripts/pretrain_ibot_sapiens2_08b_2x3090.sh
#
# Total wall-clock available = N x SBATCH --time. With N=4 and --time=30:00:00, 120 hours max.
# Gros modèles @224: MAX_STEPS=100000 (~8 links pour aller au bout). Stop-early possible :
# 4 snapshots intermédiaires sont logués dans MLflow (encoder_25000/50000/75000/encoder final),
# tu peux scancel quand le sweep Optuna révèle qu'un snapshot précoce suffit.
# Each link reuses the same MLflow run via <OUT>/mlflow_run_id.txt and resumes from <OUT>/checkpoint-XXX.

set -e

N="${1:-4}"
SCRIPT="${2:-scripts/pretrain_ibot_vith16plus_2x3090.sh}"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: script not found: $SCRIPT"
    exit 1
fi

echo "Chaining $N submissions of: $SCRIPT"
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

echo ""
JOB_NAME=$(grep -m1 -E '^#SBATCH --job-name=' "$SCRIPT" | sed 's/.*=//')

echo "Chain submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f scripts/logs/${JOB_NAME}_<jobid>.out"
echo ""
echo "Cancel the whole chain with:  scancel \$(squeue -u \$USER -h -o '%i' | tr '\\n' ' ')"
