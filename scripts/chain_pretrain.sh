#!/bin/bash
# Chain N successive SLURM submissions of the same script — each picks up where the previous left off
# via the pretrain checkpoint resumability.
#
# Usage:
#   ./scripts/chain_pretrain.sh                  # 3 links of pretrain_ibot_vith16plus_2x3090.sh
#   ./scripts/chain_pretrain.sh 5                # 5 links
#   ./scripts/chain_pretrain.sh 3 scripts/foo.sh # 3 links of foo.sh
#
# Total wall-clock available = N x SBATCH --time. With N=3 and --time=30:00:00, 90 hours max.
# Each link reuses the same MLflow run via results/pretrain/mlflow_run_id.txt
# and resumes from the latest results/pretrain/checkpoint-XXX.

set -e

N="${1:-3}"
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
echo "Chain submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f scripts/logs/face-occ-ibot-vith16plus_<jobid>.out"
echo ""
echo "Cancel the whole chain with:  scancel \$(squeue -u \$USER -h -o '%i' | tr '\\n' ' ')"
