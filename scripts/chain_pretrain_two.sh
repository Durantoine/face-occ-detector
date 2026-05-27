#!/bin/bash
# Chain N rounds of TWO pretrain scripts in INTERLEAVED order:
#   A1, B1, A2(after A1), B2(after B1), A3(after A2), B3(after B2), ...
#
# Each chain (A_i, B_i) resumes its own iBOT pretrain via the HF Trainer checkpoint +
# MLflow run_id stored in results/pretrain_<arch>/. The two chains are independent
# (different output_dir, different MLflow run).
#
# Why interleave? On a single 2x3090 node, sbatch FIFO would run all of A's chain
# (8 × 30h = 240h) before B starts. Interleaving guarantees both gros modèles
# progressent en parallèle : vith16plus et sapiens2_0.8b alternent leurs créneaux.
#
# Usage:
#   ./scripts/chain_pretrain_two.sh                                         # 8 rounds, vith16plus + sapiens 0.8b (defaults)
#   ./scripts/chain_pretrain_two.sh 4                                       # 4 rounds (lighter budget)
#   ./scripts/chain_pretrain_two.sh 6 scripts/pretrain_ibot_vith16plus_2x3090.sh scripts/pretrain_ibot_sapiens2_08b_2x3090.sh
#
# Targeting MAX_STEPS=100000 @224 → ~8 links per arch needed pour aller au bout.
# Stop-early possible : snapshots intermédiaires (encoder_25000/50000/75000) sont
# logés dans MLflow → scancel quand le sweep Optuna révèle qu'un snapshot précoce suffit.

set -e

N="${1:-8}"
SCRIPT_A="${2:-scripts/pretrain_ibot_vith16plus_2x3090.sh}"
SCRIPT_B="${3:-scripts/pretrain_ibot_sapiens2_08b_2x3090.sh}"

for s in "$SCRIPT_A" "$SCRIPT_B"; do
    if [ ! -f "$s" ]; then
        echo "ERROR: script not found: $s"
        exit 1
    fi
done

echo "Interleaved pretrain chain: $N rounds of"
echo "  A = $SCRIPT_A"
echo "  B = $SCRIPT_B"
echo ""

PREV_A=""
PREV_B=""
for i in $(seq 1 "$N"); do
    # Submit A_i (depends on A_{i-1} if exists)
    if [ -z "$PREV_A" ]; then
        A=$(sbatch --parsable "$SCRIPT_A")
        echo "  [A$i] $A  (no dependency)"
    else
        A=$(sbatch --parsable --dependency=afterany:"$PREV_A" "$SCRIPT_A")
        echo "  [A$i] $A  (depends on A$((i-1))=$PREV_A)"
    fi
    PREV_A="$A"

    # Submit B_i (depends on B_{i-1} if exists)
    if [ -z "$PREV_B" ]; then
        B=$(sbatch --parsable "$SCRIPT_B")
        echo "  [B$i] $B  (no dependency)"
    else
        B=$(sbatch --parsable --dependency=afterany:"$PREV_B" "$SCRIPT_B")
        echo "  [B$i] $B  (depends on B$((i-1))=$PREV_B)"
    fi
    PREV_B="$B"
done

JOB_NAME_A=$(grep -m1 -E '^#SBATCH --job-name=' "$SCRIPT_A" | sed 's/.*=//')
JOB_NAME_B=$(grep -m1 -E '^#SBATCH --job-name=' "$SCRIPT_B" | sed 's/.*=//')

echo ""
echo "Chains submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f scripts/logs/${JOB_NAME_A}_<jobid>.out"
echo "  tail -f scripts/logs/${JOB_NAME_B}_<jobid>.out"
echo ""
echo "Cancel everything:  scancel \$(squeue -u \$USER -h -o '%i' | tr '\\n' ' ')"
