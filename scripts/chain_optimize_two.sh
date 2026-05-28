#!/bin/bash
# Chain N rounds of TWO Optuna optimize scripts in INTERLEAVED order:
#   A1, B1, A2(after A1), B2(after B1), A3(after A2), B3(after B2), ...
#
# Each chain (A_i, B_i) resumes its own Optuna study via load_if_exists=True on
# sqlite:///optuna.db. The two studies are separate (different `architecture`).
#
# Why interleave? On a single-node cluster, sbatch FIFO would run A1→A2→A3→B1→...
# (all of A's chain before B even starts). Interleaving guarantees both archs get
# cluster time alternately — vitb16 and sapiens 0.1b progress in parallel.
#
# Usage:
#   ./scripts/chain_optimize_two.sh                                        # 3 rounds, vitb16+sapiens01b v8 (defaults)
#   ./scripts/chain_optimize_two.sh 3
#   ./scripts/chain_optimize_two.sh 4 scripts/optimize_dinov3_vitb16_v8_2x3090.sh scripts/optimize_sapiens2_01b_v8_2x3090.sh

set -e

N="${1:-3}"
SCRIPT_A="${2:-scripts/optimize_dinov3_vitb16_v8_2x3090.sh}"
SCRIPT_B="${3:-scripts/optimize_sapiens2_01b_v8_2x3090.sh}"

for s in "$SCRIPT_A" "$SCRIPT_B"; do
    if [ ! -f "$s" ]; then
        echo "ERROR: script not found: $s"
        exit 1
    fi
done

echo "Interleaved chain: $N rounds of"
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
