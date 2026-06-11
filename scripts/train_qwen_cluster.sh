#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Qwen fine-tuning — SLURM script for ecole cluster
#
# Usage:
#   # Optuna sweep on L40S (2 GPU, respectueux des autres étudiants)
#   sbatch scripts/train_qwen_cluster.sh --partition ecole-l40s --gpus 2 --sweep
#
#   # Run final 72B on H100 (2 GPU max)
#   sbatch scripts/train_qwen_cluster.sh --partition ecole-h100 --gpus 2 --config configs/architectures/qwen-72b-h100-v1.yaml
#
# ─────────────────────────────────────────────────────────────────────────────

# ── SLURM defaults (overridable via sbatch args) ──────────────────────────────
#SBATCH --job-name=qwen-finetune
#SBATCH --partition=ecole-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/qwen_finetune_%j.out
#SBATCH --error=logs/qwen_finetune_%j.err

set -euo pipefail

# ── Parse args ────────────────────────────────────────────────────────────────
PARTITION="ecole-l40s"
GPUS=2
CONFIG="configs/architectures/qwen-7b-l40s-v1.yaml"
SWEEP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --partition) PARTITION="$2"; shift 2 ;;
        --gpus)      GPUS="$2";      shift 2 ;;
        --config)    CONFIG="$2";    shift 2 ;;
        --sweep)     SWEEP=true;     shift ;;
        *)           shift ;;
    esac
done

# ── Environment ───────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs results/qwen_finetune

echo "========================================"
echo "Job:       $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Partition: $PARTITION"
echo "GPUs:      $GPUS"
echo "Config:    $CONFIG"
echo "Sweep:     $SWEEP"
echo "========================================"

# ── Check GPU ─────────────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Setup Python env ──────────────────────────────────────────────────────────
if command -v uv &>/dev/null; then
    echo "Using uv..."
    uv sync --frozen
    PYTHON="uv run python"
else
    echo "Using pip venv..."
    python -m pip install -q peft trl bitsandbytes accelerate
    PYTHON="python"
fi

# ── DeepSpeed config (ZeRO-3 pour 72B, ZeRO-2 pour 7B) ─────────────────────
if [[ "$CONFIG" == *"72b"* ]]; then
    DS_STAGE=3
    OFFLOAD="false"
else
    DS_STAGE=2
    OFFLOAD="false"
fi

cat > /tmp/ds_config_$SLURM_JOB_ID.json << EOF
{
  "zero_optimization": {
    "stage": $DS_STAGE,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "offload_optimizer": {"device": "none"},
    "offload_param": {"device": "none"}
  },
  "bf16": {"enabled": true},
  "gradient_clipping": 1.0,
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
EOF

# ── Launch ────────────────────────────────────────────────────────────────────
SWEEP_FLAG=""
if $SWEEP; then
    SWEEP_FLAG="--sweep"
fi

ACCELERATE_CMD="accelerate launch \
    --num_processes $GPUS \
    --mixed_precision bf16 \
    --dynamo_backend no"

# Add DeepSpeed only for 72B (overhead not worth it for 7B with 2 GPU)
if [[ "$CONFIG" == *"72b"* ]]; then
    ACCELERATE_CMD="$ACCELERATE_CMD \
        --use_deepspeed \
        --deepspeed_config_file /tmp/ds_config_$SLURM_JOB_ID.json"
fi

echo "Launching training..."
$PYTHON -m accelerate.commands.launch \
    --num_processes "$GPUS" \
    --mixed_precision bf16 \
    src/qwen_finetune.py \
    --config "$CONFIG" \
    --mlflow-uri "sqlite:///mlflow.db" \
    $SWEEP_FLAG

echo "Done. Job $SLURM_JOB_ID finished."
