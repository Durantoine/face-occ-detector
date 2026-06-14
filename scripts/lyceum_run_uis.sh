#!/bin/bash
# Launch MLflow + Optuna + Qualitative UIs on a Lyceum VM (non-SLURM).
#
# Reads MLFLOW_PORT, OPTUNA_PORT, QUAL_PORT env vars (defaults : 5050, 8090, 8511).
# All 3 UIs run in background; use `jobs` / `pkill -f 'mlflow ui'` to manage.
#
# Usage (from another SSH session into the VM, while training runs in the 1st session):
#   cd ~/face-occ-detector
#   bash scripts/lyceum_run_uis.sh

set -euo pipefail
cd "$(dirname "$0")/.."

MLFLOW_PORT="${MLFLOW_PORT:-5050}"
OPTUNA_PORT="${OPTUNA_PORT:-8090}"
QUAL_PORT="${QUAL_PORT:-8511}"

VENV_DIR="${HOME}/face-occ-detector/.venv"
UI_VENV="${HOME}/.face_occ_ui_venv"
LOG_DIR="${HOME}/face-occ-detector/scripts/logs"
mkdir -p "$LOG_DIR"

# === UI venv (lightweight, persists across launches) ===
# Detect mlflow version of training venv FIRST so we install the exact same in UI venv.
# Avoids schema mismatch ("Detected out-of-date database schema") when training venv is
# older than the latest mlflow that pip would install by default in the UI venv.
TRAIN_MLFLOW_VER=""
if [ -x "$VENV_DIR/bin/mlflow" ]; then
    TRAIN_MLFLOW_VER=$("$VENV_DIR/bin/mlflow" --version 2>/dev/null | awk '{print $NF}')
    echo "Detected training venv mlflow version: ${TRAIN_MLFLOW_VER:-<unknown>}"
fi
MLFLOW_PIN=""
[ -n "$TRAIN_MLFLOW_VER" ] && MLFLOW_PIN="==$TRAIN_MLFLOW_VER"

if [ ! -x "$UI_VENV/bin/streamlit" ]; then
    echo "Bootstrapping UI venv (first time — ~2 min)..."
    PYBIN=$(command -v python3.12 || command -v python3.11 || command -v python3)
    [ -z "$PYBIN" ] && { echo "no python3 in PATH"; exit 1; }
    "$PYBIN" -m venv "$UI_VENV"
    "$UI_VENV/bin/pip" install --quiet --upgrade pip
    "$UI_VENV/bin/pip" install --quiet "mlflow${MLFLOW_PIN}" streamlit optuna-dashboard pandas pillow plotly streamlit-autorefresh
    echo "  UI venv ready (mlflow${MLFLOW_PIN:-<unpinned>})"
else
    # Venv exists — re-align mlflow if version mismatch with training
    UI_MLFLOW_VER=$("$UI_VENV/bin/mlflow" --version 2>/dev/null | awk '{print $NF}')
    if [ -n "$TRAIN_MLFLOW_VER" ] && [ "$UI_MLFLOW_VER" != "$TRAIN_MLFLOW_VER" ]; then
        echo "Re-aligning UI mlflow: $UI_MLFLOW_VER → $TRAIN_MLFLOW_VER (avoid schema mismatch with training DB)"
        "$UI_VENV/bin/pip" install --quiet "mlflow==$TRAIN_MLFLOW_VER"
    fi
fi

# Kill any previous UI process on these ports
for port in $MLFLOW_PORT $OPTUNA_PORT $QUAL_PORT; do
    if lsof -ti :$port >/dev/null 2>&1; then
        echo "  Killing existing process on port $port"
        lsof -ti :$port | xargs kill -9 2>/dev/null || true
    fi
done

echo "================================================================================"
echo "Launching 3 UIs in background — Lyceum $(hostname)"
echo "  MLflow      → port $MLFLOW_PORT  (sqlite:///mlflow.db)"
echo "  Optuna      → port $OPTUNA_PORT  (sqlite:///optuna.db)"
echo "  Qualitative → port $QUAL_PORT    (streamlit, scripts/qualitative_viewer.py)"
echo "================================================================================"

# 1. MLflow UI
nohup "$UI_VENV/bin/mlflow" ui \
    --backend-store-uri sqlite:///mlflow.db \
    --host 0.0.0.0 \
    --port "$MLFLOW_PORT" \
    > "$LOG_DIR/ui-mlflow.log" 2>&1 &
MLFLOW_PID=$!
echo "  MLflow PID=$MLFLOW_PID, log → $LOG_DIR/ui-mlflow.log"

# 2. Optuna dashboard
if [ ! -f optuna.db ]; then
    echo "  ⚠ optuna.db missing yet (sweep not started?). UI will be empty until sweep creates it."
fi
# Persistent UI venvs created before optuna-dashboard was in the bootstrap won't have it;
# the 'venv exists' branch above only re-aligns mlflow -> without this the launch fails silently.
if [ ! -x "$UI_VENV/bin/optuna-dashboard" ]; then
    echo "  optuna-dashboard missing from UI venv — installing (needs internet)..."
    "$UI_VENV/bin/pip" install --quiet optuna-dashboard || echo "  ⚠ optuna-dashboard install FAILED (offline?) — dashboard will not start"
fi
nohup "$UI_VENV/bin/optuna-dashboard" sqlite:///optuna.db \
    --host 0.0.0.0 \
    --port "$OPTUNA_PORT" \
    > "$LOG_DIR/ui-optuna.log" 2>&1 &
OPTUNA_PID=$!
echo "  Optuna PID=$OPTUNA_PID, log → $LOG_DIR/ui-optuna.log"

# 3. Qualitative viewer (Streamlit)
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export FACE_OCC_TRACKING_URI="sqlite:///$(pwd)/mlflow.db"
nohup "$UI_VENV/bin/streamlit" run scripts/qualitative_viewer.py \
    --server.port "$QUAL_PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    > "$LOG_DIR/ui-qualitative.log" 2>&1 &
QUAL_PID=$!
echo "  Qualitative PID=$QUAL_PID, log → $LOG_DIR/ui-qualitative.log"

sleep 3

# Detect VM IP for the tunnel command
TUNNEL_TARGET=""
if [ -n "${SSH_CONNECTION:-}" ]; then
    TUNNEL_TARGET=$(echo "$SSH_CONNECTION" | awk '{print $3}')
fi
[ -z "$TUNNEL_TARGET" ] && TUNNEL_TARGET=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$TUNNEL_TARGET" ] && TUNNEL_TARGET="LYCEUM_VM_IP"
VM_USER="${USER:-lyceum}"

echo ""
echo "================================================================================"
echo "  ✓ 3 UIs RUNNING  —  copy-paste the tunnel command below on your Mac"
echo "================================================================================"
echo ""
echo "  --- SSH tunnel command (paste in a new Mac terminal) ---"
echo ""
echo "ssh -fNT -L ${MLFLOW_PORT}:localhost:${MLFLOW_PORT} -L ${OPTUNA_PORT}:localhost:${OPTUNA_PORT} -L ${QUAL_PORT}:localhost:${QUAL_PORT} ${VM_USER}@${TUNNEL_TARGET}"
echo ""
echo "  --- Then open in your Mac browser ---"
echo ""
echo "  http://localhost:${MLFLOW_PORT}   MLflow"
echo "  http://localhost:${OPTUNA_PORT}   Optuna dashboard"
echo "  http://localhost:${QUAL_PORT}   Qualitative + post-processing viewer"
echo ""
echo "  --- Cleanup ---"
echo ""
echo "  Stop tunnel (Mac) : pkill -f 'ssh -fNT.*${VM_USER}@${TUNNEL_TARGET}'"
echo "  Stop UIs (VM)     : kill ${MLFLOW_PID} ${OPTUNA_PID} ${QUAL_PID}"
echo "                       (logs : scripts/logs/ui-*.log)"
echo "================================================================================"
