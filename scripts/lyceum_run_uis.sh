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
MLFLOW_DB="${MLFLOW_DB:-$(pwd)/mlflow.db}"   # point at a snapshot (e.g. /tmp/mlflow_ui.db) to avoid 'database is locked' during a sweep
OPTUNA_DB="${OPTUNA_DB:-$(pwd)/optuna.db}"

# Overrides via CLI flags (env vars MLFLOW_PORT/OPTUNA_PORT/QUAL_PORT/MLFLOW_DB/OPTUNA_DB also work):
#   bash scripts/lyceum_run_uis.sh --mlflow-port 6000 --mlflow-db /tmp/mlflow_ui.db
while [ $# -gt 0 ]; do
    case "$1" in
        --mlflow-port) MLFLOW_PORT="$2"; shift 2 ;;
        --optuna-port) OPTUNA_PORT="$2"; shift 2 ;;
        --qual-port)   QUAL_PORT="$2"; shift 2 ;;
        --mlflow-db)   MLFLOW_DB="$2"; shift 2 ;;
        --optuna-db)   OPTUNA_DB="$2"; shift 2 ;;
        --snapshot)    SNAPSHOT=1; shift ;;
        -h|--help) echo "usage: $0 [--mlflow-port P] [--optuna-port P] [--qual-port P] [--snapshot] [--mlflow-db PATH] [--optuna-db PATH]"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# --snapshot: read COPIES of the dbs (in /tmp) instead of the live ones -> no "database is locked"
# while a sweep is writing. Snapshots are frozen at launch; re-run the script to refresh.
if [ "${SNAPSHOT:-0}" = "1" ]; then
    cp "$MLFLOW_DB" /tmp/ui_mlflow.db 2>/dev/null && MLFLOW_DB=/tmp/ui_mlflow.db
    cp "$OPTUNA_DB" /tmp/ui_optuna.db 2>/dev/null && OPTUNA_DB=/tmp/ui_optuna.db
    echo "  snapshot mode: UIs read /tmp/ui_*.db (lock-free; re-run to refresh)"
fi

VENV_DIR="${HOME}/face-occ-detector/.venv"
UI_VENV="${HOME}/.face_occ_ui_venv"
LOG_DIR="${HOME}/face-occ-detector/scripts/logs"
mkdir -p "$LOG_DIR"

# === UI venv (lightweight, persists across launches) ===
# Pin the UI's mlflow STACK to the training version (from uv.lock) — avoids "out-of-date database
# schema". skinny/tracing ALSO set the schema, so all THREE must match (not just mlflow).
ML_VER=$(grep -A1 'name = "mlflow"' "${HOME}/face-occ-detector/uv.lock" 2>/dev/null | grep -m1 'version' | sed -E 's/.*"([^"]+)".*/\1/')
[ -z "$ML_VER" ] && ML_VER="3.12.0"
MLFLOW_PKGS="mlflow==${ML_VER} mlflow-skinny==${ML_VER} mlflow-tracing==${ML_VER}"

if [ ! -x "$UI_VENV/bin/streamlit" ]; then
    echo "Bootstrapping UI venv (first time — ~2 min)..."
    PYBIN=$(command -v python3.12 || command -v python3.11 || command -v python3)
    [ -z "$PYBIN" ] && { echo "no python3 in PATH"; exit 1; }
    "$PYBIN" -m venv "$UI_VENV"
    "$UI_VENV/bin/pip" install --quiet --upgrade pip
    "$UI_VENV/bin/pip" install --quiet ${MLFLOW_PKGS} streamlit optuna-dashboard pandas pillow plotly streamlit-autorefresh
    echo "  UI venv ready (mlflow ${ML_VER} stack)"
else
    # Venv exists — re-align the FULL mlflow stack if it drifted (skinny sets the schema)
    UI_ML=$("$UI_VENV/bin/pip" show mlflow-skinny 2>/dev/null | awk '/^Version/{print $2}')
    if [ -n "$UI_ML" ] && [ "$UI_ML" != "$ML_VER" ]; then
        echo "Re-aligning UI mlflow stack: $UI_ML → $ML_VER (avoid DB schema mismatch)"
        "$UI_VENV/bin/pip" install --quiet ${MLFLOW_PKGS}
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
    --backend-store-uri "sqlite:///${MLFLOW_DB}" \
    --host 0.0.0.0 \
    --port "$MLFLOW_PORT" \
    > "$LOG_DIR/ui-mlflow.log" 2>&1 &
MLFLOW_PID=$!
echo "  MLflow PID=$MLFLOW_PID, log → $LOG_DIR/ui-mlflow.log"

# 2. Optuna dashboard
if [ ! -f "$OPTUNA_DB" ]; then
    echo "  ⚠ $OPTUNA_DB missing yet (sweep not started?). UI will be empty until sweep creates it."
fi
# Persistent UI venvs created before optuna-dashboard was in the bootstrap won't have it;
# the 'venv exists' branch above only re-aligns mlflow -> without this the launch fails silently.
if [ ! -x "$UI_VENV/bin/optuna-dashboard" ]; then
    echo "  optuna-dashboard missing from UI venv — installing (needs internet)..."
    "$UI_VENV/bin/pip" install --quiet optuna-dashboard || echo "  ⚠ optuna-dashboard install FAILED (offline?) — dashboard will not start"
fi
nohup "$UI_VENV/bin/optuna-dashboard" "sqlite:///${OPTUNA_DB}" \
    --host 0.0.0.0 \
    --port "$OPTUNA_PORT" \
    > "$LOG_DIR/ui-optuna.log" 2>&1 &
OPTUNA_PID=$!
echo "  Optuna PID=$OPTUNA_PID, log → $LOG_DIR/ui-optuna.log"

# 3. Qualitative viewer (Streamlit)
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export FACE_OCC_TRACKING_URI="sqlite:///${MLFLOW_DB}"
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
