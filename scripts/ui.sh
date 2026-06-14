#!/bin/bash
#SBATCH --job-name=ui-mlflow-optuna
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00

set -e

MLFLOW_PORT="${MLFLOW_PORT:-5000}"
OPTUNA_PORT="${OPTUNA_PORT:-8080}"
QUAL_PORT="${QUAL_PORT:-8501}"
PROJECT_DIR="${HOME}/face-occ-detector"
UI_VENV="${HOME}/.face_occ_ui_venv"
cd "${PROJECT_DIR}"

# Pin the UI's mlflow to the TRAINING version (from uv.lock) so the viewer never errors with
# "Detected out-of-date database schema" against a db written by a different mlflow.
ML_VER=$(grep -A1 'name = "mlflow"' uv.lock 2>/dev/null | grep -m1 'version' | sed -E 's/.*"([^"]+)".*/\1/')
[ -z "$ML_VER" ] && ML_VER="3.12.0"
MLFLOW_SPEC="mlflow==${ML_VER} mlflow-skinny==${ML_VER} mlflow-tracing==${ML_VER}"  # pin the FULL stack (skinny/tracing set the DB schema)

# === Persistent UI venv (created once, reused across SLURM jobs) ===
# Avoids re-downloading streamlit + mlflow + pandas + pillow on every job and
# avoids the uvx ephemeral-env failure mode on CPU nodes without internet.
if [ ! -x "$UI_VENV/bin/streamlit" ]; then
    echo "Bootstrapping UI venv at $UI_VENV (first time only — ~2 min)"
    rm -rf "$UI_VENV"
    PYBIN=$(command -v python3.12 || command -v python3.11 || command -v python3)
    if [ -z "$PYBIN" ]; then
        echo "FATAL: no python3 found in PATH" >&2; exit 1
    fi
    echo "  using $PYBIN"
    "$PYBIN" -m venv "$UI_VENV"
    "$UI_VENV/bin/pip" install --quiet --upgrade pip
    "$UI_VENV/bin/pip" install --quiet streamlit ${MLFLOW_SPEC} optuna-dashboard pandas pillow plotly streamlit-autorefresh
    echo "  UI venv ready: $($UI_VENV/bin/streamlit --version) | mlflow ${ML_VER}"
fi

# Re-align mlflow if an existing UI venv has a different version than the training lock (the usual
# cause of "out-of-date database schema": UI venv installed latest mlflow, db written by 3.12.0).
UI_ML=$("$UI_VENV/bin/pip" show mlflow-skinny 2>/dev/null | awk '/^Version/{print $2}')
if [ -n "$UI_ML" ] && [ "$UI_ML" != "$ML_VER" ]; then
    echo "  Aligning UI mlflow stack $UI_ML -> $ML_VER (skinny/tracing set the DB schema)"
    "$UI_VENV/bin/pip" install --quiet ${MLFLOW_SPEC}
fi

# Ensure plotly is present (older UI venvs from before the trials-comparison feature
# may not have it). Cheap no-op if already installed.
if ! "$UI_VENV/bin/python" -c "import plotly" 2>/dev/null; then
    echo "  Installing plotly into UI venv (one-time)"
    "$UI_VENV/bin/pip" install --quiet plotly
fi
# Same for streamlit-autorefresh (non-blocking JS-timer reruns).
if ! "$UI_VENV/bin/python" -c "import streamlit_autorefresh" 2>/dev/null; then
    echo "  Installing streamlit-autorefresh into UI venv (one-time)"
    "$UI_VENV/bin/pip" install --quiet streamlit-autorefresh
fi

echo "================================================================================"
echo "Combined UIs (MLflow + Optuna + Qualitative viewer) — node $(hostname)"
echo "  MLflow      port: $MLFLOW_PORT   (sqlite:///mlflow.db)"
echo "  Optuna      port: $OPTUNA_PORT   (sqlite:///optuna.db)"
echo "  Analytics   port: $QUAL_PORT    (trials comparison live + qualitative viewer)"
echo "Job ID: $SLURM_JOB_ID | Started: $(date)"
echo ""
echo "Access from laptop:"
echo "  ssh -N -L ${MLFLOW_PORT}:$(hostname):${MLFLOW_PORT} \\"
echo "         -L ${OPTUNA_PORT}:$(hostname):${OPTUNA_PORT} \\"
echo "         -L ${QUAL_PORT}:$(hostname):${QUAL_PORT} adurand-25@gpu-gw"
echo "================================================================================"

mkdir -p $HOME/tmp scripts/logs
export OMP_NUM_THREADS=1
export TMPDIR=$HOME/tmp
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false
export MLFLOW_DEPLOYMENTS_TARGET=''
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
# Streamlit config: avoid version-check (needs internet), keep config in $HOME
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
mkdir -p "$HOME/.streamlit"

# Port-conflict detection — pick the next free port if the default is taken.
# (Common on shared CPU nodes where multiple users run dashboards.)
find_free_port() {
    local start_port="$1"
    for offset in 0 1 2 3 4 5 6 7 8 9; do
        local port=$((start_port + offset))
        if ! (echo > /dev/tcp/127.0.0.1/$port) 2>/dev/null; then
            echo "$port"; return 0
        fi
    done
    echo "$start_port"  # fallback: use requested port anyway
}
MLFLOW_PORT=$(find_free_port "$MLFLOW_PORT")
OPTUNA_PORT=$(find_free_port "$OPTUNA_PORT")
QUAL_PORT=$(find_free_port "$QUAL_PORT")
echo "  Resolved ports: MLflow=$MLFLOW_PORT  Optuna=$OPTUNA_PORT  Qualitative=$QUAL_PORT"

# Pre-flight: sqlite DBs exist?
if [ ! -f mlflow.db ]; then
    echo "WARNING: mlflow.db not found in $(pwd) — MLflow UI will start empty"
fi
if [ ! -f optuna.db ]; then
    echo "WARNING: optuna.db not found in $(pwd) — Optuna dashboard will show no studies"
fi

# Start all three UIs in background, log to per-process files
ML_LOG="scripts/logs/ui-mlflow_${SLURM_JOB_ID}.log"
OP_LOG="scripts/logs/ui-optuna_${SLURM_JOB_ID}.log"
QU_LOG="scripts/logs/ui-qualitative_${SLURM_JOB_ID}.log"

"$UI_VENV/bin/mlflow" server \
    --backend-store-uri sqlite:///mlflow.db \
    --host 0.0.0.0 --port "${MLFLOW_PORT}" --workers 1 \
    > "$ML_LOG" 2>&1 &
ML_PID=$!
echo "  → MLflow started (pid=$ML_PID, log=$ML_LOG)"

# Optuna dashboard isn't in the shared UI venv (separate dep). Install on demand.
if [ ! -x "$UI_VENV/bin/optuna-dashboard" ]; then
    echo "  Installing optuna-dashboard into UI venv (first time only)..."
    "$UI_VENV/bin/pip" install --quiet optuna-dashboard
fi
"$UI_VENV/bin/optuna-dashboard" sqlite:///optuna.db \
    --host 0.0.0.0 --port "${OPTUNA_PORT}" \
    > "$OP_LOG" 2>&1 &
OP_PID=$!
echo "  → Optuna started (pid=$OP_PID, log=$OP_LOG)"

"$UI_VENV/bin/streamlit" run scripts/qualitative_viewer.py \
    --server.port "${QUAL_PORT}" --server.address 0.0.0.0 \
    --server.headless true \
    > "$QU_LOG" 2>&1 &
QU_PID=$!
echo "  → Qualitative viewer started (pid=$QU_PID, log=$QU_LOG)"

# Small delay then check each process actually started (catches immediate crashes)
sleep 3
for entry in "MLflow:$ML_PID:$ML_LOG" "Optuna:$OP_PID:$OP_LOG" "Qualitative:$QU_PID:$QU_LOG"; do
    name="${entry%%:*}"
    rest="${entry#*:}"
    pid="${rest%%:*}"
    log="${rest##*:}"
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "  ✗ $name died within 3s — see error below:"
        tail -20 "$log" 2>/dev/null | sed 's/^/      /'
    else
        echo "  ✓ $name alive (pid=$pid)"
    fi
done

# Clean shutdown only on external SIGTERM/SIGINT (e.g. scancel) — NOT on child exit.
# If one of the 3 UIs crashes, the other two keep running.
trap "echo 'Shutting down (external signal received)...'; kill $ML_PID $OP_PID $QU_PID 2>/dev/null; wait" TERM INT

# Wait for ALL three processes. If one dies, log it and keep the others alive.
# The script only exits when (a) scancel is sent, or (b) all 3 children have died.
while true; do
    sleep 30
    for entry in "MLflow:$ML_PID:$ML_LOG" "Optuna:$OP_PID:$OP_LOG" "Qualitative:$QU_PID:$QU_LOG"; do
        name="${entry%%:*}"
        rest="${entry#*:}"
        pid="${rest%%:*}"
        log="${rest##*:}"
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] $name (pid=$pid) is DEAD — last 10 lines of $log:"
            tail -10 "$log" 2>/dev/null | sed 's/^/    /'
            echo "    → other UIs still running; restart only $name with: scancel $SLURM_JOB_ID && sbatch scripts/ui.sh"
            # Mark as dead so we don't log again on next iteration
            case "$name" in
                MLflow)      ML_PID=0 ;;
                Optuna)      OP_PID=0 ;;
                Qualitative) QU_PID=0 ;;
            esac
        fi
    done
    # If all 3 dead, exit
    if [ "$ML_PID" = "0" ] && [ "$OP_PID" = "0" ] && [ "$QU_PID" = "0" ]; then
        echo "All UI processes have exited. Bye."
        break
    fi
done
