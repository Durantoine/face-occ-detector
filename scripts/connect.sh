#!/bin/bash

CLUSTER_HOST="cluster"
CLUSTER_DIR="face-occ-detector"
MLFLOW_PORT=5000
OPTUNA_PORT=8080

echo "Connexion au cluster et démarrage des dashboards..."

(
    sleep 4
    open "http://localhost:${MLFLOW_PORT}" 2>/dev/null
    open "http://localhost:${OPTUNA_PORT}" 2>/dev/null
) &

echo "Établissement des tunnels SSH..."
echo "   MLflow:  http://localhost:${MLFLOW_PORT}"
echo "   Optuna:  http://localhost:${OPTUNA_PORT}"
echo ""

ssh -t \
    -L ${MLFLOW_PORT}:localhost:${MLFLOW_PORT} \
    -L ${OPTUNA_PORT}:localhost:${OPTUNA_PORT} \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    ${CLUSTER_HOST} \
    "cd ~/${CLUSTER_DIR} && source .venv/bin/activate && mkdir -p logs && \
     (pgrep -f 'mlflow ui' > /dev/null || (echo 'Démarrage MLflow...' && nohup env OMP_NUM_THREADS=1 TMPDIR=\$HOME/tmp MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false MLFLOW_DEPLOYMENTS_TARGET='' mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port ${MLFLOW_PORT} > logs/mlflow_ui.log 2>&1 &)) && \
     (pgrep -f 'optuna_dashboard' > /dev/null || (echo 'Démarrage Optuna...' && nohup env OMP_NUM_THREADS=1 optuna-dashboard sqlite:///optuna.db --host 0.0.0.0 --port ${OPTUNA_PORT} > logs/optuna_dashboard.log 2>&1 &)) && \
     sleep 1 && echo 'Dashboards prets' && exec bash"
