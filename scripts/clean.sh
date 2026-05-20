#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Cleaning project artifacts..."

rm -f scripts/logs/*.out scripts/logs/*.err
rm -rf results/
rm -rf mlruns/
rm -f optuna.db mlflow.db optuna.db-* mlflow.db-*
rm -rf data/folds/
rm -rf configs/architectures/optuna_trials/

echo "Done."
