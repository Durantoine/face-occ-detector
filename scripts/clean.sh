#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Cleaning project artifacts..."

rm -f scripts/*.out scripts/*.err
rm -rf scripts/logs/
rm -rf results/
rm -rf mlruns/
rm -f optuna.db mlflow.db

echo "Done."
