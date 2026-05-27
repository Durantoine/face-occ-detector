from typing import Any, Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient


def get_or_create_experiment(client: MlflowClient, name: str) -> str:
    try:
        return client.create_experiment(name)
    except Exception:
        exp = client.get_experiment_by_name(name)
        if exp:
            # Auto-restore if soft-deleted via UI/API. Otherwise create_run() raises
            # MlflowException("must be in 'active' state. Current state is deleted.")
            if getattr(exp, "lifecycle_stage", "active") == "deleted":
                try:
                    client.restore_experiment(exp.experiment_id)
                    print(f"[mlflow] restored soft-deleted experiment '{name}' (id={exp.experiment_id})")
                except Exception as e:
                    print(f"[mlflow] WARNING: could not restore experiment '{name}': {e}")
            return exp.experiment_id
        return client.create_experiment(name)


def log_params(client: Optional[MlflowClient], run_id: Optional[str], params: Dict[str, Any]) -> None:
    if client and run_id:
        for k, v in params.items():
            client.log_param(run_id, k, v)
    elif mlflow.active_run():
        mlflow.log_params(params)


def log_metrics(client: Optional[MlflowClient], run_id: Optional[str], metrics: Dict[str, float]) -> None:
    if client and run_id:
        for k, v in metrics.items():
            client.log_metric(run_id, k, v)
    elif mlflow.active_run():
        mlflow.log_metrics(metrics)
