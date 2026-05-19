from typing import Any, Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient


def get_or_create_experiment(client: MlflowClient, name: str) -> str:
    try:
        return client.create_experiment(name)
    except Exception:
        exp = client.get_experiment_by_name(name)
        if exp:
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
