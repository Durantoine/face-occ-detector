import os
from typing import Any, Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient


def _is_rank_zero() -> bool:
    """True if we're rank 0 (or non-distributed). All MLflow writes guarded so that
    multi-rank DDP doesn't duplicate writes → reduces sqlite lock contention when
    multiple jobs share the same mlflow.db."""
    return int(os.environ.get("RANK", "0")) == 0


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
    if not _is_rank_zero():
        return
    if client and run_id:
        for k, v in params.items():
            client.log_param(run_id, k, v)
    elif mlflow.active_run():
        mlflow.log_params(params)


def log_metrics(client: Optional[MlflowClient], run_id: Optional[str], metrics: Dict[str, float], step: Optional[int] = None) -> None:
    if not _is_rank_zero():
        return
    if client and run_id:
        for k, v in metrics.items():
            client.log_metric(run_id, k, v, step=step) if step is not None else client.log_metric(run_id, k, v)
    elif mlflow.active_run():
        mlflow.log_metrics(metrics, step=step) if step is not None else mlflow.log_metrics(metrics)
