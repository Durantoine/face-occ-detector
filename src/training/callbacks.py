from typing import Any, Dict, Optional

from mlflow.tracking import MlflowClient
from transformers import TrainerCallback


class MlflowClientCallback(TrainerCallback):
    """Pipe HF Trainer logged metrics → MLflow via direct client (vs report_to='mlflow')."""

    def __init__(self, client: MlflowClient, run_id: str) -> None:
        self.client = client
        self.run_id = run_id

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict] = None, **kwargs: Any) -> None:
        for key, value in (logs or {}).items():
            if isinstance(value, (int, float)):
                try:
                    self.client.log_metric(self.run_id, key, value, step=state.global_step)
                except Exception:
                    pass
