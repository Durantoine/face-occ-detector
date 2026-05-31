from typing import Any, Dict, Optional

from mlflow.tracking import MlflowClient
from transformers import TrainerCallback


class MlflowClientCallback(TrainerCallback):
    """Pipe HF Trainer logged metrics → MLflow via direct client. RANK 0 ONLY
    (avoids sqlite lock contention with multiple jobs)."""

    def __init__(self, client: MlflowClient, run_id: str) -> None:
        self.client = client
        self.run_id = run_id

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict] = None, **kwargs: Any) -> None:
        if not getattr(state, "is_world_process_zero", True):
            return
        items = logs or {}
        # During double-eval (raw + EMA), the chosen-min eval_* series is kept in the
        # returned dict so HF Trainer can track best model — but we skip it from MLflow
        # to avoid a 3rd zigzag plot superposed with ema_*/noema_*.
        drop_eval = "_double_eval_marker" in items
        for key, value in items.items():
            if key == "_double_eval_marker":
                continue
            if drop_eval and key.startswith("eval_") and not key.startswith("eval_chose_"):
                continue
            if isinstance(value, (int, float)):
                try:
                    self.client.log_metric(self.run_id, key, value, step=state.global_step)
                except Exception as e:
                    print(f"[mlflow] WARNING: log_metric '{key}' failed: {e}", flush=True)
