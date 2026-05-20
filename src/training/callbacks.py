from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from mlflow.tracking import MlflowClient
from transformers import TrainerCallback


class MlflowClientCallback(TrainerCallback):
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


class EMACallback(TrainerCallback):
    def __init__(self, decay: float = 0.9998, warmup_steps: int = 200) -> None:
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.shadow: Dict[str, torch.Tensor] = {}
        self._active = False

    def _effective_decay(self, step: int) -> float:
        if step < self.warmup_steps:
            return min(self.decay, (1 + step) / (10 + step))
        return self.decay

    def on_train_begin(self, args: Any, state: Any, control: Any, model: Optional[nn.Module] = None, **kwargs: Any) -> None:
        if model is None:
            return
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self._active = True

    def on_step_end(self, args: Any, state: Any, control: Any, model: Optional[nn.Module] = None, **kwargs: Any) -> None:
        if not self._active or model is None:
            return
        d = self._effective_decay(state.global_step)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow and param.requires_grad:
                    self.shadow[name].mul_(d).add_(param.detach(), alpha=1 - d)

    def on_train_end(self, args: Any, state: Any, control: Any, model: Optional[nn.Module] = None, **kwargs: Any) -> None:
        if not self._active or model is None:
            return
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.data.copy_(self.shadow[name])
        print(f"[EMA] Swapped EMA weights into model (decay={self.decay})")


def make_ema_callback_from_cfg(cfg: Dict[str, Any]) -> Optional[EMACallback]:
    decay = float(cfg.get("ema_decay", 0.0))
    if decay <= 0:
        return None
    warmup = int(cfg.get("ema_warmup_steps", 200))
    return EMACallback(decay=decay, warmup_steps=warmup)
