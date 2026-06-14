from __future__ import annotations

import torch
import torch.nn as nn

WEIGHT_OFFSET = 1.0 / 30.0
ASYM_MAX, ASYM_BETA = 0.85, 3.5  # tilt cap (never fully one-sided) and val-gap sensitivity


class ChallengeLoss(nn.Module):
    """Weighted MSE per gender + a fixed fairness penalty, optionally TILTED by the validation gap.

        loss = (err_F + err_M)/2 + lambda * [(1+tilt)*relu(err_M-err_F) + (1-tilt)*relu(err_F-err_M)]
        err_g = sum_g w * (pred - y)^2 / sum_g w ,   w = (1/30 + y) * is_w
    tilt=0 -> symmetric |err_F-err_M|. tilt>0 -> penalize M-worse more. Set from the VALIDATION gap
    (reliable) via update_tilt(), so per-batch noise can't flip the penalty direction.
    """
    def __init__(self, lambda_gap: float = 1.0, asymmetric: bool = False) -> None:
        super().__init__()
        self.lam = float(lambda_gap)
        self.asymmetric = asymmetric
        self.tilt = 0.0  # in [-ASYM_MAX, ASYM_MAX]; >0 leans to penalize M-worse

    def update_tilt(self, val_err_m: float, val_err_f: float) -> float:
        """Lean the fairness penalty toward the gender worse ON VALIDATION (reliable, not per-batch
        noise). tilt = clip(beta * (err_M - err_F) / (|err_M|+|err_F|), -ASYM_MAX, ASYM_MAX)."""
        if self.asymmetric:
            denom = max(abs(val_err_m) + abs(val_err_f), 1e-8)
            self.tilt = float(max(-ASYM_MAX, min(ASYM_MAX, ASYM_BETA * (val_err_m - val_err_f) / denom)))
        return self.tilt

    def forward(self, pred: torch.Tensor, y: torch.Tensor, gender: torch.Tensor,
                is_w: torch.Tensor | None = None, return_gap: bool = False) -> torch.Tensor | tuple[torch.Tensor, float]:
        pred = pred.view(-1)
        w = WEIGHT_OFFSET + y
        if is_w is not None:
            w = w * is_w

        err = (pred - y) ** 2

        f, m = gender < 0.5, gender >= 0.5
        if not (f.any() and m.any()):
            loss = (w * err).sum() / w.sum().clamp(min=1e-8)
            return (loss, 0.0) if return_gap else loss

        err_f = (w[f] * err[f]).sum() / w[f].sum().clamp(min=1e-8)
        err_m = (w[m] * err[m]).sum() / w[m].sum().clamp(min=1e-8)
        gap = err_f - err_m
        # tilt=0 -> lam*|gap|; tilt>0 weights relu(err_M-err_F) more (penalize M-worse harder).
        pen = (1.0 + self.tilt) * torch.relu(-gap) + (1.0 - self.tilt) * torch.relu(gap)
        loss = (err_f + err_m) / 2.0 + self.lam * pen
        return (loss, gap.detach().item()) if return_gap else loss
