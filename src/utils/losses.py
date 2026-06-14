from __future__ import annotations

import math

import torch
import torch.nn as nn

WEIGHT_OFFSET = 1.0 / 30.0


class ChallengeLoss(nn.Module):
    """Weighted MSE per gender + adaptive Lagrangian fairness penalty.

        loss = (err_F + err_M) / 2 + lambda * |err_F - err_M|
        err_g = sum_g w * (pred - y)^2 / sum_g w ,   w = (1/30 + y) * is_w
    """
    def __init__(self, lambda_gap: float = 1.0, adaptive: bool = True, lam_lr: float = 0.5,
                 lam_min: float = 1.0, lam_max: float = 5.0, threshold: float = 1e-3) -> None:
        super().__init__()
        self.lam = float(lambda_gap)
        self.adaptive = adaptive
        self.lam_lr, self.lam_min, self.lam_max, self.thr = lam_lr, lam_min, lam_max, threshold

    def update_lambda(self, current_gap: float) -> float:
        """Classic proportional-sigmoid Lagrangian: reacts immediately to gap changes."""
        if self.adaptive:
            gap_excess = max(0, abs(current_gap) - self.thr)
            # lam_lr scales the reaction speed. 
            # 0.001 / lam_lr means higher lr -> smaller scale -> steeper sigmoid.
            scale = 0.001 / max(self.lam_lr, 1e-6)
            sig = 1.0 / (1.0 + math.exp(-gap_excess / scale))
            intensity = (sig - 0.5) * 2.0
            self.lam = self.lam_min + (self.lam_max - self.lam_min) * intensity
        return self.lam

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
        loss = (err_f + err_m) / 2.0 + self.lam * gap.abs()
        return (loss, gap.detach().item()) if return_gap else loss
