"""
Learnable anatomical zone weights for face occlusion estimation.

8 zones cover the full face surface (weights sum to 1.0):
  forehead, eye_left, eye_right, nose, cheek_left, cheek_right, mouth, chin

Each weight w_z is bounded in [w_min_z, w_max_z] (anatomical prior) and
jointly optimized with Qwen's LoRA adapters via gradient descent.
Optuna searches the bounds; gradient descent refines within them.

Parametrization (bounded softmax on simplex):
    w_raw  ∈ R^8                           (free parameters)
    w_soft = softmax(w_raw)                (sums to 1, unconstrained)
    w_clip = clamp(w_soft, w_min, w_max)  (per-zone anatomical bounds)
    w_z    = w_clip / sum(w_clip)          (renormalize onto simplex)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ZONE_NAMES = [
    "forehead",
    "eye_left",
    "eye_right",
    "nose",
    "cheek_left",
    "cheek_right",
    "mouth",
    "chin",
]

# Anatomical bounds [w_min, w_max] — sum of midpoints ≈ 1.0
# Searched by Optuna; defaults are medically-motivated.
DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "forehead":   (0.08, 0.25),
    "eye_left":   (0.06, 0.18),
    "eye_right":  (0.06, 0.18),
    "nose":       (0.08, 0.20),
    "cheek_left": (0.10, 0.25),
    "cheek_right":(0.10, 0.25),
    "mouth":      (0.06, 0.18),
    "chin":       (0.08, 0.20),
}


class ZoneWeights(nn.Module):
    """Learnable bounded zone weight vector.

    Args:
        bounds: dict mapping zone name → (w_min, w_max). Defaults to DEFAULT_BOUNDS.
        init_uniform: if True, initialize w_raw so softmax gives equal weights.
    """

    def __init__(
        self,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        init_uniform: bool = True,
    ) -> None:
        super().__init__()
        self.bounds = bounds or DEFAULT_BOUNDS
        w_min = torch.tensor([self.bounds[z][0] for z in ZONE_NAMES])
        w_max = torch.tensor([self.bounds[z][1] for z in ZONE_NAMES])
        self.register_buffer("w_min", w_min)
        self.register_buffer("w_max", w_max)

        init = torch.zeros(len(ZONE_NAMES)) if init_uniform else torch.randn(len(ZONE_NAMES)) * 0.1
        self.w_raw = nn.Parameter(init)

    def forward(self) -> torch.Tensor:
        """Returns normalized zone weights of shape (8,) summing to 1."""
        w_soft = F.softmax(self.w_raw, dim=0)
        w_clip = torch.clamp(w_soft, self.w_min, self.w_max)
        return w_clip / w_clip.sum().clamp(min=1e-8)

    def as_dict(self) -> Dict[str, float]:
        w = self.forward().detach().cpu()
        return {z: float(w[i]) for i, z in enumerate(ZONE_NAMES)}

    def weighted_zone_loss(
        self,
        occ_pred: torch.Tensor,
        occ_gt: torch.Tensor,
    ) -> torch.Tensor:
        """MSE per zone weighted by learned w_z.

        Args:
            occ_pred: (B, 8) predicted occlusion fraction per zone
            occ_gt:   (B, 8) ground-truth occlusion fraction per zone
        Returns:
            scalar loss
        """
        w = self.forward()                          # (8,)
        err = (occ_pred - occ_gt) ** 2             # (B, 8)
        return (err * w.unsqueeze(0)).sum(dim=1).mean()

    def aggregate(self, occ_zones: torch.Tensor) -> torch.Tensor:
        """Compute global occlusion from per-zone fractions.

        Args:
            occ_zones: (B, 8) or (8,) per-zone occlusion fractions
        Returns:
            (B,) or scalar global occlusion ∈ [0, 1]
        """
        w = self.forward()
        if occ_zones.dim() == 1:
            return (w * occ_zones).sum()
        return (occ_zones * w.unsqueeze(0)).sum(dim=1)


def build_zone_weights(
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> ZoneWeights:
    return ZoneWeights(bounds=bounds)


def bounds_from_optuna(trial: object) -> Dict[str, Tuple[float, float]]:
    """Sample per-zone bound ranges from an Optuna trial.

    Each zone gets a (w_min, w_max) pair where Optuna chooses the
    center and half-width, then we derive the interval.
    """
    import optuna  # type: ignore
    assert isinstance(trial, optuna.Trial)
    bounds: Dict[str, Tuple[float, float]] = {}
    for zone, (lo_default, hi_default) in DEFAULT_BOUNDS.items():
        mid_default = (lo_default + hi_default) / 2
        half_default = (hi_default - lo_default) / 2
        center = trial.suggest_float(f"wz_center_{zone}", lo_default, hi_default)
        half   = trial.suggest_float(f"wz_half_{zone}",  half_default * 0.3, half_default * 1.5)
        bounds[zone] = (max(0.01, center - half), min(0.50, center + half))
    return bounds
