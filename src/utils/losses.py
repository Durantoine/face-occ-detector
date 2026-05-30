from typing import Optional

import numpy as np
import torch
import torch.nn as nn


N_BINS = 15
BIN_WIDTH = 0.5 / N_BINS

_TEST_PMF = np.array([
    0.118034,
    0.092728,
    0.092810,
    0.108154,
    0.102878,
    0.094643,
    0.106911,
    0.091066,
    0.079910,
    0.054407,
    0.034125,
    0.015720,
    0.005424,
    0.002145,
    0.001046,
], dtype=np.float64)
_TEST_PMF = _TEST_PMF / _TEST_PMF.sum()


def compute_empirical_pmf(targets: np.ndarray, n_bins: int = N_BINS, bin_width: float = BIN_WIDTH) -> np.ndarray:
    edges = np.linspace(0.0, n_bins * bin_width, n_bins + 1)
    y = np.clip(np.asarray(targets, dtype=np.float64), 0.0, edges[-1] - 1e-9)
    hist, _ = np.histogram(y, bins=edges)
    return hist.astype(np.float64) / max(hist.sum(), 1)


def compute_empirical_pmf_cell(
    targets: np.ndarray, gender: np.ndarray,
    n_bins: int = N_BINS, bin_width: float = BIN_WIDTH,
) -> np.ndarray:
    g = (np.asarray(gender) >= 0.5).astype(int)
    b = np.clip((np.asarray(targets) / bin_width).astype(int), 0, n_bins - 1)
    counts = np.zeros((2, n_bins), dtype=np.float64)
    for gi, bi in zip(g, b):
        counts[gi, bi] += 1
    return counts / max(counts.sum(), 1)


def compute_target_weights(
    targets: np.ndarray, gender: np.ndarray,
    axis1_power: float, axis2_power: float,
    n_bins: int = N_BINS, bin_width: float = BIN_WIDTH,
    test_pmf_y: np.ndarray = _TEST_PMF,
    clip: float = 10.0,
) -> np.ndarray:
    g = (np.asarray(gender) >= 0.5).astype(int)
    b = np.clip((np.asarray(targets) / bin_width).astype(int), 0, n_bins - 1)
    p_joint = compute_empirical_pmf_cell(targets, gender, n_bins=n_bins, bin_width=bin_width)
    p_train_y = p_joint.sum(axis=0)
    safe_y = np.maximum(p_train_y, 1e-9)
    p_train_g_given_y = p_joint / safe_y[None, :]

    p_target_y = (1.0 - axis1_power) * p_train_y + axis1_power * test_pmf_y
    uniform_g = np.full_like(p_train_g_given_y, 0.5)
    p_target_g_given_y = (1.0 - axis2_power) * p_train_g_given_y + axis2_power * uniform_g
    p_target_joint = p_target_y[None, :] * p_target_g_given_y

    ratio = p_target_joint / np.maximum(p_joint, 1e-9)
    ratio = np.clip(ratio, 1.0 / clip, clip)
    sample_w = ratio[g, b]
    sample_w = sample_w / max(float(sample_w.mean()), 1e-9)
    return sample_w.astype(np.float32)


class WeightedMSELoss(nn.Module):
    """Challenge metric formula with adaptive Lagrangian λ on the fairness gap.

    Per-group weighted MSE:
        Err_g = Σ w_i·(p-y)² / Σ w_i   with w_i = 1/30 + y_i  (+ optional sample reweight)

    Training loss (Lagrangian, λ_adapt updated via gradient ascent on the constraint):
        L = (Err_F + Err_M)/2 + λ_adapt · |Err_F - Err_M|
        λ_adapt ← clip(λ_adapt + η · EMA(|Err_F - Err_M|),  0,  λ_max)

    The challenge metric (logged separately by metrics.compute_score) always uses
    λ_metric = 1.0 — only the TRAINING loss has an adaptive λ.
    """

    def __init__(
        self,
        weight_offset: float = 1.0 / 30.0,
        focal_gamma: float = 0.0,
        lambda_init: float = 1.0,
        lambda_lr: float = 0.5,
        lambda_max: float = 5.0,
        lambda_ema: float = 0.9,
    ) -> None:
        super().__init__()
        self.weight_offset = weight_offset
        self.focal_gamma = focal_gamma
        self.lambda_lr = float(lambda_lr)
        self.lambda_max = float(lambda_max)
        self.lambda_ema = float(lambda_ema)
        self.register_buffer("lambda_adapt", torch.tensor(float(lambda_init)))
        self.register_buffer("err_diff_ema", torch.tensor(0.0))

    def forward(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        sample_loss_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        preds = preds.view(-1)
        if labels.dim() == 2 and labels.size(1) >= 2:
            targets = labels[:, 0]
            gender = labels[:, 1]
        else:
            targets = labels.view(-1)
            gender = None

        err = (preds - targets) ** 2
        w = self.weight_offset + targets

        if sample_loss_weight is not None:
            w = w * sample_loss_weight.to(w.device).to(w.dtype)

        if self.focal_gamma > 0:
            w = w * (err.detach() + 0.05).pow(self.focal_gamma)

        if gender is None:
            return (w * err).sum() / w.sum().clamp(min=1e-8)

        mask_f = gender < 0.5
        mask_m = gender >= 0.5

        if not (mask_f.any() and mask_m.any()):
            return (w * err).sum() / w.sum().clamp(min=1e-8)

        err_f = (w[mask_f] * err[mask_f]).sum() / w[mask_f].sum().clamp(min=1e-8)
        err_m = (w[mask_m] * err[mask_m]).sum() / w[mask_m].sum().clamp(min=1e-8)
        err_diff = (err_f - err_m).abs()

        if self.training:
            lam = self.lambda_adapt
            with torch.no_grad():
                err_diff_sync = err_diff.detach().clone()
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(err_diff_sync, op=torch.distributed.ReduceOp.AVG)
                self.err_diff_ema.mul_(self.lambda_ema).add_(err_diff_sync * (1.0 - self.lambda_ema))
                self.lambda_adapt.add_(self.lambda_lr * self.err_diff_ema)
                self.lambda_adapt.clamp_(0.0, self.lambda_max)
        else:
            lam = torch.tensor(1.0, device=err_diff.device, dtype=err_diff.dtype)

        return (err_f + err_m) / 2.0 + lam * err_diff


def sliced_wasserstein(x: torch.Tensor, y: torch.Tensor, n_projections: int = 50) -> torch.Tensor:
    """Sliced Wasserstein-2 distance between two feature sets.

    Projects x, y onto random 1D directions, computes 1D OT (= sorted L2) per
    projection, averages. Differentiable, batch-size-agnostic (handles unequal
    sizes via linear interpolation on the empirical CDFs).
    """
    if x.shape[0] < 2 or y.shape[0] < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    d = x.shape[-1]
    proj = torch.randn(d, n_projections, device=x.device, dtype=x.dtype)
    proj = proj / proj.norm(dim=0, keepdim=True).clamp(min=1e-9)

    x_proj = (x @ proj).t()
    y_proj = (y @ proj).t()
    x_sorted, _ = x_proj.sort(dim=-1)
    y_sorted, _ = y_proj.sort(dim=-1)

    if x.shape[0] != y.shape[0]:
        n = max(x.shape[0], y.shape[0])
        x_sorted = torch.nn.functional.interpolate(x_sorted.unsqueeze(1), size=n, mode="linear", align_corners=True).squeeze(1)
        y_sorted = torch.nn.functional.interpolate(y_sorted.unsqueeze(1), size=n, mode="linear", align_corners=True).squeeze(1)

    return ((x_sorted - y_sorted) ** 2).mean()


def mmd_rbf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape[0] < 2 or y.shape[0] < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    dxy = torch.cdist(x, y).pow(2)
    sigma_sq = dxy.detach().median().clamp(min=1e-6)

    dxx = torch.cdist(x, x).pow(2)
    dyy = torch.cdist(y, y).pow(2)

    k_xx = (-dxx / sigma_sq).exp().mean()
    k_yy = (-dyy / sigma_sq).exp().mean()
    k_xy = (-dxy / sigma_sq).exp().mean()
    return (k_xx + k_yy - 2.0 * k_xy).clamp(min=0.0)
