from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# v10 : retour aux distributions empiriques avec 10 bins de width 0.05.
# Grille plus large = smoothing implicite (bin 19 anomalie disparaît, ratios stables).
# P_test extraite du PDF page 3 (29980 images), re-binnée à 10 bins.
N_BINS = 10
BIN_WIDTH = 0.05

# P_test 10 bins (somme des 20 bins originaux par paires) — raw extraction PDF.
_TEST_PMF = np.array([
    0.0967 + 0.0546,    # bin 0 : Y ∈ [0.00, 0.05)
    0.0847 + 0.0642,    # bin 1 : Y ∈ [0.05, 0.10)
    0.0812 + 0.0692,    # bin 2 : Y ∈ [0.10, 0.15)
    0.0838 + 0.0677,    # bin 3 : Y ∈ [0.15, 0.20)
    0.0790 + 0.0713,    # bin 4 : Y ∈ [0.20, 0.25)
    0.0681 + 0.0589,    # bin 5 : Y ∈ [0.25, 0.30)
    0.0448 + 0.0316,    # bin 6 : Y ∈ [0.30, 0.35)
    0.0228 + 0.0118,    # bin 7 : Y ∈ [0.35, 0.40)
    0.0056 + 0.0023,    # bin 8 : Y ∈ [0.40, 0.45)
    0.0012 + 0.0006,    # bin 9 : Y ∈ [0.45, 0.50)
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
    """Empirical joint pmf P(g, y), shape (2, n_bins). Sums to 1."""
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
    clip: float = 20.0,
) -> np.ndarray:
    """v10 — UNIFIED target weight per sample i.

    P_target(g, y) = mix_y(α1) × mix_g(α2)
      mix_y(α1) = (1-α1) × P_train_marg_y + α1 × P_test_y
      mix_g(α2) = (1-α2) × P_train(g|y) + α2 × 0.5

    Returns per-sample weight = P_target(g_i, y_i) / P_train_emp(g_i, y_i), clipped.
    """
    g = (np.asarray(gender) >= 0.5).astype(int)
    b = np.clip((np.asarray(targets) / bin_width).astype(int), 0, n_bins - 1)
    p_joint = compute_empirical_pmf_cell(targets, gender, n_bins=n_bins, bin_width=bin_width)   # (2, n_bins)
    p_train_y = p_joint.sum(axis=0)                                                              # (n_bins,)
    safe_y = np.maximum(p_train_y, 1e-9)
    p_train_g_given_y = p_joint / safe_y[None, :]                                                # (2, n_bins)

    # mix targets
    p_target_y = (1.0 - axis1_power) * p_train_y + axis1_power * test_pmf_y
    uniform_g = np.full_like(p_train_g_given_y, 0.5)
    p_target_g_given_y = (1.0 - axis2_power) * p_train_g_given_y + axis2_power * uniform_g
    p_target_joint = p_target_y[None, :] * p_target_g_given_y                                    # (2, n_bins)

    # per-cell weight = P_target / P_train_emp
    ratio = p_target_joint / np.maximum(p_joint, 1e-9)
    ratio = np.clip(ratio, 1.0 / clip, clip)
    return ratio[g, b]                                                                            # (N,) per-sample


def split_loss_aug(
    target_weight: np.ndarray, aug_share: float, k_max: int = 3, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split unified target into loss weight + aug replication count.

    loss_weight_i = target_weight_i ^ (1 - aug_share)
    aug_repli_i  = round_bernoulli(target_weight_i ^ aug_share), clipped to [1, k_max]

    En espérance : loss_weight × aug_repli = target_weight, mais aug_repli capé à k_max
    pour éviter mémorisation sur cells rares.
    """
    w = np.asarray(target_weight, dtype=np.float64)
    loss_w = np.power(w, 1.0 - aug_share)
    aug_w = np.power(w, aug_share)
    # Stochastic Bernoulli rounding preserves E[copies] when uncapped
    rng = np.random.RandomState(seed)
    floor = np.floor(aug_w).astype(int)
    frac = aug_w - floor
    extra = (rng.uniform(size=len(aug_w)) < frac).astype(int)
    aug_repli = np.clip(floor + extra, 1, k_max)
    return loss_w, aug_repli


def importance_weight_of(
    targets: torch.Tensor, sample_weights: torch.Tensor,
) -> torch.Tensor:
    """Index per-sample weights by sample id (not bin) — assumes weights aligned with batch."""
    return sample_weights


class WeightedMSELoss(nn.Module):
    """v10 — challenge metric formula + optional regularizers.

    Base (= métrique officielle PDF page 4) :
        Err_g = Σ w_i·(p-y)² / Σ w_i   avec w_i = 1/30 + y_i
        Score = (Err_F + Err_M)/2 + λ·|Err_F - Err_M|

    Optional :
      * sample_loss_weight : per-sample multiplier (axe 1+2 unified target)
      * focal_gamma > 0   : multiplier (err + 0.05)^γ (auto hard-example focus)
    """

    def __init__(
        self,
        weight_offset: float = 1.0 / 30.0,
        focal_gamma: float = 0.0,
        fairness_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight_offset = weight_offset
        self.focal_gamma = focal_gamma
        self.fairness_lambda = fairness_lambda

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

        return (err_f + err_m) / 2.0 + self.fairness_lambda * (err_f - err_m).abs()


def mmd_rbf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """RBF MMD² avec median heuristic (bandwidth auto adapté à l'échelle des features).

    v10 fix : σ² = median(||x-y||²) au lieu de sigmas fixes (1, 5, 10) qui saturaient
    à 0 pour features ViT-B 768-dim (pairwise distances ~1500).

    Robuste aux mini-batches : retourne 0 si moins de 2 samples par groupe (median
    indéfini sur 1 sample, MMD non-significatif sur singletons).
    """
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
