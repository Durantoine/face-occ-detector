from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.utils.distributions import _TEST_DIST, _TRAIN_DIST


# v9 : P_test et P_train viennent maintenant de distributions paramétriques
# (Mix(spike + Beta)) fittées une fois (cf src/utils/distributions.py). Les
# arrays ci-dessous sont les PMF évaluées sur la grille canonique 20 bins de
# width 0.025, exposés pour compat (utilisés direct par eval_pmf_ratio dans
# train.py et build_importance_weights).
_TEST_PMF_0025 = _TEST_DIST.pmf_at_bins(n_bins=20, bin_width=0.025)
_TRAIN_PMF_0025 = _TRAIN_DIST.pmf_at_bins(n_bins=20, bin_width=0.025)


def build_importance_weights(
    train_targets: Optional[np.ndarray] = None,
    test_pmf: np.ndarray = _TEST_PMF_0025,
    train_pmf: np.ndarray = _TRAIN_PMF_0025,
    clip: float = 20.0,
    power: float = 1.0,
) -> np.ndarray:
    """Per-bin importance weights w[b] = (P_test[b] / P_train[b])^power, clipped + normalized.

    v9 : P_train est lu depuis `_TRAIN_PMF_0025` (smoothed Mix(spike+Beta)). `train_targets`
    est ignoré, gardé pour signature ascendante (callers passent `y_train` positionnel).
    """
    del train_targets
    w = test_pmf / np.maximum(train_pmf, 1e-6)
    if power != 1.0:
        w = np.power(w, power)
    w = np.clip(w, 1.0 / clip, clip)
    w = w / w.mean()
    return w


def importance_weight_of(
    targets: torch.Tensor,
    pmf_ratio: torch.Tensor,
    bin_width: float = 0.025,
) -> torch.Tensor:
    n_bins = pmf_ratio.numel()
    idx = torch.clamp((targets / bin_width).long(), 0, n_bins - 1)
    return pmf_ratio[idx]


def build_cell_weights(
    train_targets: np.ndarray,
    train_gender: np.ndarray,
    n_bins: int = 20,
    bin_width: float = 0.025,
    power: float = 1.0,
    base_exp: float = 0.75,
) -> np.ndarray:
    """Cell weights soft : w = (1 / count(g,b)^base_exp)^power, normalisé à mean=1.

    v9 : base_exp passe de 0.5 (sqrt) à 0.75 (entre sqrt et inverse plein) → plus mordant
    sur cellules rares sans dépasser power=1 par mécanisme. Au max (power=1, base_exp=0.75)
    le ratio cell rare/dense ≈ 43× (vs 12× avec sqrt). À power=0 → uniform.
    """
    g = (np.asarray(train_gender) >= 0.5).astype(int)
    b = np.clip((np.asarray(train_targets) / bin_width).astype(int), 0, n_bins - 1)
    counts = np.zeros((2, n_bins), dtype=np.float64)
    for gi, bi in zip(g, b):
        counts[gi, bi] += 1
    counts = np.maximum(counts, 1.0)
    w = 1.0 / np.power(counts, base_exp)
    if power != 1.0:
        w = np.power(w, power)
    w = w / w.mean()
    return w


class WeightedMSELoss(nn.Module):
    def __init__(
        self,
        weight_offset: float = 1.0 / 30.0,
        focal_gamma: float = 0.0,
        fairness_lambda: float = 1.0,
        importance_pmf_ratio: np.ndarray | None = None,
        importance_bin_width: float = 0.025,
        gender_class_weights: np.ndarray | None = None,
        cell_class_weights: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.weight_offset = weight_offset
        self.focal_gamma = focal_gamma
        self.fairness_lambda = fairness_lambda
        self.importance_bin_width = importance_bin_width
        if importance_pmf_ratio is not None:
            self.register_buffer(
                "importance_pmf_ratio",
                torch.as_tensor(importance_pmf_ratio, dtype=torch.float32),
            )
        else:
            self.importance_pmf_ratio = None
        if gender_class_weights is not None:
            self.register_buffer(
                "gender_class_weights",
                torch.as_tensor(gender_class_weights, dtype=torch.float32),
            )
        else:
            self.gender_class_weights = None
        if cell_class_weights is not None:
            self.register_buffer(
                "cell_class_weights",
                torch.as_tensor(cell_class_weights, dtype=torch.float32),
            )
        else:
            self.cell_class_weights = None

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        preds = preds.view(-1)
        if labels.dim() == 2 and labels.size(1) >= 2:
            targets = labels[:, 0]
            gender = labels[:, 1]
        else:
            targets = labels.view(-1)
            gender = None

        err = (preds - targets) ** 2
        w = self.weight_offset + targets

        if self.importance_pmf_ratio is not None:
            w = w * importance_weight_of(targets, self.importance_pmf_ratio, self.importance_bin_width)

        if self.gender_class_weights is not None and gender is not None:
            g_idx = (gender >= 0.5).long()
            w = w * self.gender_class_weights[g_idx]

        if self.cell_class_weights is not None and gender is not None:
            n_bins = self.cell_class_weights.shape[1]
            b_idx = torch.clamp((targets / self.importance_bin_width).long(), 0, n_bins - 1)
            g_idx = (gender >= 0.5).long()
            w = w * self.cell_class_weights[g_idx, b_idx]

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


class GroupDROLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        weight_offset: float = 1.0 / 30.0,
        num_groups: int = 2,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.weight_offset = weight_offset
        self.num_groups = num_groups

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        preds = preds.view(-1)
        if labels.dim() != 2 or labels.size(1) < 2:
            targets = labels.view(-1)
            err = (preds - targets) ** 2
            w = self.weight_offset + targets
            return (w * err).sum() / w.sum().clamp(min=1e-8)

        targets = labels[:, 0]
        gender = labels[:, 1]
        err = (preds - targets) ** 2
        w = self.weight_offset + targets

        group_losses = []
        for g in range(self.num_groups):
            mask = (gender >= g - 0.5) & (gender < g + 0.5)
            if mask.any():
                num = (w[mask] * err[mask]).sum()
                den = w[mask].sum().clamp(min=1e-8)
                group_losses.append(num / den)

        if not group_losses:
            return (w * err).sum() / w.sum().clamp(min=1e-8)

        stacked = torch.stack(group_losses)
        mean_loss = stacked.mean()
        worst_loss = stacked.max()
        return (1.0 - self.alpha) * mean_loss + self.alpha * worst_loss


def mmd_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: Tuple[float, ...] = (1.0, 5.0, 10.0),
) -> torch.Tensor:
    """Multi-bandwidth RBF Maximum Mean Discrepancy² between two sets of features.

    x: (n_x, D), y: (n_y, D). Returns scalar MMD² ≥ 0.
    Returns 0 if either side is empty (no group in batch).
    """
    if x.shape[0] == 0 or y.shape[0] == 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    xx_sq = x.pow(2).sum(-1)
    yy_sq = y.pow(2).sum(-1)
    dxx = xx_sq.unsqueeze(1) + xx_sq.unsqueeze(0) - 2.0 * (x @ x.T)
    dyy = yy_sq.unsqueeze(1) + yy_sq.unsqueeze(0) - 2.0 * (y @ y.T)
    dxy = xx_sq.unsqueeze(1) + yy_sq.unsqueeze(0) - 2.0 * (x @ y.T)

    mmd = torch.zeros((), device=x.device, dtype=x.dtype)
    for sigma in sigmas:
        denom = 2.0 * sigma * sigma
        mmd = mmd + (-dxx / denom).exp().mean() + (-dyy / denom).exp().mean() - 2.0 * (-dxy / denom).exp().mean()
    return mmd / len(sigmas)


def inter_gender_mixup(
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
    bin_width: float = 0.025,
    max_bucket_distance: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """In-batch inter-gender mixup. For each F sample, pair with the M sample whose
    Y bucket is closest (within `max_bucket_distance` bins of `bin_width`). Interpolate
    image + y. Gender stays F (no label noise on the gender dimension).

    pixel_values: (B, C, H, W). labels: (B, 2) — col 0 = y, col 1 = gender ∈ {0,1}.
    F samples without a valid (close-Y) M partner are left unchanged.
    Returns (mixed_pixel_values, mixed_labels) — both same shape as inputs.
    """
    if labels.dim() != 2 or labels.size(1) < 2 or alpha <= 0:
        return pixel_values, labels

    device = pixel_values.device
    y = labels[:, 0]
    g = labels[:, 1]
    f_idx = torch.nonzero(g < 0.5, as_tuple=False).flatten()
    m_idx = torch.nonzero(g >= 0.5, as_tuple=False).flatten()
    if f_idx.numel() == 0 or m_idx.numel() == 0:
        return pixel_values, labels

    f_y = y[f_idx]
    m_y = y[m_idx]
    f_bin = (f_y / bin_width).floor()
    m_bin = (m_y / bin_width).floor()
    # For each F, find nearest M in bin space (|Δbin|), break ties by random order
    diff = (f_bin.unsqueeze(1) - m_bin.unsqueeze(0)).abs()
    perm = torch.randperm(m_idx.numel(), device=device)
    diff = diff[:, perm]
    nearest = diff.argmin(dim=1)
    nearest_dist = diff.gather(1, nearest.unsqueeze(1)).squeeze(1)
    # Only mix F samples that found a partner within max_bucket_distance bins.
    valid = nearest_dist <= float(max_bucket_distance)
    if not valid.any():
        return pixel_values, labels
    f_idx_valid = f_idx[valid]
    partner_idx = m_idx[perm[nearest[valid]]]

    mix = pixel_values.clone()
    new_labels = labels.clone()
    lam = torch.distributions.Beta(alpha, alpha).sample((f_idx_valid.numel(),)).to(
        device=device, dtype=pixel_values.dtype,
    )
    lam_x = lam.view(-1, 1, 1, 1)
    lam_y = lam.view(-1)

    mix[f_idx_valid] = lam_x * pixel_values[f_idx_valid] + (1.0 - lam_x) * pixel_values[partner_idx]
    new_labels[f_idx_valid, 0] = lam_y * y[f_idx_valid] + (1.0 - lam_y) * y[partner_idx]
    # Gender stays F (label of the original slot, not the partner's). The mixed sample is
    # "F image contaminated by M pixels", label remains F → forces features to predict Y
    # invariantly to gender, without introducing label noise on the gender dimension.
    return mix, new_labels


def occlusion_bucket(labels, n_buckets: int = 5):
    arr = np.asarray(labels, dtype=float)
    bins = np.quantile(arr, np.linspace(0, 1, n_buckets + 1)[1:-1])
    return np.digitize(arr, bins)


def stratify_key(gender, occlusion, n_buckets: int = 5):
    g = np.asarray(gender).astype(int)
    b = occlusion_bucket(occlusion, n_buckets=n_buckets).astype(int)
    return g * (n_buckets + 1) + b


def make_sampler_keys(
    df,
    strategy: str = "gender",
    n_buckets: int = 10,
) -> np.ndarray:
    if strategy == "gender":
        return np.asarray(df["gender"]).astype(int)
    if strategy == "occlusion":
        return occlusion_bucket(df["FaceOcclusion"], n_buckets=n_buckets).astype(int)
    if strategy == "gender_x_occ":
        return stratify_key(df["gender"], df["FaceOcclusion"], n_buckets=n_buckets).astype(int)
    raise ValueError(f"Unknown sampler strategy: {strategy}")
