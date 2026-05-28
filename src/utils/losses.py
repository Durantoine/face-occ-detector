from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# Extracted pixel-by-pixel from example/task_brief.pdf page 3 (test histogram, 29980 images).
# Calibration verified: same-method extraction of the train histogram has 0.9999 correlation
# with local train.csv at 100-bin resolution. Integrated test count = 29961 / 29980 (99.94%).
#
# v6 : fit d'une vraie distribution paramétrique pour éliminer le saw-tooth (artefact
# de label-rounding ~0.05 visible dans la PDF source) :
#     P_test(y) = π_0 · δ(0) + (1 − π_0) · Beta(α, β; y · 2)
# avec π_0 = 0.069 (visages "no occlusion" parfaitement), α = 1.67, β = 2.85.
# Fit par minimisation L2 vs raw histogram → L2 = 0.0334 (meilleur parmi Beta/Gamma/Exp
# seuls qui font 0.057-0.083 et écrasent le spike bin 0). Bin 0 préservé à 0.096 (−1%
# vs raw 0.0967). Interprétation : 6.9% du test = aucune occlusion, le reste suit une
# Beta avec mode ~0.12 et tail jusqu'à 0.5.
_TEST_PMF_0025_RAW = np.array([
    0.0967, 0.0546, 0.0847, 0.0642, 0.0812, 0.0692, 0.0838, 0.0677, 0.0790, 0.0713,
    0.0681, 0.0589, 0.0448, 0.0316, 0.0228, 0.0118, 0.0056, 0.0023, 0.0012, 0.0006,
], dtype=np.float64)
_TEST_PMF_0025 = np.array([
    0.0960, 0.0531, 0.0679, 0.0764, 0.0807, 0.0817, 0.0801, 0.0765, 0.0713, 0.0650,
    0.0578, 0.0500, 0.0420, 0.0339, 0.0262, 0.0189, 0.0124, 0.0070, 0.0029, 0.0005,
], dtype=np.float64)


def build_importance_weights(
    train_targets: np.ndarray,
    n_bins: int = 20,
    bin_width: float = 0.025,
    test_pmf: np.ndarray = _TEST_PMF_0025,
    clip: float = 10.0,
    power: float = 1.0,
) -> np.ndarray:
    """Per-bin importance weights w[b] = (P_test[b] / P_train[b])^power, clipped + normalized.

    `power` (paired-α design v6.5) :
      * 1.0 → full loss-side correction (default, legacy behavior)
      * 0.5 → √-strength : combined with √-strength sampler donne la correction complète
      * 0.0 → all-ones après normalisation → no loss effect (sampler à 100%)
    """
    edges = np.linspace(0.0, n_bins * bin_width, n_bins + 1)
    train_hist, _ = np.histogram(np.clip(train_targets, 0.0, edges[-1] - 1e-9), bins=edges)
    train_pmf = train_hist / max(train_hist.sum(), 1)
    w = test_pmf / np.clip(train_pmf, 1e-6, None)
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
) -> np.ndarray:
    """Cell weights soft via 1/sqrt(count(g, b)), normalisé à mean=1.

    `power` (v8 paired-α design) :
      * 1.0  → standard sqrt-soft compensation (legacy v3/v4 cell_rw)
      * 0.5  → encore plus doux : (1/sqrt(count))^0.5 = 1/count^0.25
      * 0.0  → tous les poids = 1, équivalent no-correction
    Permet à TPE de tuner l'intensité de la compensation sans switcher de mécanisme.
    """
    g = (np.asarray(train_gender) >= 0.5).astype(int)
    b = np.clip((np.asarray(train_targets) / bin_width).astype(int), 0, n_bins - 1)
    counts = np.zeros((2, n_bins), dtype=np.float64)
    for gi, bi in zip(g, b):
        counts[gi, bi] += 1
    counts = np.maximum(counts, 1.0)
    w = 1.0 / np.sqrt(counts)
    if power != 1.0:
        w = np.power(w, power)
    w = w / w.mean()
    return w


def build_cell_weights_within(
    train_targets: np.ndarray,
    train_gender: np.ndarray,
    target_pmf: Optional[np.ndarray] = None,
    n_bins: int = 20,
    bin_width: float = 0.025,
    clip: float = 10.0,
    power: float = 1.0,
) -> np.ndarray:
    """Cell weights qui égalisent F/M *intra-bin* tout en suivant `target_pmf` sur Y.

    Pour chaque cellule (g, b) :
        W[g, b] = (0.5 × target_pmf[b] × N / count(g, b))^power      (ratio r)
    Normalisé à moyenne pondérée = 1.

    `power` (paired-α design v6.5) :
      * 1.0 → correction loss-side complète (default, legacy behavior)
      * 0.5 → √-strength : combiné avec sampler √-strength = correction complète
      * 0.0 → all-ones → no loss effect (sampler fait 100% du boulot)

    Si `target_pmf=None` → utilise P_train(Y) empirique (préserve la distribution Y).
    Si `target_pmf=_TEST_PMF_0025` → matche P_test (corrige aussi le shift Y).
    """
    g = (np.asarray(train_gender) >= 0.5).astype(int)
    b = np.clip((np.asarray(train_targets) / bin_width).astype(int), 0, n_bins - 1)
    counts = np.zeros((2, n_bins), dtype=np.float64)
    for gi, bi in zip(g, b):
        counts[gi, bi] += 1

    if target_pmf is None:
        bin_counts = counts.sum(axis=0)
        target = bin_counts / max(bin_counts.sum(), 1)
    else:
        target = np.asarray(target_pmf, dtype=np.float64).flatten()
        if len(target) != n_bins:
            raise ValueError(f"target_pmf has len {len(target)}, expected {n_bins}")
        target = target / max(target.sum(), 1e-9)

    safe_counts = np.maximum(counts, 1.0)
    w = 0.5 * target[None, :] / safe_counts   # broadcast → shape (2, n_bins)
    w[counts == 0] = 0.0

    # Apply power on the per-cell ratio. We power-up before the renormalization so that
    # the mean-weighted-to-1 invariant holds for any α.
    if power != 1.0:
        nonzero_mask = w > 0
        w_pow = np.zeros_like(w)
        w_pow[nonzero_mask] = np.power(w[nonzero_mask], power)
        w = w_pow

    # Normalize so that the average weight across the actual train distribution = 1.
    # i.e. Σ_{g,b} count(g,b) × W[g,b] / N = 1
    total_count = counts.sum()
    mean_w = float((counts * w).sum() / max(total_count, 1))
    if mean_w > 0:
        w = w / mean_w

    # Clip cellules ultra-rares pour éviter gradient bruité
    nonzero = w[w > 0]
    if len(nonzero) > 0:
        median_w = float(np.median(nonzero))
        w = np.clip(w, 0.0, median_w * clip)

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
            w = w * (1.0 + err.detach().pow(self.focal_gamma))

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
