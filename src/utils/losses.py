import numpy as np
import torch
import torch.nn as nn


_TEST_PMF_0025 = np.array([
    0.105, 0.090, 0.090, 0.090, 0.092, 0.088, 0.083, 0.072, 0.067, 0.063,
    0.055, 0.045, 0.028, 0.017, 0.010, 0.003, 0.002, 0.000, 0.000, 0.000,
], dtype=np.float64)


def build_importance_weights(
    train_targets: np.ndarray,
    n_bins: int = 20,
    bin_width: float = 0.025,
    test_pmf: np.ndarray = _TEST_PMF_0025,
    clip: float = 10.0,
) -> np.ndarray:
    edges = np.linspace(0.0, n_bins * bin_width, n_bins + 1)
    train_hist, _ = np.histogram(np.clip(train_targets, 0.0, edges[-1] - 1e-9), bins=edges)
    train_pmf = train_hist / max(train_hist.sum(), 1)
    w = test_pmf / np.clip(train_pmf, 1e-6, None)
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
) -> np.ndarray:
    edges = np.linspace(0.0, n_bins * bin_width, n_bins + 1)
    g = (np.asarray(train_gender) >= 0.5).astype(int)
    b = np.clip((np.asarray(train_targets) / bin_width).astype(int), 0, n_bins - 1)
    counts = np.zeros((2, n_bins), dtype=np.float64)
    for gi, bi in zip(g, b):
        counts[gi, bi] += 1
    counts = np.maximum(counts, 1.0)
    w = 1.0 / np.sqrt(counts)
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
