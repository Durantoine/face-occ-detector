import numpy as np
import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
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
