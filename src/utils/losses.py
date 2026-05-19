from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 0.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction="none", label_smoothing=self.label_smoothing)

        if self.gamma > 0:
            probs = F.softmax(logits, dim=-1)
            pt = probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).clamp(min=1e-7, max=1.0)
            ce_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.class_weights is not None:
            weights = self.class_weights
            if weights.device != labels.device:
                weights = weights.to(labels.device)
            sample_weights = weights[labels]
            return (ce_loss * sample_weights).mean()

        return ce_loss.mean()


def compute_class_weights(df: pd.DataFrame, label_col: str = "label") -> torch.Tensor:
    num_classes = df[label_col].nunique()
    total = len(df)
    weights = torch.ones(num_classes)
    for cls in range(num_classes):
        count = (df[label_col] == cls).sum()
        if count > 0:
            weights[cls] = total / (num_classes * count)
    return weights
