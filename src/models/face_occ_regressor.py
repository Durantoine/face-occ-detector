from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import build_backbone


class AttentionPool(nn.Module):
    def __init__(self, dim: int, n_queries: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_queries = n_queries
        self.q = nn.Parameter(torch.empty(n_queries, dim))
        nn.init.orthogonal_(self.q)
        self.scale = dim ** -0.5
        self.proj = nn.Linear(n_queries * dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D), q: (K, D) -> attn: (B, K, N)
        attn = (self.q @ x.transpose(1, 2) * self.scale).softmax(dim=-1)
        # pooled: (B, K, D)
        pooled = attn @ x
        # flatten and project: (B, K*D) -> (B, D)
        return self.drop(self.proj(pooled.flatten(1)))


class GridPool(nn.Module):
    def __init__(self, dim: int, grid_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.g = grid_size
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) where N is spatial tokens
        x = self.proj(x)
        b, n, d = x.shape
        # Automatically infer grid size (e.g., 14x14 = 196)
        side = int(math.isqrt(n))
        if side * side != n:
            raise ValueError(f"GridPool input tokens {n} must be a perfect square after stripping prefix tokens.")
        
        # Reshape to (B, D, H, W) for pooling
        x = x.transpose(1, 2).reshape(b, d, side, side)
        x = F.adaptive_avg_pool2d(x, self.g)
        return self.drop(x.reshape(b, d * self.g * self.g))


class MeanPool(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average across spatial tokens
        return self.drop(self.proj(x.mean(dim=1)))


class FaceOccModel(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = True, drop_path: float = 0.1,
                 pooling_dropout: float = 0.0, head_dropout: float = 0.1,
                 pooling_type: str = "mean", grid_size: int = 4, attn_queries: int = 4,
                 target_mean: float | None = None, init_backbone_from: str | None = None,
                 head_mlp_ratio: float = 0.0) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained=pretrained, drop_path=drop_path,
                                       init_backbone_from=init_backbone_from)
        d = self.backbone.num_features
        self.n_prefix = self.backbone.num_prefix
        self.pooling_type = pooling_type

        if pooling_type == "attention":
            self.pool = AttentionPool(d, n_queries=attn_queries, dropout=pooling_dropout)
        elif pooling_type == "grid":
            self.pool = GridPool(d, grid_size, pooling_dropout)
        else:
            self.pool = MeanPool(d, pooling_dropout)

        pooled_dim = d * grid_size * grid_size if pooling_type == "grid" else d
        self.head_drop = nn.Dropout(head_dropout)
        hidden = round(head_mlp_ratio * pooled_dim)
        if hidden > 0:
            # small regularized MLP head: lets a non-linear feature->occlusion map form (e.g. to
            # separate blur from real occlusion). GELU + dropout; weight_decay applies (ndim>1).
            # hidden is a ratio of pooled_dim so it scales with the backbone width.
            self.head = nn.Sequential(nn.Linear(pooled_dim, hidden), nn.GELU(),
                                      nn.Dropout(head_dropout), nn.Linear(hidden, 1))
            final = self.head[-1]
        else:
            self.head = nn.Linear(pooled_dim, 1)
            final = self.head

        nn.init.trunc_normal_(final.weight, std=0.02)
        bias = 0.0
        if target_mean is not None:
            p = min(max(float(target_mean), 1e-4), 1.0 - 1e-4)
            bias = math.log(p / (1.0 - p))
        nn.init.constant_(final.bias, bias)

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.pool(tokens[:, self.n_prefix:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._pool(self.backbone(x))
        return torch.sigmoid(self.head(self.head_drop(feat))).squeeze(-1)
