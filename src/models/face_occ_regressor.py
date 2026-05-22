from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.models.dinov3_loader import hidden_size_of, load_dinov3
from src.models.sapiens2_loader import hidden_size_of as sapiens2_hidden_size_of
from src.models.sapiens2_loader import is_sapiens2, load_sapiens2


def _is_dinov3(model_name: str) -> bool:
    return model_name.startswith("dinov3_")


def _build_backbone(model_name: str, drop_path_rate: float = 0.0) -> Tuple[nn.Module, int]:
    if _is_dinov3(model_name):
        backbone = load_dinov3(model_name, drop_path_rate=drop_path_rate)
        return backbone, hidden_size_of(model_name)
    if is_sapiens2(model_name):
        backbone = load_sapiens2(model_name, drop_rate=drop_path_rate)
        return backbone, sapiens2_hidden_size_of(model_name)
    backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    return backbone, backbone.config.hidden_size


def _forward_backbone(backbone: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    if hasattr(backbone, "get_intermediate_layers"):
        return backbone.get_intermediate_layers(pixel_values, n=1)[0]
    out = backbone(pixel_values)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out


class AttentionPooling(nn.Module):
    """K-query attention pooling with mixed temperature initialization and learnable τ.

    K = n_focal + n_diffuse + n_free queries. Each query has its own learnable
    log-temperature so τ_k = exp(log_tau_k) > 0 strictly. Focal queries init at low τ
    (sharp attention, captures localized features like physical occluders), diffuse
    queries init at high τ (uniform attention, captures global signals like blur /
    stylization), free queries init at τ=1 (neutral).

    Output: concatenation of K pooled vectors → shape (B, K·D). Subsequent Linear head
    maps to the prediction.
    """

    def __init__(
        self,
        dim: int,
        n_focal: int = 2,
        n_diffuse: int = 2,
        n_free: int = 2,
        tau_focal_init: float = 0.1,
        tau_diffuse_init: float = 1.5,
        tau_free_init: float = 1.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        learnable_tau: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_focal = n_focal
        self.n_diffuse = n_diffuse
        self.n_free = n_free
        self.K = n_focal + n_diffuse + n_free
        assert self.K > 0, "AttentionPooling: K must be > 0"

        self.queries = nn.Parameter(torch.randn(self.K, dim) * 0.02)

        log_taus = torch.zeros(self.K)
        if n_focal > 0:
            log_taus[:n_focal] = torch.log(torch.tensor(tau_focal_init))
        if n_diffuse > 0:
            log_taus[n_focal:n_focal + n_diffuse] = torch.log(torch.tensor(tau_diffuse_init))
        if n_free > 0:
            log_taus[n_focal + n_diffuse:] = torch.log(torch.tensor(tau_free_init))
        if learnable_tau:
            self.log_tau = nn.Parameter(log_taus)
        else:
            self.register_buffer("log_tau", log_taus)

        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.norm = nn.LayerNorm(self.K * dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    @property
    def output_dim(self) -> int:
        return self.K * self.dim

    def get_taus(self) -> torch.Tensor:
        return torch.exp(self.log_tau)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, N, D)  →  pooled: (B, K·D), attn_weights: (B, K, N)"""
        D = x.size(-1)
        k = self.proj_k(x)
        v = self.proj_v(x)
        scale = D ** 0.5
        taus = torch.exp(self.log_tau).view(self.K, 1)

        scores = torch.einsum("kd,bnd->bkn", self.queries, k) / (scale * taus)
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        pooled = torch.einsum("bkn,bnd->bkd", weights, v)
        flat = pooled.flatten(start_dim=1)
        flat = self.norm(flat)
        flat = self.proj_dropout(flat)
        return flat, weights


def _query_diversity_penalty(weights: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity between attention distributions across queries.

    weights: (B, K, N). Returns a scalar in [0, 1] (penalize redundancy).
    """
    w = F.normalize(weights, p=2, dim=-1)
    K = w.size(1)
    if K < 2:
        return weights.sum() * 0.0
    sim = torch.einsum("bkn,bjn->bkj", w, w)
    mask = (1.0 - torch.eye(K, device=sim.device)).unsqueeze(0)
    return (sim * mask).sum() / (sim.size(0) * K * (K - 1))


class FaceOccRegressor(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov3_vits16",
        output_dim: int = 1,
        head_dropout: float = 0.1,
        projection_size: Optional[int] = None,
        output_activation: str = "sigmoid",
        # Backbone regularization
        backbone_drop_path_rate: float = 0.0,
        # Attention pooling structure
        n_focal: int = 2,
        n_diffuse: int = 2,
        n_free: int = 2,
        tau_focal_init: float = 0.1,
        tau_diffuse_init: float = 1.5,
        tau_free_init: float = 1.0,
        learnable_tau: bool = True,
        # Attention pooling regularization
        pool_attn_dropout: float = 0.0,
        pool_proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.head_dropout = head_dropout
        self.output_activation = output_activation

        self.backbone, hidden_size = _build_backbone(model_name, drop_path_rate=backbone_drop_path_rate)

        self.pool = AttentionPooling(
            dim=hidden_size,
            n_focal=n_focal,
            n_diffuse=n_diffuse,
            n_free=n_free,
            tau_focal_init=tau_focal_init,
            tau_diffuse_init=tau_diffuse_init,
            tau_free_init=tau_free_init,
            attn_dropout=pool_attn_dropout,
            proj_dropout=pool_proj_dropout,
            learnable_tau=learnable_tau,
        )

        if projection_size:
            self.projection: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(self.pool.output_dim, projection_size),
                nn.LayerNorm(projection_size),
                nn.GELU(),
                nn.Dropout(head_dropout),
            )
            final_size = projection_size
        else:
            self.projection = None
            final_size = self.pool.output_dim

        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(final_size, output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.head, self.pool.proj_k, self.pool.proj_v]:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        if self.projection is not None:
            for m in self.projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)

    def _activate(self, logits: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "sigmoid":
            return torch.sigmoid(logits)
        if self.output_activation == "clamp":
            return logits.clamp(0.0, 1.0)
        return logits

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kwargs)
            return
        blocks_attr = next((a for a in ("blocks", "layers") if hasattr(self.backbone, a)), None)
        if blocks_attr is None:
            print(f"[FaceOccRegressor] WARNING: cannot enable gradient_checkpointing (backbone has neither .blocks nor .layers)")
            return
        from torch.utils.checkpoint import checkpoint as ckpt
        blocks = getattr(self.backbone, blocks_attr)
        for block in blocks:
            orig = block.forward

            def _wrap(orig_forward):
                def _ckpt_forward(*args, **kw):
                    return ckpt(orig_forward, *args, use_reentrant=False, **kw)
                return _ckpt_forward
            block.forward = _wrap(orig)
        print(f"[FaceOccRegressor] gradient_checkpointing enabled on {len(blocks)} {blocks_attr}")

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

    def forward(self, pixel_values: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        hidden = _forward_backbone(self.backbone, pixel_values)
        pooled, attn_weights = self.pool(hidden)
        if self.projection is not None:
            pooled = self.projection(pooled)
        pooled = self.dropout(pooled)
        logits = self.head(pooled)
        pred = self._activate(logits).squeeze(-1) if self.output_dim == 1 else self._activate(logits)

        out = {"logits": pred}
        if self.training:
            out["attn_weights"] = attn_weights
        return out

    @classmethod
    def load_from_mlflow(cls, model_uri: str, output_dim: Optional[int] = None) -> "FaceOccRegressor":
        import mlflow
        model = mlflow.pytorch.load_model(model_uri)
        return model
