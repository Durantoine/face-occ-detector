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


def _is_timm_cnn(model_name: str) -> bool:
    """Detect timm CNN models (efficientnet*, resnet*, convnext*, etc.)."""
    prefixes = ("efficientnet", "resnet", "resnext", "convnext", "regnet", "mobilenetv", "tf_efficientnet")
    return any(model_name.startswith(p) for p in prefixes)


def _build_backbone(
    model_name: str,
    drop_path_rate: float = 0.0,
    pretrained: bool = True,
) -> Tuple[nn.Module, int]:
    if _is_dinov3(model_name):
        backbone = load_dinov3(model_name, drop_path_rate=drop_path_rate, pretrained=pretrained)
        return backbone, hidden_size_of(model_name)
    if is_sapiens2(model_name):
        backbone = load_sapiens2(model_name, drop_rate=drop_path_rate, pretrained=pretrained)
        return backbone, sapiens2_hidden_size_of(model_name)
    if _is_timm_cnn(model_name):
        import timm
        # num_classes=0 + global_pool="" : keep conv_head (which projects last-stage
        # 512ch → 2048ch for EfficientNet-B5) and the spatial map, but skip the global
        # average pool + classifier. Result: backbone(x) → (B, num_features, H, W).
        # num_features=2048 for EfficientNet-B5 (vs 512 with features_only).
        backbone = timm.create_model(model_name, pretrained=pretrained,
                                       num_classes=0, global_pool="",
                                       drop_path_rate=drop_path_rate)
        hidden = backbone.num_features
        return backbone, hidden
    if pretrained:
        backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    else:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        backbone = AutoModel.from_config(cfg, trust_remote_code=True)
    return backbone, backbone.config.hidden_size


def _forward_backbone(backbone: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """Return (B, N, D) — sequence of token/spatial features compatible with all pools.

    For ViT-like backbones: native (B, N, D) where N = num_patches + (CLS).
    For timm CNN (features_only): (B, C, H, W) → reshaped to (B, H·W, C). No CLS token,
    so CLSPooling won't work — use K-query, MHA, MIL or mean_var.
    """
    if hasattr(backbone, "get_intermediate_layers"):
        return backbone.get_intermediate_layers(pixel_values, n=1)[0]
    out = backbone(pixel_values)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    # timm CNN features_only returns list of (B, C, H, W); we already selected out_indices=(-1,)
    if isinstance(out, torch.Tensor) and out.dim() == 4:
        # (B, C, H, W) → (B, H·W, C)
        b, c, h, w = out.shape
        return out.permute(0, 2, 3, 1).reshape(b, h * w, c)
    return out


class CLSPooling(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    @property
    def output_dim(self) -> int:
        return self.dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return x[:, 0, :], None


class MeanVarPooling(nn.Module):
    """Global stats: concat(mean, std) over patches. No learned params.

    Captures global distribution of features → useful for blur / global degradation
    (high σ ≈ noisy/diverse features ≈ occlusion). Loses spatial localization.
    """
    def __init__(self, dim: int, skip_cls: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.skip_cls = skip_cls

    @property
    def output_dim(self) -> int:
        return 2 * self.dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        patches = x[:, 1:, :] if self.skip_cls else x
        mu = patches.mean(dim=1)
        sigma = patches.std(dim=1)
        return torch.cat([mu, sigma], dim=-1), None


class MILPooling(nn.Module):
    """Multi-Instance Learning: per-patch occlusion score + aggregation.

      f(h_i)  →  o_i ∈ R         (raw scalar score per patch via 2-layer MLP)
      ŷ_raw  =  g(o_1, ..., o_N)  aggregation: mean | max | topk_mean | attention

    Returns (B, 1) — head then applies Linear(1, 1) + sigmoid for final calibration.

    Conceptually aligned with face occlusion (sparse, localized) but also captures
    blur (uniform high scores) depending on aggregation:
      - mean       : OK for blur, dilutes sparse
      - max        : OK for sparse, saturates on blur
      - topk_mean  : compromise (good for both with k tuned)
      - attention  : softmax(logits) weights — flexible
    """
    def __init__(self, dim: int, hidden: int = 128, agg: str = "topk_mean",
                  k_top: int = 30, skip_cls: bool = True) -> None:
        super().__init__()
        if agg not in ("mean", "max", "topk_mean", "attention"):
            raise ValueError(f"Unknown MIL agg: {agg}")
        self.dim = dim
        self.hidden = hidden
        self.agg = agg
        self.k_top = k_top
        self.skip_cls = skip_cls
        self.scorer = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    @property
    def output_dim(self) -> int:
        return 1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        patches = x[:, 1:, :] if self.skip_cls else x
        logits = self.scorer(patches).squeeze(-1)  # (B, N)
        if self.agg == "mean":
            pooled = logits.mean(dim=1, keepdim=True)
        elif self.agg == "max":
            pooled = logits.max(dim=1, keepdim=True).values
        elif self.agg == "topk_mean":
            k = min(self.k_top, logits.size(1))
            top, _ = logits.topk(k, dim=1)
            pooled = top.mean(dim=1, keepdim=True)
        else:  # attention
            attn = F.softmax(logits, dim=1)
            pooled = (attn * logits).sum(dim=1, keepdim=True)
        return pooled, logits  # (B, 1), (B, N)


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


class MultiHeadAttentionPooling(nn.Module):
    """Standard multi-head attention pooling with a single learnable query.

    1 query ∈ ℝᴰ, split across H heads (each ∈ ℝ^(D/H)). No per-head τ. Diversity
    emerges from random init of W_q^h, W_k^h, W_v^h per head. Output: ℝᴰ.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"MultiHeadAttentionPooling: dim={dim} not divisible by num_heads={num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.norm = nn.LayerNorm(dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    @property
    def output_dim(self) -> int:
        return self.dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, N, D)  →  pooled: (B, D), attn_weights: (B, H, N)"""
        B, N, D = x.shape
        H, Hd = self.num_heads, self.head_dim

        q = self.proj_q(self.query).view(H, Hd)
        k = self.proj_k(x).view(B, N, H, Hd).transpose(1, 2)
        v = self.proj_v(x).view(B, N, H, Hd).transpose(1, 2)

        scores = torch.einsum("hd,bhnd->bhn", q, k) / (Hd ** 0.5)
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        pooled = torch.einsum("bhn,bhnd->bhd", weights, v).reshape(B, D)
        pooled = self.proj_out(pooled)
        pooled = self.norm(pooled)
        pooled = self.proj_dropout(pooled)
        return pooled, weights


def build_pooling(
    pooling_type: str,
    dim: int,
    n_focal: int = 2,
    n_diffuse: int = 2,
    n_free: int = 2,
    tau_focal_init: float = 0.1,
    tau_diffuse_init: float = 1.5,
    tau_free_init: float = 1.0,
    learnable_tau: bool = True,
    num_heads: int = 4,
    pool_attn_dropout: float = 0.0,
    pool_proj_dropout: float = 0.0,
    mil_agg: str = "topk_mean",
    mil_hidden: int = 128,
    mil_k_top: int = 30,
) -> nn.Module:
    if pooling_type == "cls":
        return CLSPooling(dim=dim)
    if pooling_type == "mean_var":
        return MeanVarPooling(dim=dim)
    if pooling_type == "mil":
        return MILPooling(dim=dim, hidden=mil_hidden, agg=mil_agg, k_top=mil_k_top)
    if pooling_type == "attention_k_query":
        return AttentionPooling(
            dim=dim, n_focal=n_focal, n_diffuse=n_diffuse, n_free=n_free,
            tau_focal_init=tau_focal_init, tau_diffuse_init=tau_diffuse_init,
            tau_free_init=tau_free_init, learnable_tau=learnable_tau,
            attn_dropout=pool_attn_dropout, proj_dropout=pool_proj_dropout,
        )
    if pooling_type == "multihead_attention":
        return MultiHeadAttentionPooling(
            dim=dim, num_heads=num_heads,
            attn_dropout=pool_attn_dropout, proj_dropout=pool_proj_dropout,
        )
    raise ValueError(f"Unknown pooling_type: {pooling_type}")


class _GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer (Ganin & Lempitsky 2015). Identity forward, sign-flipped
    gradient backward scaled by alpha. The adversarial λ is applied as a loss weight
    in compute_loss — alpha here is fixed to 1.0 so GRL is a pure sign-flipper.
    """
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, alpha)


class GenderDiscriminator(nn.Module):
    """Small MLP that predicts gender from features. Used with GRL for DANN-style
    adversarial debiasing: backbone learns features that the disc cannot classify.
    """
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _query_diversity_penalty(weights: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity between attention distributions across queries/heads.

    weights: (B, K, N) for K-query, (B, H, N) for MHA. Returns a scalar in [0, 1].
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
        # Backbone
        backbone_drop_path_rate: float = 0.0,
        pretrained: bool = True,
        # Pooling dispatch
        pooling_type: str = "attention_k_query",
        # K-query
        n_focal: int = 2,
        n_diffuse: int = 2,
        n_free: int = 2,
        tau_focal_init: float = 0.1,
        tau_diffuse_init: float = 1.5,
        tau_free_init: float = 1.0,
        learnable_tau: bool = True,
        # Multi-head
        num_heads: int = 4,
        # MIL
        mil_agg: str = "topk_mean",
        mil_hidden: int = 128,
        mil_k_top: int = 30,
        # Common pool regularization
        pool_attn_dropout: float = 0.0,
        pool_proj_dropout: float = 0.0,
        # Adversarial debiasing (DANN)
        enable_adv_disc: bool = False,
        adv_disc_hidden: int = 256,
        adv_disc_dropout: float = 0.2,
        # Init head bias so sigmoid(bias) ≈ E[Y_train] (avoids 0.5 offset at start)
        target_mean: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.head_dropout = head_dropout
        self.output_activation = output_activation
        self.pooling_type = pooling_type
        self.enable_adv_disc = enable_adv_disc

        self.backbone, hidden_size = _build_backbone(
            model_name, drop_path_rate=backbone_drop_path_rate, pretrained=pretrained,
        )

        self.pool = build_pooling(
            pooling_type=pooling_type,
            dim=hidden_size,
            n_focal=n_focal, n_diffuse=n_diffuse, n_free=n_free,
            tau_focal_init=tau_focal_init, tau_diffuse_init=tau_diffuse_init,
            tau_free_init=tau_free_init, learnable_tau=learnable_tau,
            num_heads=num_heads,
            mil_agg=mil_agg, mil_hidden=mil_hidden, mil_k_top=mil_k_top,
            pool_attn_dropout=pool_attn_dropout, pool_proj_dropout=pool_proj_dropout,
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
        self.adv_disc: Optional[GenderDiscriminator] = (
            GenderDiscriminator(in_dim=final_size, hidden=adv_disc_hidden, dropout=adv_disc_dropout)
            if enable_adv_disc else None
        )

        self._init_weights(target_mean=target_mean)

    def _init_weights(self, target_mean: Optional[float] = None) -> None:
        import math
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if target_mean is not None and self.output_activation == "sigmoid":
            eps = 1e-6
            p = max(eps, min(1.0 - eps, float(target_mean)))
            bias_init = math.log(p / (1.0 - p))
        else:
            bias_init = 0.0
        nn.init.constant_(self.head.bias, bias_init)
        for m in self.pool.modules():
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
        features = self.dropout(pooled)
        logits = self.head(features)
        pred = self._activate(logits).squeeze(-1) if self.output_dim == 1 else self._activate(logits)

        out: Dict[str, torch.Tensor] = {"logits": pred}
        if self.training:
            out["features"] = features
            if attn_weights is not None:
                out["attn_weights"] = attn_weights
            if self.adv_disc is not None:
                out["adv_logits"] = self.adv_disc(grad_reverse(features, 1.0))
        return out

    @classmethod
    def load_from_mlflow(cls, model_uri: str, output_dim: Optional[int] = None) -> "FaceOccRegressor":
        import mlflow
        model = mlflow.pytorch.load_model(model_uri)
        return model
