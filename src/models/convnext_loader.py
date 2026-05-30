"""ConvNeXt v2 backbone loader for FaceOccRegressor.

Wraps a timm ConvNeXt v2 model so its 4-D spatial output (B, C, H, W) appears
as the (B, N+1, D) token sequence that the existing pooling layers expect:

    out[:, 0,  :]  = synthesized [CLS]-like global token (mean over spatial)
    out[:, 1:, :]  = H*W tokens, one per spatial location

This keeps CLSPooling, GeMPooling, AttentionPooling and MultiHeadAttentionPooling
compatible with no further changes — the only ViT-specific assumption is the
[CLS]-at-index-0 convention, which we synthesise.
"""
from typing import Any

import timm
import torch
import torch.nn as nn
from transformers import AutoImageProcessor


_VARIANTS = {
    "convnext_v2_atto":  ("convnextv2_atto.fcmae_ft_in22k_in1k",   3.7e6,  320),
    "convnext_v2_femto": ("convnextv2_femto.fcmae_ft_in22k_in1k",  5.2e6,  384),
    "convnext_v2_pico":  ("convnextv2_pico.fcmae_ft_in22k_in1k",   9.1e6,  512),
    "convnext_v2_nano":  ("convnextv2_nano.fcmae_ft_in22k_in1k",  15.6e6,  640),
    "convnext_v2_tiny":  ("convnextv2_tiny.fcmae_ft_in22k_in1k",  28.6e6,  768),
    "convnext_v2_base":  ("convnextv2_base.fcmae_ft_in22k_in1k",  88.7e6, 1024),
    "convnext_v2_large": ("convnextv2_large.fcmae_ft_in22k_in1k", 197.9e6, 1536),
    "convnext_v2_huge":  ("convnextv2_huge.fcmae_ft_in22k_in1k",  660.3e6, 2816),
}

_HIDDEN_SIZES = {k: v[2] for k, v in _VARIANTS.items()}


def is_convnext(model_name: str) -> bool:
    return model_name.startswith("convnext_")


def hidden_size_of(arch: str) -> int:
    if arch not in _HIDDEN_SIZES:
        raise ValueError(f"Unknown ConvNeXt arch \"{arch}\". Available: {list(_HIDDEN_SIZES)}")
    return _HIDDEN_SIZES[arch]


class _ConvNextTokenAdapter(nn.Module):
    """Adapts timm ConvNeXt spatial output (B, C, H, W) → (B, H*W+1, C) tokens."""

    def __init__(self, timm_model: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.model = timm_model
        self.hidden_size = hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(pixel_values)
        if feats.dim() != 4:
            raise RuntimeError(f"Expected 4-D ConvNeXt feature map, got shape {tuple(feats.shape)}")
        # Normalize to channel-first if timm ever returns (B, H, W, C).
        if feats.shape[1] != self.hidden_size and feats.shape[-1] == self.hidden_size:
            feats = feats.permute(0, 3, 1, 2)
        B, C, H, W = feats.shape
        patches = feats.flatten(2).transpose(1, 2)   # (B, H*W, C)
        cls = patches.mean(dim=1, keepdim=True)      # (B, 1, C) synthesised global token
        return torch.cat([cls, patches], dim=1)       # (B, H*W+1, C)

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        if hasattr(self.model, "set_grad_checkpointing"):
            self.model.set_grad_checkpointing(enable=True)
            return
        for stage in getattr(self.model, "stages", []):
            if hasattr(stage, "grad_checkpointing"):
                stage.grad_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.model, "set_grad_checkpointing"):
            self.model.set_grad_checkpointing(enable=False)
            return
        for stage in getattr(self.model, "stages", []):
            if hasattr(stage, "grad_checkpointing"):
                stage.grad_checkpointing = False


def load_convnext(
    arch: str = "convnext_v2_base",
    drop_path_rate: float = 0.0,
    pretrained: bool = True,
) -> nn.Module:
    if arch not in _VARIANTS:
        raise ValueError(f"Unknown ConvNeXt arch \"{arch}\". Available: {list(_VARIANTS)}")
    timm_name, _, hidden = _VARIANTS[arch]
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="",
        drop_path_rate=drop_path_rate,
    )
    return _ConvNextTokenAdapter(model, hidden_size=hidden)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_convnext_image_processor() -> Any:
    """Returns a 224x224 ImageNet-normalized processor matching ConvNeXt training."""
    proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    proc.image_mean = list(IMAGENET_MEAN)
    proc.image_std = list(IMAGENET_STD)
    return proc
