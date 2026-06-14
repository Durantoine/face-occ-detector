from __future__ import annotations

import re

import torch
import torch.nn as nn

from src.models.sapiens2_loader import hidden_size_of, is_sapiens2, load_sapiens2


# Cache the pretrained backbone weights in memory so an Optuna sweep (one process, many trials,
# same backbone) doesn't reload ~hundreds of MB from disk on every trial.
_TIMM_PRETRAINED_CACHE: dict = {}


class _TimmTokens(nn.Module):
    def __init__(self, name: str, pretrained: bool, drop_path: float) -> None:
        super().__init__()
        import timm
        cached = _TIMM_PRETRAINED_CACHE.get(name) if pretrained else None
        self.m = timm.create_model(name, pretrained=(pretrained and cached is None),
                                   num_classes=0, global_pool="", drop_path_rate=drop_path)
        if cached is not None:
            self.m.load_state_dict(cached, strict=True)
        elif pretrained:
            _TIMM_PRETRAINED_CACHE[name] = {k: v.detach().cpu() for k, v in self.m.state_dict().items()}
        self.num_features = self.m.num_features
        self.num_prefix = int(getattr(self.m, "num_prefix_tokens", 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.m.forward_features(x)
        if t.dim() == 4:
            b, c, h, w = t.shape
            t = t.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return t


class _SapiensTokens(nn.Module):
    def __init__(self, name: str, pretrained: bool, drop_path: float) -> None:
        super().__init__()
        self.num_features = hidden_size_of(name)
        # Sapiens-2: 1 CLS token + 8 register tokens = 9 prefix tokens
        # This leaves 196 spatial tokens for 224x224 images (14x14 grid)
        self.num_prefix = 9
        self.m = load_sapiens2(name, image_size=224, drop_rate=drop_path, pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.m(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("last_hidden_state", next(iter(out.values())))
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out


class _DinoTokens(nn.Module):
    def __init__(self, name: str, pretrained: bool, drop_path: float) -> None:
        super().__init__()
        from src.models.dinov3_loader import hidden_size_of as dino_hidden, load_dinov3
        self.m = load_dinov3(name, drop_path_rate=drop_path, pretrained=pretrained)
        self.num_features = dino_hidden(name)
        self.num_prefix = 0  # we return patch tokens only (cls/register tokens already dropped)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.m.forward_features(x) if hasattr(self.m, "forward_features") else self.m(x)
        if isinstance(out, dict):
            return out["x_norm_patchtokens"]  # (B, N=196, C) spatial tokens, N a perfect square
        return out


def _is_dinov3(name: str) -> bool:
    return name.startswith("dinov3_")


def _load_ibot(backbone: nn.Module, uri: str) -> tuple[int, int, str | None]:
    import mlflow
    print(f"[backbone] loading iBOT model from {uri}...", flush=True)
    pretrained_model = mlflow.pytorch.load_model(uri)
    print(f"[backbone] iBOT model loaded. Updating backbone state_dict...", flush=True)
    missing, unexpected = backbone.m.load_state_dict(pretrained_model.state_dict(), strict=False)
    m = re.match(r"runs:/([^/]+)/", uri)
    return len(missing), len(unexpected), (m.group(1) if m else None)


def _make_backbone(name: str, pretrained: bool, drop_path: float) -> nn.Module:
    if _is_dinov3(name):
        return _DinoTokens(name, pretrained, drop_path)
    if is_sapiens2(name):
        return _SapiensTokens(name, pretrained, drop_path)
    return _TimmTokens(name, pretrained, drop_path)


def build_backbone(name: str, pretrained: bool = True, drop_path: float = 0.1,
                   init_backbone_from: str | None = None) -> nn.Module:
    if init_backbone_from:
        backbone = _make_backbone(name, False, drop_path)  # iBOT weights overwrite the init anyway
        miss, unexp, run_id = _load_ibot(backbone, init_backbone_from)
        print(f"[backbone] iBOT init from {init_backbone_from}: {miss} missing / {unexp} unexpected", flush=True)
        backbone.init_pretrain_run_id = run_id
        backbone.init_missing_keys = miss
        backbone.init_unexpected_keys = unexp
        return backbone
    return _make_backbone(name, pretrained, drop_path)
