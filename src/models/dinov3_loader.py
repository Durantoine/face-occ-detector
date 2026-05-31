from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoImageProcessor


_REPO = Path(__file__).parent / "dinov3_repo"
_WEIGHTS_DIR = Path(__file__).parent / "weights"

_AVAILABLE_WEIGHTS = {
    "dinov3_vits16":     _WEIGHTS_DIR / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vits16plus": _WEIGHTS_DIR / "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "dinov3_vitb16":     _WEIGHTS_DIR / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16":     _WEIGHTS_DIR / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vitl16plus": _WEIGHTS_DIR / "dinov3_vitl16plus_pretrain_lvd1689m-46503df0.pth",
    "dinov3_vith16plus": _WEIGHTS_DIR / "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
}

_VARIANTS = {
    "dinov3_vits16":     (22e6,  384,  2),
    "dinov3_vits16plus": (29e6,  384,  3),
    "dinov3_vitb16":     (86e6,  768,  5),
    "dinov3_vitl16":     (300e6, 1024, 10),
    "dinov3_vitl16plus": (400e6, 1024, 13),
    "dinov3_vith16plus": (600e6, 1280, 17),
}

_HIDDEN_SIZES = {k: v[1] for k, v in _VARIANTS.items()}


def hidden_size_of(arch: str) -> int:
    if arch not in _HIDDEN_SIZES:
        raise ValueError(f"Unknown DINOv3 arch '{arch}'. Available: {list(_HIDDEN_SIZES)}")
    return _HIDDEN_SIZES[arch]


def _set_drop_path_rate(model: torch.nn.Module, drop_path_rate: float) -> None:
    """Override stochastic depth in DINOv3 ViT blocks via linear scaling 0 → drop_path_rate.

    DINOv3 uses sample-level stochastic depth stored as `block.sample_drop_ratio` (not
    a DropPath module). We mutate that attribute directly.
    """
    if drop_path_rate <= 0 or not hasattr(model, "blocks"):
        return
    blocks = list(model.blocks)
    n = max(len(blocks) - 1, 1)
    applied = 0
    for i, block in enumerate(blocks):
        target = drop_path_rate * i / n
        if hasattr(block, "sample_drop_ratio"):
            block.sample_drop_ratio = target
            applied += 1
    if applied == 0:
        print(f"WARNING: backbone_drop_path_rate={drop_path_rate} requested but blocks have no sample_drop_ratio attribute (no-op)")
    else:
        print(f"DINOv3 drop_path_rate={drop_path_rate} applied (linear 0 → {drop_path_rate:.3f} across {applied} blocks via sample_drop_ratio)")


def load_dinov3(
    arch: str = "dinov3_vits16",
    device: Optional[torch.device] = None,
    drop_path_rate: float = 0.0,
    pretrained: bool = True,
) -> torch.nn.Module:
    if not _REPO.exists():
        raise FileNotFoundError(f"DINOv3 repo not found at {_REPO}")

    model = torch.hub.load(str(_REPO), arch, source="local", pretrained=False)
    _set_drop_path_rate(model, drop_path_rate)

    if not pretrained:
        print(f"DINOv3 {arch}: random init (pretrained=False)")
    else:
        weights_path = _AVAILABLE_WEIGHTS.get(arch)
        if weights_path and weights_path.exists():
            state = torch.load(weights_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state, strict=True)
            print(f"Loaded DINOv3 weights: {weights_path.name}")
        else:
            print(f"WARNING: no weights for {arch} at {weights_path} — model is randomly initialized")

    if device is not None:
        model = model.to(device)
    return model


def recommend_dinov3_variant(headroom_gb: float = 2.0) -> str:
    if not torch.cuda.is_available():
        return "dinov3_vits16"
    total_gb = sum(
        torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        for i in range(torch.cuda.device_count())
    )
    budget = total_gb - headroom_gb
    best = "dinov3_vits16"
    for name, (_, _, vram_need) in _VARIANTS.items():
        if vram_need <= budget and _VARIANTS[name][0] > _VARIANTS[best][0]:
            best = name
    print(f"VRAM available: {total_gb:.1f} GB → recommended: {best}")
    return best


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_image_processor(model_name: str) -> Any:
    if model_name.startswith("dinov3_") or "sapiens" in model_name.lower():
        proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        proc.image_mean = list(IMAGENET_MEAN)
        proc.image_std = list(IMAGENET_STD)
        return proc
    # timm CNN backbones (efficientnet*, resnet*, convnext*, etc.) — use ImageNet stats
    # via a generic ViT processor (just rescale + normalize, resize handled to 224x224)
    timm_prefixes = ("efficientnet", "resnet", "resnext", "convnext", "regnet", "mobilenetv", "tf_efficientnet")
    if any(model_name.startswith(p) for p in timm_prefixes):
        proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        proc.image_mean = list(IMAGENET_MEAN)
        proc.image_std = list(IMAGENET_STD)
        return proc
    return AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
