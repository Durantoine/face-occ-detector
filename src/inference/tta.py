from typing import Any, Callable, List, Optional

import torch
from PIL import Image


def hflip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def identity(img: Image.Image) -> Image.Image:
    return img


def default_tta() -> List[Callable[[Image.Image], Image.Image]]:
    return [identity, hflip]


@torch.no_grad()
def predict_tta_batch(
    model: Any,
    processor: Any,
    image_paths: List[str],
    transforms: Optional[List[Callable]] = None,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    image_base_dir: Optional[str] = None,
) -> torch.Tensor:
    from pathlib import Path
    if transforms is None:
        transforms = default_tta()
    device = device or next(model.parameters()).device
    n_tta = len(transforms)
    img_per_batch = max(1, batch_size // n_tta)
    base = Path(image_base_dir) if image_base_dir else None

    out: List[torch.Tensor] = []
    for i in range(0, len(image_paths), img_per_batch):
        chunk = image_paths[i:i + img_per_batch]
        imgs = []
        for p in chunk:
            full = base / p if (base and not Path(p).is_absolute()) else Path(p)
            imgs.append(Image.open(full).convert("RGB"))
        variants = [t(img) for img in imgs for t in transforms]
        enc = processor(images=variants, return_tensors="pt")
        preds = model(pixel_values=enc["pixel_values"].to(device))["logits"]
        preds = preds.view(len(imgs), n_tta).mean(dim=1)
        out.append(preds.detach().cpu())
    return torch.cat(out, dim=0).clamp(0.0, 1.0)
