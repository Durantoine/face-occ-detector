from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image


def hflip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def identity(img: Image.Image) -> Image.Image:
    return img


def _zoom_crop(scale: float) -> Callable[[Image.Image], Image.Image]:
    def t(img: Image.Image) -> Image.Image:
        if scale == 1.0:
            return img
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        resized = img.resize((new_w, new_h), Image.BICUBIC)
        if scale > 1.0:
            left, top = (new_w - w) // 2, (new_h - h) // 2
            return resized.crop((left, top, left + w, top + h))
        canvas = Image.new("RGB", (w, h))
        canvas.paste(resized, ((w - new_w) // 2, (h - new_h) // 2))
        return canvas
    return t


def default_tta() -> List[Callable[[Image.Image], Image.Image]]:
    return [identity, hflip]


def extended_tta() -> List[Callable[[Image.Image], Image.Image]]:
    transforms = []
    for scale in (0.95, 1.0, 1.05):
        zoom = _zoom_crop(scale)
        transforms.append(zoom)
        transforms.append(lambda img, z=zoom: hflip(z(img)))
    return transforms


@torch.no_grad()
def predict_tta_batch(
    model: Any,
    processor: Any,
    image_paths: List[str],
    transforms: Optional[List[Callable]] = None,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if transforms is None:
        transforms = default_tta()
    device = device or next(model.parameters()).device
    n_tta = len(transforms)
    img_per_batch = max(1, batch_size // n_tta)

    out_probs: List[torch.Tensor] = []
    for i in range(0, len(image_paths), img_per_batch):
        chunk = image_paths[i:i + img_per_batch]
        pil_images = [Image.open(p).convert("RGB") for p in chunk]
        variants = [t(img) for img in pil_images for t in transforms]
        enc = processor(images=variants, return_tensors="pt")
        logits = model(pixel_values=enc["pixel_values"].to(device))["logits"]
        probs = F.softmax(logits, dim=-1).view(len(pil_images), n_tta, -1).mean(dim=1)
        out_probs.append(probs.cpu())
    return torch.cat(out_probs, dim=0)
