from typing import Callable

from PIL import Image


def _identity(img: Image.Image) -> Image.Image:
    return img


def build_train_transform(level: str = "medium") -> Callable[[Image.Image], Image.Image]:
    if level == "none":
        return _identity

    try:
        from torchvision import transforms as T
    except ImportError as e:
        raise ImportError("torchvision is required for augmentation. Install it.") from e

    ops: list[Callable[[Image.Image], Image.Image]] = []

    if level in ("light", "medium", "strong"):
        ops.append(T.RandomHorizontalFlip(p=0.5))

    if level in ("medium", "strong"):
        ops.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02))
        ops.append(T.RandomRotation(degrees=8, fill=0))

    if level == "strong":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))

    return T.Compose(ops) if ops else _identity
