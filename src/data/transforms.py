from typing import Callable

from PIL import Image


def _identity(img: Image.Image) -> Image.Image:
    return img


def build_train_transform(level: str = "light") -> Callable[[Image.Image], Image.Image]:
    if level == "none":
        return _identity

    try:
        from torchvision import transforms as T
    except ImportError as e:
        raise ImportError("torchvision is required for augmentation. Install it.") from e

    ops: list[Callable[[Image.Image], Image.Image]] = []

    if level in ("light", "medium", "strong"):
        ops.append(T.RandomHorizontalFlip(p=0.5))
        ops.append(T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.02))

    if level in ("medium", "strong"):
        ops.append(T.RandomRotation(degrees=10, fill=0))
        ops.append(T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)))

    if level == "strong":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))

    return T.Compose(ops) if ops else _identity
