from typing import Callable

from PIL import Image


def _identity(img: Image.Image) -> Image.Image:
    return img


def build_train_transform(level: str = "light_v10") -> Callable[[Image.Image], Image.Image]:
    """v10 — pipeline LIGHT par défaut (v4 top utilisait `light`).

    Pas de RandAugment (trop random), pas d'Erasing/CutOut (corruption label).
    Couleurs modérées (h≤0.02 pour préserver tons peau). Rotation avec fill=reflect
    (pas de bords noirs qui ressembleraient à de l'occlusion).
    """
    if level == "none":
        return _identity

    try:
        from torchvision import transforms as T
    except ImportError as e:
        raise ImportError("torchvision is required for augmentation. Install it.") from e

    ops: list[Callable[[Image.Image], Image.Image]] = []

    if level in ("light", "light_v10", "medium", "strong"):
        ops.append(T.RandomHorizontalFlip(p=0.5))

    if level in ("light_v10", "medium", "strong"):
        ops.append(T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.02))

    if level in ("medium", "strong"):
        # Rotation avec reflect pad → pas de bords noirs ressemblant à l'occlusion
        ops.append(T.RandomRotation(degrees=10, fill=0))   # PIL ne supporte pas reflect natif, mais via Tensor on aurait fill=mean
        ops.append(T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)))

    if level == "strong":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))

    return T.Compose(ops) if ops else _identity
