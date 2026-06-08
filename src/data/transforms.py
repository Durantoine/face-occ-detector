from typing import Callable

from PIL import Image


def _identity(img: Image.Image) -> Image.Image:
    return img


def build_train_transform(level: str = "light") -> Callable[[Image.Image], Image.Image]:
    """Build a train-time PIL->PIL transform.

    Levels (back-compat with the v19 broad/refined sweeps):
      - "none"         : identity (no augmentation)
      - "light"        : HFlip + mild ColorJitter
      - "medium"       : "light" + RandomRotation + GaussianBlur
      - "strong"       : "medium" + RandAugment
      - "tier1_safe"   : enriched safe-only ops (NEW). Every op preserves the
                         FaceOcclusion label because it never adds/removes
                         occlusion, never crops the face, never erases pixels.
                         Designed for the augmentation experiment alongside
                         the synthetic-occluder dataset (Tier 2).
    """
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

    if level == "tier1_safe":
        # === Geometry: horizontal flip ===
        # Safe by face symmetry (occlusion area is invariant to L<->R mirror).
        ops.append(T.RandomHorizontalFlip(p=0.5))

        # === Colour / photometric — all pixel-level, never touch geometry ===
        # Wider ColorJitter than v19's "light" (0.30/0.30/0.20/0.05 vs 0.15/0.15/0.08/0.02).
        # Simulates real-world lighting / camera variation.
        ops.append(T.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.20, hue=0.05))

        # Random desaturation: forces the model to rely on shape, not colour.
        ops.append(T.RandomGrayscale(p=0.10))

        # Auto-contrast and equalize add tonal variation. Both are pixel-level
        # remappings that preserve geometry.
        ops.append(T.RandomAutocontrast(p=0.20))
        ops.append(T.RandomEqualize(p=0.20))

        # Sharpness variation: simulates camera focus / motion blur diff.
        ops.append(T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.30))

        # Random Gaussian blur — slightly wider sigma than v19's medium.
        ops.append(T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)))

        # Posterize (colour quantisation) at low probability — robust to
        # heavy JPEG / low-bit-depth artefacts seen in some test images.
        ops.append(T.RandomPosterize(bits=4, p=0.10))

        # === No rotation, no crop, no erasing in tier1_safe ===
        # Rationale: rotation can push face pixels out of the 224x224 frame,
        # crops change the visible face area, RandomErasing literally adds
        # occlusion. We want strict label invariance here. Geometric and
        # erasing augmentations belong to Tier 2 (synthetic occluder dataset),
        # which generates the new label explicitly.

    return T.Compose(ops) if ops else _identity
