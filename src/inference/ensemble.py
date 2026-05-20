from typing import List, Optional

import torch

from src.inference.tta import default_tta, predict_tta_batch
from src.predict import load_model, predict_images


def ensemble_predict(
    model_uris: List[str],
    image_paths: List[str],
    tracking_uri: str = "sqlite:///mlflow.db",
    use_tta: bool = True,
    batch_size: int = 32,
    image_base_dir: Optional[str] = None,
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    if weights is not None and len(weights) != len(model_uris):
        raise ValueError("len(weights) must match len(model_uris)")
    w = torch.tensor(weights, dtype=torch.float32) if weights else None
    if w is not None:
        w = w / w.sum()

    transforms = default_tta()
    acc: Optional[torch.Tensor] = None
    for i, uri in enumerate(model_uris):
        print(f"[ensemble {i + 1}/{len(model_uris)}] loading {uri}")
        model, processor = load_model(uri, tracking_uri)
        if use_tta:
            preds = predict_tta_batch(
                model, processor, image_paths, transforms,
                batch_size=batch_size, image_base_dir=image_base_dir,
            )
        else:
            scalars = predict_images(model, processor, image_paths, image_base_dir, batch_size)
            preds = torch.tensor(scalars, dtype=torch.float32)
        coef = float(w[i]) if w is not None else 1.0 / len(model_uris)
        acc = coef * preds if acc is None else acc + coef * preds
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert acc is not None
    return acc.clamp(0.0, 1.0)
