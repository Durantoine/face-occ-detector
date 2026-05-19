from typing import List, Optional, Tuple

import numpy as np
import torch

from src.inference.tta import default_tta, extended_tta, predict_tta_batch
from src.predict import load_model, predict_images


def ensemble_predict(
    model_uris: List[str],
    image_paths: List[str],
    tracking_uri: str = "sqlite:///mlflow.db",
    use_tta: bool = True,
    tta_mode: str = "default",
    batch_size: int = 16,
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    if weights is not None and len(weights) != len(model_uris):
        raise ValueError("len(weights) must match len(model_uris)")
    w = torch.tensor(weights, dtype=torch.float32) if weights else None
    if w is not None:
        w = w / w.sum()

    transforms = extended_tta() if tta_mode == "extended" else default_tta()

    acc: Optional[torch.Tensor] = None
    for i, uri in enumerate(model_uris):
        print(f"[ensemble {i + 1}/{len(model_uris)}] loading {uri}")
        model, processor = load_model(uri, tracking_uri)
        if use_tta:
            probs = predict_tta_batch(model, processor, image_paths, transforms, batch_size=batch_size)
        else:
            res = predict_images(model, processor, image_paths, batch_size=batch_size)
            probs = torch.tensor(res["probabilities"])
        coef = float(w[i]) if w is not None else 1.0 / len(model_uris)
        acc = coef * probs if acc is None else acc + coef * probs
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert acc is not None
    return acc


def threshold_sweep(
    probs: torch.Tensor,
    labels: torch.Tensor,
    metric_fn,
    n: int = 81,
) -> Tuple[float, float]:
    if probs.shape[1] != 2:
        raise ValueError("threshold_sweep only supports binary classification")
    pos = probs[:, 1].numpy()
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    best_tau, best_score = 0.5, -1.0
    for tau in np.linspace(0.1, 0.9, n):
        preds = (pos >= tau).astype(int)
        s = metric_fn(preds, labels_np)
        if s > best_score:
            best_tau, best_score = float(tau), float(s)
    return best_tau, best_score
