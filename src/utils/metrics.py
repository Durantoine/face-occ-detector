from typing import Any, Dict

import numpy as np


def _weighted_err(pred: np.ndarray, gt: np.ndarray, weight_offset: float = 1.0 / 30.0) -> float:
    if len(gt) == 0:
        return 0.0
    w = weight_offset + gt
    num = float((w * (pred - gt) ** 2).sum())
    den = float(w.sum())
    return num / den if den > 0 else 0.0


def compute_score(pred: np.ndarray, gt: np.ndarray, gender: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred).astype(np.float64).flatten()
    gt = np.asarray(gt).astype(np.float64).flatten()
    gender = np.asarray(gender).astype(np.float64).flatten()

    mask_f = gender < 0.5
    mask_m = gender >= 0.5

    err_f = _weighted_err(pred[mask_f], gt[mask_f])
    err_m = _weighted_err(pred[mask_m], gt[mask_m])
    score = (err_f + err_m) / 2.0 + abs(err_f - err_m)

    return {
        "score": score,
        "err_F": err_f,
        "err_M": err_m,
        "err_diff": abs(err_f - err_m),
        "mse": float(((pred - gt) ** 2).mean()),
        "mae": float(np.abs(pred - gt).mean()),
    }


def compute_metrics(p: Any) -> Dict[str, float]:
    preds = np.asarray(p.predictions).astype(np.float64).flatten()
    labels = np.asarray(p.label_ids).astype(np.float64)

    if labels.ndim == 2 and labels.shape[1] >= 2:
        gt = labels[:, 0]
        gender = labels[:, 1]
    else:
        gt = labels.flatten()
        gender = np.zeros_like(gt)

    return compute_score(preds, gt, gender)
