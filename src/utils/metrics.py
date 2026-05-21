from typing import Any, Dict, Optional

import numpy as np


def _weighted_err(
    pred: np.ndarray,
    gt: np.ndarray,
    weight_offset: float = 1.0 / 30.0,
    w_imp: Optional[np.ndarray] = None,
) -> float:
    if len(gt) == 0:
        return 0.0
    w = weight_offset + gt
    if w_imp is not None:
        w = w * w_imp
    num = float((w * (pred - gt) ** 2).sum())
    den = float(w.sum())
    return num / den if den > 0 else 0.0


def compute_score(
    pred: np.ndarray,
    gt: np.ndarray,
    gender: np.ndarray,
    importance_pmf_ratio: Optional[np.ndarray] = None,
    bin_width: float = 0.025,
) -> Dict[str, float]:
    pred = np.asarray(pred).astype(np.float64).flatten()
    gt = np.asarray(gt).astype(np.float64).flatten()
    gender = np.asarray(gender).astype(np.float64).flatten()

    w_imp = None
    if importance_pmf_ratio is not None:
        ratio = np.asarray(importance_pmf_ratio).astype(np.float64).flatten()
        idx = np.clip((gt / bin_width).astype(int), 0, len(ratio) - 1)
        w_imp = ratio[idx]

    mask_f = gender < 0.5
    mask_m = gender >= 0.5

    err_f = _weighted_err(pred[mask_f], gt[mask_f], w_imp=(w_imp[mask_f] if w_imp is not None else None))
    err_m = _weighted_err(pred[mask_m], gt[mask_m], w_imp=(w_imp[mask_m] if w_imp is not None else None))
    score = (err_f + err_m) / 2.0 + abs(err_f - err_m)

    return {
        "score": score,
        "err_F": err_f,
        "err_M": err_m,
        "err_diff": abs(err_f - err_m),
        "mse": float(((pred - gt) ** 2).mean()),
        "mae": float(np.abs(pred - gt).mean()),
    }


def make_compute_metrics(importance_pmf_ratio: Optional[np.ndarray] = None, bin_width: float = 0.025):
    def compute_metrics(p: Any, compute_result: bool = True, **kwargs: Any) -> Dict[str, float]:
        preds_raw = p.predictions
        if isinstance(preds_raw, (tuple, list)):
            preds_raw = preds_raw[0]
        preds = np.asarray(preds_raw).astype(np.float64).flatten()
        labels = np.asarray(p.label_ids).astype(np.float64)

        if labels.ndim == 2 and labels.shape[1] >= 2:
            gt = labels[:, 0]
            gender = labels[:, 1]
        else:
            gt = labels.flatten()
            gender = np.zeros_like(gt)

        return compute_score(preds, gt, gender, importance_pmf_ratio=importance_pmf_ratio, bin_width=bin_width)

    return compute_metrics


def compute_metrics(p: Any, compute_result: bool = True, **kwargs: Any) -> Dict[str, float]:
    return make_compute_metrics()(p, compute_result=compute_result, **kwargs)
