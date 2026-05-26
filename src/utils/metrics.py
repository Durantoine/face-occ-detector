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

    err_f_raw = _weighted_err(pred[mask_f], gt[mask_f], w_imp=None)
    err_m_raw = _weighted_err(pred[mask_m], gt[mask_m], w_imp=None)
    score_raw = (err_f_raw + err_m_raw) / 2.0 + abs(err_f_raw - err_m_raw)

    err_f = _weighted_err(pred[mask_f], gt[mask_f], w_imp=(w_imp[mask_f] if w_imp is not None else None))
    err_m = _weighted_err(pred[mask_m], gt[mask_m], w_imp=(w_imp[mask_m] if w_imp is not None else None))
    score = (err_f + err_m) / 2.0 + abs(err_f - err_m)

    abs_err = np.abs(pred - gt)
    sq_err = (pred - gt) ** 2

    return {
        "challenge_score_test_estimated": score,
        "err_F_test_estimated":           err_f,
        "err_M_test_estimated":           err_m,
        "err_diff_test_estimated":        abs(err_f - err_m),
        "challenge_score_val":            score_raw,
        "err_F_val":                      err_f_raw,
        "err_M_val":                      err_m_raw,
        "err_diff_val":                   abs(err_f_raw - err_m_raw),
        "mse_val":                _weighted_mean(sq_err, None),
        "mae_val":                _weighted_mean(abs_err, None),
        "mae_pct_val":            float(abs_err.mean() * 100.0),
        "mae_pct_test_estimated": _weighted_mean(abs_err, w_imp) * 100.0,
        "r2_val":                 _r2_weighted(pred, gt, None),
        "r2_test_estimated":      _r2_weighted(pred, gt, w_imp),
    }


def _weighted_mean(values: np.ndarray, weights: Optional[np.ndarray]) -> float:
    if len(values) == 0:
        return 0.0
    if weights is None:
        return float(values.mean())
    den = float(weights.sum())
    return float((weights * values).sum() / den) if den > 0 else 0.0


def _r2_weighted(pred: np.ndarray, gt: np.ndarray, weights: Optional[np.ndarray]) -> float:
    """R² = 1 - SS_res / SS_tot, with optional sample weights.
    Returns 0.0 when var(gt) is zero (no signal to explain).
    """
    if len(gt) == 0:
        return 0.0
    if weights is None:
        mean_y = float(gt.mean())
        ss_res = float(((pred - gt) ** 2).sum())
        ss_tot = float(((gt - mean_y) ** 2).sum())
    else:
        wsum = float(weights.sum())
        if wsum <= 0:
            return 0.0
        mean_y = float((weights * gt).sum() / wsum)
        ss_res = float((weights * (pred - gt) ** 2).sum())
        ss_tot = float((weights * (gt - mean_y) ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


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
