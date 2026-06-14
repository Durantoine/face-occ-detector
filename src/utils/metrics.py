from __future__ import annotations

import numpy as np

WEIGHT_OFFSET = 1.0 / 30.0


def _err_g(pred: np.ndarray, gt: np.ndarray) -> float:
    if len(gt) == 0:
        return 0.0
    w = WEIGHT_OFFSET + gt
    return float((w * (pred - gt) ** 2).sum() / max(w.sum(), 1e-12))


def _r2(pred: np.ndarray, gt: np.ndarray) -> float:
    if len(gt) == 0:
        return 0.0
    ss_res = float(((pred - gt) ** 2).sum())
    ss_tot = float(((gt - gt.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def challenge_score(pred: np.ndarray, gt: np.ndarray, gender: np.ndarray) -> dict:
    pred = np.asarray(pred, float).ravel()
    gt = np.asarray(gt, float).ravel()
    g = np.asarray(gender, float).ravel()
    f, m = g < 0.5, g >= 0.5
    err_f, err_m = _err_g(pred[f], gt[f]), _err_g(pred[m], gt[m])
    abs_err = np.abs(pred - gt)
    gap_signed = err_f - err_m
    return {
        "challenge_score": (err_f + err_m) / 2.0 + abs(gap_signed),
        "err_F": err_f,
        "err_M": err_m,
        "err_diff": abs(gap_signed),
        "err_gap": gap_signed, # Signed gap
        "mae_pct": float(abs_err.mean() * 100.0),
        "r2": _r2(pred, gt),
    }
