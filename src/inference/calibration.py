from typing import Tuple

import numpy as np

from src.utils.metrics import _weighted_err, compute_score


def find_optimal_bias(
    preds: np.ndarray,
    gt: np.ndarray,
    gender: np.ndarray,
    delta_range: Tuple[float, float] = (-0.10, 0.10),
    n_steps: int = 41,
) -> Tuple[float, float, float]:
    preds = np.asarray(preds, dtype=np.float64).flatten()
    gt = np.asarray(gt, dtype=np.float64).flatten()
    gender = np.asarray(gender, dtype=np.float64).flatten()

    mask_f = gender < 0.5
    mask_m = gender >= 0.5
    p_f, gt_f = preds[mask_f], gt[mask_f]
    p_m, gt_m = preds[mask_m], gt[mask_m]

    grid = np.linspace(delta_range[0], delta_range[1], n_steps)
    best = (0.0, 0.0, float("inf"))
    for df in grid:
        err_f = _weighted_err(np.clip(p_f + df, 0.0, 1.0), gt_f)
        for dm in grid:
            err_m = _weighted_err(np.clip(p_m + dm, 0.0, 1.0), gt_m)
            score = (err_f + err_m) / 2.0 + abs(err_f - err_m)
            if score < best[2]:
                best = (float(df), float(dm), float(score))
    return best


def apply_bias(preds: np.ndarray, gender: np.ndarray, delta_f: float, delta_m: float) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.float64).flatten()
    gender = np.asarray(gender, dtype=np.float64).flatten()
    out = preds.copy()
    out[gender < 0.5] += delta_f
    out[gender >= 0.5] += delta_m
    return np.clip(out, 0.0, 1.0)


def calibrate_and_report(
    preds: np.ndarray,
    gt: np.ndarray,
    gender: np.ndarray,
    delta_range: Tuple[float, float] = (-0.10, 0.10),
    n_steps: int = 41,
) -> dict:
    before = compute_score(preds, gt, gender)
    delta_f, delta_m, _ = find_optimal_bias(preds, gt, gender, delta_range, n_steps)
    corrected = apply_bias(preds, gender, delta_f, delta_m)
    after = compute_score(corrected, gt, gender)
    return {
        "delta_f": delta_f, "delta_m": delta_m,
        "before": before, "after": after,
        "improvement": before["score"] - after["score"],
    }
