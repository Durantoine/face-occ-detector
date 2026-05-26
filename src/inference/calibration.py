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


def quantile_match_to_test_pmf(
    predictions: np.ndarray,
    test_pmf: np.ndarray,
    bin_width: float = 0.025,
) -> np.ndarray:
    """Post-hoc histogram matching: transform predictions so their empirical
    marginal distribution matches `test_pmf` (the known/estimated P_test(Y)).

    Monotonic transformation T(ŷ) = F_test⁻¹(F_pred(ŷ)), where F_pred is the
    empirical CDF of the predictions and F_test is the CDF built from test_pmf.

    Effect:
      * Corrects systematic shift in the prediction distribution (e.g. model
        under-predicting high-Y values because they were rare in train)
      * Preserves the RANK ORDER of predictions (monotonic)
      * Reduces bias if the model's predictions have wrong moments; does NOT
        help if errors are purely random noise

    Use as the very last step before saving the submission CSV. Free win when
    the model has been trained on a distribution shifted vs test.
    """
    preds = np.asarray(predictions, dtype=np.float64).flatten()
    test = np.asarray(test_pmf, dtype=np.float64).flatten()
    if len(preds) == 0:
        return preds.copy()

    # Target distribution: build a piecewise linear CDF on bin upper edges.
    # bin b spans [b·δ, (b+1)·δ], target P_test[b] mass per bin.
    n_bins = len(test)
    upper_edges = (np.arange(n_bins) + 1) * bin_width
    target_cdf = np.cumsum(test)
    if target_cdf[-1] <= 0:
        return preds.copy()
    target_cdf = target_cdf / target_cdf[-1]
    # Prepend the origin (cdf=0 at y=0) to make np.interp well-defined for low quantiles
    target_cdf = np.concatenate([[0.0], target_cdf])
    target_values = np.concatenate([[0.0], upper_edges])

    # Empirical quantile of each prediction. Using ranks with the (k+0.5)/n
    # "plotting position" convention (unbiased for symmetric distributions).
    n = len(preds)
    ranks = np.argsort(np.argsort(preds))  # 0..n-1
    empirical_quantiles = (ranks + 0.5) / n

    # Map each quantile to its corresponding y-value under target distribution
    mapped = np.interp(empirical_quantiles, target_cdf, target_values)
    return np.clip(mapped, 0.0, 1.0)


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
