from typing import Any, Dict, Optional

import numpy as np


def _weighted_err(
    pred: np.ndarray,
    gt: np.ndarray,
    weight_offset: float = 1.0 / 30.0,
) -> float:
    if len(gt) == 0:
        return 0.0
    w = weight_offset + gt
    num = float((w * (pred - gt) ** 2).sum())
    den = float(w.sum())
    return num / den if den > 0 else 0.0


def _r2_weighted(pred: np.ndarray, gt: np.ndarray) -> float:
    if len(gt) == 0:
        return 0.0
    mean_y = float(gt.mean())
    ss_res = float(((pred - gt) ** 2).sum())
    ss_tot = float(((gt - mean_y) ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_score(
    pred: np.ndarray,
    gt: np.ndarray,
    gender: np.ndarray,
) -> Dict[str, float]:
    """Standard challenge metric on samples drawn from the eval distribution.

    Use this when val/test samples already follow the target distribution
    (e.g., test holdout matching P_test, or val resampled to P_test).
    For val drawn iid from P_train, use compute_score_stratified_is below.
    """
    pred = np.asarray(pred).astype(np.float64).flatten()
    gt = np.asarray(gt).astype(np.float64).flatten()
    gender = np.asarray(gender).astype(np.float64).flatten()

    mask_f = gender < 0.5
    mask_m = gender >= 0.5

    err_f = _weighted_err(pred[mask_f], gt[mask_f])
    err_m = _weighted_err(pred[mask_m], gt[mask_m])
    score = (err_f + err_m) / 2.0 + abs(err_f - err_m)

    abs_err = np.abs(pred - gt)
    return {
        "challenge_score": score,
        "err_F": err_f,
        "err_M": err_m,
        "err_diff": abs(err_f - err_m),
        "mae_pct": float(abs_err.mean() * 100.0),
        "r2": _r2_weighted(pred, gt),
        "mse": float(((pred - gt) ** 2).mean()),
        "mae": float(abs_err.mean()),
    }


def compute_score_stratified_is(
    pred: np.ndarray,
    gt: np.ndarray,
    gender: np.ndarray,
    test_pmf_joint: np.ndarray,
    bin_width: float,
    n_bins: int,
    weight_offset: float = 1.0 / 30.0,
) -> Dict[str, float]:
    """Stratified Importance Sampling estimator of the challenge metric on test
    distribution, using samples drawn iid from P_train (val set).

    Formula per group g:
        Err_g^test  =  sum_b  P_test(g, b) · E[w·err | g, b]
                      ──────────────────────────────────────────
                       sum_b  P_test(g, b) · E[w     | g, b]

    where E[.|g,b] is the empirical mean over val samples in cell (g, b).

    This is Rao-Blackwellised IS: instead of per-sample ratios r_i = P_test(x_i)/P_train(x_i)
    (high variance when ratios are extreme on rare bins), we average per cell first then
    weight by known P_test(g, b). Variance is bounded by the within-cell variance.

    Args:
        pred, gt, gender: arrays from val (drawn iid from P_train)
        test_pmf_joint: shape (2, n_bins), joint P_test(g, y_bin) estimated via H_C
        bin_width, n_bins: binning of Y
    """
    pred = np.asarray(pred).astype(np.float64).flatten()
    gt = np.asarray(gt).astype(np.float64).flatten()
    gender = np.asarray(gender).astype(np.float64).flatten()

    g_int = (gender >= 0.5).astype(int)
    bin_idx = np.clip((gt / bin_width).astype(int), 0, n_bins - 1)
    err = (pred - gt) ** 2
    w = weight_offset + gt

    err_per_group = np.zeros(2, dtype=np.float64)
    skipped_cells = 0
    for gi in (0, 1):
        num = 0.0
        den = 0.0
        for b in range(n_bins):
            mask = (g_int == gi) & (bin_idx == b)
            n_in_cell = int(mask.sum())
            p_cell = float(test_pmf_joint[gi, b])
            if p_cell <= 0:
                continue
            if n_in_cell == 0:
                skipped_cells += 1
                continue
            mean_w_err = float((w[mask] * err[mask]).sum()) / n_in_cell
            mean_w = float(w[mask].sum()) / n_in_cell
            num += p_cell * mean_w_err
            den += p_cell * mean_w
        err_per_group[gi] = num / den if den > 0 else 0.0

    err_f, err_m = err_per_group
    score = (err_f + err_m) / 2.0 + abs(err_f - err_m)

    abs_err = np.abs(pred - gt)
    out = {
        "challenge_score": float(score),
        "err_F": float(err_f),
        "err_M": float(err_m),
        "err_diff": float(abs(err_f - err_m)),
        "mae_pct": float(abs_err.mean() * 100.0),
        "r2": _r2_weighted(pred, gt),
        "mse": float(((pred - gt) ** 2).mean()),
        "mae": float(abs_err.mean()),
        "_is_stratified_skipped_cells": float(skipped_cells),
    }
    return out


def make_compute_metrics(test_pmf_joint: Optional[np.ndarray] = None,
                          bin_width: float = 1.0 / 30.0,
                          n_bins: int = 15):
    """Build compute_metrics function for HF Trainer.

    If test_pmf_joint provided: val is iid from P_train → use stratified IS to estimate
    the test-distribution metric. Otherwise: standard per-sample compute_score.
    """
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

        if test_pmf_joint is not None:
            return compute_score_stratified_is(preds, gt, gender, test_pmf_joint, bin_width, n_bins)
        return compute_score(preds, gt, gender)
    return compute_metrics
