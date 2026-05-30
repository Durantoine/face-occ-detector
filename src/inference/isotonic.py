from typing import Dict

import numpy as np
from sklearn.isotonic import IsotonicRegression


class GenderConditionalIsotonic:
    """Per-gender isotonic regression on (pred -> y) calibration.

    Fit on val (preds_val, y_val, gender_val) — one IsotonicRegression per gender.
    Apply at inference: needs gender at test time (TODO: from MID lookup + Sapiens probe).

    Use weighted IsotonicRegression with the official metric's per-sample weight
    w_i = 1/30 + y_i so the calibration optimizes the challenge metric, not raw MSE.
    """

    def __init__(self, weight_offset: float = 1.0 / 30.0, out_min: float = 0.0, out_max: float = 1.0) -> None:
        self.weight_offset = weight_offset
        self.out_min = out_min
        self.out_max = out_max
        self.fitted: Dict[int, IsotonicRegression] = {}

    def fit(self, preds: np.ndarray, gt: np.ndarray, gender: np.ndarray) -> "GenderConditionalIsotonic":
        preds = np.asarray(preds, dtype=np.float64).flatten()
        gt = np.asarray(gt, dtype=np.float64).flatten()
        g = (np.asarray(gender).flatten() >= 0.5).astype(int)
        for gi in (0, 1):
            mask = g == gi
            if mask.sum() < 10:
                self.fitted[gi] = None
                continue
            w = self.weight_offset + gt[mask]
            iso = IsotonicRegression(y_min=self.out_min, y_max=self.out_max, out_of_bounds="clip")
            iso.fit(preds[mask], gt[mask], sample_weight=w)
            self.fitted[gi] = iso
        return self

    def transform(self, preds: np.ndarray, gender: np.ndarray) -> np.ndarray:
        preds = np.asarray(preds, dtype=np.float64).flatten()
        g = (np.asarray(gender).flatten() >= 0.5).astype(int)
        out = preds.copy()
        for gi in (0, 1):
            mask = g == gi
            iso = self.fitted.get(gi)
            if iso is not None and mask.any():
                out[mask] = iso.transform(preds[mask])
        return np.clip(out, self.out_min, self.out_max)
