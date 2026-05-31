"""Post-hoc per-gender calibration of regression outputs.

Three calibrators implemented, all per-gender (separate fit for F and M):

1. IsotonicCalibrator      : non-parametric monotone (PAV algorithm). Most flexible.
2. LinearCalibrator        : y_calib = a·pred + b. Simplest, parametric.
3. PCHIPSplineCalibrator   : monotone cubic Hermite spline, smoother than isotonic.

All support optional Importance Sampling reweighting (`use_is_weight=True`) which
corrects for the shift P_train(y|g) → P_test(y|g) induced by the marginal P(y) shift
between train and test. See docs/v12_theory.md §4.5 for derivation.

Theoretical sample weights:
    w_i = (1/30 + y_i)              ← challenge weight per sample (always)
    w_i × ratio_i                   ← additionally if use_is_weight
    ratio_i = P_test(y_i) / P_train_emp(y_i)

    Note: under H_C, P_test(y|g)/P_train(y|g) = [P_test(y)/P_train(y)] × [P_train(g)/P_test(g)].
    The per-g constant simplifies (calibrator fit per gender uses relative weights only).
"""

from typing import Dict, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression


# ============================================================================
# IS ratio computation (used by all calibrators when use_is_weight=True)
# ============================================================================

def compute_is_ratios(gt: np.ndarray) -> np.ndarray:
    """ratio_i = P_test(y_i) / P_train_emp(y_i).

    Both P_test and P_train_emp computed on the same binning (15 bins × 0.0333).
    Clipped to [1/10, 10] then normalized to mean=1 (echelle hygiene, cf v3 trick).
    """
    from src.utils.distribution import _TEST_PMF, N_BINS, BIN_WIDTH, empirical_pmf_y
    bin_idx = np.clip((np.asarray(gt) / BIN_WIDTH).astype(int), 0, N_BINS - 1)
    p_train_emp = empirical_pmf_y(gt, n_bins=N_BINS, bin_width=BIN_WIDTH)
    safe = np.maximum(p_train_emp, 1e-9)
    ratios = np.asarray(_TEST_PMF)[bin_idx] / safe[bin_idx]
    ratios = np.clip(ratios, 0.1, 10.0)
    ratios = ratios / max(ratios.mean(), 1e-9)
    return ratios.astype(np.float64)


# ============================================================================
# Base class
# ============================================================================

class PerGenderCalibratorBase:
    """Abstract per-gender calibrator."""

    name: str = "base"

    def __init__(self, weight_offset: float = 1.0 / 30.0, use_is_weight: bool = True,
                  out_min: float = 0.0, out_max: float = 1.0) -> None:
        self.weight_offset = weight_offset
        self.use_is_weight = use_is_weight
        self.out_min = out_min
        self.out_max = out_max
        self.fitted: Dict[int, object] = {}

    def _sample_weights(self, gt: np.ndarray) -> np.ndarray:
        w = self.weight_offset + gt
        if self.use_is_weight:
            w = w * compute_is_ratios(gt)
        return w

    def _fit_one(self, preds: np.ndarray, gt: np.ndarray, w: np.ndarray) -> object:
        raise NotImplementedError

    def _transform_one(self, model: object, preds: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, preds: np.ndarray, gt: np.ndarray, gender: np.ndarray) -> "PerGenderCalibratorBase":
        preds = np.asarray(preds, dtype=np.float64).flatten()
        gt = np.asarray(gt, dtype=np.float64).flatten()
        g = (np.asarray(gender).flatten() >= 0.5).astype(int)
        for gi in (0, 1):
            mask = g == gi
            if mask.sum() < 10:
                self.fitted[gi] = None
                continue
            w = self._sample_weights(gt[mask])
            self.fitted[gi] = self._fit_one(preds[mask], gt[mask], w)
        return self

    def transform(self, preds: np.ndarray, gender: np.ndarray) -> np.ndarray:
        preds = np.asarray(preds, dtype=np.float64).flatten()
        g = (np.asarray(gender).flatten() >= 0.5).astype(int)
        out = preds.copy()
        for gi in (0, 1):
            mask = g == gi
            model = self.fitted.get(gi)
            if model is not None and mask.any():
                out[mask] = self._transform_one(model, preds[mask])
        return np.clip(out, self.out_min, self.out_max)


# ============================================================================
# 1. Isotonic (non-parametric monotone, PAV)
# ============================================================================

class IsotonicCalibrator(PerGenderCalibratorBase):
    """Pool Adjacent Violators (Best 1955; sklearn IsotonicRegression).

    Fits a piecewise-constant non-decreasing function y_calib = f(pred).
    Most flexible of the 3 (no parametric assumption). Risk: stairs in mapping.
    """
    name = "isotonic"

    def _fit_one(self, preds: np.ndarray, gt: np.ndarray, w: np.ndarray) -> object:
        iso = IsotonicRegression(y_min=self.out_min, y_max=self.out_max, out_of_bounds="clip")
        iso.fit(preds, gt, sample_weight=w)
        return iso

    def _transform_one(self, model: object, preds: np.ndarray) -> np.ndarray:
        return model.transform(preds)


# ============================================================================
# 2. Linear (Platt-like for regression)
# ============================================================================

class LinearCalibrator(PerGenderCalibratorBase):
    """Weighted linear regression: y_calib = clip(a·pred + b, 0, 1).

    Closed-form weighted least squares. 2 params per gender. Monotone if a>0
    (typically true if model is at all predictive). Simplest baseline.
    """
    name = "linear"

    def _fit_one(self, preds: np.ndarray, gt: np.ndarray, w: np.ndarray) -> object:
        ws = float(w.sum())
        x_mean = float((w * preds).sum() / ws)
        y_mean = float((w * gt).sum() / ws)
        xc = preds - x_mean
        yc = gt - y_mean
        cov = float((w * xc * yc).sum() / ws)
        var = float((w * xc * xc).sum() / ws)
        a = cov / max(var, 1e-12)
        b = y_mean - a * x_mean
        return (a, b)

    def _transform_one(self, model: object, preds: np.ndarray) -> np.ndarray:
        a, b = model
        return a * preds + b


# ============================================================================
# 3. PCHIP monotone spline (smoother than isotonic, parametric in knots)
# ============================================================================

class PCHIPSplineCalibrator(PerGenderCalibratorBase):
    """Monotone cubic Hermite spline (Fritsch-Carlson 1980, scipy.PchipInterpolator).

    Strategy:
      1. Bin preds into ~20 equal-frequency bins (deciles + finer)
      2. Per bin, weighted mean(pred) → x; weighted mean(gt) → y
      3. Apply Pool Adjacent Violators on (x, y) pairs to enforce monotonicity
      4. Fit PCHIP through monotonic (x, y) → smooth monotonic interpolator

    Compared to isotonic: smoother (cubic instead of piecewise-constant). Less prone
    to "stair" overfitting at the cost of slightly less flexibility.
    """
    name = "pchip"

    def __init__(self, n_knots: int = 20, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_knots = int(n_knots)

    def _fit_one(self, preds: np.ndarray, gt: np.ndarray, w: np.ndarray) -> object:
        # Bin preds into equal-frequency bins → knot positions
        # (skip empty bins, ensure at least 4 knots for PCHIP)
        sort_idx = np.argsort(preds)
        preds_sorted = preds[sort_idx]
        gt_sorted = gt[sort_idx]
        w_sorted = w[sort_idx]

        n = len(preds_sorted)
        if n < self.n_knots:
            n_knots = max(4, n // 2)
        else:
            n_knots = self.n_knots
        # Cumulative weight → equal-weight bin boundaries
        w_cum = np.cumsum(w_sorted)
        w_total = w_cum[-1]
        boundaries = np.linspace(0, w_total, n_knots + 1)[1:-1]
        bin_idx = np.searchsorted(w_cum, boundaries)
        bin_idx = np.unique(np.concatenate(([0], bin_idx, [n])))
        x_knots, y_knots = [], []
        for i in range(len(bin_idx) - 1):
            lo, hi = bin_idx[i], bin_idx[i + 1]
            if hi <= lo:
                continue
            w_local = w_sorted[lo:hi]
            w_sum = w_local.sum()
            if w_sum <= 0:
                continue
            x_knots.append(float((preds_sorted[lo:hi] * w_local).sum() / w_sum))
            y_knots.append(float((gt_sorted[lo:hi] * w_local).sum() / w_sum))
        x_knots = np.asarray(x_knots)
        y_knots = np.asarray(y_knots)

        # Deduplicate identical x_knots (PCHIP requires strictly increasing x)
        if len(x_knots) >= 2:
            keep = np.concatenate(([True], np.diff(x_knots) > 1e-9))
            x_knots, y_knots = x_knots[keep], y_knots[keep]

        if len(x_knots) < 4:
            # Fallback to linear
            from scipy.interpolate import interp1d
            if len(x_knots) >= 2:
                return ("interp1d", interp1d(x_knots, y_knots, kind="linear",
                                                fill_value=(y_knots[0], y_knots[-1]), bounds_error=False))
            else:
                return ("constant", float(y_knots[0]) if len(y_knots) > 0 else 0.0)

        # Enforce monotonicity on y_knots via PAV (sklearn IsotonicRegression on knots)
        iso = IsotonicRegression(out_of_bounds="clip")
        y_knots_mono = iso.fit_transform(x_knots, y_knots)

        # PCHIP interpolation through monotone knots
        from scipy.interpolate import PchipInterpolator
        pchip = PchipInterpolator(x_knots, y_knots_mono, extrapolate=True)
        return ("pchip", pchip, float(x_knots[0]), float(x_knots[-1]),
                float(y_knots_mono[0]), float(y_knots_mono[-1]))

    def _transform_one(self, model: object, preds: np.ndarray) -> np.ndarray:
        kind = model[0]
        if kind == "pchip":
            _, pchip, x_lo, x_hi, y_lo, y_hi = model
            out = pchip(preds)
            out = np.where(preds < x_lo, y_lo, out)
            out = np.where(preds > x_hi, y_hi, out)
            return np.asarray(out)
        if kind == "interp1d":
            return model[1](preds)
        if kind == "constant":
            return np.full_like(preds, model[1])
        return preds


# ============================================================================
# 4. Isotonic per (gender × y-regime) — stretches high-y tail
# ============================================================================

class IsotonicRegimeCalibrator(PerGenderCalibratorBase):
    """Per-(gender × y-regime) isotonic with smooth blend at the regime boundary.

    Motivation: global isotonic shrinks toward identity in sparse regions; on the
    rare high-y tail (y > 0.2, ~15% of P_test mass), the model systematically under-
    predicts but global isotonic can't stretch enough due to few support points.
    Fitting a SEPARATE isotonic on the high-y subset gives it more pull there.

    Fit:  partitions samples by GT into low (y < thr) and high (y ≥ thr) regimes,
          fits one isotonic per regime per gender.
    Transform: uses PREDICTED value to pick regime (we don't have GT at inference).
          Smooth linear blend in [thr - hw, thr + hw] to avoid stair at the boundary.
    """
    name = "isotonic_regime"

    def __init__(self, y_threshold: float = 0.20, blend_halfwidth: float = 0.05,
                 min_samples_per_regime: int = 50, **kwargs) -> None:
        super().__init__(**kwargs)
        self.y_threshold = float(y_threshold)
        self.blend_halfwidth = float(blend_halfwidth)
        self.min_samples = int(min_samples_per_regime)

    def _fit_one(self, preds: np.ndarray, gt: np.ndarray, w: np.ndarray) -> object:
        mask_low = gt < self.y_threshold
        mask_high = ~mask_low
        iso_low = iso_high = None
        if mask_low.sum() >= self.min_samples:
            iso_low = IsotonicRegression(y_min=self.out_min, y_max=self.out_max, out_of_bounds="clip")
            iso_low.fit(preds[mask_low], gt[mask_low], sample_weight=w[mask_low])
        if mask_high.sum() >= self.min_samples:
            iso_high = IsotonicRegression(y_min=self.out_min, y_max=self.out_max, out_of_bounds="clip")
            iso_high.fit(preds[mask_high], gt[mask_high], sample_weight=w[mask_high])
        # Fallback: global isotonic when one regime is too sparse
        iso_global = None
        if iso_low is None or iso_high is None:
            iso_global = IsotonicRegression(y_min=self.out_min, y_max=self.out_max, out_of_bounds="clip")
            iso_global.fit(preds, gt, sample_weight=w)
        return (iso_low, iso_high, iso_global)

    def _transform_one(self, model: object, preds: np.ndarray) -> np.ndarray:
        iso_low, iso_high, iso_global = model
        if iso_low is None and iso_high is None:
            return iso_global.transform(preds) if iso_global is not None else preds
        if iso_low is None:
            return iso_high.transform(preds)
        if iso_high is None:
            return iso_low.transform(preds)
        out_low = iso_low.transform(preds)
        out_high = iso_high.transform(preds)
        hw = self.blend_halfwidth
        w_high = np.clip((preds - (self.y_threshold - hw)) / (2.0 * hw), 0.0, 1.0)
        return (1.0 - w_high) * out_low + w_high * out_high


# ============================================================================
# 5. Isotonic with tail-boost weights — single fit, no regime boundary
# ============================================================================

class IsotonicTailBoostCalibrator(PerGenderCalibratorBase):
    """Single per-gender isotonic with weights boosted on high-y samples.

    Same one-shot PAV as IsotonicCalibrator but the sample weights are:
        w_i = (1/30 + y_i) × IS_ratio_i × (1 + boost · max(0, y_i - y_pivot))

    The (1 + boost · (y - y_pivot)+) term amplifies the influence of samples with
    y > y_pivot during the fit, pushing the isotonic mapping to better match the
    sparse high-y tail (where the model typically under-predicts and standard
    isotonic shrinks toward identity due to few support points).

    No regime boundary, no blend zone, smooth monotonic mapping (no artefact step).
    Cleaner alternative to IsotonicRegimeCalibrator.

    Defaults: y_pivot=0.20 (start boosting), boost=5.0 (samples at y=0.5 get ~2.5×
    extra weight on top of standard weights).
    """
    name = "isotonic_tailboost"

    def __init__(self, y_pivot: float = 0.20, boost: float = 5.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.y_pivot = float(y_pivot)
        self.boost = float(boost)

    def _sample_weights(self, gt: np.ndarray) -> np.ndarray:
        w = super()._sample_weights(gt)
        tail = np.maximum(gt - self.y_pivot, 0.0)
        return w * (1.0 + self.boost * tail)

    def _fit_one(self, preds: np.ndarray, gt: np.ndarray, w: np.ndarray) -> object:
        iso = IsotonicRegression(y_min=self.out_min, y_max=self.out_max, out_of_bounds="clip")
        iso.fit(preds, gt, sample_weight=w)
        return iso

    def _transform_one(self, model: object, preds: np.ndarray) -> np.ndarray:
        return model.transform(preds)


# ============================================================================
# Convenience: fit all calibrators (with IS-weighting)
# ============================================================================

def fit_all_calibrators(
    preds: np.ndarray, gt: np.ndarray, gender: np.ndarray,
    use_is_weight: bool = True,
) -> Dict[str, PerGenderCalibratorBase]:
    """Fit all per-gender calibrators on (preds, gt, gender).
    Returns dict keyed by calibrator name. All use IS-weighting by default.
    """
    out: Dict[str, PerGenderCalibratorBase] = {}
    for cls in [IsotonicCalibrator, LinearCalibrator, PCHIPSplineCalibrator,
                IsotonicRegimeCalibrator, IsotonicTailBoostCalibrator]:
        try:
            cal = cls(use_is_weight=use_is_weight).fit(preds, gt, gender)
            out[cal.name] = cal
        except Exception as e:
            print(f"[calibrators] WARNING: {cls.__name__} fit failed: {e}")
    return out
