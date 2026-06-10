"""Continuous (KDE-based) importance weighting — ported from the v36 pipeline.

A single regularized importance ratio, computed on CONTINUOUS densities (no binning,
so no tail-clamp artefact):

    w(y, g) = P_test_target(y, g) / (P_train(y, g) + lam)   clipped to [W_FLOOR, W_CEIL]

- target "hc"    : global densities, P_test_target = P_test.
- target "joint" : per-gender (P_test_target = 0.5*P_test, P_train = P_train_g * prior_g)
                   => the CONTINUOUS analogue of v4's binned `cell_joint` reweighting.

The same `weight_fn` is meant to be shared by the loss and the eval metric. P_train is a
Gaussian KDE (reflected at y=0); P_test is a spline of the official PMF, clamped to its
support. `lam` (Config.is_lambda) is a regularization floor that tames the high-variance
tail ratio (P_train estimated from only tens of high-occlusion images).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.neighbors import KernelDensity

N_BINS = 200
CENTERS = 0.5 * (np.linspace(0.0, 1.0, N_BINS + 1)[:-1] + np.linspace(0.0, 1.0, N_BINS + 1)[1:])
_GRID = np.linspace(0.0, 1.0, 1001)
DEFAULT_LAMBDA = 0.10
W_FLOOR, W_CEIL = 0.05, 5.0
W_HI_FLOOR, Y_LOW = 1.0, 0.102   # past Y_LOW no variant falls below 1.0 (kept high-occ never ignored)

# Official P_test PMF (100 bins over [0,0.5], zero above).
_TEST_PMF_05 = np.array([
    0.023880, 0.021935, 0.018046, 0.018046, 0.014993, 0.013726, 0.011194, 0.011194, 0.008277, 0.011940,
    0.019267, 0.019267, 0.016282, 0.015671, 0.014450, 0.014450, 0.013704, 0.013161, 0.012618, 0.012618,
    0.017096, 0.016926, 0.016757, 0.016757, 0.015332, 0.014823, 0.014314, 0.014314, 0.012822, 0.015468,
    0.018113, 0.018113, 0.016757, 0.015909, 0.015061, 0.014608, 0.013704, 0.013229, 0.012754, 0.014269,
    0.017299, 0.016485, 0.015671, 0.015694, 0.015739, 0.015264, 0.014789, 0.014156, 0.012890, 0.013839,
    0.014789, 0.013941, 0.013093, 0.012890, 0.012686, 0.012449, 0.012211, 0.011601, 0.010990, 0.010651,
    0.010312, 0.009362, 0.008412, 0.007734, 0.007055, 0.006682, 0.006309, 0.006038, 0.005766, 0.005178,
    0.004885, 0.004613, 0.004342, 0.003347, 0.002849, 0.002578, 0.002307, 0.002035, 0.011900, 0.001492,
    0.001085, 0.000950, 0.000882, 0.000814, 0.000746, 0.000339, 0.000339, 0.000373, 0.000407, 0.000339,
    0.000339, 0.000237, 0.000136, 0.000136, 0.000136, 0.000170, 0.000204, 0.000068, 0.000068, 0.000068,
], dtype=np.float64)
_TEST_PMF_05[78] = 0.001900
_TEST_PMF_05[92] = 0.000136
P_TEST_PMF = np.zeros(N_BINS, dtype=np.float64)
P_TEST_PMF[:100] = _TEST_PMF_05
P_TEST_PMF = P_TEST_PMF / P_TEST_PMF.sum()
Y_SUPPORT = float(CENTERS[np.max(np.where(P_TEST_PMF > 0))])  # P_test = 0 strictly above this y


def _fit_kde(data: np.ndarray, name: str, bw: float):
    if len(data) == 0:
        return None
    data_ext = np.concatenate([data, -data])   # reflect at y=0 boundary
    return KernelDensity(kernel="gaussian", bandwidth=bw, algorithm="ball_tree").fit(data_ext[:, None])


def _grid_density(kde):
    return (np.exp(kde.score_samples(_GRID[:, None])) * 2.0) if kde is not None else None


@lru_cache(maxsize=8)
def _pr_grid_global(y_bytes: bytes, bw: float) -> np.ndarray:
    y = np.frombuffer(y_bytes, dtype=np.float64)
    return _grid_density(_fit_kde(y, "global", bw))


@lru_cache(maxsize=8)
def _pr_grid_gender(y_bytes: bytes, g_bytes: bytes, bw: float):
    y = np.frombuffer(y_bytes, dtype=np.float64)
    g = np.frombuffer(g_bytes, dtype=np.float64)
    f = g < 0.5
    p_f, p_m = f.mean(), (~f).mean()
    return _grid_density(_fit_kde(y[f], "female", bw)), _grid_density(_fit_kde(y[~f], "male", bw)), p_f, p_m


def get_p_test_spline(s: float = 1e-4):
    return UnivariateSpline(CENTERS, P_TEST_PMF, k=3, s=s, ext=3)


def _p_test_density(y: np.ndarray, spline) -> np.ndarray:
    y = np.asarray(y, float)
    p = np.clip(spline(y), 0.0, None) * N_BINS
    p[y > Y_SUPPORT] = 0.0
    return p


def get_joint_weight_fn(y_train: np.ndarray, g_train: np.ndarray | None,
                        target: str = "joint", alpha: float = 1.0, lam: float = DEFAULT_LAMBDA, bw: float = 0.018):
    """Build the continuous importance weight_fn(y, g). See module docstring."""
    g_safe = g_train if g_train is not None else np.zeros_like(y_train)
    y_bytes = np.asarray(y_train, np.float64).tobytes()
    prg_all = _pr_grid_global(y_bytes, bw)
    prg_f = prg_m = p_f = p_m = None
    if target == "joint":
        prg_f, prg_m, p_f, p_m = _pr_grid_gender(y_bytes, np.asarray(g_safe, np.float64).tobytes(), bw)
    spline = get_p_test_spline(s=1e-4)

    def weight_fn(y: np.ndarray, g: np.ndarray | None = None) -> np.ndarray:
        y = np.asarray(y, float)
        p_t_full = _p_test_density(y, spline)
        if target == "hc" or g is None:
            p_t = p_t_full
            p_r = np.interp(y, _GRID, prg_all)
        else:
            g_idx = (np.asarray(g) >= 0.5).astype(int)
            p_t = 0.5 * p_t_full
            p_r = np.zeros_like(y)
            for gi, prg, prior in [(0, prg_f, p_f), (1, prg_m, p_m)]:
                mask = (g_idx == gi)
                if mask.any():
                    dens = np.interp(y[mask], _GRID, prg) if prg is not None else np.interp(y[mask], _GRID, prg_all)
                    p_r[mask] = dens * prior
        w = np.clip(p_t / (p_r + lam), W_FLOOR, W_CEIL)
        w = np.where(y > Y_LOW, np.maximum(w, W_HI_FLOOR), w)
        if alpha < 1.0:
            w = (1.0 - alpha) * 1.0 + alpha * w
        return w.astype(np.float32)

    return weight_fn
