from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.neighbors import KernelDensity
from functools import lru_cache

# Constants
N_BINS = 200
BIN_WIDTH = 1.0 / N_BINS
CENTERS = 0.5 * (np.linspace(0.0, 1.0, N_BINS + 1)[:-1] + np.linspace(0.0, 1.0, N_BINS + 1)[1:])
_GRID = np.linspace(0.0, 1.0, 1001)  # KDEs scored on this grid, then interpolated
DEFAULT_LAMBDA = 0.10  # regularization floor on P_train in the importance ratio (Config.is_lambda)
W_FLOOR, W_CEIL = 0.05, 5.0  # guard clip on the weight (natural max ~3.2 at lambda=0.10)
W_HI_FLOOR, Y_LOW = 1.0, 0.102  # Y_LOW = raw P_train=P_test crossover (lambda-independent); past it no variant may fall below W_HI_FLOOR

# Official P_test PMF
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
_TEST_PMF_05[78] = 0.001900; _TEST_PMF_05[92] = 0.000136
P_TEST_PMF = np.zeros(N_BINS, dtype=np.float64); P_TEST_PMF[:100] = _TEST_PMF_05
P_TEST_PMF = P_TEST_PMF / P_TEST_PMF.sum()

# P_test has STRICTLY ZERO mass above this y. The spline rings (spurious +/- bumps) past the
# support, which over a near-zero P_train fabricates a fake ratio explosion (~11 at y~0.74, on
# 2 train images) and a creux-then-spike at y=0.5. We clamp the density to [0, Y_SUPPORT] and
# clip the spline at 0 (not 1e-12), so beyond the support the weight cleanly floors out.
Y_SUPPORT = float(CENTERS[np.max(np.where(P_TEST_PMF > 0))])


def _fit_kde(data: np.ndarray, name: str, bw: float):
    if len(data) == 0:
        return None
    # No subsampling: the sparse tail (y>0.5) has only ~tens of samples; dropping them
    # zeroes P_train there and breaks the ratio. Reflection handles the y=0 boundary.
    data_ext = np.concatenate([data, -data])
    print(f"  [dist] Fitting KDE for {name} ({len(data_ext)} points)...", flush=True)
    return KernelDensity(kernel='gaussian', bandwidth=bw, algorithm='ball_tree').fit(data_ext[:, None])


def _grid_density(kde) -> np.ndarray | None:
    return (np.exp(kde.score_samples(_GRID[:, None])) * 2.0) if kde is not None else None


@lru_cache(maxsize=8)
def _pr_grid_global(y_bytes: bytes, bw: float) -> np.ndarray:
    """Global P_train density on _GRID (integrates to ~1 over [0,1]). The only fit HC needs."""
    y = np.frombuffer(y_bytes, dtype=np.float64)
    return _grid_density(_fit_kde(y, "global", bw))


@lru_cache(maxsize=8)
def _pr_grid_gender(y_bytes: bytes, g_bytes: bytes, bw: float):
    """Per-gender P_train densities + priors on _GRID. Fitted only for target='joint'."""
    y = np.frombuffer(y_bytes, dtype=np.float64); g = np.frombuffer(g_bytes, dtype=np.float64)
    f_mask = g < 0.5; p_f, p_m = f_mask.mean(), (~f_mask).mean()
    return _grid_density(_fit_kde(y[f_mask], "female", bw)), _grid_density(_fit_kde(y[~f_mask], "male", bw)), p_f, p_m


def get_p_test_spline(s: float = 1e-4):
    return UnivariateSpline(CENTERS, P_TEST_PMF, k=3, s=s, ext=3)


def _p_test_density(y: np.ndarray, spline) -> np.ndarray:
    """P_test density at y: spline clipped at 0 and forced to 0 beyond the PMF support."""
    y = np.asarray(y, float)
    p = np.clip(spline(y), 0.0, None) * N_BINS
    p[y > Y_SUPPORT] = 0.0
    return p


def get_joint_weight_fn(y_train: np.ndarray, g_train: np.ndarray | None,
                        target: str = "hc", alpha: float = 1.0, lam: float = DEFAULT_LAMBDA):
    """Single regularized importance ratio, shared by sampler, loss and eval.

        w(y, g) = P_test_target(y, g) / (P_train(y, g) + lam)        clipped to [W_FLOOR, W_CEIL]

    - target "hc": global densities, P_test_target = P_test.
    - target "joint": per-gender, P_test_target = 0.5 * P_test and P_train = P_train_g * prior_g,
      so each gender's occlusion marginal is steered to half the test shape (balanced + matched).
    The lam floor tames the high-variance tail ratio (P_train estimated from tens of images).
    Only the low-occ surplus (y<=Y_LOW) keeps w<1 (down-weight, complementary to the sampler);
    past Y_LOW every variant is floored to >=W_HI_FLOOR so kept high-occ frames are never ignored
    (the PMF tail under-states the real high-occ mass). alpha<1 blends the weight toward 1.0.
    """
    g_safe = g_train if g_train is not None else np.zeros_like(y_train)
    y_bytes = y_train.astype(np.float64).tobytes()
    bw = 0.018
    # HC needs only the global ratio; joint additionally needs the per-gender densities + priors.
    prg_all = _pr_grid_global(y_bytes, bw)
    prg_f = prg_m = p_f = p_m = None
    if target == "joint":
        prg_f, prg_m, p_f, p_m = _pr_grid_gender(y_bytes, g_safe.astype(np.float64).tobytes(), bw)
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
                    p_r[mask] = (np.interp(y[mask], _GRID, prg) if prg is not None
                                 else np.interp(y[mask], _GRID, prg_all)) * prior
        w = np.clip(p_t / (p_r + lam), W_FLOOR, W_CEIL)
        # alpha<1 = LINEAR blend toward 1.0 -> target P_target = (1-alpha)*P_train + alpha*P_test.
        if alpha < 1.0:
            w = (1.0 - alpha) * 1.0 + alpha * w
        # Floor LAST so the weight never drops below 1.0 at high occlusion (past Y_LOW),
        # regardless of alpha (high-occ frames are never down-weighted).
        w = np.where(y > Y_LOW, np.maximum(w, W_HI_FLOOR), w)
        return w.astype(np.float32)

    return weight_fn


def p_test_val_indices(y: np.ndarray, gender: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    w_fn = get_joint_weight_fn(y, gender, target="hc")
    w = w_fn(y); rng = np.random.RandomState(seed); u = rng.rand(len(y)); v = u ** (1.0 / np.maximum(w, 1e-9))
    return np.argsort(-v)[:n]


def p_train_val_indices(y: np.ndarray, gender: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    # PROPORTIONAL stratified sampling, NO tail floor. Each stratum is sampled at the SAME
    # rate pi_s = take/total ~= n/M, so the inclusion probability is constant across strata. That
    # constant cancels in the self-normalised IS score, so the P_test/P_train weight is applied
    # without sampling bias. 20 fine bins for 0-40% (2% each), then 1 'Super-Strate' for 40-100%.
    g = (np.asarray(gender) >= 0.5).astype(int)
    y_feat = np.asarray(y, float)
    b = np.where(y_feat < 0.4, (y_feat * 50).astype(int), 20)
    n_strat_bins = 21
    strata = g * n_strat_bins + b; rng = np.random.RandomState(seed); out = []
    total_counts = np.bincount(strata, minlength=n_strat_bins * 2)
    exact = total_counts / len(y) * n
    take_counts = np.minimum(exact.astype(int), total_counts)
    remainder = n - take_counts.sum()
    if remainder > 0:  # largest-remainder, but never exceed a stratum's capacity
        for s in np.argsort(-(exact - exact.astype(int))):
            if remainder <= 0:
                break
            if take_counts[s] < total_counts[s]:
                take_counts[s] += 1; remainder -= 1
    for s in range(n_strat_bins * 2):
        pool = np.where(strata == s)[0]; k = min(int(take_counts[s]), len(pool))
        if k > 0: out.extend(rng.choice(pool, k, replace=False).tolist())
    return np.array(out, dtype=int)


def stratified_is_score(pred: np.ndarray, gt: np.ndarray, gender: np.ndarray, weight_fn=None) -> dict:
    weight_offset = 1.0 / 30.0; pred = np.asarray(pred, float).ravel(); gt = np.asarray(gt, float).ravel()
    g = (np.asarray(gender) >= 0.5).astype(int); is_w = weight_fn(gt, g) if weight_fn is not None else np.ones_like(gt)
    w = (weight_offset + gt) * is_w; err = (pred - gt) ** 2; werr = w * err; f, m = g == 0, g == 1
    def weighted_err(mask):
        if not mask.any(): return 0.0
        return float(werr[mask].sum() / np.maximum(w[mask].sum(), 1e-8))
    ef, em = weighted_err(f), weighted_err(m); abs_err = np.abs(pred - gt)
    gap_signed = ef - em
    return {"challenge_score": (ef + em) / 2 + abs(gap_signed), "err_F": ef, "err_M": em,
            "err_diff": abs(gap_signed), "err_gap": gap_signed, "mae_pct": float(abs_err.mean() * 100.0)}
