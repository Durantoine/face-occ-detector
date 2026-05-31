"""Train/test distribution utilities, centralized.

This module owns everything that depends on Y binning and the train↔test
covariate shift estimation. The rest of the codebase imports from here.

Conventions:
    Y ∈ [0, 0.5]  : occlusion fraction in test set support (P_test(Y > 0.5) ≈ 0)
    G ∈ {0, 1}    : gender; 0 = Female, 1 = Male
    bins          : N_BINS uniform bins over [0, N_BINS · BIN_WIDTH]

Hypothesis H_C (= "H1 covariate shift Y-only" in fairness.md historical doc):
    P_test(G | Y) = P_train(G | Y)
    → P_test(G, Y) = P_train(G | Y) · P_test(Y)
    → marginale P_test(G) automatiquement dérivée du shift en Y
    → empiriquement validée via MID lookup (cf scripts/estimate_test_gender.py et
      docs/v12_theory.md §2-3) à 0.5 pt d'écart sur la marginale F.
"""

from typing import Tuple

import numpy as np


# ============================================================================
# Binning + extracted test PMF (P_test(Y) measured from PDF via pixel reading)
# ============================================================================

N_BINS: int = 15
BIN_WIDTH: float = 0.5 / N_BINS

# P_test(Y) — 15 bins × 0.0333. Extracted from PDF page 3 via pdf-render-to-image
# + per-column blue-bar height detection (cf docs/v12_theory.md §6.3).
# Sums to 1.0 within float precision.
_TEST_PMF: np.ndarray = np.array([
    0.118034,
    0.092728,
    0.092810,
    0.108154,
    0.102878,
    0.094643,
    0.106911,
    0.091066,
    0.079910,
    0.054407,
    0.034125,
    0.015720,
    0.005424,
    0.002145,
    0.001046,
], dtype=np.float64)
_TEST_PMF = _TEST_PMF / _TEST_PMF.sum()

# P_test marginal gender estimated via hybrid MID lookup + DINOv3 linear probe gender
# classifier, 100% coverage on test_students.csv (29980 samples):
#   - MID lookup (93.5%): deterministic gender from train.csv MID→gender mapping
#   - DINOv3 probe (6.5%): logistic regression on DINOv3 ViT-B features, trained on
#     3000 train samples with known gender (train acc 1.0 — easy task)
# Result: P_test(F) = 0.4879, P_test(M) = 0.5121 (vs MID-only: 0.4766, 0.5234)
# Build script: src.inference.gender_classifier.GenderClassifier.fit_or_load
_TEST_P_GENDER = np.array([0.4879, 0.5121], dtype=np.float64)  # [F, M]


# ============================================================================
# Empirical distributions on a labeled set (train or val)
# ============================================================================

def empirical_pmf_y(targets: np.ndarray, n_bins: int = N_BINS, bin_width: float = BIN_WIDTH) -> np.ndarray:
    """Marginal P_emp(Y) over n_bins."""
    edges = np.linspace(0.0, n_bins * bin_width, n_bins + 1)
    y = np.clip(np.asarray(targets, dtype=np.float64), 0.0, edges[-1] - 1e-9)
    hist, _ = np.histogram(y, bins=edges)
    return hist.astype(np.float64) / max(hist.sum(), 1)


def empirical_pmf_joint(
    targets: np.ndarray, gender: np.ndarray,
    n_bins: int = N_BINS, bin_width: float = BIN_WIDTH,
) -> np.ndarray:
    """Joint P_emp(G, Y) shape (2, n_bins). Sums to 1."""
    g = (np.asarray(gender) >= 0.5).astype(int)
    b = np.clip((np.asarray(targets) / bin_width).astype(int), 0, n_bins - 1)
    counts = np.zeros((2, n_bins), dtype=np.float64)
    for gi, bi in zip(g, b):
        counts[gi, bi] += 1
    return counts / max(counts.sum(), 1)


def empirical_p_g_given_y(
    targets: np.ndarray, gender: np.ndarray,
    n_bins: int = N_BINS, bin_width: float = BIN_WIDTH,
) -> np.ndarray:
    """P_emp(G | Y) shape (2, n_bins). Each column sums to 1."""
    joint = empirical_pmf_joint(targets, gender, n_bins=n_bins, bin_width=bin_width)
    col_sums = joint.sum(axis=0, keepdims=True)
    return joint / (col_sums + 1e-12)


# ============================================================================
# Test joint estimation via H_C
# ============================================================================

def estimate_test_pmf_joint(
    targets: np.ndarray, gender: np.ndarray,
    n_bins: int = N_BINS, bin_width: float = BIN_WIDTH,
    test_pmf_y: np.ndarray = _TEST_PMF,
) -> np.ndarray:
    """Estimate P_test(G, Y) joint distribution under hypothesis H_C.

    H_C: P_test(G | Y) = P_train(G | Y).
        → P_test(G, Y) = P_train(G | Y) · P_test(Y)

    Why H_C: empirically validated (0.5 pt error on observed marginal P_test(F)
    via MID lookup), see docs/v12_theory.md §2-3 for rigorous tests of H_A, H_B, H_C.

    Args:
        targets, gender: labeled samples used to estimate P(G|Y). Typically val data
            since val mirrors train distribution (under v12 iid split).
        test_pmf_y: known P_test(Y) extracted from PDF (defaults to _TEST_PMF).

    Returns:
        Array shape (2, n_bins) summing to 1.0. test_pmf_joint[g, b] = P_test(G=g, Y∈b).
    """
    p_g_given_y = empirical_p_g_given_y(targets, gender, n_bins=n_bins, bin_width=bin_width)
    return np.asarray(test_pmf_y, dtype=np.float64)[None, :] * p_g_given_y


# ============================================================================
# Unified rebalancing target (axes 1 + 2)
# ============================================================================

def compute_target_weights(
    targets: np.ndarray, gender: np.ndarray,
    axis1_power: float, axis2_power: float,
    n_bins: int = N_BINS, bin_width: float = BIN_WIDTH,
    test_pmf_y: np.ndarray = _TEST_PMF,
    test_p_gender: np.ndarray = _TEST_P_GENDER,
    clip: float = 10.0,
) -> np.ndarray:
    """Per-sample loss weight for the unified rebalancing target.

        P_target(g, y) = mix_y(α1) × mix_g(α2)
            mix_y(α1) = (1-α1)·P_train(y) + α1·P_test(y)              ← axe 1
            mix_g(α2) = (1-α2)·P_train(g|y) + α2·P_test(g)            ← axe 2 (v13)

        sample_weight_i = clip(P_target(g_i, y_i) / P_train(g_i, y_i), 1/clip, clip)
        sample_weight  ← sample_weight / mean(sample_weight)          ← v3 hygiene trick

    v13 change: axe 2 target = P_test(g) ≈ (0.48, 0.52) from MID lookup, instead of
    uniform (0.5, 0.5). Aligned with empirical P_test(g) — marginal 2% correction
    but theoretically clean (uniform was a proxy when P_test(g) was unknown).

    Returns per-sample float32 array of same length as `targets`.
    """
    g = (np.asarray(gender) >= 0.5).astype(int)
    b = np.clip((np.asarray(targets) / bin_width).astype(int), 0, n_bins - 1)
    p_joint = empirical_pmf_joint(targets, gender, n_bins=n_bins, bin_width=bin_width)
    p_train_y = p_joint.sum(axis=0)
    safe_y = np.maximum(p_train_y, 1e-9)
    p_train_g_given_y = p_joint / safe_y[None, :]

    p_target_y = (1.0 - axis1_power) * p_train_y + axis1_power * test_pmf_y
    # axe 2 target = P_test(g) repeated across y bins (g-only, not depending on y bin)
    test_p_g_broadcast = np.asarray(test_p_gender, dtype=np.float64)[:, None] * np.ones((1, n_bins))
    p_target_g_given_y = (1.0 - axis2_power) * p_train_g_given_y + axis2_power * test_p_g_broadcast
    p_target_joint = p_target_y[None, :] * p_target_g_given_y

    ratio = p_target_joint / np.maximum(p_joint, 1e-9)
    ratio = np.clip(ratio, 1.0 / clip, clip)
    sample_w = ratio[g, b]
    sample_w = sample_w / max(float(sample_w.mean()), 1e-9)
    return sample_w.astype(np.float32)


# ============================================================================
# Stratified sampling matching P_test joint (used for test holdout in v12)
# ============================================================================

def sample_indices_matching_test_pmf(
    targets: np.ndarray, gender: np.ndarray,
    n_target: int, seed: int = 42,
    n_bins: int = N_BINS, bin_width: float = BIN_WIDTH,
    test_pmf_y: np.ndarray = _TEST_PMF,
) -> np.ndarray:
    """Stratified sampling of n_target indices such that resulting joint matches H_C:
    P_target(g, y) = P_train(g | y) × P_test(y).

    Per cell (g, b), takes ceil(P_target(g, b) × n_target) samples without replacement.
    If a cell has fewer samples than needed, takes all available.
    """
    g_int = (np.asarray(gender) >= 0.5).astype(int) if gender is not None else np.zeros(len(targets), dtype=int)
    bin_idx = np.clip((np.asarray(targets) / bin_width).astype(int), 0, n_bins - 1)

    # P_target(g, b) = P_train(g | b) × P_test(b)
    counts = np.zeros((2, n_bins), dtype=np.float64)
    for gi, bi in zip(g_int, bin_idx):
        counts[gi, bi] += 1
    p_train_g_given_y = counts / (counts.sum(axis=0, keepdims=True) + 1e-12)
    p_target_joint = np.asarray(test_pmf_y, dtype=np.float64)[None, :] * p_train_g_given_y

    target_per_cell = (p_target_joint * n_target).astype(int)
    rng = np.random.RandomState(seed)
    selected = []
    for gi in range(2):
        for b in range(n_bins):
            in_cell = np.where((g_int == gi) & (bin_idx == b))[0]
            need = int(target_per_cell[gi, b])
            if need == 0 or len(in_cell) == 0:
                continue
            if len(in_cell) >= need:
                selected.extend(rng.choice(in_cell, need, replace=False).tolist())
            else:
                selected.extend(in_cell.tolist())
    return np.array(selected, dtype=int)


def split_train_val_test(
    targets: np.ndarray, gender: np.ndarray,
    val_ratio: float, test_ratio: float, seed: int = 42,
    n_bins: int = N_BINS, bin_width: float = BIN_WIDTH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-way index split for v12 tri-partite design.

    1. Sample test holdout matching P_test (H_C) — taken first to ensure stratification.
    2. Sample val iid from remaining (preserves P_train → train keeps rare-bin samples).
    3. Train = the rest.

    Returns (train_idx, val_idx, test_idx).
    """
    n = len(targets)
    test_size = max(int(n * test_ratio), 1)
    val_size = max(int(n * val_ratio), 1)

    test_idx = sample_indices_matching_test_pmf(targets, gender, test_size, seed=seed,
                                                  n_bins=n_bins, bin_width=bin_width)
    remaining_mask = np.ones(n, dtype=bool)
    remaining_mask[test_idx] = False
    remaining = np.where(remaining_mask)[0]

    rng = np.random.RandomState(seed + 1)
    val_local = rng.choice(len(remaining), min(val_size, len(remaining)), replace=False)
    val_idx = remaining[val_local]

    val_mask = np.zeros(n, dtype=bool)
    val_mask[val_idx] = True
    train_idx = np.where(remaining_mask & ~val_mask)[0]
    return train_idx, val_idx, test_idx


def split_val_matching_test_pmf(
    targets: np.ndarray, gender: np.ndarray,
    val_ratio: float, seed: int = 42,
    n_bins: int = N_BINS, bin_width: float = BIN_WIDTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy two-way split (v11 and prior): val resampled to match P_test (H_C).
    Returns (train_idx, val_idx). Train = setdiff(all, val).
    """
    n = len(targets)
    val_size = max(int(n * val_ratio), 1)
    val_idx = sample_indices_matching_test_pmf(targets, gender, val_size, seed=seed,
                                                 n_bins=n_bins, bin_width=bin_width)
    train_idx = np.setdiff1d(np.arange(n), val_idx)
    return train_idx, val_idx
