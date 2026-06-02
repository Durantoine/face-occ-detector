"""Build data/raw/test_students_with_gender.csv = test_students.csv + predicted gender.

Pipeline:
1. Fit (or load cached) GenderClassifier on FULL labeled train (CV + grid search on C,
   refit on all data for production model).
2. Validate probe on MID-known test samples (real-world accuracy estimate, NOT just CV).
3. Predict gender for every test sample (MID lookup primary, Sapiens probe fallback).
4. Write CSV with columns: filename, gender_predicted (0=F, 1=M), gender_source,
   gender_proba_M (NaN for mid_lookup rows, prob for sapiens_probe rows).
5. Validate H_C marginal: compute observed P_test(F) and compare with H_C-derived expected
   value (Σ_y P_train(F|y) · P_test(y)). Diverge significant ⇒ H_C suspect.
6. Print suggested update to _TEST_P_GENDER in src/utils/distribution.py.

Usage:
    python scripts/build_test_gender_csv.py
    python scripts/build_test_gender_csv.py --force-refit
    python scripts/build_test_gender_csv.py --sample-size 20000     # subsample for fast iteration
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.gender_classifier import (
    GenderClassifier, build_mid_gender_mapping, extract_mid, extract_sapiens_features,
    train_sapiens_probe,
)
from src.utils.distribution import _TEST_PMF, N_BINS, BIN_WIDTH, empirical_p_g_given_y


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test_students.csv")
    parser.add_argument("--image-base-dir", default="data/raw")
    parser.add_argument("--out-csv", default="data/raw/test_students_with_gender.csv")
    parser.add_argument("--cache-path", default="cache/gender_classifier.pkl")
    parser.add_argument("--sample-size", type=int, default=None,
                         help="Subsample train for fast iteration; None = full")
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--model-name", default="dinov3_vitl16",
                         help="Backbone for feature extraction. Options: dinov3_vitb16, "
                              "dinov3_vitl16 (default, 300M D=1024), dinov3_vith16plus (600M D=1280), "
                              "sapiens2_0.1b, sapiens2_0.4b, sapiens2_0.8b")
    args = parser.parse_args()

    # === 1. Fit / load classifier ===
    print("=" * 80)
    print(f"STEP 1 — Fit or load GenderClassifier (cache: {args.cache_path})")
    print("=" * 80)
    clf = GenderClassifier.fit_or_load(
        train_csv=args.train_csv, image_base_dir=args.image_base_dir,
        cache_path=args.cache_path, sample_size=args.sample_size,
        sapiens_model_name=args.model_name,
        force_refit=args.force_refit,
    )

    # === 2. Validate probe on MID-known test samples ===
    print()
    print("=" * 80)
    print("STEP 2 — Probe accuracy on MID-known test samples (= real-world estimate)")
    print("=" * 80)
    val_metrics = clf.validate_against_mid_lookup(args.test_csv, args.image_base_dir)

    # === 3. Predict gender for ALL test samples ===
    print()
    print("=" * 80)
    print("STEP 3 — Generate predictions for ALL test samples")
    print("=" * 80)
    df = pd.read_csv(args.test_csv)
    df["mid"] = df["filename"].apply(extract_mid)
    df["gender_mid"] = df["mid"].map(clf.mid_mapping)

    has_mid = df["gender_mid"].notna().values
    n_mid = int(has_mid.sum())
    n_probe = len(df) - n_mid
    print(f"MID coverage: {n_mid:,}/{len(df):,} ({100*n_mid/len(df):.2f}%)")
    print(f"To predict via Sapiens probe: {n_probe:,}")

    gender_predicted = np.full(len(df), -1, dtype=int)
    gender_source = np.array(["mid_lookup"] * len(df), dtype=object)
    gender_proba_M = np.full(len(df), np.nan, dtype=float)

    gender_predicted[has_mid] = df.loc[has_mid, "gender_mid"].astype(int).values

    if n_probe > 0:
        if clf.probe is None or clf.probe_norm is None:
            print("WARNING: no Sapiens probe — defaulting unknown to majority M=1")
            gender_predicted[~has_mid] = 1
            gender_source[~has_mid] = "majority_fallback"
        else:
            unknown_paths = df.loc[~has_mid, "filename"].tolist()
            feats = extract_sapiens_features(unknown_paths, args.image_base_dir,
                                              model_name=clf.sapiens_model_name)
            feats_norm = (feats - clf.probe_norm[0:1]) / (clf.probe_norm[1:2] + 1e-8)
            preds = clf.probe.predict(feats_norm)
            probas = clf.probe.predict_proba(feats_norm)[:, 1]  # P(M)
            gender_predicted[~has_mid] = preds
            gender_source[~has_mid] = "sapiens_probe"
            gender_proba_M[~has_mid] = probas

    # === 4. Write CSV ===
    out_df = df[["filename"]].copy()
    out_df["gender_predicted"] = gender_predicted
    out_df["gender_source"] = gender_source
    out_df["gender_proba_M"] = gender_proba_M
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print()
    print(f"Wrote {args.out_csv}  ({len(out_df):,} rows)")

    # === 5. H_C marginal validation ===
    print()
    print("=" * 80)
    print("STEP 5 — Validate H_C: compare observed P_test(F) vs H_C-derived expected")
    print("=" * 80)
    obs_f = int((gender_predicted == 0).sum())
    obs_m = int((gender_predicted == 1).sum())
    p_test_f_obs = obs_f / len(df)
    p_test_m_obs = obs_m / len(df)
    print(f"Observed P_test(F) = {p_test_f_obs:.4f}  ({obs_f:,}/{len(df):,})")
    print(f"Observed P_test(M) = {p_test_m_obs:.4f}  ({obs_m:,}/{len(df):,})")

    # H_C-derived expected: Σ_y P_train(F|y) · P_test(y) (= what marginal SHOULD be under H_C)
    train_df = pd.read_csv(args.train_csv).dropna(subset=["gender", "FaceOcclusion"])
    train_y = train_df["FaceOcclusion"].values
    train_g = (train_df["gender"].astype(float).values >= 0.5).astype(int)
    p_g_given_y = empirical_p_g_given_y(train_y, train_g)  # shape (2, N_BINS)
    p_test_f_expected = float((p_g_given_y[0] * _TEST_PMF).sum())
    p_test_m_expected = float((p_g_given_y[1] * _TEST_PMF).sum())
    p_train_f = float((train_g == 0).mean())
    print(f"Expected P_test(F) under H_C = Σ_y P_train(F|y)·P_test(y) = {p_test_f_expected:.4f}")
    print(f"  (vs P_train(F) = {p_train_f:.4f}, shows the H_C-implied marginal shift)")

    delta_obs_hc = p_test_f_obs - p_test_f_expected
    print(f"\nDelta (observed − H_C expected) = {delta_obs_hc:+.4f}")
    if abs(delta_obs_hc) < 0.01:
        print("  → H_C strongly supported (≤ 1pp gap)")
    elif abs(delta_obs_hc) < 0.03:
        print("  → H_C plausible (1-3pp gap, within noise/classifier error)")
    elif abs(delta_obs_hc) < 0.07:
        print("  → H_C borderline (3-7pp gap), investigate classifier accuracy")
    else:
        print("  → H_C SUSPECT (>7pp gap), revise hypothesis")

    # === 6. Suggested update for _TEST_P_GENDER ===
    print()
    print("=" * 80)
    print("STEP 6 — Suggested update for src/utils/distribution.py")
    print("=" * 80)
    print("Current:   _TEST_P_GENDER = np.array([0.4879, 0.5121], dtype=np.float64)")
    print(f"Observed:  _TEST_P_GENDER = np.array([{p_test_f_obs:.4f}, {p_test_m_obs:.4f}], dtype=np.float64)  # v18 full-train probe")

    if val_metrics:
        print(f"\nProbe quality (sanity): test_mid_acc = {val_metrics.get('test_mid_acc', '?'):.4f}, "
              f"F = {val_metrics.get('test_mid_acc_F', '?'):.4f}, "
              f"M = {val_metrics.get('test_mid_acc_M', '?'):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
