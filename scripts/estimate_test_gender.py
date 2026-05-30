"""Estimate test set gender distribution via MID (Freebase identifier) lookup.

Usage: python scripts/estimate_test_gender.py

Filename format: database3/database3/{MID}/{frame}_align.webp
MID is the 3rd path segment. For test samples whose MID also appears in train,
we know the gender (Freebase ID = identity = fixed gender). For unknown MIDs,
gender is unresolved — output rate gives us the lower bound coverage of MID lookup.

This validates hypothesis H1 used for the val split: P_test(g|y) ≈ P_train(g|y).
First-order check: compare global F/M ratios test vs train. If close → H1 plausible.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def extract_mid(path: str) -> str:
    parts = Path(path).parts
    for p in parts:
        if p.startswith("m.") and len(p) > 2:
            return p
    return ""


def main() -> int:
    train_csv = "data/raw/train.csv"
    test_csv = "data/raw/test_students.csv"

    train_df = pd.read_csv(train_csv)
    train_df["mid"] = train_df["filename"].apply(extract_mid)
    train_df = train_df[train_df["mid"] != ""]
    print(f"Train: {len(train_df):,} samples, {train_df['mid'].nunique():,} unique MIDs")

    mid_genders = train_df.groupby("mid")["gender"].agg(lambda s: s.mode().iloc[0])
    train_g = train_df["gender"].astype(int)
    print(f"Train gender: F={int((train_g==0).sum()):,} ({(train_g==0).mean()*100:.2f}%)  "
          f"M={int((train_g==1).sum()):,} ({(train_g==1).mean()*100:.2f}%)")
    print(f"Per-MID majority: {(mid_genders==0).sum():,} F MIDs, {(mid_genders==1).sum():,} M MIDs")
    print()

    test_df = pd.read_csv(test_csv)
    test_df["mid"] = test_df["filename"].apply(extract_mid)
    n_test = len(test_df)
    n_no_mid = int((test_df["mid"] == "").sum())
    print(f"Test: {n_test:,} samples, {n_no_mid:,} have no parseable MID")
    test_df = test_df[test_df["mid"] != ""].copy()

    test_df["gender_lookup"] = test_df["mid"].map(mid_genders)
    n_known = int(test_df["gender_lookup"].notna().sum())
    n_unknown = int(test_df["gender_lookup"].isna().sum())
    print(f"MID lookup coverage: known={n_known:,} ({100*n_known/n_test:.2f}%)  "
          f"unknown={n_unknown + n_no_mid:,} ({100*(n_unknown+n_no_mid)/n_test:.2f}%)")
    print()

    g_test_known = test_df["gender_lookup"].dropna().astype(int)
    n_f = int((g_test_known == 0).sum())
    n_m = int((g_test_known == 1).sum())
    print(f"=== Test gender (MID-known subset, n={len(g_test_known):,}) ===")
    print(f"  F: {n_f:,} ({100*n_f/len(g_test_known):.2f}%)")
    print(f"  M: {n_m:,} ({100*n_m/len(g_test_known):.2f}%)")
    print()

    p_train_f = (train_g == 0).mean()
    p_test_f_known = n_f / len(g_test_known)
    delta = p_test_f_known - p_train_f
    print(f"=== Hypothesis H1 first-order check ===")
    print(f"  P_train(F) = {p_train_f:.4f}")
    print(f"  P_test(F)  = {p_test_f_known:.4f}  (MID-known subset)")
    print(f"  delta      = {delta:+.4f}")
    if abs(delta) < 0.03:
        print(f"  -> Plausible: marginals match within ±3%, H1 holds at the 1st-order")
    elif abs(delta) < 0.07:
        print(f"  -> Borderline: marginals diverge by 3-7%, H1 partially holds")
    else:
        print(f"  -> CAUTION: marginals diverge by >7%, H1 likely broken — revise val split")

    return 0


if __name__ == "__main__":
    sys.exit(main())
