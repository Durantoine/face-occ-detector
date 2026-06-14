"""Aggregate a K-fold run of a single config.

Each fold #k was trained by train.py with --fold k and wrote:
  results/<name>_fold{k}/val_predictions.csv   (gt, pred, gender on that fold's held-out frames)
  results/<name>_fold{k}/test_predictions.csv  (filename, FaceOcclusion from that fold's best.pt)

Because the K val folds are disjoint and tile the dataset exactly once, concatenating the
val_predictions gives a true out-of-fold (OOF) prediction for every training frame -> a single,
low-variance cross-validated challenge score directly comparable to the single-split validation.

The K test predictions are averaged into an ensemble submission.

Usage:
  python scripts/kfold_oof.py configs/architectures/config_v39_trial6_resolved.yaml --n-folds 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.utils.distribution import get_joint_weight_fn, stratified_is_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="Path/name of the architecture config used for the K-fold run")
    ap.add_argument("--n-folds", type=int, required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=None, help="Submission CSV path (default: results/<name>_kfold_submission.csv)")
    args = ap.parse_args()

    cfg, _ = load_config(args.config)
    base = cfg.name
    rdir = Path(args.results_dir)

    # Weight function fit on the FULL real train distribution (val is drawn from real data), exactly
    # as train.py fits weight_fn_score / weight_fn_full -> the OOF number is comparable to the run logs.
    df = pd.read_csv(cfg.data_csv)
    rm = getattr(cfg, "remove_csv", "") or ""
    if rm and Path(rm).exists():
        df = df[~df[cfg.image_col].isin(set(pd.read_csv(rm)[cfg.image_col]))].reset_index(drop=True)
    y_all, g_all = df[cfg.label_col].values, df[cfg.gender_col].values
    wf_interim = get_joint_weight_fn(y_all, g_all, target=cfg.is_target, alpha=cfg.eval_alpha, lam=cfg.is_lambda)
    wf_final = get_joint_weight_fn(y_all, g_all, target=cfg.is_target, alpha=1.0, lam=cfg.is_lambda)

    val_parts, test_cols, fold_score, per_fold = [], {}, {}, []
    for k in range(args.n_folds):
        fdir = rdir / f"{base}_fold{k}"
        vp = fdir / "val_predictions.csv"
        tp = fdir / "test_predictions.csv"
        if not vp.exists():
            raise FileNotFoundError(f"missing {vp} -- fold {k} did not finish?")
        v = pd.read_csv(vp)
        s = stratified_is_score(v["pred"].values, v["gt"].values, v["gender"].values, weight_fn=wf_interim)
        per_fold.append((k, len(v), s["challenge_score"], s["err_gap"]))
        fold_score[k] = s["challenge_score"]
        val_parts.append(v)
        if tp.exists():
            t = pd.read_csv(tp).rename(columns={cfg.label_col: f"pred_{k}"})
            test_cols[k] = t.set_index(cfg.image_col)[f"pred_{k}"]

    oof = pd.concat(val_parts, ignore_index=True)
    s_int = stratified_is_score(oof["pred"].values, oof["gt"].values, oof["gender"].values, weight_fn=wf_interim)
    s_fin = stratified_is_score(oof["pred"].values, oof["gt"].values, oof["gender"].values, weight_fn=wf_final)

    print("=" * 72)
    print(f"K-FOLD OOF REPORT  --  {base}  ({args.n_folds} folds)")
    print("=" * 72)
    print(f"{'fold':>4} | {'n_val':>7} | {'score(interim)':>14} | {'gap':>9}")
    for k, n, sc, gp in per_fold:
        print(f"{k:>4} | {n:>7,} | {sc:>14.5f} | {gp:>+9.5f}")
    scores = np.array([p[2] for p in per_fold])
    print("-" * 72)
    print(f"per-fold mean={scores.mean():.5f}  std={scores.std():.5f}")
    print(f"OOF (all {len(oof):,} frames pooled):")
    print(f"  interim (alpha={cfg.eval_alpha}): score={s_int['challenge_score']:.5f}  "
          f"gap={s_int['err_gap']:+.5f} (F={s_int['err_F']:.4f}, M={s_int['err_M']:.4f})")
    print(f"  final   (alpha=1.0): score={s_fin['challenge_score']:.5f}  "
          f"gap={s_fin['err_gap']:+.5f} (F={s_fin['err_F']:.4f}, M={s_fin['err_M']:.4f})")
    print("=" * 72)

    if test_cols:
        ks = sorted(test_cols)
        mat = pd.concat([test_cols[k] for k in ks], axis=1)          # rows=filename, cols=folds
        # weight ~ 1/fold_score (better fold -> more weight), normalised; same scheme as ensemble_challenge.py
        w = np.array([1.0 / max(fold_score[k], 1e-9) for k in ks]); w /= w.sum()
        out_unw = rdir / f"{base}_kfold_submission_unweighted.csv"
        out_wei = Path(args.out) if args.out else rdir / f"{base}_kfold_submission_weighted.csv"
        for tag, vals, path in [
            ("unweighted", mat.mean(axis=1).values, out_unw),
            ("weighted",   (mat.values * w).sum(axis=1), out_wei),
        ]:
            sub = pd.DataFrame({cfg.image_col: mat.index, cfg.label_col: np.clip(vals, 0.0, 1.0)})
            sub.to_csv(path, index=False)
            print(f"ensemble [{tag:10}] ({len(ks)} folds) -> {path}  "
                  f"(mean={sub[cfg.label_col].mean():.4f}, std={sub[cfg.label_col].std():.4f})")
        print("  fold weights:", {k: round(float(wi), 3) for k, wi in zip(ks, w)})
    else:
        print("no test_predictions.csv found in any fold dir -- no submission written.")


if __name__ == "__main__":
    main()
