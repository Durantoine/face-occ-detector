"""Compare the predicted-occlusion distribution of the k-fold ensemble submission vs a single fold
(default: the best fold), with the P_test target overlaid. Saves a PNG + prints summary stats.

Usage (on the cluster, where results/ live):
  python scripts/plot_kfold_dist.py \
    --submission results/convnextv2-large-H100-v39_trial6_kfold_submission_weighted.csv \
    --fold-pred  results/convnextv2-large-H100-v39_trial6_fold0/test_predictions.csv \
    --out results/kfold_ensemble_vs_fold0.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.distribution import BIN_WIDTH, CENTERS, P_TEST_PMF


def _preds(path: str, col: str) -> np.ndarray:
    df = pd.read_csv(path)
    c = col if col in df.columns else df.columns[-1]
    return np.clip(df[c].astype(float).values, 0.0, 1.0)


def _stats(x: np.ndarray) -> str:
    return (f"n={len(x):,} mean={x.mean():.4f} std={x.std():.4f} median={np.median(x):.4f} "
            f">0.3={np.mean(x > 0.3):.3f} >0.5={np.mean(x > 0.5):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--fold-pred", required=True)
    ap.add_argument("--label-col", default="FaceOcclusion")
    ap.add_argument("--out", default="results/kfold_ensemble_vs_fold.png")
    args = ap.parse_args()

    ens = _preds(args.submission, args.label_col)
    fold = _preds(args.fold_pred, args.label_col)
    print(f"[ensemble] {_stats(ens)}")
    print(f"[fold    ] {_stats(fold)}")

    bins = np.linspace(0, 1, 61)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    for a, (lo, hi, title) in zip(ax, [(0.0, 1.0, "full range"), (0.0, 0.5, "zoom 0-0.5 (P_test support)")]):
        a.hist(fold, bins=bins, density=True, histtype="step", lw=2, color="tab:orange", label="best fold")
        a.hist(ens, bins=bins, density=True, histtype="step", lw=2, color="tab:blue", label="ensemble (weighted)")
        a.plot(CENTERS, P_TEST_PMF / BIN_WIDTH, color="k", ls="--", lw=1.5, alpha=.7, label="P_test target")
        a.set_xlim(lo, hi); a.set_xlabel("predicted occlusion"); a.set_ylabel("density")
        a.set_title(title); a.legend(); a.grid(alpha=.3)

    fig.suptitle("K-fold ensemble vs best fold — test prediction distribution")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
