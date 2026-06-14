from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.distribution import P_TEST_PMF, N_BINS, CENTERS


def _target_quantile_fn(pmf: np.ndarray):
    pmf = np.clip(np.asarray(pmf, float), 0.0, None)
    pmf = pmf / pmf.sum()
    edges = np.linspace(0.0, 1.0, len(pmf) + 1)
    cdf = np.concatenate([[0.0], np.cumsum(pmf)])
    cdf = cdf / cdf[-1]
    cdf, keep = np.unique(cdf, return_index=True)
    edges = edges[keep]
    return lambda u: np.interp(np.asarray(u, float), cdf, edges)


def _ranks(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p, kind="mergesort")
    u = np.empty(n, float)
    u[order] = (np.arange(n) + 0.5) / n
    return u


def quantile_map(p: np.ndarray, pmf: np.ndarray, strength: float = 1.0) -> np.ndarray:
    qfn = _target_quantile_fn(pmf)
    mapped = np.clip(qfn(_ranks(p)), 0.0, 1.0)
    out = (1.0 - strength) * np.asarray(p, float) + strength * mapped
    return np.clip(out, 0.0, 1.0)


def _per_gender_pmfs(train_csv: str, label_col: str, gender_col: str, groups: list):
    df = pd.read_csv(train_csv)
    y = np.clip(df[label_col].values.astype(float), 0.0, 1.0 - 1e-9)
    b = (y * N_BINS).astype(int)
    out = {}
    for grp in groups:
        mask = df[gender_col].astype(str).values == str(grp)
        cnt_g = np.bincount(b[mask], minlength=N_BINS).astype(float)
        cnt_all = np.bincount(b, minlength=N_BINS).astype(float)
        p_g_given_y = np.divide(cnt_g, cnt_all, out=np.zeros_like(cnt_g), where=cnt_all > 0)
        pmf = p_g_given_y * P_TEST_PMF
        s = pmf.sum()
        out[grp] = pmf / s if s > 0 else P_TEST_PMF.copy()
    return out


def _fit_isotonic(pred: np.ndarray, gt: np.ndarray, weighted: bool = True):
    from sklearn.isotonic import IsotonicRegression
    pred = np.asarray(pred, float)
    gt = np.asarray(gt, float)
    w = (1.0 / 30.0 + gt) if weighted else None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(pred, gt, sample_weight=w)
    return iso


def _weighted_err(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, float)
    gt = np.asarray(gt, float)
    w = 1.0 / 30.0 + gt
    return float((w * (pred - gt) ** 2).sum() / max(w.sum(), 1e-12))


def _w1_to_ptest(p: np.ndarray, pmf=P_TEST_PMF) -> float:
    grid = np.linspace(0.0, 1.0, 1001)
    hp, _ = np.histogram(np.clip(p, 0, 1), bins=1000, range=(0, 1), density=True)
    cdf_p = np.concatenate([[0.0], np.cumsum(hp) / hp.sum()])
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    cdf_t_bins = np.concatenate([[0.0], np.cumsum(pmf) / pmf.sum()])
    cdf_t = np.interp(grid, edges, cdf_t_bins)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz(np.abs(cdf_p - cdf_t), grid))


def main():
    ap = argparse.ArgumentParser(description="Monotone recalibration of predictions. Default: quantile-map the marginal onto the official P_test PMF (unsupervised, 1D OT). --isotonic: weighted isotonic regression fit on a labeled holdout (supervised, metric-aligned).")
    ap.add_argument("input_csv")
    ap.add_argument("--out", default=None, help="output CSV (default: <input>_ptest.csv)")
    ap.add_argument("--col", default="FaceOcclusion", help="prediction column to transform")
    ap.add_argument("--strength", type=float, default=1.0, help="0=identity, 1=full transform (default 1.0)")
    ap.add_argument("--per-gender", action="store_true", help="calibrate per gender (needs real gender values)")
    ap.add_argument("--gender-col", default="gender")
    ap.add_argument("--train-csv", default="data/raw/train.csv")
    ap.add_argument("--label-col", default="FaceOcclusion")
    ap.add_argument("--isotonic", action="store_true", help="supervised isotonic calibration instead of quantile map (requires --fit-csv)")
    ap.add_argument("--fit-csv", default=None, help="labeled holdout (pred + gt) to fit isotonic on")
    ap.add_argument("--fit-pred-col", default=None, help="prediction column in --fit-csv (default: --col)")
    ap.add_argument("--fit-label-col", default="gt", help="ground-truth column in --fit-csv (default: gt)")
    ap.add_argument("--no-weight", action="store_true", help="isotonic: unweighted MSE instead of metric weights w=1/30+GT")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    p = df[args.col].values.astype(float)
    out = df.copy()
    fit_before = fit_after = None

    if args.isotonic:
        if not args.fit_csv:
            sys.exit("--isotonic requires --fit-csv (a labeled holdout with prediction + ground-truth columns)")
        fit = pd.read_csv(args.fit_csv)
        fp = fit[args.fit_pred_col or args.col].values.astype(float)
        fg = fit[args.fit_label_col].values.astype(float)
        weighted = not args.no_weight
        if args.per_gender:
            g = df[args.gender_col].astype(str).values
            fgen = fit[args.gender_col].astype(str).values
            groups = sorted(set(g) - {"x", "X", "nan"})
            if len(groups) < 2:
                sys.exit(f"--per-gender: need >=2 real gender values, found {sorted(set(g))}")
            isos = {grp: _fit_isotonic(fp[fgen == grp], fg[fgen == grp], weighted) for grp in groups}
            calibrated = p.copy(); fit_cal = fp.copy()
            for grp in groups:
                calibrated[g == grp] = isos[grp].predict(p[g == grp])
                fit_cal[fgen == grp] = isos[grp].predict(fp[fgen == grp])
        else:
            iso = _fit_isotonic(fp, fg, weighted)
            calibrated = iso.predict(p); fit_cal = iso.predict(fp)
        fit_before, fit_after = _weighted_err(fp, fg), _weighted_err(fit_cal, fg)
        mode = f"isotonic(weighted={weighted}, fit_n={len(fp)})"
    else:
        if args.per_gender:
            g = df[args.gender_col].astype(str).values
            groups = sorted(set(g) - {"x", "X", "nan"})
            if len(groups) < 2:
                sys.exit(f"--per-gender: need >=2 real gender values, found {sorted(set(g))}")
            pmfs = _per_gender_pmfs(args.train_csv, args.label_col, args.gender_col, groups)
            calibrated = p.copy()
            for grp in groups:
                calibrated[g == grp] = quantile_map(p[g == grp], pmfs[grp], 1.0)
        else:
            calibrated = quantile_map(p, P_TEST_PMF, 1.0)
        mode = "quantile-map"

    final = np.clip((1.0 - args.strength) * p + args.strength * np.asarray(calibrated, float), 0.0, 1.0)
    out[args.col] = final

    w1_before, w1_after = _w1_to_ptest(p), _w1_to_ptest(final)
    out_path = args.out or str(Path(args.input_csv).with_name(Path(args.input_csv).stem + "_ptest.csv"))
    out.to_csv(out_path, index=False)

    print(f"  in : {args.input_csv}  (n={len(df)})")
    print(f"  out: {out_path}")
    print(f"  mode={mode}  strength={args.strength}  per_gender={args.per_gender}")
    print(f"  pred mean {p.mean():.4f} -> {final.mean():.4f}   (P_test mean {(CENTERS*P_TEST_PMF).sum()/P_TEST_PMF.sum():.4f})")
    print(f"  frac<0.01  {np.mean(p<0.01):.3f} -> {np.mean(final<0.01):.3f}")
    print(f"  W1 to P_test: {w1_before:.5f} -> {w1_after:.5f}  ({(w1_after-w1_before)/max(w1_before,1e-9)*100:+.1f}%)")
    if fit_before is not None:
        print(f"  fit-set weighted err (in-sample): {fit_before:.6f} -> {fit_after:.6f}  ({(fit_after-fit_before)/max(fit_before,1e-12)*100:+.1f}%)")


if __name__ == "__main__":
    main()
