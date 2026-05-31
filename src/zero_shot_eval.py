"""
Zero-shot evaluation of face occlusion estimation (SAM2 + Qwen2.5-VL).

Runs the pipeline on a stratified sample of the labeled training set and
produces global + per-group metrics and a Fairness Report.

Stratification: gender (Female/Male) × occlusion level (low/medium/high)
Fairness warning: if |MAE_Female - MAE_Male| / mean(MAE) > 10%

Usage:
    python src/zero_shot_eval.py                          # 500 samples
    python src/zero_shot_eval.py --n-samples 100          # quick sanity check
    python src/zero_shot_eval.py --n-samples 2000 --seed 7
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.metrics import compute_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Fairness thresholds ───────────────────────────────────────────────────────

GENDER_LABELS: Dict[int, str] = {0: "Female", 1: "Male"}
OCC_BINS: List[Tuple[str, float, float]] = [
    ("low",    0.00, 0.10),
    ("medium", 0.10, 0.30),
    ("high",   0.30, 1.01),
]
# Warn if relative gap > 10% OR absolute gap > 0.02 MAE
FAIRNESS_REL_THRESHOLD = 0.10
FAIRNESS_ABS_THRESHOLD = 0.02


# ── Stratified sampling ───────────────────────────────────────────────────────


def _occ_level(v: float) -> str:
    for name, lo, hi in OCC_BINS:
        if lo <= v < hi:
            return name
    return "high"


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Sample n rows stratified by gender × occlusion_level, preserving natural proportions."""
    df = df.copy()
    df["_stratum"] = df["gender"].astype(int).astype(str) + "_" + df["FaceOcclusion"].apply(_occ_level)

    strata_counts = df["_stratum"].value_counts()
    total = len(df)

    # Proportional allocation (at least 1 per stratum)
    n_per_stratum: Dict[str, int] = {
        s: max(1, round(n * count / total))
        for s, count in strata_counts.items()
    }

    # Integer adjustment to hit exactly n
    delta = n - sum(n_per_stratum.values())
    for s in sorted(strata_counts.index, key=lambda x: strata_counts[x], reverse=True):
        if delta == 0:
            break
        step = 1 if delta > 0 else -1
        n_per_stratum[s] = max(1, n_per_stratum[s] + step)
        delta -= step

    rng = np.random.default_rng(seed)
    parts: List[pd.DataFrame] = []
    for stratum, k in n_per_stratum.items():
        subset = df[df["_stratum"] == stratum]
        k = min(k, len(subset))
        idx = rng.choice(len(subset), size=k, replace=False)  # type: ignore[arg-type]
        parts.append(subset.iloc[idx])

    result = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return result.drop(columns=["_stratum"])


# ── Evaluation loop ───────────────────────────────────────────────────────────


def run_evaluation(pipeline, df: pd.DataFrame, image_base_dir: Path) -> pd.DataFrame:
    """Run zero-shot pipeline over df; return a results DataFrame."""
    records = []
    n = len(df)
    for i, row in enumerate(df.itertuples(index=False), 1):
        if i == 1 or i % 50 == 0:
            logger.info("Progress: %d / %d", i, n)

        image_path = image_base_dir / row.filename
        result = pipeline.predict(image_path)

        records.append({
            "filename":               row.filename,
            "gt":                     float(row.FaceOcclusion),
            "pred":                   float(result.prediction),
            # Signal A
            "signal_a_geom":          float(result.signal_a_geom),
            "signal_a_conf":          float(result.signal_a_conf),
            # Signal B
            "signal_b_orientation":   result.signal_b_orientation,
            "signal_b_pose_occ":      float(result.signal_b_pose_occ),
            "signal_b_missing":       result.signal_b_missing,
            # Signal C
            "signal_c_type":          result.signal_c_type,
            "signal_c_first_imp":     float(result.signal_c_first_imp),
            "signal_c_description":   result.signal_c_description,
            # Judge
            "sam2_reliable":          bool(result.sam2_reliable),
            "mediapipe_reliable":     bool(result.mediapipe_reliable),
            "dominant_signal":        result.dominant_signal,
            "reasoning":              result.reasoning,
            # Compat
            "gender":                 int(row.gender),
            "geometric_occ":          float(result.geometric_occ) if result.geometric_occ is not None else float("nan"),
            "used_mask":              bool(result.used_mask),
            "failure":                result.failure or "",
        })

    return pd.DataFrame(records)


# ── Metrics & fairness report ─────────────────────────────────────────────────


def compute_fairness_report(results: pd.DataFrame) -> dict:
    preds  = results["pred"].astype(float).values
    gt     = results["gt"].astype(float).values
    gender = results["gender"].astype(float).values

    # ── Global challenge metrics ──────────────────────────────────────────────
    global_m = compute_score(preds, gt, gender)
    global_section = {
        "n":               len(results),
        "challenge_score": round(global_m["challenge_score"], 6),
        "mae":             round(global_m["mae"], 6),
        "r2":              round(global_m["r2"], 4),
        "err_F":           round(global_m["err_F"], 6),
        "err_M":           round(global_m["err_M"], 6),
        "err_diff":        round(global_m["err_diff"], 6),
    }

    # ── Per-gender ────────────────────────────────────────────────────────────
    per_gender: Dict[str, dict] = {}
    for g_code, g_label in GENDER_LABELS.items():
        mask = gender == g_code
        if mask.sum() == 0:
            continue
        p_g, gt_g = preds[mask], gt[mask]
        err = p_g - gt_g
        corr = float(np.corrcoef(p_g, gt_g)[0, 1]) if mask.sum() > 1 else float("nan")
        per_gender[g_label] = {
            "n":           int(mask.sum()),
            "mae":         round(float(np.abs(err).mean()), 6),
            "correlation": round(corr, 4),
            "mean_bias":   round(float(err.mean()), 6),   # positive = over-prediction
            "std_error":   round(float(err.std()), 6),
        }

    # ── Per occlusion level ───────────────────────────────────────────────────
    per_occ: Dict[str, dict] = {}
    for level, lo, hi in OCC_BINS:
        mask = (gt >= lo) & (gt < hi)
        if mask.sum() == 0:
            continue
        per_occ[level] = {
            "n":   int(mask.sum()),
            "mae": round(float(np.abs(preds[mask] - gt[mask]).mean()), 6),
        }

    # ── Per stratum (gender × occ_level) ──────────────────────────────────────
    per_stratum: Dict[str, dict] = {}
    for g_code, g_label in GENDER_LABELS.items():
        for level, lo, hi in OCC_BINS:
            mask = (gender == g_code) & (gt >= lo) & (gt < hi)
            if mask.sum() == 0:
                continue
            key = f"{g_label}_{level}"
            per_stratum[key] = {
                "n":   int(mask.sum()),
                "mae": round(float(np.abs(preds[mask] - gt[mask]).mean()), 6),
            }

    # ── Fairness gap ──────────────────────────────────────────────────────────
    mae_f = per_gender.get("Female", {}).get("mae", 0.0)
    mae_m = per_gender.get("Male",   {}).get("mae", 0.0)
    mean_mae = (mae_f + mae_m) / 2.0
    abs_gap  = abs(mae_f - mae_m)
    rel_gap  = abs_gap / max(mean_mae, 1e-9)
    fairness_warning = rel_gap > FAIRNESS_REL_THRESHOLD or abs_gap > FAIRNESS_ABS_THRESHOLD

    # ── Failures (SAM2/Qwen errors) ───────────────────────────────────────────
    failures_df = results[results["failure"] != ""]
    failure_log = [
        {
            "filename": row["filename"],
            "gender":   GENDER_LABELS.get(int(row["gender"]), "unknown"),
            "gt":       round(float(row["gt"]), 4),
            "failure":  row["failure"],
        }
        for _, row in failures_df.iterrows()
    ]

    # ── Pipeline stats ────────────────────────────────────────────────────────
    mask_used  = results["used_mask"].sum()
    a_conf_col = "signal_a_conf" if "signal_a_conf" in results.columns else "sam2_confidence"
    sam2_valid = results[a_conf_col].gt(0).sum() if a_conf_col in results.columns else 0
    mean_conf  = (
        round(float(results[a_conf_col].mean()), 4)
        if a_conf_col in results.columns else 0.0
    )

    pipeline_stats = {
        "sam2_mask_used":       int(mask_used),
        "sam2_detected":        int(sam2_valid),
        "total":                len(results),
        "mask_use_rate_pct":    round(100 * mask_used / max(len(results), 1), 1),
        "mean_sam2_confidence": mean_conf,
        "failure_count":        len(failures_df),
        "failure_by_gender": {
            GENDER_LABELS[g]: int((failures_df["gender"] == g).sum())
            for g in GENDER_LABELS
        },
    }

    # ── Dominant signal distribution + MAE per signal ────────────────────────
    dom_dist: Dict[str, int] = {}
    dom_mae:  Dict[str, dict] = {}
    if "dominant_signal" in results.columns:
        dom_dist = results["dominant_signal"].value_counts().to_dict()
        for sig in results["dominant_signal"].dropna().unique():
            m = results["dominant_signal"] == sig
            if m.sum() == 0:
                continue
            dom_mae[str(sig)] = {
                "n":   int(m.sum()),
                "mae": round(float(np.abs(preds[m] - gt[m]).mean()), 6),
            }

    # ── Signal C type distribution ────────────────────────────────────────────
    signal_c_dist: Dict[str, int] = {}
    if "signal_c_type" in results.columns:
        signal_c_dist = results["signal_c_type"].value_counts().to_dict()

    return {
        "global":            global_section,
        "per_gender":        per_gender,
        "per_occ_level":     per_occ,
        "per_stratum":       per_stratum,
        "fairness_warning":  fairness_warning,
        "fairness_gap": {
            "mae_female":  mae_f,
            "mae_male":    mae_m,
            "abs_gap":     round(abs_gap, 6),
            "rel_gap_pct": round(100 * rel_gap, 2),
        },
        "failures":          failure_log,
        "pipeline_stats":    pipeline_stats,
        "dominant_signal_distribution": dom_dist,
        "dominant_signal_mae":          dom_mae,
        "signal_c_type_distribution":   signal_c_dist,
    }


# ── Diagnostic analysis ──────────────────────────────────────────────────────


def analyze_results(results: pd.DataFrame, error_threshold: float = 0.2) -> None:
    """Answer three diagnostic questions after evaluation:
    1. Which images have |error| > error_threshold?  (SAM2 over-segmented?)
    2. What range does Qwen predict vs the GT distribution?
    3. Is there a systematic bias (over- or under-prediction)?
    """
    preds  = results["pred"].astype(float).values
    gt     = results["gt"].astype(float).values
    errors = np.abs(preds - gt)
    signed = preds - gt
    gender = results["gender"].astype(int).values

    LINE = "─" * 72

    # ── 1. Worst predictions ──────────────────────────────────────────────────
    print(f"\n{LINE}")
    print(f"  DIAGNOSTIC 1 — WORST PREDICTIONS  (|error| > {error_threshold})")
    print(LINE)
    bad_mask = errors > error_threshold
    n_bad = int(bad_mask.sum())
    print(f"  {n_bad} / {len(results)} samples exceed threshold ({100*n_bad/len(results):.1f}%)")

    if n_bad > 0:
        bad = results[bad_mask].copy()
        bad["abs_error"]  = errors[bad_mask]
        bad["signed_err"] = signed[bad_mask]
        bad = bad.sort_values("abs_error", ascending=False)

        a_conf_col = "signal_a_conf" if "signal_a_conf" in bad.columns else "sam2_confidence"
        has_desc   = "signal_c_description" in bad.columns
        has_dom    = "dominant_signal" in bad.columns

        print(f"\n  {'G':1}  {'gt':>5}  {'pred':>5}  {'err':>5}  "
              f"{'A_c':>5}  {'A_g%':>5}  {'B_or':>12}  {'C_1st':>5}  {'dom':>11}  filename")
        for _, row in bad.head(20).iterrows():
            g_label  = GENDER_LABELS.get(int(row["gender"]), "?")[:1]
            a_conf   = row.get(a_conf_col, float("nan"))
            geom     = row.get("geometric_occ", float("nan"))
            geom_str = f"{geom*100:.0f}%" if not np.isnan(geom) else "N/A"
            b_orient = str(row.get("signal_b_orientation", ""))[:12]
            c_first  = row.get("signal_c_first_imp", float("nan"))
            dom      = str(row.get("dominant_signal", ""))[:11]
            print(
                f"  {g_label}  {row['gt']:>5.3f}  {row['pred']:>5.3f}  "
                f"{row['abs_error']:>5.3f}  {a_conf:>5.3f}  {geom_str:>5}  "
                f"{b_orient:>12}  {c_first:>5.2f}  {dom:>11}  "
                f"{Path(row['filename']).name}"
            )
            if has_desc and row.get("signal_c_description", ""):
                print(f"       C: \"{str(row['signal_c_description'])[:110]}\"")
            if has_dom and row.get("reasoning", ""):
                print(f"       J: \"{str(row['reasoning'])[:110]}\"")
        if n_bad > 20:
            print(f"  … {n_bad - 20} more in the CSV")

    # ── 2. Distribution comparison (ASCII histogram) ──────────────────────────
    print(f"\n{LINE}")
    print("  DIAGNOSTIC 2 — PREDICTION DISTRIBUTION vs GROUND TRUTH")
    print(LINE)
    bins = np.arange(0.0, 1.025, 0.05)   # 20 bins of width 0.05
    pred_hist, _ = np.histogram(preds, bins=bins)
    gt_hist,   _ = np.histogram(gt,    bins=bins)
    max_count = max(pred_hist.max(), gt_hist.max(), 1)
    bar_width  = 30  # characters

    print(f"  {'bin':>10}  {'GT':^{bar_width}}  {'Pred':^{bar_width}}")
    print(f"  {'':>10}  {'(ground truth)':^{bar_width}}  {'(Qwen output)':^{bar_width}}")
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        gt_bar   = round(gt_hist[i]   / max_count * bar_width)
        pred_bar = round(pred_hist[i] / max_count * bar_width)
        print(
            f"  [{lo:.2f}-{hi:.2f})  "
            f"{'█' * gt_bar:{bar_width}}  "
            f"{'█' * pred_bar:{bar_width}}  "
            f"GT={gt_hist[i]:3d}  Pred={pred_hist[i]:3d}"
        )

    print(f"\n  GT   range : [{gt.min():.3f}, {gt.max():.3f}]  "
          f"mean={gt.mean():.3f}  median={np.median(gt):.3f}")
    print(f"  Pred range : [{preds.min():.3f}, {preds.max():.3f}]  "
          f"mean={preds.mean():.3f}  median={np.median(preds):.3f}")

    # ── 3. Bias analysis ───────────────────────────────────────────────────────
    print(f"\n{LINE}")
    print("  DIAGNOSTIC 3 — SYSTEMATIC BIAS")
    print(LINE)
    global_bias = float(signed.mean())
    direction   = "over-predicts" if global_bias > 0 else "under-predicts"
    print(f"  Global bias (mean signed error): {global_bias:+.4f}  → Qwen {direction} occlusion")

    for g_code, g_label in GENDER_LABELS.items():
        m = gender == g_code
        if m.sum() == 0:
            continue
        bias = float(signed[m].mean())
        d    = "over" if bias > 0 else "under"
        print(f"  {g_label:8s} bias                : {bias:+.4f}  → {d}-prediction")

    # By occlusion level
    for level, lo, hi in OCC_BINS:
        m = (gt >= lo) & (gt < hi)
        if m.sum() == 0:
            continue
        bias = float(signed[m].mean())
        d    = "over" if bias > 0 else "under"
        print(f"  {level:8s} occ bias (GT∈[{lo:.2f},{hi:.2f}]) : {bias:+.4f}  → {d}-prediction")

    # Check if Qwen compresses dynamic range (predicts too close to mean)
    pred_std = float(preds.std())
    gt_std   = float(gt.std())
    ratio    = pred_std / max(gt_std, 1e-6)
    print(f"\n  Pred std / GT std = {pred_std:.4f} / {gt_std:.4f} = {ratio:.2f}")
    if ratio < 0.5:
        print("  ⚠ Qwen compresses output range significantly (regression to mean)")
    elif ratio > 1.5:
        print("  ⚠ Qwen over-spreads predictions (higher variance than GT)")
    else:
        print("  Qwen output spread looks reasonable vs GT spread")

    print(f"\n{LINE}\n")


# ── Console printer ───────────────────────────────────────────────────────────


def print_fairness_report(report: dict) -> None:
    g  = report["global"]
    pg = report["per_gender"]
    po = report["per_occ_level"]
    fw = report["fairness_warning"]
    fgap = report["fairness_gap"]
    ps = report["pipeline_stats"]
    failures = report["failures"]

    SEP  = "=" * 72
    LINE = "─" * 72

    print(f"\n{SEP}")
    print("  ZERO-SHOT REPORT — SAM2 + Qwen2.5-VL-7B  (Training-Free Track)")
    print(SEP)

    print(f"\n  Samples evaluated : {g['n']}")
    print(f"  SAM2 face detected: {ps['sam2_detected']} / {ps['total']}")
    print(f"  SAM2 mask used    : {ps['sam2_mask_used']} ({ps['mask_use_rate_pct']:.1f}%)")
    print(f"  Mean SAM2 conf.   : {ps['mean_sam2_confidence']:.3f}")
    print(f"  Failures          : {ps['failure_count']}  "
          f"(F={ps['failure_by_gender'].get('Female',0)}, "
          f"M={ps['failure_by_gender'].get('Male',0)})")

    # Dominant signal distribution
    total = g["n"]
    dom_dist = report.get("dominant_signal_distribution", {})
    dom_mae  = report.get("dominant_signal_mae", {})
    if dom_dist:
        print("\n  Dominant signal distribution :")
        for sig in ("SAM2", "MediaPipe", "Description", "Combined"):
            n_sig = dom_dist.get(sig, 0)
            if n_sig == 0:
                continue
            pct_sig = 100 * n_sig / max(total, 1)
            mae_sig = dom_mae.get(sig, {}).get("mae", float("nan"))
            bar = "█" * max(1, round(pct_sig / 100 * 30))
            print(f"    {sig:<12} : {n_sig:3d} ({pct_sig:4.1f}%)  MAE={mae_sig:.4f}  {bar}")

    # Signal C type distribution
    c_dist = report.get("signal_c_type_distribution", {})
    if c_dist:
        print("  Signal C types : " + "  ".join(f"{k}={v}" for k, v in c_dist.items()))

    print(f"\n{LINE}")
    print("  GLOBAL METRICS")
    print(LINE)
    print(f"  Challenge score   : {g['challenge_score']:.5f}   (lower = better)")
    print(f"  MAE               : {g['mae']:.5f}")
    print(f"  R²                : {g['r2']:.4f}")
    print(f"  Weighted Err_F    : {g['err_F']:.5f}")
    print(f"  Weighted Err_M    : {g['err_M']:.5f}")
    print(f"  |Err_F - Err_M|   : {g['err_diff']:.5f}")

    print(f"\n{LINE}")
    print("  FAIRNESS REPORT — PER GENDER")
    print(LINE)
    for g_label, gm in pg.items():
        print(
            f"  {g_label:8s}  n={gm['n']:5d}  "
            f"MAE={gm['mae']:.5f}  corr={gm['correlation']:.3f}  "
            f"bias={gm['mean_bias']:+.4f}  σ={gm['std_error']:.4f}"
        )
    print(f"\n  MAE gap |F-M|     : {fgap['abs_gap']:.5f}  ({fgap['rel_gap_pct']:.1f}% relative)")
    if fw:
        print()
        print(f"  *** FAIRNESS WARNING: gender gap exceeds {100*FAIRNESS_REL_THRESHOLD:.0f}% threshold ***")
        print(f"      MAE_Female = {fgap['mae_female']:.5f}  |  MAE_Male = {fgap['mae_male']:.5f}")
        print(f"      Relative gap = {fgap['rel_gap_pct']:.1f}%")
        print("      → Pipeline may be systematically biased on one gender.")
        print("        Inspect per-stratum results and failure log for root cause.")

    print(f"\n{LINE}")
    print("  PERFORMANCE BY OCCLUSION LEVEL")
    print(LINE)
    for level, lm in po.items():
        bar = "█" * max(1, round(lm["mae"] * 100))
        print(f"  {level:8s}  n={lm['n']:5d}  MAE={lm['mae']:.5f}  {bar}")

    print(f"\n{LINE}")
    print("  PERFORMANCE BY STRATUM (gender × occlusion level)")
    print(LINE)
    for stratum, sm in report["per_stratum"].items():
        print(f"  {stratum:22s}  n={sm['n']:5d}  MAE={sm['mae']:.5f}")

    if failures:
        print(f"\n{LINE}")
        print(f"  FAILURE LOG  ({len(failures)} total — showing first 20)")
        print(LINE)
        for f in failures[:20]:
            short_name = Path(f["filename"]).name
            print(
                f"  [{f['gender']:6s}]  gt={f['gt']:.3f}  "
                f"{f['failure'][:55]:<55}  {short_name}"
            )
        if len(failures) > 20:
            print(f"  … {len(failures) - 20} more — see JSON report.")

    print(f"\n{SEP}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


# ── Multi-version comparison ──────────────────────────────────────────────────


def _metrics_from_csv(path: Path) -> Dict[str, float]:
    """Load a predictions CSV and compute key metrics (no need for full pipeline)."""
    df = pd.read_csv(path)
    preds  = df["pred"].astype(float).values
    gt     = df["gt"].astype(float).values
    gender = df["gender"].astype(float).values
    m = compute_score(preds, gt, gender)
    errors = np.abs(preds - gt)
    return {
        "challenge_score": m["challenge_score"],
        "mae":             m["mae"],
        "err_F":           m["err_F"],
        "err_M":           m["err_M"],
        "err_diff":        m["err_diff"],
        "n_worst_02":      int((errors > 0.2).sum()),
        "n":               len(df),
    }


def compare_runs(current: pd.DataFrame, baselines: Dict[str, Path]) -> None:
    """Print a side-by-side comparison table across multiple run versions."""
    LINE = "─" * 82
    rows: Dict[str, Dict[str, float]] = {}

    for name, path in baselines.items():
        if not path.exists():
            logger.warning("Baseline CSV not found: %s (skipping)", path)
            continue
        rows[name] = _metrics_from_csv(path)

    # Add current run
    preds  = current["pred"].astype(float).values
    gt     = current["gt"].astype(float).values
    gender = current["gender"].astype(float).values
    m = compute_score(preds, gt, gender)
    rows["v_current"] = {
        "challenge_score": m["challenge_score"],
        "mae":             m["mae"],
        "err_F":           m["err_F"],
        "err_M":           m["err_M"],
        "err_diff":        m["err_diff"],
        "n_worst_02":      int((np.abs(preds - gt) > 0.2).sum()),
        "n":               len(current),
    }

    if len(rows) < 2:
        return  # Nothing to compare against

    print(f"\n{LINE}")
    print("  VERSION COMPARISON")
    print(LINE)
    header = f"  {'version':>12}  {'score':>7}  {'MAE':>6}  {'err_F':>6}  "
    header += f"{'err_M':>6}  {'|F-M|':>6}  {'worst>0.2':>9}"
    print(header)
    print(f"  {'':>12}  {'':>7}  {'':>6}  {'':>6}  {'':>6}  {'':>6}  {'':>9}")

    base_score = None
    for name, r in rows.items():
        tag = " ← current" if name == "v_current" else ""
        delta = ""
        if base_score is not None:
            diff = r["challenge_score"] - base_score
            delta = f"({diff:+.5f})"
        else:
            base_score = r["challenge_score"]
        print(
            f"  {name:>12}  {r['challenge_score']:.5f}  {r['mae']:.4f}  "
            f"{r['err_F']:.4f}  {r['err_M']:.4f}  {r['err_diff']:.4f}  "
            f"{r['n_worst_02']:>9}  {delta}{tag}"
        )
    print(LINE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zero-shot face occlusion evaluation (SAM2 + Qwen2.5-VL)."
    )
    p.add_argument("--csv",            default="data/train.csv",
                   help="Labeled CSV with columns: filename, FaceOcclusion, gender")
    p.add_argument("--image-base-dir",
                   default="/home/matt/Programmation/704_IADATA_ML_avance/datachallenge/"
                           "DataChallenge2026/occlusion_datasets/raw/Crop_224_5fp_100K",
                   help="Base directory prepended to filename paths in the CSV")
    p.add_argument("--n-samples",      type=int,   default=500,
                   help="Number of samples (stratified by gender × occlusion level)")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--output-dir",     default="results/zero_shot",
                   help="Directory for predictions CSV and fairness report JSON")
    p.add_argument("--sam2-model",     default="facebook/sam2.1-hiera-large")
    p.add_argument("--qwen-model",     default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--sam2-device",    default="cuda")
    p.add_argument("--qwen-device",    default="auto")
    p.add_argument(
        "--compare-csv",
        metavar="NAME:PATH",
        action="append",
        default=[],
        help=(
            "Previous run CSV to compare against. Format: 'name:path'. "
            "Can be specified multiple times. "
            "Example: --compare-csv v1:results/zero_shot/predictions_20260529_2200.csv"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Load and stratify dataset ─────────────────────────────────────────────
    logger.info("Loading dataset from %s", args.csv)
    df = pd.read_csv(args.csv).dropna(subset=["filename", "FaceOcclusion", "gender"])
    df["gender"] = pd.to_numeric(df["gender"], errors="coerce").fillna(0.5)
    df = df[df["gender"].isin([0, 1])]  # drop ambiguous rows

    logger.info("Stratified sampling: %d from %d total", args.n_samples, len(df))
    sample_df = stratified_sample(df, args.n_samples, seed=args.seed)

    gender_dist = sample_df["gender"].value_counts().sort_index()
    for g, count in gender_dist.items():
        label = GENDER_LABELS.get(int(g), str(g))
        logger.info("  %s: %d (%.1f%%)", label, count, 100 * count / len(sample_df))

    # ── Load pipeline ─────────────────────────────────────────────────────────
    # Import here so the module can be imported without GPU
    from src.inference.zero_shot_pipeline import ZeroShotOcclusionPipeline  # noqa: PLC0415

    pipeline = ZeroShotOcclusionPipeline(
        sam2_model_id=args.sam2_model,
        qwen_model_id=args.qwen_model,
        sam2_device=args.sam2_device,
        qwen_device=args.qwen_device,
    )

    # ── Run evaluation ────────────────────────────────────────────────────────
    results = run_evaluation(pipeline, sample_df, Path(args.image_base_dir))

    # ── Fairness report + diagnostic analysis ────────────────────────────────
    report = compute_fairness_report(results)
    print_fairness_report(report)
    analyze_results(results)

    # ── Version comparison ────────────────────────────────────────────────────
    baselines: Dict[str, Path] = {}
    for spec in args.compare_csv:
        if ":" not in spec:
            logger.warning("--compare-csv format must be 'name:path', got %r", spec)
            continue
        name, path_str = spec.split(":", 1)
        baselines[name] = Path(path_str)
    if baselines:
        compare_runs(results, baselines)

    # ── Save outputs ──────────────────────────────────────────────────────────
    results_csv  = output_dir / f"predictions_{timestamp}.csv"
    report_json  = output_dir / f"fairness_report_{timestamp}.json"

    results.to_csv(results_csv, index=False)
    with open(report_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    logger.info("Predictions → %s", results_csv)
    logger.info("Fairness report → %s", report_json)


if __name__ == "__main__":
    main()
