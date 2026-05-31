"""
Qwen-7B seul — baseline un seul appel, aucun signal externe.

Run A : stratifié seed=42, n=100  (référence rapide)
Run B : test-like 1/3-1/3-1/3, n=150, seed=7  (estimation honnête)

Usage:
    uv run python src/qwen_only_baseline.py
    uv run python src/qwen_only_baseline.py --run-a-only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference.zero_shot_pipeline import QwenEstimator, QWEN_MODEL_ID
from src.utils.metrics import compute_score
from src.zero_shot_eval import OCC_BINS

IMAGE_BASE = Path(
    "/home/matt/Programmation/704_IADATA_ML_avance/datachallenge"
    "/DataChallenge2026/occlusion_datasets/raw/Crop_224_5fp_100K"
)

# ── Prompt (défini par le brief) ──────────────────────────────────────────────

PROMPT = (
    "You are estimating face occlusion on a cropped 224x224 face image.\n\n"
    "Occlusion = the FRACTION OF THE FACE SURFACE that is hidden, from 0.00 "
    "(fully visible) to 1.00 (fully hidden). What counts as hidden: anything "
    "covering the face surface — a hand, mask, glasses, scarf, object, heavy blur, "
    "deep shadow, and hair where it covers part of the face area. A thin strand of "
    "hair covers little surface (small value); a thick fringe over the forehead "
    "covers real surface (moderate value). A clear, fully visible frontal face = 0.00.\n\n"
    "Look carefully and estimate what fraction of the face surface is occluded.\n\n"
    'Respond ONLY as JSON:\n'
    '{"observation": "1-2 sentences on what you see", "percentage": <float 0.00-1.00>}'
)


# ── Parsers (inline — no import from pipeline to keep this script self-contained) ──

def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    depth, start = 0, -1
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = -1
    return None


def _parse_occlusion_value(text: str) -> float | None:
    for m in re.finditer(r"\b(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\b", text):
        try:
            v = float(m.group())
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            continue
    return None


def parse_response(raw: str) -> tuple[float, str]:
    """Returns (percentage, observation)."""
    data = _extract_json(raw)
    if data is not None and "percentage" in data:
        try:
            pct = float(np.clip(float(str(data["percentage"])), 0.0, 1.0))
            obs = str(data.get("observation", ""))[:300]
            return pct, obs
        except (ValueError, TypeError):
            pass
    val = _parse_occlusion_value(raw)
    return (val if val is not None else 0.10), ""


# ── Sampling helpers ──────────────────────────────────────────────────────────

def stratified_sample_standard(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Proportional stratified sample (gender × occ_level) — same as zero_shot_eval."""
    from src.zero_shot_eval import stratified_sample
    return stratified_sample(df, n, seed=seed)


def testlike_sample(df: pd.DataFrame, n_per_bin: int, seed: int) -> pd.DataFrame:
    """Equal-weight sample across OCC_BINS (~1/3 each)."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, lo, hi in OCC_BINS:
        pool = df[(df["FaceOcclusion"] >= lo) & (df["FaceOcclusion"] < hi)]
        k = min(n_per_bin, len(pool))
        parts.append(pool.iloc[rng.choice(len(pool), k, replace=False)])
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# ── Inference loop ────────────────────────────────────────────────────────────

def run_inference(qwen: QwenEstimator, sample: pd.DataFrame) -> pd.DataFrame:
    records = []
    n = len(sample)
    t0 = time.time()
    for i, row in enumerate(sample.itertuples(index=False), 1):
        if i == 1 or i % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (n - i)
            print(f"  {i:3d}/{n}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)
        img_path = IMAGE_BASE / row.filename
        try:
            image = Image.open(img_path).convert("RGB")
            raw = qwen._run_qwen(image, PROMPT, max_new_tokens=256)
            pct, obs = parse_response(raw)
        except Exception as exc:
            print(f"  ERROR {row.filename}: {exc}")
            pct, obs = 0.10, f"error: {exc}"
        records.append({
            "filename": row.filename,
            "gt":       float(row.FaceOcclusion),
            "pred":     float(pct),
            "observation": obs,
            "gender":   int(row.gender),
        })
    return pd.DataFrame(records)


# ── Reporting ─────────────────────────────────────────────────────────────────

def report(df: pd.DataFrame, label: str) -> None:
    preds  = df["pred"].values
    gt     = df["gt"].values
    gender = df["gender"].values
    errors = np.abs(preds - gt)
    signed = preds - gt

    m = compute_score(preds, gt, gender.astype(float))

    SEP  = "=" * 70
    LINE = "─" * 70
    print(f"\n{SEP}")
    print(f"  {label}")
    print(SEP)
    print(f"  n={len(df)}")
    print(f"  challenge_score : {m['challenge_score']:.5f}  (lower = better)")
    print(f"  MAE             : {m['mae']:.5f}")
    print(f"  biais signé     : {signed.mean():+.5f}")
    print(f"  Err_F / Err_M   : {m['err_F']:.5f} / {m['err_M']:.5f}")
    print(f"  |Err_F - Err_M| : {m['err_diff']:.5f}")
    print(f"  worst >0.2      : {int((errors>0.2).sum())} / {len(df)}")

    print(f"\n  Distribution des prédictions :")
    print(f"    min={preds.min():.3f}  max={preds.max():.3f}"
          f"  mean={preds.mean():.3f}  median={np.median(preds):.3f}  std={preds.std():.3f}")

    # Flag ancrage
    mode_counts = pd.Series(np.round(preds, 1)).value_counts()
    top_mode, top_count = mode_counts.index[0], mode_counts.iloc[0]
    if top_count / len(df) > 0.3:
        print(f"    ⚠ ANCRAGE DÉTECTÉ : {top_count}/{len(df)} prédictions ≈ {top_mode:.1f}")

    print(f"\n{LINE}")
    print(f"  PAR BIN D'OCCLUSION")
    print(LINE)
    for level, lo, hi in OCC_BINS:
        mask = (gt >= lo) & (gt < hi)
        if mask.sum() == 0:
            continue
        p_b, gt_b = preds[mask], gt[mask]
        mae_b  = float(np.abs(p_b - gt_b).mean())
        w_b    = 1/30 + gt_b
        werr_b = float(np.sum(w_b * (p_b - gt_b)**2) / np.sum(w_b))
        bias_b = float((p_b - gt_b).mean())
        print(f"  {level:<8} n={mask.sum():3d}  MAE={mae_b:.4f}  wErr={werr_b:.5f}  bias={bias_b:+.5f}")

    # 5 exemples bin haut
    high_mask = gt >= 0.30
    if high_mask.sum() > 0:
        print(f"\n{LINE}")
        print(f"  5 EXEMPLES — BIN HAUT (gt ≥ 0.30)")
        print(LINE)
        df_high = df[high_mask].copy()
        df_high["abs_err"] = np.abs(df_high["pred"] - df_high["gt"])
        for _, r in df_high.sort_values("gt", ascending=False).head(5).iterrows():
            print(f"  gt={r['gt']:.4f}  pred={r['pred']:.4f}  err={r['abs_err']:.4f}")
            if r["observation"]:
                print(f"    obs: \"{r['observation'][:100]}\"")

    # Baseline constante
    print(f"\n{LINE}")
    print(f"  BASELINE CONSTANTE (repère diagnostique)")
    print(LINE)
    best_c, best_s = 0.0, 999.0
    for c in np.arange(0.0, 0.51, 0.01):
        s = compute_score(
            np.full_like(preds, c), gt, gender.astype(float)
        )["challenge_score"]
        if s < best_s:
            best_s, best_c = s, c
    print(f"  Meilleure constante : c={best_c:.2f}  score={best_s:.5f}")
    print(f"  Qwen vs constante   : {m['challenge_score']:.5f} vs {best_s:.5f}  "
          + ("✓ Qwen gagne" if m["challenge_score"] < best_s else "✗ Qwen ne bat PAS la constante"))
    print(SEP)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a-only", action="store_true")
    ap.add_argument("--run-b-only", action="store_true")
    ap.add_argument("--qwen-model", default=QWEN_MODEL_ID)
    args = ap.parse_args()

    df_full = pd.read_csv("data/train.csv").dropna(
        subset=["filename", "FaceOcclusion", "gender"]
    )
    df_full["gender"] = pd.to_numeric(df_full["gender"], errors="coerce").fillna(0.5)
    df_full = df_full[df_full["gender"].isin([0, 1])]

    print(f"Loading QwenEstimator ({args.qwen_model}) …")
    qwen = QwenEstimator(args.qwen_model, device="auto")
    print("Qwen loaded.\n")

    out_dir = Path("results/zero_shot")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.run_b_only:
        print("=" * 70)
        print("  RUN A — stratifié seed=42, n=100")
        print("=" * 70)
        sample_a = stratified_sample_standard(df_full, 100, seed=42)
        df_a = run_inference(qwen, sample_a)
        df_a.to_csv(out_dir / "qwen_only_run_a.csv", index=False)
        report(df_a, "RUN A — stratifié seed=42, n=100")

    if not args.run_a_only:
        print("\n" + "=" * 70)
        print("  RUN B — test-like 1/3-1/3-1/3, seed=7, n=150")
        print("=" * 70)
        sample_b = testlike_sample(df_full, n_per_bin=50, seed=7)
        print(f"  Distribution : "
              + "  ".join(
                  f"{lv}={((sample_b['FaceOcclusion']>=lo)&(sample_b['FaceOcclusion']<hi)).sum()}"
                  for lv, lo, hi in OCC_BINS
              ))
        df_b = run_inference(qwen, sample_b)
        df_b.to_csv(out_dir / "qwen_only_run_b.csv", index=False)
        report(df_b, "RUN B — test-like 1/3-1/3-1/3, seed=7, n=150  ← LE SCORE QUI COMPTE")


if __name__ == "__main__":
    main()
