"""
Baseline brute : pred = 1 - skin_ratio (face-parser, zero-shot).
Aucun paramètre appris. Formule fixée a priori.

Éval : test-like seed=7 n=150 (même que tous les runs précédents).

Usage:
    uv run python src/skin_ratio_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LinearRegression
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.metrics import compute_score
from src.zero_shot_eval import OCC_BINS

IMAGE_BASE = Path(
    "/home/matt/Programmation/704_IADATA_ML_avance/datachallenge"
    "/DataChallenge2026/occlusion_datasets/raw/Crop_224_5fp_100K"
)

# skin class index in jonathandinu/face-parsing (CelebAMask-HQ)
SKIN_CLS = 1
IMAGE_AREA = 224 * 224


def build_testlike_sample(seed: int = 7, n_per_bin: int = 50) -> pd.DataFrame:
    df = pd.read_csv("data/train.csv").dropna(
        subset=["filename", "FaceOcclusion", "gender"]
    )
    df["gender"] = pd.to_numeric(df["gender"], errors="coerce").fillna(0.5)
    df = df[df["gender"].isin([0, 1])]
    rng = np.random.default_rng(seed)
    parts = []
    for _, lo, hi in OCC_BINS:
        pool = df[(df["FaceOcclusion"] >= lo) & (df["FaceOcclusion"] < hi)]
        k = min(n_per_bin, len(pool))
        parts.append(pool.iloc[rng.choice(len(pool), k, replace=False)])
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    out_dir = Path("results/zero_shot")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = build_testlike_sample(seed=7, n_per_bin=50)
    print(f"Test-like sample : n={len(sample)}")
    for _, lo, hi in OCC_BINS:
        n = ((sample["FaceOcclusion"] >= lo) & (sample["FaceOcclusion"] < hi)).sum()
        print(f"  [{lo:.2f},{hi:.2f}) : n={n}")

    print("\nLoading face-parser (jonathandinu/face-parsing) …")
    processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "jonathandinu/face-parsing"
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Loaded on {device}.\n")

    records = []
    for i, row in enumerate(sample.itertuples(index=False), 1):
        if i == 1 or i % 30 == 0:
            print(f"  {i:3d}/{len(sample)}", flush=True)
        img = Image.open(IMAGE_BASE / row.filename).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        seg = F.interpolate(logits, size=(224, 224), mode="bilinear", align_corners=False)
        seg = seg.argmax(1).squeeze().cpu().numpy()

        skin_ratio = float((seg == SKIN_CLS).sum() / IMAGE_AREA)
        pred = float(np.clip(1.0 - skin_ratio, 0.0, 1.0))

        records.append({
            "filename":   row.filename,
            "gt":         float(row.FaceOcclusion),
            "pred":       pred,
            "skin_ratio": skin_ratio,
            "gender":     int(row.gender),
        })

    df_out = pd.DataFrame(records)
    csv_path = out_dir / "skin_ratio_testlike.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")

    preds  = df_out["pred"].values
    gt     = df_out["gt"].values
    gender = df_out["gender"].values
    errors = np.abs(preds - gt)
    signed = preds - gt

    m = compute_score(preds, gt, gender.astype(float))

    SEP  = "=" * 70
    LINE = "─" * 70
    print(f"\n{SEP}")
    print("  skin_ratio baseline : pred = 1 − skin_pct  (zero-shot, no params)")
    print(SEP)
    print(f"  n                  : {len(df_out)}")
    print(f"  challenge_score    : {m['challenge_score']:.5f}  (lower = better)")
    print(f"  MAE                : {m['mae']:.5f}")
    print(f"  biais signé        : {signed.mean():+.5f}")
    print(f"  Err_F / Err_M      : {m['err_F']:.5f} / {m['err_M']:.5f}")
    print(f"  |Err_F - Err_M|    : {m['err_diff']:.5f}")
    print(f"  worst >0.2         : {int((errors>0.2).sum())} / {len(df_out)}")

    print(f"\n  Distribution des prédictions :")
    print(f"    pred : min={preds.min():.3f}  max={preds.max():.3f}"
          f"  mean={preds.mean():.3f}  median={np.median(preds):.3f}  std={preds.std():.3f}")
    print(f"    GT   : min={gt.min():.3f}  max={gt.max():.3f}"
          f"  mean={gt.mean():.3f}  median={np.median(gt):.3f}  std={gt.std():.3f}")

    print(f"\n{LINE}")
    print("  PAR BIN")
    print(LINE)
    for level, lo, hi in OCC_BINS:
        mask = (gt >= lo) & (gt < hi)
        if mask.sum() == 0:
            continue
        p_b, gt_b = preds[mask], gt[mask]
        mae_b  = float(np.abs(p_b - gt_b).mean())
        w_b    = 1 / 30 + gt_b
        werr_b = float(np.sum(w_b * (p_b - gt_b) ** 2) / np.sum(w_b))
        bias_b = float((p_b - gt_b).mean())
        print(f"  {level:<8} n={mask.sum():3d}  MAE={mae_b:.4f}  wErr={werr_b:.5f}  bias={bias_b:+.5f}")

    # ── Régression descriptive pred ~ gt ─────────────────────────────────────
    print(f"\n{LINE}")
    print("  RÉGRESSION DESCRIPTIVE pred ~ gt  (mesure du biais, PAS un modèle)")
    print(LINE)
    lr = LinearRegression().fit(gt.reshape(-1, 1), preds)
    slope  = float(lr.coef_[0])
    offset = float(lr.intercept_)
    r2     = float(lr.score(gt.reshape(-1, 1), preds))
    print(f"  pred = {slope:.3f} × gt + {offset:.3f}   R²={r2:.3f}")
    if abs(slope - 1.0) < 0.15 and abs(offset) < 0.05:
        print("  → Pente ≈ 1, offset ≈ 0 : biais faible, formule bien calibrée.")
    elif abs(slope - 1.0) < 0.20:
        print(f"  → Pente ≈ 1 mais offset={offset:+.3f} : biais systématique d'offset.")
        print(f"     Cause probable : %skin normalisé par aire IMAGE, pas aire VISAGE.")
    else:
        print(f"  → Pente={slope:.2f} ≠ 1 : compression ou expansion des prédictions.")

    # ── Scatter plot pred vs gt ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#3498db" if g == 0 else "#e74c3c" for g in gender]
    ax.scatter(gt, preds, c=colors, alpha=0.55, s=22, linewidths=0)
    # Régression descriptive
    x_line = np.array([0, 1])
    ax.plot(x_line, slope * x_line + offset, "k--", lw=1.5,
            label=f"fit : pred = {slope:.2f}×gt + {offset:.2f}")
    ax.plot([0, 1], [0, 1], "gray", lw=0.8, linestyle=":", label="pred = gt (idéal)")
    ax.set_xlabel("gt (FaceOcclusion)", fontsize=12)
    ax.set_ylabel("pred = 1 − skin_ratio", fontsize=12)
    ax.set_title(f"Scatter pred vs gt — skin_ratio baseline\n"
                 f"score={m['challenge_score']:.4f}  MAE={m['mae']:.4f}  bias={signed.mean():+.4f}",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    # Blue=Female, Red=Male legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0],marker='o',color='w',markerfacecolor='#3498db',markersize=8,label='Female'),
               Line2D([0],[0],marker='o',color='w',markerfacecolor='#e74c3c',markersize=8,label='Male')]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0][:-1],
              loc="upper left", fontsize=8)
    scatter_path = out_dir / "skin_ratio_scatter.png"
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"\n  Scatter → {scatter_path}")

    # ── 5 pires cas ──────────────────────────────────────────────────────────
    print(f"\n{LINE}")
    print("  5 PIRES CAS (|pred - gt| maximum)")
    print(LINE)
    df_out["abs_err"] = errors
    df_out["signed"]  = signed
    worst = df_out.nlargest(5, "abs_err")
    for _, r in worst.iterrows():
        print(f"  gt={r['gt']:.4f}  pred={r['pred']:.4f}  err={r['abs_err']:.4f}"
              f"  skin={r['skin_ratio']:.3f}  {Path(r['filename']).name}")

    # ── Constante plancher ────────────────────────────────────────────────────
    print(f"\n{LINE}")
    best_c, best_s = 0.0, 999.0
    for c in np.arange(0.0, 0.60, 0.01):
        s = compute_score(np.full_like(preds, c), gt, gender.astype(float))["challenge_score"]
        if s < best_s:
            best_s, best_c = s, c
    print(f"  Constante plancher : c={best_c:.2f}  score={best_s:.5f}")
    print(f"  {'✓ skin_ratio bat la constante' if m['challenge_score'] < best_s else '✗ skin_ratio ne bat PAS la constante'}"
          f"  ({m['challenge_score']:.5f} vs {best_s:.5f})")
    print(SEP)


if __name__ == "__main__":
    main()
