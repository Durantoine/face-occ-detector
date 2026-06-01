"""
Qwen-7B few-shot ICL — 2 variantes comparées sur distribution test-like.

Exemples ICL : 5 images couvrant le spectre d'occlusion, sélectionnées par
médiane de chaque tranche, fixées a priori (aucune optimisation sur le score).
Légalité : "petit sous-ensemble NON optimisé" — les images sont choisies pour
couvrir le spectre, pas pour maximiser le score de validation.

Variante CONTINU : exemples avec leur valeur numérique réelle (2 décimales).
Variante ORDINAL : exemples avec une classe (none/light/moderate/heavy/severe).

Évaluation : test-like 1/3-1/3-1/3 seed=7 n=150 (même que Run B zéro-shot),
images ICL EXCLUES du set d'éval.

Usage:
    uv run python src/qwen_fewshot_baseline.py
    uv run python src/qwen_fewshot_baseline.py --variant continu
    uv run python src/qwen_fewshot_baseline.py --variant ordinal
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
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference.zero_shot_pipeline import QwenEstimator, QWEN_MODEL_ID
from src.utils.metrics import compute_score
from src.zero_shot_eval import OCC_BINS

IMAGE_BASE = Path(
    "/home/matt/Programmation/704_IADATA_ML_avance/datachallenge"
    "/DataChallenge2026/occlusion_datasets/raw/Crop_224_5fp_100K"
)

# ── Exemples ICL ──────────────────────────────────────────────────────────────
# Sélectionnés par médiane de tranche, seed=7 eval exclus, aucune optimisation.
# Tranche severe ajustée à [0.65, 1.01) car le train a 0 image dans [0.80, 0.90).

ICL_EXAMPLES = [
    {"class": "none",     "gt": 0.0180, "pct": 0.02,
     "filename": "database3/database3/m.01ypf3/118-FaceId-0_align.webp"},
    {"class": "light",    "gt": 0.1388, "pct": 0.14,
     "filename": "database3/database3/m.0280j8s/9-FaceId-0_align.webp"},
    {"class": "moderate", "gt": 0.3312, "pct": 0.33,
     "filename": "database3/database3/m.01gvx4/77-FaceId-0_align.webp"},
    {"class": "heavy",    "gt": 0.5304, "pct": 0.53,
     "filename": "database3/database3/m.0256sx/61-FaceId-0_align.webp"},
    {"class": "severe",   "gt": 0.7823, "pct": 0.78,
     "filename": "database3/database3/m.01c56w/50-FaceId-54_align.webp"},
]

ORDINAL_MAP = {"none": 0.02, "light": 0.12, "moderate": 0.32,
               "heavy": 0.57, "severe": 0.85}

ORDINAL_CLASSES = list(ORDINAL_MAP.keys())

# ── Prompts ───────────────────────────────────────────────────────────────────

OCCLUSION_DEF = (
    "Occlusion = the FRACTION OF THE FACE SURFACE that is hidden, "
    "from 0.00 (fully visible) to 1.00 (fully hidden). "
    "What counts as hidden: hand, mask, glasses, scarf, heavy blur, deep shadow, "
    "and hair where it covers part of the face area. "
    "A thin strand of hair = small value; thick fringe = moderate value; "
    "fully visible frontal face = 0.00."
)

PROMPT_CONTINU_FINAL = (
    "Estimate the occluded surface fraction of the LAST face image, "
    "using the examples above as a calibrated scale. "
    "Respond ONLY as JSON: "
    '{"observation": "1-2 sentences", "percentage": <float 0.00-1.00>}'
)

PROMPT_ORDINAL_FINAL = (
    "Classify the occlusion of the LAST face image into exactly one of: "
    "none, light, moderate, heavy, severe — using the examples above as reference. "
    "Respond ONLY as JSON: "
    '{"observation": "1-2 sentences", "class": "<one of the five classes>"}'
)


# ── Multi-image Qwen call ─────────────────────────────────────────────────────

def run_qwen_fewshot(
    qwen: QwenEstimator,
    example_images: list[Image.Image],
    example_texts: list[str],
    target_image: Image.Image,
    final_prompt: str,
    max_new_tokens: int = 128,
) -> str:
    """Single Qwen call with N example (image, text) pairs + 1 target image."""
    content = []
    for img_txt in example_texts:
        content.append({"type": "image"})
        content.append({"type": "text", "text": img_txt})
    content.append({"type": "image"})
    content.append({"type": "text", "text": final_prompt})

    messages = [{"role": "user", "content": content}]
    text = qwen.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    all_images = example_images + [target_image]
    inputs = qwen.processor(
        text=[text], images=all_images, padding=True, return_tensors="pt"
    ).to(qwen._device)
    with torch.no_grad():
        gen = qwen.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    return qwen.processor.decode(
        gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


# ── Parsers ───────────────────────────────────────────────────────────────────

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


def _parse_float(text: str) -> float | None:
    for m in re.finditer(r"\b(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\b", text):
        try:
            v = float(m.group())
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            continue
    return None


def parse_continu(raw: str) -> tuple[float, str]:
    data = _extract_json(raw)
    if data and "percentage" in data:
        try:
            pct = float(np.clip(float(str(data["percentage"])), 0.0, 1.0))
            return pct, str(data.get("observation", ""))[:200]
        except (ValueError, TypeError):
            pass
    val = _parse_float(raw)
    return (val if val is not None else 0.10), ""


def parse_ordinal(raw: str) -> tuple[float, str]:
    data = _extract_json(raw)
    if data and "class" in data:
        cls = str(data["class"]).strip().lower()
        if cls in ORDINAL_MAP:
            return ORDINAL_MAP[cls], str(data.get("observation", ""))[:200]
        # fuzzy match
        for k in ORDINAL_MAP:
            if k in cls:
                return ORDINAL_MAP[k], str(data.get("observation", ""))[:200]
    # fallback: scan raw text for class name
    for cls in reversed(ORDINAL_CLASSES):  # longest match wins
        if cls in raw.lower():
            return ORDINAL_MAP[cls], ""
    return 0.32, ""  # moderate as default


# ── Sampling ──────────────────────────────────────────────────────────────────

def build_eval_set(df: pd.DataFrame, exclude_filenames: set[str]) -> pd.DataFrame:
    """Reconstruct the test-like seed=7 sample, excluding ICL examples."""
    rng = np.random.default_rng(7)
    parts = []
    for _, lo, hi in OCC_BINS:
        pool = df[(df["FaceOcclusion"] >= lo) & (df["FaceOcclusion"] < hi)]
        k = min(50, len(pool))
        parts.append(pool.iloc[rng.choice(len(pool), k, replace=False)])
    sample = pd.concat(parts).sample(frac=1, random_state=7).reset_index(drop=True)
    # Remove any ICL example that might be in the sample (safety check)
    before = len(sample)
    sample = sample[~sample["filename"].isin(exclude_filenames)].reset_index(drop=True)
    if len(sample) < before:
        print(f"  [Safety] Removed {before-len(sample)} ICL image(s) from eval set.")
    return sample


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
    std_gt = df["gt"].std()
    print(f"    pred : min={preds.min():.3f}  max={preds.max():.3f}"
          f"  mean={preds.mean():.3f}  median={np.median(preds):.3f}  std={preds.std():.3f}")
    print(f"    GT   : min={gt.min():.3f}  max={gt.max():.3f}"
          f"  mean={gt.mean():.3f}  median={np.median(gt):.3f}  std={std_gt:.3f}")
    ratio = preds.std() / max(std_gt, 1e-6)
    flag = "✓ proche" if 0.6 < ratio < 1.4 else "⚠ compression" if ratio < 0.6 else "⚠ expansion"
    print(f"    std(pred)/std(GT) = {ratio:.2f}  {flag}")

    print(f"\n{LINE}")
    print(f"  PAR BIN")
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
            obs = r.get("observation", "")
            if obs:
                print(f"    obs: \"{obs[:90]}\"")

    # Constante plancher
    best_c, best_s = 0.0, 999.0
    for c in np.arange(0.0, 0.51, 0.01):
        s = compute_score(np.full_like(preds, c), gt, gender.astype(float))["challenge_score"]
        if s < best_s:
            best_s, best_c = s, c
    print(f"\n{LINE}")
    print(f"  Constante plancher : c={best_c:.2f}  score={best_s:.5f}")
    print(f"  {'✓ Qwen bat la constante' if m['challenge_score'] < best_s else '✗ Qwen ne bat PAS la constante'}"
          f"  ({m['challenge_score']:.5f} vs {best_s:.5f})")
    print(SEP)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["continu", "ordinal", "both"], default="both")
    ap.add_argument("--qwen-model", default=QWEN_MODEL_ID)
    args = ap.parse_args()

    # Load data
    df_full = pd.read_csv("data/train.csv").dropna(
        subset=["filename", "FaceOcclusion", "gender"]
    )
    df_full["gender"] = pd.to_numeric(df_full["gender"], errors="coerce").fillna(0.5)
    df_full = df_full[df_full["gender"].isin([0, 1])]

    icl_filenames = {e["filename"] for e in ICL_EXAMPLES}
    eval_df = build_eval_set(df_full, icl_filenames)
    print(f"Eval set : n={len(eval_df)}")

    # Load ICL example images
    print("Loading ICL example images …")
    ex_images = [
        Image.open(IMAGE_BASE / e["filename"]).convert("RGB")
        for e in ICL_EXAMPLES
    ]
    print(f"  {len(ex_images)} images loaded.")

    # Load Qwen
    print(f"Loading QwenEstimator ({args.qwen_model}) …")
    qwen = QwenEstimator(args.qwen_model, device="auto")
    print("Qwen loaded.\n")

    out_dir = Path("results/zero_shot")
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_variant(variant: str) -> None:
        if variant == "continu":
            ex_texts = [
                f"[Occlusion scale example — {OCCLUSION_DEF}]\n"
                f"This face: {{\"percentage\": {e['pct']:.2f}}}"
                for e in ICL_EXAMPLES
            ]
            final_prompt = PROMPT_CONTINU_FINAL
            parse_fn = parse_continu
        else:  # ordinal
            ex_texts = [
                f"[Occlusion scale example — {OCCLUSION_DEF}]\n"
                f"This face: {{\"class\": \"{e['class']}\"}}"
                for e in ICL_EXAMPLES
            ]
            final_prompt = PROMPT_ORDINAL_FINAL
            parse_fn = parse_ordinal

        print("=" * 70)
        print(f"  VARIANTE {variant.upper()} — test-like 1/3-1/3-1/3 seed=7 n={len(eval_df)}")
        print("=" * 70)

        records = []
        n = len(eval_df)
        t0 = time.time()
        for i, row in enumerate(eval_df.itertuples(index=False), 1):
            if i == 1 or i % 25 == 0:
                elapsed = time.time() - t0
                eta = elapsed / i * (n - i)
                print(f"  {i:3d}/{n}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)
            try:
                target = Image.open(IMAGE_BASE / row.filename).convert("RGB")
                raw = run_qwen_fewshot(
                    qwen, ex_images, ex_texts, target, final_prompt,
                    max_new_tokens=128,
                )
                pct, obs = parse_fn(raw)
            except Exception as exc:
                print(f"  ERROR {row.filename}: {exc}")
                pct, obs = 0.32, f"error: {exc}"
            records.append({
                "filename":    row.filename,
                "gt":          float(row.FaceOcclusion),
                "pred":        float(pct),
                "observation": obs,
                "gender":      int(row.gender),
            })

        df_out = pd.DataFrame(records)
        csv_path = out_dir / f"qwen_fewshot_{variant}.csv"
        df_out.to_csv(csv_path, index=False)
        print(f"  Saved → {csv_path}")
        report(df_out, f"Qwen-7B few-shot {variant.upper()} — test-like seed=7")

    variants = ["continu", "ordinal"] if args.variant == "both" else [args.variant]
    for v in variants:
        run_variant(v)

    # Tableau comparatif
    print("\n" + "=" * 70)
    print("  TABLEAU COMPARATIF — test-like seed=7")
    print("=" * 70)
    print(f"  {'méthode':<30} {'score':>7}  {'biais_low':>10}  {'biais_high':>11}  {'std_pred':>9}")
    rows = [
        ("constante c=0.32",       0.030,  None,   None,   None),
        ("Qwen-7B zéro-shot",      0.065, +0.190, -0.027,  0.198),
    ]
    for v in variants:
        csv_path = out_dir / f"qwen_fewshot_{v}.csv"
        if csv_path.exists():
            df_v = pd.read_csv(csv_path)
            preds, gt, gender = df_v["pred"].values, df_v["gt"].values, df_v["gender"].values
            sc = compute_score(preds, gt, gender.astype(float))["challenge_score"]
            low_mask  = gt < 0.10
            high_mask = gt >= 0.30
            bias_low  = float((preds[low_mask] - gt[low_mask]).mean()) if low_mask.sum() else float("nan")
            bias_high = float((preds[high_mask] - gt[high_mask]).mean()) if high_mask.sum() else float("nan")
            rows.append((f"Qwen-7B few-shot {v}", sc, bias_low, bias_high, float(preds.std())))

    for name, sc, bl, bh, std in rows:
        bl_str  = f"{bl:+.3f}" if bl is not None else "  —  "
        bh_str  = f"{bh:+.3f}" if bh is not None else "  —  "
        std_str = f"{std:.3f}" if std is not None else "  —  "
        flag = " ✓ < constante" if sc < 0.030 else ""
        print(f"  {name:<30} {sc:>7.5f}  {bl_str:>10}  {bh_str:>11}  {std_str:>9}{flag}")
    print("=" * 70)


if __name__ == "__main__":
    main()
