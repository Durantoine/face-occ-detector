"""
ÉTAPE 1 — Routeur Qwen : classification de la cause d'occlusion.

Qwen sort UNE catégorie (object/profile/hair/none) + blur (bool).
Aucun chiffre d'occlusion. Validation par proxies objectifs.

Usage:
    uv run python src/qwen_router.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import urllib.request
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference.zero_shot_pipeline import QwenEstimator, QWEN_MODEL_ID
from src.zero_shot_eval import OCC_BINS

IMAGE_BASE = Path(
    "/home/matt/Programmation/704_IADATA_ML_avance/datachallenge"
    "/DataChallenge2026/occlusion_datasets/raw/Crop_224_5fp_100K"
)

# ── Routing prompt ─────────────────────────────────────────────────────────────

ROUTING_PROMPT = (
    "Look at this face. What is the MAIN thing reducing visibility of the face skin, if any?\n"
    "Choose ONE: 'object' (hand, mask, glasses, hat, scarf, mic), "
    "'profile' (face turned away), "
    "'hair' (hair covering the face), "
    "'none' (face mostly clear).\n"
    "Also: is the image notably blurry? yes/no.\n"
    "Respond ONLY as JSON: "
    '{\"cause\":\"object|profile|hair|none\",\"blur\":true|false}'
)

VALID_CAUSES = {"object", "profile", "hair", "none"}

# ── Parser / MediaPipe setup ──────────────────────────────────────────────────

HAIR_CLS = 13
_MP_URL   = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task")
_MP_CACHE = Path.home() / ".cache" / "mediapipe" / "face_landmarker.task"


def _load_mediapipe():
    if not _MP_CACHE.exists():
        _MP_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MP_URL, _MP_CACHE)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_MP_CACHE)),
        num_faces=1, min_face_detection_confidence=0.3, min_face_presence_confidence=0.3,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def get_yaw(img_rgb: np.ndarray, lmk) -> float | None:
    H, W = img_rgb.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = lmk.detect(mp_img)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    left_x, right_x, nose_x = lm[33].x, lm[263].x, lm[1].x
    eye_cx = (left_x + right_x) / 2
    face_w = max(abs(right_x - left_x), 0.01)
    return float(np.clip((nose_x - eye_cx) / face_w * 90.0, -90, 90))


# ── Parsers ────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    depth, start = 0, -1
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0: start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try: return json.loads(text[start:i+1])
                except json.JSONDecodeError: start = -1
    return None


def parse_route(raw: str) -> tuple[str, bool]:
    data = _extract_json(raw)
    if data:
        cause = str(data.get("cause", "none")).lower().strip()
        if cause not in VALID_CAUSES:
            # fuzzy fallback
            for v in VALID_CAUSES:
                if v in cause:
                    cause = v; break
            else:
                cause = "none"
        blur = bool(data.get("blur", False))
        return cause, blur
    # last-resort text scan
    text = raw.lower()
    for c in ("object", "profile", "hair", "none"):
        if c in text:
            return c, "blur" in text and "true" in text
    return "none", False


# ── Sample builder ────────────────────────────────────────────────────────────

def build_testlike(seed=7, n_per_bin=50) -> pd.DataFrame:
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = Path("results/zero_shot")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = build_testlike()
    print(f"Test-like sample : n={len(sample)}")

    # Load models
    print("Loading Qwen …")
    qwen = QwenEstimator(QWEN_MODEL_ID, device="auto")
    print("Loading face-parser …")
    proc = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    fpm  = SegformerForSemanticSegmentation.from_pretrained(
        "jonathandinu/face-parsing"
    ).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fpm.to(dev)
    print("Loading MediaPipe …")
    lmk = _load_mediapipe()
    print("All models loaded.\n")

    records = []
    n = len(sample)
    t0 = time.time()
    for i, row in enumerate(sample.itertuples(index=False), 1):
        if i == 1 or i % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (n - i)
            print(f"  {i:3d}/{n}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

        img = Image.open(IMAGE_BASE / row.filename).convert("RGB")
        img_rgb = np.array(img)
        gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Qwen routing
        raw   = qwen._run_qwen(img, ROUTING_PROMPT, max_new_tokens=64)
        cause, blur_pred = parse_route(raw)

        # Proxy 1 : yaw (MediaPipe)
        yaw = get_yaw(img_rgb, lmk)

        # Proxy 2 : Laplacian (sharpness)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Proxy 3 : %hair (face-parser)
        inputs = proc(images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = fpm(**inputs).logits
        seg = F.interpolate(logits, size=(224,224), mode="bilinear",
                            align_corners=False).argmax(1).squeeze().cpu().numpy()
        pct_hair = float((seg == HAIR_CLS).sum() / (224 * 224))

        records.append({
            "filename":   row.filename,
            "gt":         float(row.FaceOcclusion),
            "gender":     int(row.gender),
            "cause":      cause,
            "blur_pred":  blur_pred,
            "yaw":        yaw if yaw is not None else float("nan"),
            "lap_var":    lap_var,
            "pct_hair":   pct_hair,
            "raw_route":  raw[:120],
        })

    df = pd.DataFrame(records)
    csv = out_dir / "qwen_router_results.csv"
    df.to_csv(csv, index=False)
    print(f"\nSaved → {csv}")

    # ── Distribution des causes ───────────────────────────────────────────────
    SEP  = "=" * 68
    LINE = "─" * 68
    print(f"\n{SEP}")
    print("  DISTRIBUTION DES CAUSES")
    print(SEP)
    cause_counts = df["cause"].value_counts()
    for c, n_c in cause_counts.items():
        print(f"  {c:<10} : {n_c:3d}  ({100*n_c/len(df):.1f}%)")
    print(f"  blur=True  : {df['blur_pred'].sum():3d}  ({100*df['blur_pred'].mean():.1f}%)")

    # ── Matrice cause → gt stats ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  MATRICE cause → GT  (la clé : 'none' doit avoir le gt le plus bas)")
    print(SEP)
    print(f"  {'cause':<10}  {'n':>4}  {'gt_mean':>8}  {'gt_median':>10}  {'gt_min':>7}  {'gt_max':>7}")
    for c in ["none","hair","profile","object"]:
        mask = df["cause"] == c
        if mask.sum() == 0:
            print(f"  {c:<10}  {'0':>4}  —")
            continue
        sub = df[mask]["gt"]
        print(f"  {c:<10}  {mask.sum():>4}  {sub.mean():>8.4f}  {sub.median():>10.4f}"
              f"  {sub.min():>7.4f}  {sub.max():>7.4f}")

    # ── Taux d'accord cause ↔ proxies ────────────────────────────────────────
    print(f"\n{SEP}")
    print("  TAUX D'ACCORD cause ↔ PROXIES OBJECTIFS")
    print(SEP)

    # profile ↔ |yaw| > 25°
    df_yaw = df.dropna(subset=["yaw"])
    profile_pred = df_yaw["cause"] == "profile"
    proxy_profile = df_yaw["yaw"].abs() > 25
    agree_profile = int((profile_pred == proxy_profile).sum())
    n_profile = len(df_yaw)
    print(f"  profile↔|yaw|>25°  : accord {agree_profile}/{n_profile} ({100*agree_profile/n_profile:.1f}%)")
    print(f"    Qwen='profile' n={profile_pred.sum()}  |yaw|>25 n={proxy_profile.sum()}")
    # breakdown
    tp = int((profile_pred & proxy_profile).sum())
    fp = int((profile_pred & ~proxy_profile).sum())
    fn = int((~profile_pred & proxy_profile).sum())
    tn = int((~profile_pred & ~proxy_profile).sum())
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # blur ↔ lap_var < 100
    BLUR_THRESH = 100.0
    proxy_blur = df["lap_var"] < BLUR_THRESH
    agree_blur = int((df["blur_pred"] == proxy_blur).sum())
    print(f"\n  blur↔lap<{BLUR_THRESH:.0f}      : accord {agree_blur}/{len(df)} ({100*agree_blur/len(df):.1f}%)")
    print(f"    Qwen=blur n={df['blur_pred'].sum()}  lap<{BLUR_THRESH:.0f} n={proxy_blur.sum()}")
    tp_b = int((df["blur_pred"] & proxy_blur).sum())
    fp_b = int((df["blur_pred"] & ~proxy_blur).sum())
    fn_b = int((~df["blur_pred"] & proxy_blur).sum())
    print(f"    TP={tp_b}  FP={fp_b}  FN={fn_b}")

    # hair ↔ %hair > 0.25
    HAIR_THRESH = 0.25
    proxy_hair = df["pct_hair"] > HAIR_THRESH
    hair_pred  = df["cause"] == "hair"
    agree_hair = int((hair_pred == proxy_hair).sum())
    print(f"\n  hair↔%hair>{HAIR_THRESH}    : accord {agree_hair}/{len(df)} ({100*agree_hair/len(df):.1f}%)")
    print(f"    Qwen='hair' n={hair_pred.sum()}  %hair>{HAIR_THRESH} n={proxy_hair.sum()}")
    tp_h = int((hair_pred & proxy_hair).sum())
    fp_h = int((hair_pred & ~proxy_hair).sum())
    fn_h = int((~hair_pred & proxy_hair).sum())
    print(f"    TP={tp_h}  FP={fp_h}  FN={fn_h}")

    # Corrélation %hair avec cause=hair (distributional)
    print(f"\n  %hair moyen par cause :")
    for c in ["none","hair","profile","object"]:
        m = df[df["cause"]==c]["pct_hair"]
        if len(m)==0: continue
        print(f"    {c:<10} : %hair_mean={m.mean():.3f}  median={m.median():.3f}")

    # ── Crops pour inspection manuelle ───────────────────────────────────────
    for cat_label, n_show in [("object", 5), ("none", 5)]:
        sub = df[df["cause"] == cat_label].head(n_show)
        if len(sub) == 0:
            continue
        n_cols = min(n_show, len(sub))
        fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 5))
        if n_cols == 1: axes = [axes]
        for ax, (_, row) in zip(axes, sub.iterrows()):
            img = Image.open(IMAGE_BASE / row["filename"]).convert("RGB")
            ax.imshow(img)
            ax.set_title(f"gt={row['gt']:.3f}\n{Path(row['filename']).name[:20]}"
                         f"\nyaw={row['yaw']:+.0f}° hair={row['pct_hair']:.2f}",
                         fontsize=8)
            ax.axis("off")
        fig.suptitle(f"5 crops cause='{cat_label}'", fontsize=12)
        plt.tight_layout()
        out = out_dir / f"router_crops_{cat_label}.png"
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"\n  Crops '{cat_label}' → {out}")

    print(f"\n{SEP}")
    print("  VERDICT ROUTEUR")
    print(SEP)
    gt_none   = df[df["cause"]=="none"]["gt"].mean()   if (df["cause"]=="none").sum()   else float("nan")
    gt_others = df[df["cause"]!="none"]["gt"].mean()   if (df["cause"]!="none").sum()   else float("nan")
    if gt_none < gt_others - 0.05:
        print(f"  ✓ 'none' gt_mean={gt_none:.4f} < autres={gt_others:.4f} → routeur sépare les causes")
        print(f"    → Brancher les outils sur les branches non-none.")
    else:
        print(f"  ✗ 'none' gt_mean={gt_none:.4f} ≥ autres={gt_others:.4f} → routeur ne sépare PAS")
        print(f"    → À corriger avant de brancher les outils.")
    print(SEP)


if __name__ == "__main__":
    main()
