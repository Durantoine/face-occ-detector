"""
Baseline : occlusion = 1 − skin_visible / aire_visage, 3 définitions de l'aire.

D1 = face_bbox MediaPipe         → dénominateur géométrique (bbox)
D2 = enveloppe convexe MediaPipe → dénominateur géométrique (hull)
D3 = (skin+hair+hat) in face_bbox → dénominateur sémantique stable

Aucun coefficient appris sur le gt. Formules fixées a priori.
Éval : test-like seed=7 n=150.

Usage:
    uv run python src/face_area_baseline.py
"""
from __future__ import annotations

import sys
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
from sklearn.linear_model import LinearRegression
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.metrics import compute_score
from src.zero_shot_eval import OCC_BINS

IMAGE_BASE = Path(
    "/home/matt/Programmation/704_IADATA_ML_avance/datachallenge"
    "/DataChallenge2026/occlusion_datasets/raw/Crop_224_5fp_100K"
)
IMAGE_AREA = 224 * 224

# Parser class indices (jonathandinu/face-parsing, CelebAMask-HQ)
SKIN_CLS = 1
HAIR_CLS = 13
HAT_CLS  = 14

# Face classes for D3 denominator (skin+hair+hat = "visage entier y compris caché")
FACE_RELATED = {SKIN_CLS, HAIR_CLS, HAT_CLS}

# Fallback zone when MediaPipe fails : central 80% crop [22:202, 22:202]
FALLBACK_Y1, FALLBACK_Y2 = 22, 202
FALLBACK_X1, FALLBACK_X2 = 22, 202
FALLBACK_AREA = (FALLBACK_Y2 - FALLBACK_Y1) * (FALLBACK_X2 - FALLBACK_X1)

# MediaPipe landmarker
_MP_URL   = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task")
_MP_CACHE = Path.home() / ".cache" / "mediapipe" / "face_landmarker.task"
_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is not None:
        return _landmarker
    if not _MP_CACHE.exists():
        _MP_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MP_URL, _MP_CACHE)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_MP_CACHE)),
        num_faces=1, min_face_detection_confidence=0.3, min_face_presence_confidence=0.3,
    )
    _landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


def get_landmark_pixels(img_rgb: np.ndarray) -> np.ndarray | None:
    """Return (N, 2) array of [x_px, y_px] landmark positions, or None if not detected."""
    H, W = img_rgb.shape[:2]
    lmk = _get_landmarker()
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = lmk.detect(mp_img)
    if not result.face_landmarks:
        return None
    pts = np.array([[lm.x * W, lm.y * H] for lm in result.face_landmarks[0]], dtype=np.float32)
    return pts


def compute_d1_d2(seg_map: np.ndarray, pts: np.ndarray | None) -> tuple[float, float]:
    """pred_D1 (bbox) and pred_D2 (convex hull) from landmark points."""
    H, W = seg_map.shape
    skin = seg_map == SKIN_CLS

    if pts is None:
        # MediaPipe failed → fallback: central crop as fixed area
        fb_mask = np.zeros((H, W), dtype=bool)
        fb_mask[FALLBACK_Y1:FALLBACK_Y2, FALLBACK_X1:FALLBACK_X2] = True
        s = skin[fb_mask].sum()
        pred = float(np.clip(1.0 - s / FALLBACK_AREA, 0.0, 1.0))
        return pred, pred  # same fallback for D1 and D2

    # D1 — bounding box
    x1, y1 = np.clip(pts.min(axis=0).astype(int), 0, [W-1, H-1])
    x2, y2 = np.clip(pts.max(axis=0).astype(int), 0, [W-1, H-1])
    bbox_area = max(1, int((y2 - y1 + 1) * (x2 - x1 + 1)))
    bbox_mask = np.zeros((H, W), dtype=bool)
    bbox_mask[y1:y2+1, x1:x2+1] = True
    s_d1 = int(skin[bbox_mask].sum())
    pred_d1 = float(np.clip(1.0 - s_d1 / bbox_area, 0.0, 1.0))

    # D2 — convex hull
    hull_pts = cv2.convexHull(pts.astype(np.int32))
    hull_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull_pts, 1)
    hull_area = max(1, int(hull_mask.sum()))
    s_d2 = int(skin[hull_mask.astype(bool)].sum())
    pred_d2 = float(np.clip(1.0 - s_d2 / hull_area, 0.0, 1.0))

    return pred_d1, pred_d2


def compute_d3(seg_map: np.ndarray, pts: np.ndarray | None) -> float:
    """pred_D3 : skin / (skin+hair+hat) in face zone.

    Dénominateur = pixels classés comme face-related (skin+hair+hat) dans la zone faciale.
    Ne rétrécit pas sous occlusion : hair/hat inclus même quand ils couvrent le visage.
    Zone faciale = face_bbox MediaPipe si disponible, sinon central crop [22:202].
    """
    H, W = seg_map.shape
    face_related = np.isin(seg_map, list(FACE_RELATED))  # skin | hair | hat
    skin = seg_map == SKIN_CLS

    if pts is not None:
        x1, y1 = np.clip(pts.min(axis=0).astype(int), 0, [W-1, H-1])
        x2, y2 = np.clip(pts.max(axis=0).astype(int), 0, [W-1, H-1])
        zone = np.zeros((H, W), dtype=bool)
        zone[y1:y2+1, x1:x2+1] = True
    else:
        zone = np.zeros((H, W), dtype=bool)
        zone[FALLBACK_Y1:FALLBACK_Y2, FALLBACK_X1:FALLBACK_X2] = True

    denom = int((face_related & zone).sum())
    numer = int((skin & zone).sum())
    if denom < 100:  # less than ~0.2% of image → parser failed badly
        denom = FALLBACK_AREA
        numer = int(skin[FALLBACK_Y1:FALLBACK_Y2, FALLBACK_X1:FALLBACK_X2].sum())
    return float(np.clip(1.0 - numer / denom, 0.0, 1.0))


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


def report_and_scatter(df: pd.DataFrame, pred_col: str, label: str, out_path: Path) -> dict:
    preds  = df[pred_col].values
    gt     = df["gt"].values
    gender = df["gender"].values
    errors = np.abs(preds - gt)
    signed = preds - gt

    m = compute_score(preds, gt, gender.astype(float))

    SEP  = "=" * 68
    LINE = "─" * 68
    print(f"\n{SEP}")
    print(f"  {label}")
    print(SEP)
    print(f"  challenge_score : {m['challenge_score']:.5f}")
    print(f"  MAE             : {m['mae']:.5f}")
    print(f"  biais signé     : {signed.mean():+.5f}")
    print(f"  worst >0.2      : {int((errors>0.2).sum())} / {len(df)}")
    print(f"  Distribution pred : min={preds.min():.3f}  max={preds.max():.3f}"
          f"  mean={preds.mean():.3f}  median={np.median(preds):.3f}  std={preds.std():.3f}")

    print(f"\n{LINE}")
    print(f"  {'BIN':<8} {'n':>4}  {'MAE':>7}  {'wErr':>7}  {'biais':>8}")
    for level, lo, hi in OCC_BINS:
        mask = (gt >= lo) & (gt < hi)
        if mask.sum() == 0: continue
        p_b, gt_b = preds[mask], gt[mask]
        mae_b  = float(np.abs(p_b - gt_b).mean())
        w_b    = 1/30 + gt_b
        werr_b = float(np.sum(w_b * (p_b - gt_b)**2) / np.sum(w_b))
        bias_b = float((p_b - gt_b).mean())
        print(f"  {level:<8} {mask.sum():>4}  {mae_b:>7.4f}  {werr_b:>7.5f}  {bias_b:>+8.5f}")

    # Régression descriptive
    lr     = LinearRegression().fit(gt.reshape(-1, 1), preds)
    slope  = float(lr.coef_[0])
    offset = float(lr.intercept_)
    r2     = float(lr.score(gt.reshape(-1, 1), preds))
    print(f"\n  Régression descriptive : pred = {slope:.3f}·gt + {offset:.3f}   R²={r2:.3f}")

    # 5 pires
    print(f"\n  5 pires cas :")
    df2 = df.copy(); df2["_err"] = errors; df2["_signed"] = signed
    for _, r in df2.nlargest(5, "_err").iterrows():
        print(f"    gt={r['gt']:.4f}  pred={r[pred_col]:.4f}  err={r['_err']:.4f}"
              f"  mp={'✓' if r['mp_ok'] else '✗'}  {Path(r['filename']).name}")

    # Scatter
    fig, ax = plt.subplots(figsize=(6, 5.5))
    colors = ["#3498db" if g == 0 else "#e74c3c" for g in gender]
    ax.scatter(gt, preds, c=colors, alpha=0.5, s=20)
    x_line = np.array([0, 1])
    ax.plot(x_line, slope*x_line+offset, "k--", lw=1.5,
            label=f"pred={slope:.2f}·gt+{offset:.2f}  R²={r2:.2f}")
    ax.plot([0,1],[0,1],"gray",lw=0.8,linestyle=":")
    ax.set_xlabel("gt"); ax.set_ylabel(f"pred ({pred_col})")
    ax.set_title(f"{label}\nscore={m['challenge_score']:.4f}  bias={signed.mean():+.4f}", fontsize=10)
    ax.legend(fontsize=8); ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()
    print(f"  Scatter → {out_path}")

    return {"label": label, "score": m["challenge_score"],
            "slope": slope, "offset": offset, "r2": r2,
            "bias_low":  float((preds[(gt<0.1)]      - gt[gt<0.1]).mean()      if (gt<0.1).sum()      else 0),
            "bias_high": float((preds[(gt>=0.3)]     - gt[gt>=0.3]).mean()     if (gt>=0.3).sum()     else 0),
            "pred_std": float(preds.std()), "floor": float(preds[gt<0.05].mean())}


def main() -> None:
    out_dir = Path("results/zero_shot")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = build_testlike_sample(seed=7, n_per_bin=50)
    print(f"Test-like sample : n={len(sample)}")

    # Load face-parser
    print("Loading face-parser …")
    proc = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    fpm  = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fpm.to(device)
    print(f"Face-parser on {device}. Loading MediaPipe …")
    _get_landmarker()
    print("MediaPipe loaded.\n")

    records = []
    n = len(sample)
    for i, row in enumerate(sample.itertuples(index=False), 1):
        if i == 1 or i % 30 == 0:
            print(f"  {i:3d}/{n}", flush=True)
        img     = Image.open(IMAGE_BASE / row.filename).convert("RGB")
        img_rgb = np.array(img)

        # Face-parser
        inputs  = proc(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = fpm(**inputs).logits
        seg = F.interpolate(logits, size=(224,224), mode="bilinear", align_corners=False)
        seg = seg.argmax(1).squeeze().cpu().numpy()

        # MediaPipe landmarks
        pts    = get_landmark_pixels(img_rgb)
        mp_ok  = pts is not None

        pred_d1, pred_d2 = compute_d1_d2(seg, pts)
        pred_d3          = compute_d3(seg, pts)

        records.append({
            "filename": row.filename,
            "gt":       float(row.FaceOcclusion),
            "gender":   int(row.gender),
            "mp_ok":    mp_ok,
            "pred_d1":  pred_d1,
            "pred_d2":  pred_d2,
            "pred_d3":  pred_d3,
        })

    df = pd.DataFrame(records)
    csv_path = out_dir / "face_area_testlike.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")
    print(f"MediaPipe détecté : {df['mp_ok'].sum()} / {len(df)}")

    # Per-definition reports
    summaries = []
    for pred_col, label, fname in [
        ("pred_d1", "D1 — face_bbox MediaPipe",          "face_area_scatter_d1.png"),
        ("pred_d2", "D2 — convex hull MediaPipe",        "face_area_scatter_d2.png"),
        ("pred_d3", "D3 — (skin+hair+hat)/face_bbox",    "face_area_scatter_d3.png"),
    ]:
        s = report_and_scatter(df, pred_col, label, out_dir / fname)
        summaries.append(s)

    # Constante plancher
    gt_arr = df["gt"].values; gender_arr = df["gender"].values.astype(float)
    best_c, best_s = 0.0, 999.0
    for c in np.arange(0.0, 0.60, 0.01):
        sc = compute_score(np.full(len(gt_arr), c), gt_arr, gender_arr)["challenge_score"]
        if sc < best_s: best_s, best_c = sc, c

    # Tableau comparatif
    SEP = "=" * 82
    print(f"\n{SEP}")
    print("  TABLEAU COMPARATIF — test-like seed=7")
    print(SEP)
    print(f"  {'méthode':<30} {'score':>7}  {'pente':>6}  {'offset':>7}  "
          f"{'floor(gt<0.05)':>14}  {'bias_low':>9}  {'bias_high':>10}")
    print(f"  {'constante c=0.32':<30} {best_s:>7.5f}  {'—':>6}  {'—':>7}  {'—':>14}  {'—':>9}  {'—':>10}")
    print(f"  {'skin/image (brut)':<30} {'0.20552':>7}  {'0.475':>6}  {'+0.525':>7}  {'0.614':>14}  {'+0.497':>9}  {'+0.348':>10}")
    for s in summaries:
        print(f"  {s['label']:<30} {s['score']:>7.5f}  {s['slope']:>6.3f}  "
              f"{s['offset']:>+7.3f}  {s['floor']:>14.3f}  "
              f"{s['bias_low']:>+9.4f}  {s['bias_high']:>+10.4f}")
    print(SEP)
    best = min(summaries, key=lambda x: x["score"])
    print(f"\n  Meilleure définition : {best['label']}  (score={best['score']:.5f})")
    if best["score"] < best_s:
        print(f"  ✓ Bat la constante ({best_s:.5f})")
    else:
        print(f"  ✗ Ne bat pas la constante ({best_s:.5f})")


if __name__ == "__main__":
    main()
