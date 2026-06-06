"""
Précomputation des signaux déterministes pour le fine-tuning de Qwen.

Signaux calculés par image :
  - blur_score  : variance du Laplacien (netteté, CPU ~0.5 ms/img)
  - hair_pct    : fraction de l'image classée "cheveux" par Segformer (GPU ~30 ms/img)
  - yaw_deg     : rotation horizontale de tête (MediaPipe, CPU ~10 ms/img)
  - pitch_deg   : rotation verticale de tête (MediaPipe, CPU ~10 ms/img)
  - pose_occ    : occlusion géométrique estimée par MediaPipe
  - mp_detected : MediaPipe a-t-il détecté un visage ?
  - orientation : frontal | slight | rotated | profile

Output : data/signals_precomputed.csv

Usage:
    uv run python src/precompute_signals.py
    uv run python src/precompute_signals.py --n 5000 --seed 42     # sous-ensemble
    uv run python src/precompute_signals.py --no-segformer          # skip hair (rapide)
    uv run python src/precompute_signals.py --csv data/train.csv    # tout le dataset
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.zero_shot_eval import OCC_BINS, stratified_sample

IMAGE_BASE = Path(
    "/home/matt/Programmation/704_IADATA_ML_avance/datachallenge"
    "/DataChallenge2026/occlusion_datasets/raw/Crop_224_5fp_100K"
)
DATA_CSV        = Path("data/train.csv")
OUTPUT_CSV      = Path("data/signals_precomputed.csv")
SEGFORMER_REPO  = "jonathandinu/face-parsing"
HAIR_CLS        = 13

def laplacian_blur(img: Image.Image) -> float:
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def load_segformer(device: str):
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    proc  = SegformerImageProcessor.from_pretrained(SEGFORMER_REPO)
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_REPO).eval().to(device)
    return proc, model


def hair_pct_batch(images: list[Image.Image], seg_proc, seg_model, device: str) -> list[float]:
    """Calcule hair_pct pour un batch d'images (plus efficace que image par image)."""
    inputs = seg_proc(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = seg_model(**inputs).logits
    seg = F.interpolate(logits, size=(224, 224), mode="bilinear",
                        align_corners=False).argmax(1).cpu().numpy()
    return [float((seg[i] == HAIR_CLS).sum() / (224 * 224)) for i in range(len(images))]


def mediapipe_signals(img: Image.Image):
    """Retourne (yaw_deg, pitch_deg, pose_occ, detected, orientation)."""
    from src.inference.zero_shot_pipeline import analyze_face_pose
    pose = analyze_face_pose(np.array(img.convert("RGB")))
    if not pose.detected:
        return 0.0, 0.0, 0.0, False, "unknown"
    return pose.yaw_deg, pose.pitch_deg, pose.pose_occlusion, True, pose.orientation

def _testlike_sample(df: pd.DataFrame, n_per_bin: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for _, lo, hi in OCC_BINS:
        pool = df[(df["FaceOcclusion"] >= lo) & (df["FaceOcclusion"] < hi)]
        k = min(n_per_bin, len(pool))
        parts.append(pool.iloc[rng.choice(len(pool), k, replace=False)])
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

def compute_signals(
    df: pd.DataFrame,
    image_base: Path,
    use_segformer: bool,
    seg_batch: int,
    device: str,
) -> pd.DataFrame:

    seg_proc = seg_model = None
    if use_segformer:
        print(f"Loading Segformer on {device} …")
        seg_proc, seg_model = load_segformer(device)
        print("Segformer ready.")

    n = len(df)
    records = []
    t0 = time.time()
    batch_imgs: list[Image.Image] = []
    batch_rows: list = []

    def _flush_segformer():
        if not batch_imgs:
            return
        hairs = hair_pct_batch(batch_imgs, seg_proc, seg_model, device)
        for rec, hp in zip(batch_rows, hairs):
            rec["hair_pct"] = hp
        batch_imgs.clear()
        batch_rows.clear()

    for i, row in enumerate(df.itertuples(index=False), 1):
        if i % 200 == 0 or i == 1:
            elapsed = time.time() - t0
            eta = elapsed / i * (n - i)
            print(f"  {i:5d}/{n}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

        img_path = image_base / row.filename
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  WARN cannot open {img_path}: {e}")
            records.append({
                "filename":    row.filename,
                "blur_score":  -1.0,
                "hair_pct":    -1.0,
                "yaw_deg":     0.0,
                "pitch_deg":   0.0,
                "pose_occ":    0.0,
                "mp_detected": False,
                "orientation": "unknown",
            })
            continue

        blur  = laplacian_blur(img)
        yaw, pitch, pose_occ, detected, orient = mediapipe_signals(img)

        rec = {
            "filename":    row.filename,
            "blur_score":  round(blur, 2),
            "hair_pct":    0.0,  
            "yaw_deg":     yaw,
            "pitch_deg":   pitch,
            "pose_occ":    round(pose_occ, 4),
            "mp_detected": detected,
            "orientation": orient,
        }
        records.append(rec)

        if use_segformer:
            batch_imgs.append(img)
            batch_rows.append(rec)
            if len(batch_imgs) >= seg_batch:
                _flush_segformer()

    if use_segformer:
        _flush_segformer()

    if not use_segformer:
        for rec in records:
            rec["hair_pct"] = -1.0  

    return pd.DataFrame(records)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv",         type=Path, default=DATA_CSV)
    p.add_argument("--out",         type=Path, default=OUTPUT_CSV)
    p.add_argument("--image-base",  type=Path, default=IMAGE_BASE)
    p.add_argument("--n",           type=int,  default=0,
                   help="Nombre de samples (0 = tout le dataset)")
    p.add_argument("--n-per-bin",   type=int,  default=0,
                   help="Samples par bin occ (priorité sur --n si >0)")
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--seg-batch",   type=int,  default=16,
                   help="Batch size Segformer")
    p.add_argument("--no-segformer", action="store_true",
                   help="Passer Segformer (hair_pct = -1)")
    p.add_argument("--append",      action="store_true",
                   help="Ajouter au CSV existant (skip filenames déjà calculés)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.csv} …")
    df_full = pd.read_csv(args.csv).dropna(subset=["filename", "FaceOcclusion", "gender"])
    df_full["gender"] = pd.to_numeric(df_full["gender"], errors="coerce").fillna(0.5)
    df_full = df_full[df_full["gender"].isin([0, 1])].reset_index(drop=True)
    print(f"  Total: {len(df_full)} samples")

    if args.n_per_bin > 0:
        df = _testlike_sample(df_full, args.n_per_bin, args.seed)
    elif args.n > 0:
        df = _testlike_sample(df_full, args.n // 3, args.seed)
    else:
        df = df_full.copy()

    if args.append and args.out.exists():
        existing = pd.read_csv(args.out)["filename"].tolist()
        n_before = len(df)
        df = df[~df["filename"].isin(existing)].reset_index(drop=True)
        print(f"  --append : {n_before - len(df)} filenames déjà calculés, {len(df)} restants")

    if len(df) == 0:
        print("Aucun sample à calculer.")
        return

    print(f"  Calcul sur {len(df)} samples  (device={device})")
    print(f"  Segformer: {'oui' if not args.no_segformer else 'non (--no-segformer)'}  "
          f"seg_batch={args.seg_batch}")

    result = compute_signals(
        df, args.image_base,
        use_segformer=not args.no_segformer,
        seg_batch=args.seg_batch,
        device=device,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.append and args.out.exists():
        result = pd.concat([pd.read_csv(args.out), result], ignore_index=True)

    result.to_csv(args.out, index=False)
    elapsed = time.time()
    print(f"\nSignaux sauvegardés → {args.out}  ({len(result)} lignes)")
    print(f"Colonnes : {list(result.columns)}")
    print(result.describe())


if __name__ == "__main__":
    main()
