"""Generate a synthetic occluder dataset from clean train faces.

Workflow:
    1. Load train.csv, keep rows where FaceOcclusion < `clean_label_max` (default 0.05).
       Clean faces are a near-zero occlusion baseline — drawing a new occluder
       on them gives a label of (new_occluder_area) with negligible overlap
       with the original occlusion.

    2. For each sampled clean image, pick one occluder type at random:
         - "rect"    : black rectangle (sunglasses / censor bar)
         - "ellipse" : black ellipse (covers eye region / mouth)
         - "mask"    : medical-mask-like rectangle in the lower half of the
                       face, with a randomized colour (white / pale blue /
                       dark grey).

    3. The occluder area is sampled uniformly from
       `occluder_area_range` (default [0.10, 0.40] fraction of the image
       area). Each occluder function returns the *actual* pixel area
       fraction it drew, which is then used as the synthetic label.

    4. Save the augmented image as a WebP file (matching the original train
       set encoding) and append a row to `synthetic.csv` with columns:

         filename, FaceOcclusion, gender, source_filename,
         occluder_type, actual_area, original_label

       `filename` is RELATIVE to the project root so the file plugs
       straight into the v19 `_load_train_val` path resolver via the
       `extra_train_csv` YAML field.

USAGE
    # Default settings (10k samples, mixed occluder types, seed 42):
    python scripts/generate_synthetic_occluders.py

    # Customise:
    python scripts/generate_synthetic_occluders.py \\
        --n-samples 20000 \\
        --output-dir data/synthetic/occluder_mask \\
        --types mask \\
        --clean-label-max 0.03

OUTPUTS
    <output_dir>/
        images/syn_000000.webp
        images/syn_000001.webp
        ...
        synthetic.csv

INTEGRATION WITH TRAINING
    In a YAML, point `data.extra_train_csv` at the produced CSV:

        data:
          data_csv: data/raw/train.csv
          extra_train_csv: data/synthetic/occluder_mixed/synthetic.csv

    v19's `_load_train_val` concatenates the two CSVs automatically.

NOTES
    - The occluder is drawn assuming the image IS the face crop (which is
      true for the aligned 224x224 dataset). new_label = original_label +
      actual_occluder_area_fraction (capped at 1.0).
    - For multi-occluder realism, run the script several times with
      different `--types` and different output dirs, then point the YAML
      at whichever one(s) you want to A/B test.
    - Determinism: the script seeds Python random + numpy random with
      `--seed` (default 42). Re-running with the same seed produces the
      same dataset.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Occluder drawing primitives
# Each returns (augmented_image, actual_area_fraction). The fraction is the
# RATIO of drawn pixels to total image pixels — we treat the image as fully
# face, since the dataset is aligned face crops.
# ---------------------------------------------------------------------------

def occluder_rect(image: Image.Image, target_area_frac: float) -> tuple[Image.Image, float]:
    """Black axis-aligned rectangle at a random position.

    Aspect ratio is sampled in [0.5, 2.0] so the rectangle can be a
    horizontal bar (sunglasses) or a vertical band (censor bar).
    """
    w, h = image.size
    total = w * h
    target_pixels = total * target_area_frac

    aspect = random.uniform(0.5, 2.0)
    rect_h = int(np.sqrt(target_pixels / aspect))
    rect_w = int(rect_h * aspect)

    rect_w = max(2, min(rect_w, w - 1))
    rect_h = max(2, min(rect_h, h - 1))

    x0 = random.randint(0, w - rect_w)
    y0 = random.randint(0, h - rect_h)
    x1 = x0 + rect_w
    y1 = y0 + rect_h

    img = image.copy()
    ImageDraw.Draw(img).rectangle((x0, y0, x1, y1), fill=(0, 0, 0))

    return img, (rect_w * rect_h) / total


def occluder_ellipse(image: Image.Image, target_area_frac: float) -> tuple[Image.Image, float]:
    """Black ellipse at a random position. Aspect ratio in [0.6, 1.5]."""
    w, h = image.size
    total = w * h
    # Ellipse area = pi * (a/2) * (b/2). Solve for a, b given target area
    # and aspect ratio.
    target_pixels = total * target_area_frac
    base = 2 * np.sqrt(target_pixels / np.pi)
    aspect = random.uniform(0.6, 1.5)
    a = int(base * np.sqrt(aspect))
    b = int(base / np.sqrt(aspect))

    a = max(4, min(a, w - 1))
    b = max(4, min(b, h - 1))

    x0 = random.randint(0, w - a)
    y0 = random.randint(0, h - b)
    x1 = x0 + a
    y1 = y0 + b

    img = image.copy()
    ImageDraw.Draw(img).ellipse((x0, y0, x1, y1), fill=(0, 0, 0))

    return img, (np.pi * a * b / 4) / total


def occluder_mask(image: Image.Image, target_area_frac: float) -> tuple[Image.Image, float]:
    """Medical-mask-like rectangle in the lower half of the face.

    Width is 60-90% of the image, height computed from target_area_frac.
    Colour sampled in {white, pale blue, dark grey}.
    """
    w, h = image.size
    total = w * h
    target_pixels = total * target_area_frac

    mask_w = int(w * random.uniform(0.6, 0.9))
    mask_h = max(8, min(int(target_pixels / max(mask_w, 1)), h - 1))

    # Centre horizontally (+/- a few px jitter) and place around y = h/2.
    x0 = (w - mask_w) // 2 + random.randint(-8, 8)
    y0 = int(h * 0.55) + random.randint(-12, 12)
    x0 = max(0, min(x0, w - mask_w))
    y0 = max(0, min(y0, h - mask_h))
    x1 = x0 + mask_w
    y1 = y0 + mask_h

    colour = random.choice([
        (245, 245, 245),  # white
        (200, 215, 235),  # pale blue
        (60, 60, 60),     # dark grey
    ])

    img = image.copy()
    ImageDraw.Draw(img).rectangle((x0, y0, x1, y1), fill=colour)

    return img, (mask_w * mask_h) / total


OCCLUDER_FUNCS = {
    "rect": occluder_rect,
    "ellipse": occluder_ellipse,
    "mask": occluder_mask,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train-csv", default="data/raw/train.csv")
    ap.add_argument("--image-dir", default="data/raw",
                    help="Directory that the train.csv `filename` column resolves against.")
    ap.add_argument("--output-dir", default="data/synthetic/occluder_mixed",
                    help="Output directory (will be created). Images go in <dir>/images/, CSV at <dir>/synthetic.csv.")
    ap.add_argument("--n-samples", type=int, default=10000,
                    help="Number of synthetic samples to generate.")
    ap.add_argument("--clean-label-max", type=float, default=0.05,
                    help="Only use source images with FaceOcclusion strictly less than this.")
    ap.add_argument("--occluder-area-min", type=float, default=0.10,
                    help="Lower bound on the occluder area fraction (relative to total image area).")
    ap.add_argument("--occluder-area-max", type=float, default=0.40,
                    help="Upper bound on the occluder area fraction.")
    ap.add_argument("--types", nargs="+", default=list(OCCLUDER_FUNCS.keys()),
                    choices=list(OCCLUDER_FUNCS.keys()),
                    help="Which occluder types to sample from.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quality", type=int, default=85,
                    help="WebP quality for the saved images.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    clean = df[df["FaceOcclusion"] < args.clean_label_max].copy()
    print(f"[load] {len(df)} train rows, {len(clean)} clean (FaceOcclusion < {args.clean_label_max})")
    if len(clean) == 0:
        sys.exit("ERROR: no clean source samples — relax --clean-label-max")

    replace = len(clean) < args.n_samples
    sampled = clean.sample(n=args.n_samples, replace=replace, random_state=args.seed).reset_index(drop=True)
    if replace:
        print(f"[sample] sampling with replacement ({args.n_samples} > {len(clean)} clean sources)")

    rows: list[dict] = []
    n_skipped = 0
    for i, row in enumerate(sampled.itertuples(index=False)):
        original_label = float(row.FaceOcclusion)
        gender = row.gender
        source_path = Path(args.image_dir) / row.filename

        try:
            img = Image.open(source_path).convert("RGB")
        except Exception as e:
            n_skipped += 1
            if n_skipped <= 5:
                print(f"[skip] {source_path}: {e}")
            continue

        occluder_type = random.choice(args.types)
        target_area = random.uniform(args.occluder_area_min, args.occluder_area_max)
        img_aug, actual_area = OCCLUDER_FUNCS[occluder_type](img, target_area)

        # No overlap assumption: clean face means original occlusion ≈ 0.
        new_label = min(original_label + actual_area, 1.0)

        new_filename = f"syn_{i:06d}.webp"
        rel_path = (output_dir / "images" / new_filename).as_posix()
        img_aug.save(images_dir / new_filename, format="WEBP", quality=args.quality)

        rows.append({
            "filename": rel_path,
            "FaceOcclusion": round(new_label, 6),
            "gender": gender,
            "source_filename": row.filename,
            "occluder_type": occluder_type,
            "actual_area": round(actual_area, 6),
            "original_label": round(original_label, 6),
        })

        if (i + 1) % 1000 == 0:
            print(f"[gen]  {i + 1}/{args.n_samples}")

    out_df = pd.DataFrame(rows)
    csv_path = output_dir / "synthetic.csv"
    out_df.to_csv(csv_path, index=False)

    print("\n[done]")
    print(f"  generated   : {len(out_df)} rows")
    print(f"  skipped     : {n_skipped}")
    print(f"  images dir  : {images_dir}")
    print(f"  csv path    : {csv_path}")
    if len(out_df):
        print(f"  label range : min={out_df.FaceOcclusion.min():.3f}  "
              f"mean={out_df.FaceOcclusion.mean():.3f}  "
              f"max={out_df.FaceOcclusion.max():.3f}")
        print(f"  per-type    : {out_df.occluder_type.value_counts().to_dict()}")
        print(f"  per-gender  : {out_df.gender.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
