"""
Visualize SAM2 failure modes from a zero-shot evaluation CSV.

Three failure categories:
  A "rates"    : sam2_conf > 0.80, geom < 5%,  gt > 0.15
                 (SAM2 confident but misses real occlusion)
  B "invents"  : sam2_conf > 0.80, geom > 25%, gt < 0.10
                 (SAM2 confident but over-segments a clear face)
  C "absent"   : sam2_conf < 0.50
                 (SAM2 found nothing)

For each failure: 3-panel figure (original | binary mask | cyan contour overlay).
Outputs: PNG per image + self-contained HTML report.

Usage:
    python src/visualize_sam2_failures.py \\
        --predictions-csv results/zero_shot/predictions_TIMESTAMP.csv
"""

from __future__ import annotations

import argparse
import base64
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

_IMAGE_BASE_DIR = (
    "/home/matt/Programmation/704_IADATA_ML_avance/datachallenge"
    "/DataChallenge2026/occlusion_datasets/raw/Crop_224_5fp_100K"
)

_CAT_LABELS = {
    "A_rates":   "Category A — SAM2 rates   (conf > 0.80, geom < 5%,  gt > 0.15)",
    "B_invents": "Category B — SAM2 invents (conf > 0.80, geom > 25%, gt < 0.10)",
    "C_absent":  "Category C — SAM2 absent  (conf < 0.50)",
}
_CAT_COLORS = {"A_rates": "#ff6666", "B_invents": "#ffaa33", "C_absent": "#8888ff"}


# ── Categorisation ─────────────────────────────────────────────────────────────


def categorize_failures(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    geo  = df["geometric_occ"].fillna(np.nan).astype(float)
    conf = df["sam2_confidence"].fillna(0.0).astype(float)
    gt   = df["gt"].astype(float)

    # Categories are mutually exclusive: C requires low conf, A/B require high conf
    cat_c = df[conf < 0.50].copy()
    cat_a = df[(conf > 0.80) & (geo < 0.05)  & (gt > 0.15)].copy()
    cat_b = df[(conf > 0.80) & (geo > 0.25)  & (gt < 0.10)].copy()

    return {"A_rates": cat_a, "B_invents": cat_b, "C_absent": cat_c}


# ── SAM2 mask generation ───────────────────────────────────────────────────────


def _run_sam2(image: Image.Image, segmenter) -> tuple[np.ndarray | None, float]:
    try:
        result = segmenter.segment(image)
        if result is None:
            return None, 0.0
        return result.mask, result.confidence
    except Exception as exc:
        logger.warning("SAM2 failed: %s", exc)
        return None, 0.0


# ── Figure generation ──────────────────────────────────────────────────────────


def _make_figure(
    image: Image.Image,
    mask: np.ndarray | None,
    row: pd.Series,
    category: str,
) -> plt.Figure:
    img = np.array(image.convert("RGB"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    fig.patch.set_facecolor("#1a1a1a")

    # ── Col 1: original ────────────────────────────────────────────────────────
    axes[0].imshow(img)
    axes[0].set_title("Original", color="white", fontsize=10)

    # ── Col 2: binary mask ─────────────────────────────────────────────────────
    if mask is not None:
        axes[1].imshow(np.stack([mask.astype(np.uint8) * 255] * 3, axis=-1))
    else:
        placeholder = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        axes[1].imshow(placeholder)
        axes[1].text(
            img.shape[1] // 2, img.shape[0] // 2,
            "no mask", color="red", ha="center", va="center", fontsize=12,
        )
    axes[1].set_title("SAM2 mask (binary)", color="white", fontsize=10)

    # ── Col 3: overlay with cyan contour ───────────────────────────────────────
    overlay = img.astype(float)
    if mask is not None:
        border = mask & ~binary_erosion(mask, iterations=2)
        # Darken masked region slightly
        overlay[mask] = overlay[mask] * 0.6 + np.array([0, 50, 80]) * 0.4
        # Bright cyan contour
        overlay[border] = [0, 255, 255]
    axes[2].imshow(overlay.clip(0, 255).astype(np.uint8))
    axes[2].set_title("Contour overlay (cyan)", color="white", fontsize=10)

    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#1a1a1a")

    geom_str = (
        f"{row['geometric_occ']*100:.1f}%"
        if not pd.isna(row["geometric_occ"])
        else "N/A"
    )
    err = abs(float(row["pred"]) - float(row["gt"]))
    title = (
        f"gt={float(row['gt']):.3f}  pred={float(row['pred']):.3f}  "
        f"err={err:.3f}  |  geom={geom_str}  conf={float(row['sam2_confidence']):.2f}"
        f"  |  CAT:{category}"
    )
    fig.suptitle(title, color="white", fontsize=11, y=1.01)
    plt.tight_layout()
    return fig


# ── HTML report ────────────────────────────────────────────────────────────────


def _generate_html(figures_by_cat: dict[str, list], output_dir: Path) -> Path:
    sections: list[str] = []

    for cat, items in figures_by_cat.items():
        if not items:
            continue
        color = _CAT_COLORS.get(cat, "#aaa")
        label = _CAT_LABELS.get(cat, cat)

        cards: list[str] = []
        for png_path, row in items:
            img_b64 = base64.b64encode(png_path.read_bytes()).decode()
            err = abs(float(row["pred"]) - float(row["gt"]))
            geom_str = (
                f"{row['geometric_occ']*100:.1f}%"
                if not pd.isna(row["geometric_occ"]) else "N/A"
            )
            caption = (
                f"{Path(row['filename']).name}<br>"
                f"gt={float(row['gt']):.3f} &nbsp;|&nbsp; pred={float(row['pred']):.3f} "
                f"&nbsp;|&nbsp; err={err:.3f} &nbsp;|&nbsp; "
                f"geom={geom_str} &nbsp;|&nbsp; conf={float(row['sam2_confidence']):.2f}"
            )
            cards.append(
                f'<div class="card">'
                f'<img src="data:image/png;base64,{img_b64}" width="620">'
                f'<div class="caption">{caption}</div>'
                f"</div>"
            )

        sections.append(
            f'<section>'
            f'<h2 style="color:{color}">{label} &nbsp;({len(items)} shown)</h2>'
            f'<div class="grid">{"".join(cards)}</div>'
            f"</section>"
        )

    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        "<title>SAM2 Failure Analysis</title>\n"
        "<style>\n"
        "body{font-family:monospace;background:#1a1a1a;color:#eee;padding:24px}\n"
        "h1{color:#7af}h2{border-bottom:1px solid #444;padding-bottom:6px}\n"
        ".grid{display:flex;flex-wrap:wrap;gap:14px;margin-top:10px}\n"
        ".card{background:#2a2a2a;padding:8px;border-radius:4px}\n"
        ".caption{color:#bbb;font-size:11px;margin-top:5px;line-height:1.5}\n"
        "</style>\n</head>\n<body>\n"
        "<h1>SAM2 Failure Analysis</h1>\n"
        + "\n".join(sections)
        + "\n</body>\n</html>"
    )

    out = output_dir / "sam2_failures.html"
    out.write_text(html, encoding="utf-8")
    return out


# ── Terminal summary ──────────────────────────────────────────────────────────


def _print_summary(categories: dict[str, pd.DataFrame]) -> None:
    LINE = "─" * 64
    print(f"\n{LINE}")
    print("  SAM2 FAILURE SUMMARY")
    print(LINE)
    for cat, subset in categories.items():
        label = {
            "A_rates":   "A rates   (misses real occ)",
            "B_invents": "B invents (over-segments)  ",
            "C_absent":  "C absent  (no detection)   ",
        }[cat]
        print(f"  Cat {label}: {len(subset):3d} images")

    all_bad = (
        pd.concat(list(categories.values()))
        .drop_duplicates(subset=["filename"])
        .assign(abs_err=lambda x: (x["pred"].astype(float) - x["gt"].astype(float)).abs())
        .nlargest(3, "abs_err")
    )
    if len(all_bad):
        print(f"\n  Top 3 worst errors (all categories combined):")
        for _, row in all_bad.iterrows():
            geom_str = (
                f"{row['geometric_occ']*100:.1f}%" if not pd.isna(row["geometric_occ"]) else "N/A"
            )
            print(
                f"    {Path(row['filename']).name:<40}  "
                f"gt={row['gt']:.3f}  geom={geom_str:<6}  conf={row['sam2_confidence']:.2f}"
            )
    print(LINE)


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize SAM2 failure modes.")
    p.add_argument(
        "--predictions-csv", required=True,
        help="Predictions CSV from zero_shot_eval.py",
    )
    p.add_argument("--image-base-dir", default=_IMAGE_BASE_DIR)
    p.add_argument("--output-dir",     default="results/sam2_failures")
    p.add_argument(
        "--max-per-cat", type=int, default=10,
        help="Max images per category, sorted by worst |pred - gt| (default: 10)",
    )
    p.add_argument("--sam2-model",  default="facebook/sam2.1-hiera-large")
    p.add_argument("--sam2-device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_base = Path(args.image_base_dir)

    df = pd.read_csv(args.predictions_csv)
    categories = categorize_failures(df)
    _print_summary(categories)

    # Load SAM2 once for all regenerations
    logger.info("Loading SAM2 for mask regeneration…")
    from src.inference.zero_shot_pipeline import SAM2Segmenter  # noqa: PLC0415
    segmenter = SAM2Segmenter(model_id=args.sam2_model, device=args.sam2_device)

    figures_by_cat: dict[str, list] = {cat: [] for cat in categories}
    total = sum(min(len(s), args.max_per_cat) for s in categories.values())
    done = 0

    for cat, subset in categories.items():
        # Sort by worst prediction error, take top N
        subset = (
            subset.assign(abs_err=lambda x: (x["pred"].astype(float) - x["gt"].astype(float)).abs())
            .nlargest(args.max_per_cat, "abs_err")
        )

        for idx, (_, row) in enumerate(subset.iterrows()):
            done += 1
            img_path = image_base / row["filename"]
            if not img_path.exists():
                logger.warning("[%d/%d] NOT FOUND: %s", done, total, img_path)
                continue

            logger.info("[%d/%d] %s  (CAT:%s)", done, total, img_path.name, cat)
            image = Image.open(img_path).convert("RGB")
            mask, _ = _run_sam2(image, segmenter)

            fig = _make_figure(image, mask, row, cat)

            safe_stem = Path(row["filename"]).stem.replace("/", "_")[:60]
            png_path = output_dir / f"{cat}_{idx:02d}_{safe_stem}.png"
            fig.savefig(png_path, dpi=120, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            figures_by_cat[cat].append((png_path, row))

    html_path = _generate_html(figures_by_cat, output_dir)
    n_png = sum(len(v) for v in figures_by_cat.values())
    logger.info("Saved %d PNGs + HTML report", n_png)
    print(f"\n  PNGs  → {output_dir}/")
    print(f"  HTML  → {html_path}\n")


if __name__ == "__main__":
    main()
