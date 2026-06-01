"""Extract P_test(Y) PMF from task_brief.pdf page 3 (test histogram).

v18: re-extraction avec resolution plus fine (20 ou 25 bins vs 15 en v17-).

Methodologie (cf docs/v12_theory.md §4.7.3) :
1. Render PDF page 3 a 300 DPI via PyMuPDF
2. Detecter axis frame (lignes noires) + tick marks pour calibrer axes
3. Bar pixel detection (RGB ~ (142, 186, 217), matplotlib default blue)
4. Pour chaque colonne pixel : trouver le top du bar -> count via y calibration
5. Aggregation par mean sur chaque target bin

Sortie: 2 PMFs (20 bins et 25 bins) couvrant [0, 0.5], pretes a coller dans
src/utils/distribution.py.
"""
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image


def main() -> None:
    pdf_path = Path(__file__).parent.parent / "example" / "task_brief.pdf"
    if not pdf_path.exists():
        print(f"ERR: {pdf_path} not found", file=sys.stderr)
        sys.exit(1)

    # === 1. Render page 3 (index 2) at 300 DPI ===
    doc = fitz.open(str(pdf_path))
    page = doc[2]
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[..., :3]
    print(f"Rendered page 3: {img.shape[1]}x{img.shape[0]} px")

    H, W, _ = img.shape

    # === 2. Detect axis frame in right subplot (test) ===
    gray_max = img.max(axis=-1)
    gray_min = img.min(axis=-1)
    dark = (gray_max < 100) & (gray_max - gray_min < 30)
    right_dark = dark.copy()
    right_dark[:, : W // 2] = False
    row_sum = right_dark.sum(axis=1)
    col_sum = right_dark.sum(axis=0)

    # Frame top + bottom from rows with most dark pixels
    cand_rows = np.argsort(row_sum)[::-1][:4]
    Y_top = int(min(cand_rows))
    Y_bot = int(max(cand_rows))
    cand_cols = np.argsort(col_sum)[::-1][:4]
    X_left = int(min(cand_cols))
    X_right = int(max(cand_cols))
    print(f"Test plot frame: x[{X_left}, {X_right}]  y[{Y_top}, {Y_bot}]")

    # === 3. Detect Y-tick rows (notches left of y-axis) ===
    y_strip = dark[Y_top - 5 : Y_bot + 5, X_left - 25 : X_left - 5]
    y_tick_rows_loc = np.where(y_strip.sum(axis=1) > 5)[0]
    y_ticks = _group(y_tick_rows_loc, offset=Y_top - 5)
    print(f"Y-tick pixel rows (top->bot): {y_ticks}")
    assert len(y_ticks) == 8, f"Expected 8 Y ticks (0..1400 step 200), got {len(y_ticks)}"

    # === 4. Detect X-tick cols ===
    x_strip = dark[Y_bot + 1 : Y_bot + 25, X_left - 5 : X_right + 5]
    x_tick_cols_loc = np.where(x_strip.sum(axis=0) > 5)[0]
    x_ticks = _group(x_tick_cols_loc, offset=X_left - 5)
    print(f"X-tick pixel cols (left->right): {x_ticks}")
    assert len(x_ticks) == 6, f"Expected 6 X ticks (0.0..1.0 step 0.2), got {len(x_ticks)}"

    # === 5. Calibration ===
    # Y-axis: y_ticks[0] = 1400 count, y_ticks[-1] = 0 count
    y_count_per_px = 1400.0 / (y_ticks[-1] - y_ticks[0])
    print(f"Y calibration: {y_count_per_px:.4f} count/pixel ({(y_ticks[-1]-y_ticks[0])} px = 1400 counts)")
    # X-axis: x_ticks[0] = 0.0, x_ticks[-1] = 1.0
    x_value_per_px = 1.0 / (x_ticks[-1] - x_ticks[0])
    print(f"X calibration: {x_value_per_px:.6f} value/pixel ({(x_ticks[-1]-x_ticks[0])} px = 1.0)")

    # === 6. Bar detection: pixel color of matplotlib default blue ~ (142, 186, 217) ===
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    bar_mask = (
        (R >= 100) & (R <= 180) & (G >= 160) & (G <= 220) & (B >= 200) & (B <= 240) & (B > R)
    )
    bar_mask[: Y_top, :] = False
    bar_mask[Y_bot:, :] = False
    bar_mask[:, : X_left] = False
    bar_mask[:, X_right + 1 :] = False
    print(f"Bar pixels in test frame: {bar_mask.sum()}")

    # === 7. For each pixel col, find top of bar -> convert to count ===
    counts_per_px = np.zeros(W, dtype=np.float64)
    for x in range(X_left, X_right + 1):
        col = bar_mask[:, x]
        ys = np.where(col)[0]
        if len(ys) == 0:
            continue
        top_y = ys.min()
        height_px = y_ticks[-1] - top_y
        counts_per_px[x] = max(0.0, height_px * y_count_per_px)

    # === 8. Aggregate into target bins covering [0, 0.5] ===
    x0_data = x_ticks[0]
    x1_data = x_ticks[0] + (x_ticks[-1] - x_ticks[0]) * 0.5  # pixel for value=0.5

    for n_bins in [20, 25]:
        print(f"\n=== {n_bins} bins covering [0, 0.5] (bin_width = {0.5/n_bins:.4f}) ===")
        edges_px = np.linspace(x0_data, x1_data, n_bins + 1)
        bin_counts = np.zeros(n_bins, dtype=np.float64)
        for i in range(n_bins):
            x_start = int(np.round(edges_px[i]))
            x_end = int(np.round(edges_px[i + 1]))
            if x_end <= x_start:
                x_end = x_start + 1
            col_vals = counts_per_px[x_start:x_end]
            if col_vals.size == 0:
                continue
            bin_counts[i] = col_vals.mean()
        pmf = bin_counts / bin_counts.sum()
        print("PMF:")
        for i, p in enumerate(pmf):
            print(f"  bin {i:2d}  y=[{i*0.5/n_bins:.4f}, {(i+1)*0.5/n_bins:.4f}]  count={bin_counts[i]:6.1f}  P={p:.6f}")
        print(f"\nSum check: {pmf.sum():.6f}")
        print(f"\n# === Paste-ready for src/utils/distribution.py ({n_bins} bins) ===")
        print(f"N_BINS: int = {n_bins}")
        print(f"BIN_WIDTH: float = 0.5 / N_BINS  # = {0.5/n_bins}")
        print("_TEST_PMF: np.ndarray = np.array([")
        for p in pmf:
            print(f"    {p:.6f},")
        print("], dtype=np.float64)")


def _group(idx: np.ndarray, offset: int = 0, gap: int = 3) -> list:
    """Group consecutive indices (within `gap`), return mean of each cluster, sorted."""
    if len(idx) == 0:
        return []
    groups = []
    cur = [idx[0]]
    for k in idx[1:]:
        if k - cur[-1] <= gap:
            cur.append(k)
        else:
            groups.append(int(np.mean(cur)) + offset)
            cur = [k]
    groups.append(int(np.mean(cur)) + offset)
    return sorted(groups)


if __name__ == "__main__":
    main()
