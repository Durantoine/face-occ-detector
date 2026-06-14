from pathlib import Path
from typing import Any, Dict

import pandas as pd

CONFIG: Dict[str, Any] = {
    "mafa_root": "data/extra/mafa_raw/",
    "annot_file": "LabelTrainAll.txt",
    "image_subdir": "train-images",
    "output_csv": "data/extra/mafa.csv",
    "default_gender": 0.5,
    "default_occlusion": 0.55,
    "use_bbox_ratio": True,
}


def _parse_mafa_line(line: str):
    parts = line.strip().split()
    if len(parts) < 19:
        return None
    fname = parts[0]
    face_x, face_y, face_w, face_h = (int(x) for x in parts[1:5])
    occ_x, occ_y, occ_w, occ_h = (int(x) for x in parts[14:18])
    return fname, (face_x, face_y, face_w, face_h), (occ_x, occ_y, occ_w, occ_h)


def _bbox_intersection_area(a, b) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    return max(0, x2 - x1) * max(0, y2 - y1)


def prepare_mafa(
    mafa_root: str,
    annot_file: str,
    image_subdir: str,
    output_csv: str,
    default_gender: float,
    default_occlusion: float,
    use_bbox_ratio: bool,
) -> None:
    root = Path(mafa_root)
    annot = root / annot_file
    if not annot.exists():
        raise FileNotFoundError(f"MAFA annotation not found: {annot}")

    rows = []
    with annot.open() as f:
        for line in f:
            parsed = _parse_mafa_line(line)
            if parsed is None:
                continue
            fname, face_box, occ_box = parsed
            face_area = max(1, face_box[2] * face_box[3])
            if use_bbox_ratio:
                inter = _bbox_intersection_area(face_box, occ_box)
                ratio = min(1.0, inter / face_area)
            else:
                ratio = default_occlusion
            rows.append({
                "filename": f"{image_subdir}/{fname}",
                "FaceOcclusion": float(ratio),
                "gender": float(default_gender),
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df):,} rows to {output_csv}")
    print(f"  FaceOcclusion: mean={df['FaceOcclusion'].mean():.3f} std={df['FaceOcclusion'].std():.3f}")
    print(f"  gender imputed to {default_gender}: balanced sampler will rebalance Idemia data with these")


if __name__ == "__main__":
    prepare_mafa(
        mafa_root=CONFIG["mafa_root"],
        annot_file=CONFIG["annot_file"],
        image_subdir=CONFIG["image_subdir"],
        output_csv=CONFIG["output_csv"],
        default_gender=CONFIG["default_gender"],
        default_occlusion=CONFIG["default_occlusion"],
        use_bbox_ratio=CONFIG["use_bbox_ratio"],
    )
