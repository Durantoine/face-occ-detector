from pathlib import Path
from typing import Any, Dict

import pandas as pd

CONFIG: Dict[str, Any] = {
    "celeba_root": "data/extra/celeba_raw/",
    "attr_file": "list_attr_celeba.csv",
    "image_subdir": "img_align_celeba",
    "output_csv": "data/extra/celeba_occ.csv",
    "occlusion_attrs_weights": {
        "Eyeglasses": 0.10,
        "Wearing_Hat": 0.10,
        "Wearing_Necktie": 0.05,
        "Wearing_Necklace": 0.03,
        "Sideburns": 0.02,
        "Mustache": 0.03,
        "Goatee": 0.03,
        "No_Beard": -0.05,
    },
    "clip_range": (0.0, 0.5),
}


def prepare_celeba(
    celeba_root: str,
    attr_file: str,
    image_subdir: str,
    output_csv: str,
    occlusion_attrs_weights: Dict[str, float],
    clip_range: tuple,
) -> None:
    root = Path(celeba_root)
    df = pd.read_csv(root / attr_file)
    if "image_id" in df.columns:
        df = df.rename(columns={"image_id": "filename"})

    score = pd.Series(0.0, index=df.index)
    for attr, w in occlusion_attrs_weights.items():
        if attr in df.columns:
            score = score + ((df[attr] == 1).astype(float) * w)

    lo, hi = clip_range
    score = score.clip(lo, hi)

    out = pd.DataFrame({
        "filename": df["filename"].apply(lambda f: f"{image_subdir}/{f}"),
        "FaceOcclusion": score.astype(float),
        "gender": (df["Male"] == 1).astype(float) if "Male" in df.columns else 0.5,
    })
    out.to_csv(output_csv, index=False)
    print(f"Wrote {len(out):,} rows to {output_csv}")
    print(f"  FaceOcclusion: mean={out['FaceOcclusion'].mean():.3f} std={out['FaceOcclusion'].std():.3f}")
    print(f"  gender: F={(out['gender'] < 0.5).sum():,} M={(out['gender'] >= 0.5).sum():,}")


if __name__ == "__main__":
    prepare_celeba(
        celeba_root=CONFIG["celeba_root"],
        attr_file=CONFIG["attr_file"],
        image_subdir=CONFIG["image_subdir"],
        output_csv=CONFIG["output_csv"],
        occlusion_attrs_weights=CONFIG["occlusion_attrs_weights"],
        clip_range=CONFIG["clip_range"],
    )
