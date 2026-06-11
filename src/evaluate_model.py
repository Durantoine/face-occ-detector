from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.predict import load_model, predict_images
from src.utils.metrics import compute_score

CONFIG: Dict[str, Any] = {
    "model_uri": "runs:/RUN_ID/model",
    "data_csv": "data/train.csv",
    "image_col": "filename",
    "label_col": "FaceOcclusion",
    "gender_col": "gender",
    "image_base_dir": "data/raw",
    "tracking_uri": "sqlite:///mlflow.db",
    "batch_size": 64,
    "save_worst_k": 0,
    "worst_output_dir": "results/worst",
}


def _save_worst_k(
    df: pd.DataFrame,
    preds: np.ndarray,
    gt: np.ndarray,
    gender: np.ndarray,
    k: int,
    output_dir: str,
    image_col: str,
    image_base_dir: Optional[str],
) -> None:
    w = 1.0 / 30.0 + gt
    per_sample_err = w * (preds - gt) ** 2
    abs_err = np.abs(preds - gt)
    order = np.argsort(-per_sample_err)[:k]
    out = pd.DataFrame({
        "filename": df[image_col].values[order],
        "gt": gt[order],
        "pred": preds[order],
        "abs_err": abs_err[order],
        "weighted_err": per_sample_err[order],
        "gender": gender[order],
    })
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "worst.csv"
    out.to_csv(csv_path, index=False)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    base = Path(image_base_dir) if image_base_dir else None
    for rank, row in enumerate(out.itertuples(index=False), start=1):
        src = base / row.filename if (base and not Path(row.filename).is_absolute()) else Path(row.filename)
        if not src.exists():
            continue
        dst = img_dir / f"{rank:03d}_gt{row.gt:.2f}_pred{row.pred:.2f}_g{int(row.gender)}_{src.name}"
        if dst.exists():
            dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            import shutil
            shutil.copy(src, dst)
    print(f"Saved {len(out)} worst predictions to {csv_path} (images in {img_dir})")


def evaluate(
    model_uri: str,
    data_csv: str,
    image_col: str = "filename",
    label_col: str = "FaceOcclusion",
    gender_col: str = "gender",
    image_base_dir: Optional[str] = None,
    tracking_uri: str = "sqlite:///mlflow.db",
    batch_size: int = 64,
    save_worst_k: int = 0,
    worst_output_dir: str = "results/worst",
) -> Dict[str, float]:
    model, processor = load_model(model_uri, tracking_uri)
    df = pd.read_csv(data_csv).dropna(subset=[image_col, label_col, gender_col])
    preds = np.array(predict_images(model, processor, df[image_col].tolist(), image_base_dir, batch_size))
    gt = df[label_col].astype(float).values
    gender = pd.to_numeric(df[gender_col], errors="coerce").fillna(0.5).astype(float).values
    metrics = compute_score(preds, gt, gender)
    print(f"Score    : {metrics['score']:.5f}")
    print(f"Err_F    : {metrics['err_F']:.5f}")
    print(f"Err_M    : {metrics['err_M']:.5f}")
    print(f"|Err diff| : {metrics['err_diff']:.5f}")
    print(f"MSE      : {metrics['mse']:.5f}")
    print(f"MAE      : {metrics['mae']:.5f}")
    if save_worst_k > 0:
        _save_worst_k(df, preds, gt, gender, save_worst_k, worst_output_dir, image_col, image_base_dir)
    return metrics


if __name__ == "__main__":
    evaluate(**CONFIG)
