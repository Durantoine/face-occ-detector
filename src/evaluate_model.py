from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.predict import load_model, predict_images
from src.utils.metrics import compute_score

CONFIG: Dict[str, Any] = {
    "model_uri": "runs:/RUN_ID/model",
    "data_csv": "data/raw/train.csv",
    "image_col": "filename",
    "label_col": "FaceOcclusion",
    "gender_col": "gender",
    "image_base_dir": "data/raw",
    "tracking_uri": "sqlite:///mlflow.db",
    "batch_size": 64,
}


def evaluate(
    model_uri: str,
    data_csv: str,
    image_col: str = "filename",
    label_col: str = "FaceOcclusion",
    gender_col: str = "gender",
    image_base_dir: Optional[str] = None,
    tracking_uri: str = "sqlite:///mlflow.db",
    batch_size: int = 64,
) -> Dict[str, float]:
    model, processor = load_model(model_uri, tracking_uri)
    df = pd.read_csv(data_csv).dropna(subset=[image_col, label_col, gender_col])
    preds = predict_images(model, processor, df[image_col].tolist(), image_base_dir, batch_size)
    metrics = compute_score(np.array(preds), df[label_col].values, df[gender_col].values)
    print(f"Score    : {metrics['score']:.5f}")
    print(f"Err_F    : {metrics['err_F']:.5f}")
    print(f"Err_M    : {metrics['err_M']:.5f}")
    print(f"|Err diff| : {metrics['err_diff']:.5f}")
    print(f"MSE      : {metrics['mse']:.5f}")
    print(f"MAE      : {metrics['mae']:.5f}")
    return metrics


if __name__ == "__main__":
    evaluate(
        model_uri=CONFIG["model_uri"],
        data_csv=CONFIG["data_csv"],
        image_col=CONFIG["image_col"],
        label_col=CONFIG["label_col"],
        gender_col=CONFIG["gender_col"],
        image_base_dir=CONFIG["image_base_dir"],
        tracking_uri=CONFIG["tracking_uri"],
        batch_size=CONFIG["batch_size"],
    )
