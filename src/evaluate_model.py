import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.predict import load_model, predict_images


def evaluate(
    model_uri: str,
    data_csv: str,
    image_col: str = "image_path",
    label_col: str = "label",
    image_base_dir: str = None,
    tracking_uri: str = "sqlite:///mlflow.db",
    batch_size: int = 32,
) -> None:
    model, processor = load_model(model_uri, tracking_uri)
    df = pd.read_csv(data_csv)
    paths = df[image_col].astype(str).tolist()
    if image_base_dir:
        base = Path(image_base_dir)
        paths = [str(base / p) if not Path(p).is_absolute() else p for p in paths]

    res = predict_images(model, processor, paths, batch_size)
    preds = np.array(res["predictions"])
    labels = df[label_col].values

    print(classification_report(labels, preds, digits=4))
    print(f"F1 macro : {f1_score(labels, preds, average='macro'):.4f}")
    print(f"Accuracy : {accuracy_score(labels, preds):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--image-col", default="image_path")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--image-base-dir", default=None)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    evaluate(
        args.model_uri, args.data_csv, args.image_col, args.label_col,
        args.image_base_dir, args.tracking_uri, args.batch_size,
    )
