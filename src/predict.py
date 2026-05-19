from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
import torch
from transformers import AutoImageProcessor

CONFIG: Dict[str, Any] = {
    "mode": "csv",          # csv | dir | interactive | single
    "model_uri": "runs:/RUN_ID/model",
    "tracking_uri": "sqlite:///mlflow.db",
    "input_csv": None,
    "output_csv": "predictions.csv",
    "input_dir": None,
    "image_col": "image_path",
    "max_length": 224,
    "batch_size": 32,
    "threshold": 0.5,
    "text": None,
}


def load_model(model_uri: str, tracking_uri: str = "sqlite:///mlflow.db"):
    mlflow.set_tracking_uri(tracking_uri)
    kwargs = {} if torch.cuda.is_available() else {"map_location": "cpu"}
    model = mlflow.pytorch.load_model(model_uri, **kwargs)
    model.eval()

    processor = None
    if "runs:/" in model_uri:
        try:
            proc_uri = model_uri.rsplit("/", 1)[0] + "/processor"
            local = mlflow.artifacts.download_artifacts(proc_uri)
            processor = AutoImageProcessor.from_pretrained(local)
        except Exception:
            pass

    if processor is None:
        model_name = getattr(model, "model_name", "google/vit-base-patch16-224")
        processor = AutoImageProcessor.from_pretrained(model_name)

    return model, processor


def predict_images(model: Any, processor: Any, image_paths: List[str], batch_size: int = 32) -> Dict[str, Any]:
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch = [Image.open(p).convert("RGB") for p in image_paths[i:i + batch_size]]
            enc = processor(images=batch, return_tensors="pt", padding=True)
            pixel_values = enc["pixel_values"].to(device)
            logits = model(pixel_values=pixel_values)["logits"]
            probs = torch.softmax(logits, dim=-1)
            all_preds.extend(probs.argmax(dim=-1).cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return {"predictions": all_preds, "probabilities": all_probs}


def predict_csv(model_uri: str, input_csv: str, output_csv: str, image_col: str = "image_path",
                tracking_uri: str = "sqlite:///mlflow.db", batch_size: int = 32) -> None:
    model, processor = load_model(model_uri, tracking_uri)
    df = pd.read_csv(input_csv)
    res = predict_images(model, processor, df[image_col].tolist(), batch_size)
    df["prediction"] = res["predictions"]
    df["prob_class0"] = [p[0] for p in res["probabilities"]]
    df["prob_class1"] = [p[1] for p in res["probabilities"]]
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} predictions to {output_csv}")


def predict_dir(model_uri: str, input_dir: str, output_csv: str,
                tracking_uri: str = "sqlite:///mlflow.db", batch_size: int = 32) -> None:
    model, processor = load_model(model_uri, tracking_uri)
    paths = sorted(Path(input_dir).glob("**/*.jpg")) + sorted(Path(input_dir).glob("**/*.png"))
    if not paths:
        raise ValueError(f"No images found in {input_dir}")
    res = predict_images(model, processor, [str(p) for p in paths], batch_size)
    df = pd.DataFrame({
        "image_path": [str(p) for p in paths],
        "prediction": res["predictions"],
        "prob_class0": [p[0] for p in res["probabilities"]],
        "prob_class1": [p[1] for p in res["probabilities"]],
    })
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} predictions to {output_csv}")


def predict_single(model_uri: str, image_path: str, tracking_uri: str = "sqlite:///mlflow.db") -> None:
    model, processor = load_model(model_uri, tracking_uri)
    res = predict_images(model, processor, [image_path], batch_size=1)
    print(f"Image: {image_path}")
    print(f"Prediction: {res['predictions'][0]}")
    print(f"Probabilities: {[f'{p:.2%}' for p in res['probabilities'][0]]}")


def predict_interactive(model_uri: str, tracking_uri: str = "sqlite:///mlflow.db") -> None:
    model, processor = load_model(model_uri, tracking_uri)
    print("Interactive mode — enter image paths (or 'quit')")
    while True:
        path = input("Image > ").strip()
        if path.lower() in ("quit", "exit", "q"):
            break
        if not path:
            continue
        res = predict_images(model, processor, [path], batch_size=1)
        print(f"  class={res['predictions'][0]} | probs={[f'{p:.2%}' for p in res['probabilities'][0]]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", default=None)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--image", default=None)
    args = parser.parse_args()

    if args.model_uri:
        CONFIG["model_uri"] = args.model_uri
    if args.input_csv:
        CONFIG["input_csv"] = args.input_csv
    if args.input_dir:
        CONFIG["input_dir"] = args.input_dir
    if args.output_csv:
        CONFIG["output_csv"] = args.output_csv
    if args.image:
        CONFIG["text"] = args.image

    mode = CONFIG["mode"]
    uri, tracking = CONFIG["model_uri"], CONFIG["tracking_uri"]

    if mode == "csv":
        predict_csv(uri, CONFIG["input_csv"], CONFIG["output_csv"], CONFIG["image_col"], tracking, CONFIG["batch_size"])
    elif mode == "dir":
        predict_dir(uri, CONFIG["input_dir"], CONFIG["output_csv"], tracking, CONFIG["batch_size"])
    elif mode == "single":
        predict_single(uri, CONFIG["text"], tracking)
    elif mode == "interactive":
        predict_interactive(uri, tracking)
