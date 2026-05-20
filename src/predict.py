from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor

from src.inference.calibration import apply_bias
from src.models.dinov3_loader import get_image_processor

CONFIG: Dict[str, Any] = {
    "model_uri": "runs:/RUN_ID/model",
    "tracking_uri": "sqlite:///mlflow.db",
    "input_csv": "data/raw/test_students.csv",
    "output_csv": "test_predictions.csv",
    "image_col": "filename",
    "gender_col": "gender",
    "image_base_dir": "data/raw",
    "batch_size": 64,
    "submission_format": True,
    "use_tta": True,
    "bias_correction": None,
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
        model_name = getattr(model, "model_name", "dinov3_vits16")
        processor = get_image_processor(model_name)

    return model, processor


@torch.no_grad()
def predict_images(
    model: Any,
    processor: Any,
    image_paths: List[str],
    image_base_dir: Optional[str] = None,
    batch_size: int = 64,
) -> List[float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    base = Path(image_base_dir) if image_base_dir else None
    preds: List[float] = []
    for i in range(0, len(image_paths), batch_size):
        chunk = image_paths[i:i + batch_size]
        batch = []
        for p in chunk:
            full = base / p if (base and not Path(p).is_absolute()) else Path(p)
            batch.append(Image.open(full).convert("RGB"))
        enc = processor(images=batch, return_tensors="pt")
        out = model(pixel_values=enc["pixel_values"].to(device))["logits"]
        preds.extend(out.detach().cpu().view(-1).tolist())
    return [max(0.0, min(1.0, float(p))) for p in preds]


def predict_csv(
    model_uri: str,
    input_csv: str,
    output_csv: str,
    image_col: str = "filename",
    image_base_dir: Optional[str] = None,
    tracking_uri: str = "sqlite:///mlflow.db",
    batch_size: int = 64,
    submission_format: bool = True,
    use_tta: bool = True,
    bias_correction: Optional[Dict[str, float]] = None,
    gender_col: str = "gender",
) -> None:
    model, processor = load_model(model_uri, tracking_uri)
    df = pd.read_csv(input_csv).dropna(subset=[image_col])

    if use_tta:
        from src.inference.tta import predict_tta_batch
        preds_t = predict_tta_batch(model, processor, df[image_col].tolist(),
                                    batch_size=batch_size, image_base_dir=image_base_dir)
        preds = preds_t.numpy().astype(float)
    else:
        preds = np.array(predict_images(model, processor, df[image_col].tolist(), image_base_dir, batch_size))

    if bias_correction and gender_col in df.columns:
        g = pd.to_numeric(df[gender_col], errors="coerce").fillna(0.5).astype(float)
        preds = apply_bias(preds, g.values,
                           bias_correction.get("delta_f", 0.0),
                           bias_correction.get("delta_m", 0.0))

    df["FaceOcclusion"] = preds
    if submission_format:
        df["gender"] = "x"
        df[[image_col, "FaceOcclusion", "gender"]].to_csv(output_csv, index=False)
    else:
        df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} predictions to {output_csv}  (TTA={use_tta}, bias={bias_correction})")


if __name__ == "__main__":
    predict_csv(
        model_uri=CONFIG["model_uri"],
        input_csv=CONFIG["input_csv"],
        output_csv=CONFIG["output_csv"],
        image_col=CONFIG["image_col"],
        image_base_dir=CONFIG["image_base_dir"],
        tracking_uri=CONFIG["tracking_uri"],
        batch_size=CONFIG["batch_size"],
        submission_format=CONFIG["submission_format"],
        use_tta=CONFIG["use_tta"],
        bias_correction=CONFIG["bias_correction"],
        gender_col=CONFIG["gender_col"],
    )
