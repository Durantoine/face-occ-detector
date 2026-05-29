import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor

from src.inference.calibration import apply_bias, quantile_match_to_test_pmf
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
    "match_test_pmf": False,
}


def _collect_metadata(
    model_uri: str,
    tracking_uri: str,
    predict_options: Dict[str, Any],
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "model_uri": model_uri,
        "tracking_uri": tracking_uri,
        "predict_options": predict_options,
    }
    m = re.match(r"runs:/([^/]+)/", model_uri)
    if not m:
        return metadata
    run_id = m.group(1)
    metadata["mlflow_run_id"] = run_id
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=tracking_uri)
        run = client.get_run(run_id)
        metadata["mlflow_run_name"] = run.info.run_name
        metadata["mlflow_status"] = run.info.status
        metadata["mlflow_experiment_id"] = run.info.experiment_id
        params = dict(run.data.params)
        metadata["mlflow_params"] = params
        relevant_keys = (
            "challenge_score", "err_F", "err_M", "err_diff",
            "mae", "mse", "r2_", "eval_loss",
        )
        metadata["mlflow_metrics_summary"] = {
            k: float(v) for k, v in run.data.metrics.items()
            if any(s in k for s in relevant_keys)
        }
        metadata["mlflow_tags"] = {
            k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")
        }
        metadata["key_params"] = {
            "architecture":         params.get("architecture"),
            "model_name":           params.get("model_name"),
            "pretrained":           params.get("pretrained"),
            "init_backbone_from":   params.get("init_backbone_from"),
            "pooling_type":         params.get("pooling_type"),
            # v9 axes (2-way stick-breaking)
            "axis1_power":           params.get("axis1_power"),
            "axis1_sampler_share":   params.get("axis1_sampler_share"),
            "sampler_power":         params.get("sampler_power"),
            "loss_power":            params.get("loss_power"),
            "axis2_power":           params.get("axis2_power"),
            "feature_fairness":      params.get("feature_fairness"),
            "loss_focal_gamma":      params.get("loss_focal_gamma"),
            "loss_fairness_lambda":  params.get("loss_fairness_lambda"),
            "loss_type":             params.get("loss_type"),
            "learning_rate":         params.get("learning_rate"),
            "num_train_epochs":      params.get("num_train_epochs"),
            "augmentation_level":    params.get("augmentation_level"),
            "layer_decay":           params.get("layer_decay"),
        }
    except Exception as e:
        metadata["mlflow_fetch_error"] = str(e)
    return metadata


def _save_metadata(metadata: Dict[str, Any], output_csv: str) -> str:
    meta_path = str(Path(output_csv).with_suffix("")) + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
    return meta_path


def _print_metadata_summary(metadata: Dict[str, Any]) -> None:
    key = metadata.get("key_params", {})
    opts = metadata.get("predict_options", {})
    print("─" * 70)
    print("Submission metadata summary :")
    print(f"  Model URI         : {metadata.get('model_uri')}")
    print(f"  Architecture      : {key.get('architecture')}")
    print(f"  Backbone          : {key.get('model_name')}  pretrained={key.get('pretrained')}")
    init = key.get("init_backbone_from") or "—"
    print(f"  iBOT init         : {init}")
    print(f"  Pooling           : {key.get('pooling_type')}")
    # v9 axes
    print(f"  Axis 1 (Y shift) : strength={key.get('axis1_power')} "
          f"(sampler={key.get('sampler_power')}, loss={key.get('loss_power')})")
    print(f"  Axis 2 (cell_rw) : power={key.get('axis2_power')}")
    print(f"  Feature fairness : {key.get('feature_fairness')}")
    print(f"  Focal γ          : {key.get('loss_focal_gamma')}")
    print(f"  λ_fairness        : {key.get('loss_fairness_lambda')}")
    print(f"  Post-hoc          : TTA={opts.get('use_tta')}  "
          f"bias={'on' if opts.get('bias_correction') else 'off'}  "
          f"quantile_match={opts.get('match_test_pmf')}")
    print("─" * 70)


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
    match_test_pmf: bool = False,
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

    if match_test_pmf:
        from src.utils.losses import _TEST_PMF_0025
        before_mean = float(preds.mean())
        preds = quantile_match_to_test_pmf(preds, _TEST_PMF_0025)
        after_mean = float(preds.mean())
        print(f"Quantile-matched to P_test : mean shift {before_mean:.3f} → {after_mean:.3f}")

    df["FaceOcclusion"] = preds
    if submission_format:
        df["gender"] = "x"
        df[[image_col, "FaceOcclusion", "gender"]].to_csv(output_csv, index=False)
    else:
        df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} predictions to {output_csv}")

    metadata = _collect_metadata(
        model_uri=model_uri,
        tracking_uri=tracking_uri,
        predict_options={
            "use_tta": use_tta,
            "bias_correction": bias_correction,
            "match_test_pmf": match_test_pmf,
            "batch_size": batch_size,
            "submission_format": submission_format,
            "input_csv": input_csv,
        },
    )
    meta_path = _save_metadata(metadata, output_csv)
    _print_metadata_summary(metadata)
    print(f"Metadata saved to {meta_path}")


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
        match_test_pmf=CONFIG["match_test_pmf"],
    )
