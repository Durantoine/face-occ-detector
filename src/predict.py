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
    # v19: per-gender calibration via pickled artifact from training run.
    # If apply_calibration=True, predict.py downloads calibrators/calibrators.pkl
    # from the MLflow run + reads gender from gender_csv (test_students_with_gender.csv)
    # and applies best (cal, α) blend per sample.
    "apply_calibration": True,
    "gender_csv": "data/raw/test_students_with_gender.csv",
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
            # v16: single-axis correction (fallback to v10 axis1/2 for old runs)
            "correction_strength":   params.get("correction_strength",
                                                 params.get("axis1_power")),
            "feature_fairness":      params.get("feature_fairness"),
            "ot_method":             params.get("ot_method"),
            "ot_lambda":             params.get("ot_lambda"),
            "sinkhorn_eps":          params.get("sinkhorn_eps"),
            "adv_lambda":            params.get("adv_lambda"),
            "mmd_lambda":            params.get("mmd_lambda"),
            "loss_focal_gamma":      params.get("loss_focal_gamma"),
            "loss_lambda_max":       params.get("loss_lambda_max"),
            "learning_rate":         params.get("learning_rate"),
            "num_train_epochs":      params.get("num_train_epochs"),
            "augmentation_level":    params.get("augmentation_level"),
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
    print(f"  Correction α       : {key.get('correction_strength')}    (v16: single-axis under H_C)")
    print(f"  Feature fairness   : {key.get('feature_fairness')}  "
          f"ot_method={key.get('ot_method')}  λ={key.get('ot_lambda') or key.get('adv_lambda')}")
    if key.get("sinkhorn_eps"):
        print(f"  Sinkhorn ε         : {key.get('sinkhorn_eps')}")
    print(f"  Focal γ            : {key.get('loss_focal_gamma')}")
    print(f"  Lagrangien λ_max   : {key.get('loss_lambda_max')}")
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


def _load_calibrators_from_run(model_uri: str, tracking_uri: str) -> Optional[Dict[str, Any]]:
    """v19: download calibrators.pkl artifact from training run. Returns the unpickled
    payload {calibrators, best_cal_name, best_alpha, ...} or None if absent/error.
    """
    m = re.match(r"runs:/([^/]+)/", model_uri)
    if not m:
        print("  Cannot extract run_id from model_uri → skipping calibrator load")
        return None
    run_id = m.group(1)
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=tracking_uri)
        local_dir = client.download_artifacts(run_id, "calibrators")
        pkl_path = Path(local_dir) / "calibrators.pkl"
        if not pkl_path.exists():
            print(f"  Run {run_id[:8]} has no calibrators.pkl artifact → skipping per-gender cal")
            return None
        import pickle as _pkl
        with open(pkl_path, "rb") as f:
            payload = _pkl.load(f)
        print(f"  Loaded calibrators: {list(payload['calibrators'].keys())}, "
              f"best=({payload['best_cal_name']}, α={payload['best_alpha']:.2f})")
        return payload
    except Exception as e:
        print(f"  WARNING: failed to load calibrators artifact: {e}")
        return None


def _load_test_gender(gender_csv: str, filenames: List[str]) -> Optional[np.ndarray]:
    """Load gender_predicted column from gender_csv, aligned to `filenames` order.
    Returns int array {0=F, 1=M} or None if file missing / column missing.
    """
    csv_path = Path(gender_csv)
    if not csv_path.exists():
        print(f"  gender_csv {gender_csv} not found → skipping per-gender cal")
        return None
    df = pd.read_csv(csv_path)
    if "gender_predicted" not in df.columns or "filename" not in df.columns:
        print(f"  {gender_csv} missing required columns → skipping per-gender cal")
        return None
    # Align to input filenames order via lookup
    mapping = dict(zip(df["filename"].astype(str), df["gender_predicted"].astype(int)))
    out = np.array([mapping.get(str(fn), -1) for fn in filenames], dtype=int)
    n_unknown = int((out < 0).sum())
    if n_unknown > 0:
        print(f"  WARNING: {n_unknown:,}/{len(filenames):,} test filenames not in {gender_csv}; "
              f"defaulting to majority M=1")
        out[out < 0] = 1
    n_f, n_m = int((out == 0).sum()), int((out == 1).sum())
    print(f"  Gender from {gender_csv}: F={n_f:,} M={n_m:,}")
    return out


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
    apply_calibration: bool = True,
    gender_csv: str = "data/raw/test_students_with_gender.csv",
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

    # === v19: per-gender calibration apply ===
    # Loads the calibrators artifact from the training run + the gender CSV (predicted via
    # MID lookup + DINOv3/Sapiens linear probe). Applies best (cal, α) blend per sample.
    # Graceful fallback to old behavior if either artifact is missing.
    if apply_calibration:
        print("=== Applying per-gender calibration (v19) ===")
        cal_payload = _load_calibrators_from_run(model_uri, tracking_uri)
        gender_test = _load_test_gender(gender_csv, df[image_col].tolist()) if cal_payload else None
        if cal_payload and gender_test is not None:
            best_cal_name = cal_payload["best_cal_name"]
            best_alpha = float(cal_payload["best_alpha"])
            cal = cal_payload["calibrators"][best_cal_name]
            preds_cal = cal.transform(preds, gender_test.astype(float))
            preds_blend = best_alpha * preds_cal + (1.0 - best_alpha) * preds
            preds_blend = np.clip(preds_blend, 0.0, 1.0)
            mean_shift = float(preds_blend.mean() - preds.mean())
            print(f"  Applied {best_cal_name} (α={best_alpha:.2f}, per-gender blend). "
                  f"Mean pred shift: {mean_shift:+.4f}")
            preds = preds_blend
        else:
            print("  → fallback: no per-gender calibration applied (artifact or gender_csv missing)")

    if bias_correction and gender_col in df.columns:
        g = pd.to_numeric(df[gender_col], errors="coerce").fillna(0.5).astype(float)
        preds = apply_bias(preds, g.values,
                           bias_correction.get("delta_f", 0.0),
                           bias_correction.get("delta_m", 0.0))

    if match_test_pmf:
        from src.utils.losses import _TEST_PMF
        before_mean = float(preds.mean())
        preds = quantile_match_to_test_pmf(preds, _TEST_PMF)
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
        apply_calibration=CONFIG.get("apply_calibration", True),
        gender_csv=CONFIG.get("gender_csv", "data/raw/test_students_with_gender.csv"),
    )
