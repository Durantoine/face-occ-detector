import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from sklearn.model_selection import StratifiedKFold

from src.data.dataset import (
    DEFAULT_GENDER_COL,
    DEFAULT_IMAGE_COL,
    DEFAULT_LABEL_COL,
    _load_data,
)
from src.train import train
from src.utils.config import load_architecture_config
from src.utils.environment import setup_environment
from src.utils.losses import stratify_key
from src.utils.mlflow_utils import get_or_create_experiment

setup_environment()

CONFIG: Dict[str, Any] = {
    "architecture": os.environ.get("FACE_OCC_ARCH", "dinov3-vits16-face-occ_optuna_best"),
    "n_folds": 5,
    "data_csv": "data/raw/train.csv",
    "output_dir": "./results/ensemble",
    "tracking_uri": "sqlite:///mlflow.db",
    "mlflow_experiment": "face-occ-ensemble",
    "seed": 42,
    "test_data_csv": None,
}

FOLDS_DIR = Path("data/folds")


def _build_folds(df: pd.DataFrame, n_folds: int, seed: int) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    keys = stratify_key(df["gender"], df["FaceOcclusion"], n_buckets=10)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [
        (df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True))
        for train_idx, val_idx in skf.split(df, keys)
    ]


def _write_folds(
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    arch: str,
    image_col: str,
    label_col: str,
    gender_col: str,
) -> List[Tuple[str, str]]:
    out_dir = FOLDS_DIR / arch
    out_dir.mkdir(parents=True, exist_ok=True)
    rename = {"image_path": image_col, "FaceOcclusion": label_col, "gender": gender_col}
    paths = []
    for i, (tr, va) in enumerate(folds):
        tr_csv = out_dir / f"fold_{i}_train.csv"
        va_csv = out_dir / f"fold_{i}_val.csv"
        tr.rename(columns=rename).to_csv(tr_csv, index=False)
        va.rename(columns=rename).to_csv(va_csv, index=False)
        paths.append((str(tr_csv), str(va_csv)))
    return paths


def ensemble_train(
    architecture: str,
    n_folds: int = 5,
    data_csv: Optional[str] = None,
    output_dir: str = "./results/ensemble",
    tracking_uri: str = "sqlite:///mlflow.db",
    mlflow_experiment: str = "face-occ-ensemble",
    seed: int = 42,
    test_data_csv: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_architecture_config(architecture).to_dict()
    data_cfg = cfg.get("data", {})
    image_col = data_cfg.get("image_col", DEFAULT_IMAGE_COL)
    label_col = data_cfg.get("label_col", DEFAULT_LABEL_COL)
    gender_col = data_cfg.get("gender_col", DEFAULT_GENDER_COL)

    data_path = data_csv or data_cfg.get("data_csv")
    if not data_path:
        raise ValueError("No data path: provide data_csv or set it in the YAML")

    df_norm, _ = _load_data(
        data_path,
        image_col=image_col, label_col=label_col, gender_col=gender_col,
        split_ratio=0,
    )
    if not {"FaceOcclusion", "gender", "image_path"}.issubset(df_norm.columns):
        raise ValueError("Loaded data must have image_path, FaceOcclusion, gender after normalisation")

    folds = _build_folds(df_norm, n_folds=n_folds, seed=seed)
    fold_paths = _write_folds(folds, architecture, image_col, label_col, gender_col)
    print(f"Wrote {n_folds} fold CSVs to {FOLDS_DIR / architecture}")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    exp_id = get_or_create_experiment(client, mlflow_experiment)

    parent = client.create_run(
        experiment_id=exp_id,
        run_name=f"ensemble_{architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parent_run_id = parent.info.run_id
    client.log_param(parent_run_id, "architecture", architecture)
    client.log_param(parent_run_id, "n_folds", n_folds)
    print(f"Ensemble parent run: {parent_run_id}")

    fold_results: List[Dict[str, Any]] = []
    for i, (tr_csv, va_csv) in enumerate(fold_paths):
        child = client.create_run(
            experiment_id=exp_id,
            run_name=f"{architecture}_fold{i}",
            tags={MLFLOW_PARENT_RUN_ID: parent_run_id, "fold": str(i), "ensemble_member": str(i)},
        )
        run_id = child.info.run_id
        client.log_param(run_id, "fold", i)

        eval_loss, score, err_diff, model_uri = train(
            architecture_name=architecture,
            data_csv=tr_csv,
            val_data_csv=va_csv,
            output_dir=f"{output_dir}/fold_{i}",
            mlflow_tracking_uri=tracking_uri,
            mlflow_run_id=run_id,
            seed=seed + i,
            test_data_csv=test_data_csv,
        )
        client.log_metric(run_id, "fold_score", score)
        client.log_metric(run_id, "fold_err_diff", err_diff)
        client.log_metric(run_id, "fold_eval_loss", eval_loss)
        client.set_terminated(run_id, "FINISHED")
        fold_results.append({
            "fold": i, "score": score, "eval_loss": eval_loss,
            "err_diff": err_diff, "model_uri": model_uri,
        })
        print(f"Fold {i}: score={score:.5f}  err_diff={err_diff:.5f}")

    cv_score = float(np.mean([r["score"] for r in fold_results]))
    cv_score_std = float(np.std([r["score"] for r in fold_results]))
    client.log_metric(parent_run_id, "cv_score", cv_score)
    client.log_metric(parent_run_id, "cv_score_std", cv_score_std)
    client.set_terminated(parent_run_id, "FINISHED")

    print(f"\nCV score = {cv_score:.5f} ± {cv_score_std:.5f}")
    return {"parent_run_id": parent_run_id, "cv_score": cv_score, "cv_score_std": cv_score_std, "folds": fold_results}


if __name__ == "__main__":
    ensemble_train(
        architecture=CONFIG["architecture"],
        n_folds=CONFIG["n_folds"],
        data_csv=CONFIG["data_csv"],
        output_dir=CONFIG["output_dir"],
        tracking_uri=CONFIG["tracking_uri"],
        mlflow_experiment=CONFIG["mlflow_experiment"],
        seed=CONFIG["seed"],
        test_data_csv=CONFIG["test_data_csv"],
    )
