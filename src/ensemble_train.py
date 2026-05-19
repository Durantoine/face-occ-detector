from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from sklearn.model_selection import StratifiedKFold

from src.data.dataset import _load_data
from src.train import train
from src.utils.config import load_architecture_config
from src.utils.environment import setup_environment
from src.utils.mlflow_utils import get_or_create_experiment

setup_environment()

CONFIG: Dict[str, Any] = {
    "architecture": "vit-base-face-occ_optuna_best",
    "n_folds": 5,
    "data_dir": None,
    "data_csv": None,
    "output_dir": "./results/ensemble",
    "tracking_uri": "sqlite:///mlflow.db",
    "mlflow_experiment": "face-occ-ensemble",
    "seed": 42,
    "test_data_csv": None,
}

FOLDS_DIR = Path("data/folds")


def _resolve_data_path(architecture_name: str, data_dir: Optional[str], data_csv: Optional[str]) -> str:
    if data_csv:
        return data_csv
    if data_dir:
        return data_dir
    data_cfg = load_architecture_config(architecture_name).to_dict().get("data", {})
    return data_cfg.get("data_csv") or data_cfg.get("data_dir") or ""


def _build_folds(df: pd.DataFrame, n_folds: int, seed: int) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [
        (df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True))
        for train_idx, val_idx in skf.split(df, df["label"])
    ]


def _write_fold_csvs(folds: List[Tuple[pd.DataFrame, pd.DataFrame]], arch: str) -> List[Tuple[str, str]]:
    out_dir = FOLDS_DIR / arch
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (train_df, val_df) in enumerate(folds):
        train_csv = out_dir / f"fold_{i}_train.csv"
        val_csv = out_dir / f"fold_{i}_val.csv"
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        paths.append((str(train_csv), str(val_csv)))
    return paths


def ensemble_train(
    architecture: str,
    n_folds: int = 5,
    data_dir: Optional[str] = None,
    data_csv: Optional[str] = None,
    output_dir: str = "./results/ensemble",
    tracking_uri: str = "sqlite:///mlflow.db",
    mlflow_experiment: str = "face-occ-ensemble",
    seed: int = 42,
    test_data_csv: Optional[str] = None,
) -> Dict[str, Any]:
    data_path = _resolve_data_path(architecture, data_dir, data_csv)
    if not data_path:
        raise ValueError("No data path: provide data_csv/data_dir or set them in the YAML")

    df_all = pd.concat(_load_data(data_path, split_ratio=0), ignore_index=True)
    if "label" not in df_all.columns:
        raise ValueError("Loaded data must have a 'label' column after _load_data")

    folds = _build_folds(df_all, n_folds=n_folds, seed=seed)
    fold_paths = _write_fold_csvs(folds, arch=architecture)
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
    for i, (train_csv, val_csv) in enumerate(fold_paths):
        child = client.create_run(
            experiment_id=exp_id,
            run_name=f"{architecture}_fold{i}",
            tags={MLFLOW_PARENT_RUN_ID: parent_run_id, "fold": str(i), "ensemble_member": str(i)},
        )
        run_id = child.info.run_id
        client.log_param(run_id, "fold", i)
        client.log_param(run_id, "ensemble_member", i)

        eval_loss, f1, gap, train_loss, model_uri = train(
            architecture_name=architecture,
            data_csv=train_csv,
            val_data_csv=val_csv,
            output_dir=f"{output_dir}/fold_{i}",
            mlflow_tracking_uri=tracking_uri,
            mlflow_run_id=run_id,
            seed=seed + i,
            test_data_csv=test_data_csv,
        )
        client.log_metric(run_id, "fold_f1_macro", f1)
        client.log_metric(run_id, "fold_eval_loss", eval_loss)
        client.set_terminated(run_id, "FINISHED")
        fold_results.append({
            "fold": i, "f1_macro": f1, "eval_loss": eval_loss,
            "overfit_gap": gap, "model_uri": model_uri,
        })
        print(f"Fold {i}: F1={f1:.4f}  loss={eval_loss:.4f}")

    cv_f1 = float(np.mean([r["f1_macro"] for r in fold_results]))
    cv_f1_std = float(np.std([r["f1_macro"] for r in fold_results]))
    client.log_metric(parent_run_id, "cv_f1_macro", cv_f1)
    client.log_metric(parent_run_id, "cv_f1_macro_std", cv_f1_std)
    client.set_terminated(parent_run_id, "FINISHED")

    print(f"\nCV F1 = {cv_f1:.4f} ± {cv_f1_std:.4f}")
    return {"parent_run_id": parent_run_id, "cv_f1": cv_f1, "cv_f1_std": cv_f1_std, "folds": fold_results}


if __name__ == "__main__":
    ensemble_train(
        architecture=CONFIG["architecture"],
        n_folds=CONFIG["n_folds"],
        data_dir=CONFIG["data_dir"],
        data_csv=CONFIG["data_csv"],
        output_dir=CONFIG["output_dir"],
        tracking_uri=CONFIG["tracking_uri"],
        mlflow_experiment=CONFIG["mlflow_experiment"],
        seed=CONFIG["seed"],
        test_data_csv=CONFIG["test_data_csv"],
    )
