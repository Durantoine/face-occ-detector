import os
import shutil
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

from src.data.dataset import (
    DEFAULT_GENDER_COL,
    DEFAULT_IMAGE_COL,
    DEFAULT_LABEL_COL,
    FaceOccDataset,
    _load_data,
    create_balanced_sampler,
)
from src.data.transforms import build_train_transform
from src.models.dinov3_loader import get_image_processor
from src.models.face_occ_classifier import FaceOccRegressor
from src.training.callbacks import MlflowClientCallback, make_ema_callback_from_cfg
from src.utils.config import load_architecture_config
from src.utils.environment import setup_environment
from src.utils.losses import (
    GroupDROLoss,
    WeightedMSELoss,
    build_cell_weights,
    build_importance_weights,
    make_sampler_keys,
)
from src.utils.metrics import compute_metrics
from src.utils.mlflow_utils import log_metrics as ml_log_metrics
from src.utils.mlflow_utils import log_params as ml_log_params

setup_environment()

CONFIG: Dict[str, Any] = {
    "architecture": os.environ.get("FACE_OCC_ARCH", "dinov3-vits16-face-occ"),
    "data_csv": "data/raw/train.csv",
    "val_data_csv": None,
    "output_dir": "./results",
    "tracking_uri": "sqlite:///mlflow.db",
    "use_mlflow": True,
    "resume_from": None,
}

_NON_HF_TRAIN_KEYS = {
    "early_stopping_patience", "metric_for_best_model", "greater_is_better", "seed",
    "augmentation_level", "ema_decay", "ema_warmup_steps",
    "sampler_strategy", "loss_type", "loss_focal_gamma", "loss_fairness_lambda",
    "loss_importance_reweight", "loss_gender_reweight", "loss_cell_reweight",
    "loss_patch_mil_alpha",
    "group_dro_alpha", "layer_decay",
}


def _unwrap(module: Any) -> Any:
    return getattr(module, "module", module)


class WeightedMSETrainer(Trainer):
    def __init__(
        self,
        loss_type: str = "weighted_mse",
        focal_gamma: float = 0.0,
        fairness_lambda: float = 0.0,
        group_dro_alpha: float = 0.5,
        importance_pmf_ratio: Optional[Any] = None,
        gender_class_weights: Optional[Any] = None,
        cell_class_weights: Optional[Any] = None,
        patch_mil_alpha: float = 0.0,
        train_sampler: Optional[Any] = None,
        layer_decay: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._custom_train_sampler = train_sampler
        self._layer_decay = layer_decay
        self._patch_mil_alpha = float(patch_mil_alpha)
        if loss_type == "group_dro":
            self.loss_fct = GroupDROLoss(alpha=group_dro_alpha)
            print(f"GroupDROLoss: alpha={group_dro_alpha}")
        else:
            self.loss_fct = WeightedMSELoss(
                focal_gamma=focal_gamma,
                fairness_lambda=fairness_lambda,
                importance_pmf_ratio=importance_pmf_ratio,
                gender_class_weights=gender_class_weights,
                cell_class_weights=cell_class_weights,
            )
            tags = []
            if importance_pmf_ratio is not None:
                tags.append(f"importance_reweight=on (mean={float(importance_pmf_ratio.mean()):.2f})")
            if gender_class_weights is not None:
                tags.append(f"gender_reweight=on (F={gender_class_weights[0]:.2f}, M={gender_class_weights[1]:.2f})")
            if cell_class_weights is not None:
                tags.append(f"cell_reweight=on (max={float(cell_class_weights.max()):.2f}, min={float(cell_class_weights.min()):.2f})")
            extra = ", " + ", ".join(tags) if tags else ""
            print(f"WeightedMSELoss: focal_gamma={focal_gamma}, fairness_lambda={fairness_lambda}{extra}")

    def _get_train_sampler(self) -> Any:
        return self._custom_train_sampler if self._custom_train_sampler is not None else super()._get_train_sampler()

    def compute_loss(self, model: Any, inputs: Dict[str, Any], return_outputs: bool = False) -> Any:
        labels = inputs["labels"]
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        preds = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        loss = self.loss_fct(preds, labels)
        if self._patch_mil_alpha > 0 and isinstance(outputs, dict) and "patch_pred" in outputs:
            aux = self.loss_fct(outputs["patch_pred"], labels)
            loss = loss + self._patch_mil_alpha * aux
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is not None:
            return self.optimizer
        if self._layer_decay >= 1.0:
            return super().create_optimizer()

        base_lr = self.args.learning_rate
        wd = self.args.weight_decay
        inner = _unwrap(self.model)
        backbone = inner.backbone

        if hasattr(backbone, "blocks"):
            n_layers = len(backbone.blocks)
            block_attr = "blocks"
        elif hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
            n_layers = len(backbone.encoder.layer)
            block_attr = "encoder.layer"
        else:
            return super().create_optimizer()

        decay = self._layer_decay
        groups: List[Dict[str, Any]] = []

        embed_params = [p for n, p in self.model.named_parameters()
                        if "backbone" in n and any(k in n for k in ("embed", "pos_embed", "cls_token"))]
        if embed_params:
            groups.append({"params": embed_params, "lr": base_lr * decay ** (n_layers + 1), "weight_decay": wd})

        for i in range(n_layers):
            pattern = f"backbone.{block_attr}.{i}."
            layer_params = [p for n, p in self.model.named_parameters() if pattern in n]
            if layer_params:
                groups.append({"params": layer_params, "lr": base_lr * decay ** (n_layers - i), "weight_decay": wd})

        head_params = [p for n, p in self.model.named_parameters() if "backbone" not in n]
        if head_params:
            groups.append({"params": head_params, "lr": base_lr, "weight_decay": wd})

        print(f"LLRD: {len(groups)} groups, LR ∈ [{groups[0]['lr']:.2e}, {base_lr:.2e}]")
        self.optimizer = torch.optim.AdamW(
            groups, lr=base_lr, weight_decay=wd,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
        )
        return self.optimizer


def _resolve_data_path(data_cfg: Dict[str, Any], data_csv: Optional[str]) -> str:
    return data_csv or data_cfg.get("data_csv") or ""


def _load_train_val(
    data_cfg: Dict[str, Any],
    data_csv: Optional[str],
    val_data_csv: Optional[str],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    image_col = data_cfg.get("image_col", DEFAULT_IMAGE_COL)
    label_col = data_cfg.get("label_col", DEFAULT_LABEL_COL)
    gender_col = data_cfg.get("gender_col", DEFAULT_GENDER_COL)
    extra_train = data_cfg.get("extra_train_csv")

    train_path = _resolve_data_path(data_cfg, data_csv)
    if not train_path:
        raise ValueError("Provide data_csv via arg or YAML data.data_csv")

    common = dict(image_col=image_col, label_col=label_col, gender_col=gender_col)
    if val_data_csv:
        train_df, _ = _load_data(train_path, **common, extra_train_csv=extra_train, split_ratio=0)
        val_df, _ = _load_data(val_data_csv, **common, split_ratio=0)
        print(f"Pre-split: train={len(train_df):,} val={len(val_df):,}")
        return train_df, val_df

    return _load_data(train_path, **common, extra_train_csv=extra_train, seed=seed)


def _build_datasets(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    processor: Any,
    image_base_dir: Optional[str],
    augmentation_level: str,
) -> Tuple[FaceOccDataset, FaceOccDataset]:
    transform = build_train_transform(augmentation_level)
    train_ds = FaceOccDataset(
        image_paths=train_data["image_path"].tolist(),
        targets=train_data["FaceOcclusion"].astype(float).tolist(),
        genders=train_data["gender"].astype(float).tolist(),
        processor=processor,
        image_base_dir=image_base_dir,
        transform=transform,
    )
    val_ds = FaceOccDataset(
        image_paths=val_data["image_path"].tolist(),
        targets=val_data["FaceOcclusion"].astype(float).tolist(),
        genders=val_data["gender"].astype(float).tolist(),
        processor=processor,
        image_base_dir=image_base_dir,
        transform=None,
    )
    return train_ds, val_ds


def _start_or_attach_run(
    cfg: Dict[str, Any],
    tracking_uri: str,
    mlflow_run_id: Optional[str],
    mlflow_experiment: str,
    use_mlflow: bool,
) -> Tuple[Optional[MlflowClient], Optional[str], bool]:
    if not use_mlflow:
        return None, None, False
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow_run_id is not None:
        return MlflowClient(tracking_uri=tracking_uri), mlflow_run_id, True
    mlflow.set_experiment(mlflow_experiment)
    run_name = f"{cfg['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.start_run(run_name=run_name)
    return None, mlflow.active_run().info.run_id, False


def _save_model_to_mlflow(
    trainer: Trainer,
    processor: Any,
    run_id: str,
    model_register_name: str,
) -> str:
    import torch.nn as _nn

    while mlflow.active_run():
        mlflow.end_run()
    mlflow.start_run(run_id=run_id)

    raw = _unwrap(trainer.model).cpu() if not hasattr(trainer, "accelerator") or not trainer.accelerator else trainer.accelerator.unwrap_model(trainer.model).cpu()

    _orig = _nn.Module.__dict__.get("__getstate__")
    _nn.Module.__getstate__ = lambda self: {k: v for k, v in self.__dict__.items()}
    try:
        info = mlflow.pytorch.log_model(
            pytorch_model=raw,
            artifact_path="model",
            registered_model_name=model_register_name,
        )
        model_uri = info.model_uri
    finally:
        if _orig is not None:
            _nn.Module.__getstate__ = _orig
        else:
            try:
                delattr(_nn.Module, "__getstate__")
            except AttributeError:
                pass

    with tempfile.TemporaryDirectory() as tmp:
        processor.save_pretrained(tmp)
        mlflow.log_artifacts(tmp, "processor")

    mlflow.end_run()
    return model_uri


def _safe_mean(series) -> float:
    v = float(series.mean()) if len(series) > 0 else 0.0
    return 0.0 if v != v else v


def train(
    architecture_name: str,
    data_csv: Optional[str] = None,
    val_data_csv: Optional[str] = None,
    output_dir: str = "./results",
    mlflow_tracking_uri: str = "sqlite:///mlflow.db",
    mlflow_experiment: str = "face-occlusion-detection",
    use_mlflow: bool = True,
    resume_from_checkpoint: Optional[str] = None,
    seed: int = 42,
    val_seed: Optional[int] = None,
    mlflow_run_id: Optional[str] = None,
    test_data_csv: Optional[str] = None,
) -> Tuple[float, float, float, str]:
    cfg = load_architecture_config(architecture_name).to_dict()
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    output_dim = model_cfg.get("output_dim", 1)
    best_metric = train_cfg.get("metric_for_best_model", "eval_score")
    greater_is_better = bool(train_cfg.get("greater_is_better", False))
    image_base_dir = data_cfg.get("image_base_dir")
    augmentation_level = train_cfg.get("augmentation_level", "medium")
    sampler_strategy = train_cfg.get("sampler_strategy", "gender")
    loss_type = train_cfg.get("loss_type", "weighted_mse")

    client, run_id, use_client = _start_or_attach_run(
        cfg, mlflow_tracking_uri, mlflow_run_id, mlflow_experiment, use_mlflow,
    )

    model_name = model_cfg.get("model_name", "dinov3_vits16")
    ml_log_params(client, run_id, {"architecture": cfg["name"]})
    ml_log_params(client, run_id, {f"model_{k}": v for k, v in model_cfg.items()})
    ml_log_params(client, run_id, {f"train_{k}": v for k, v in train_cfg.items()})
    ml_log_params(client, run_id, {f"data_{k}": v for k, v in data_cfg.items()})

    processor = get_image_processor(model_name)
    train_data, val_data = _load_train_val(data_cfg, data_csv, val_data_csv, val_seed or seed)
    train_dataset, val_dataset = _build_datasets(
        train_data, val_data, processor, image_base_dir, augmentation_level,
    )

    train_sampler = None
    if sampler_strategy != "none" and "gender" in train_data.columns:
        keys = make_sampler_keys(train_data, strategy=sampler_strategy, n_buckets=10)
        n_groups = int(keys.max()) + 1
        train_sampler = create_balanced_sampler(keys.tolist(), num_groups=n_groups)
        print(f"Sampler '{sampler_strategy}': {n_groups} groups, {train_sampler.num_samples} samples/epoch")

    importance_pmf_ratio = None
    if train_cfg.get("loss_importance_reweight", False) and loss_type == "weighted_mse":
        label_col = data_cfg.get("label_col", DEFAULT_LABEL_COL)
        importance_pmf_ratio = build_importance_weights(train_data[label_col].astype(float).values)
        ml_log_params(client, run_id, {
            "loss_importance_reweight": True,
            "loss_importance_pmf_ratio": ",".join(f"{x:.3f}" for x in importance_pmf_ratio.tolist()),
        })
        print(f"Importance reweight ratios (test/train) per 0.025-bin (20 bins): {importance_pmf_ratio.round(3).tolist()}")

    gender_class_weights = None
    if train_cfg.get("loss_gender_reweight", False) and loss_type == "weighted_mse" and "gender" in train_data.columns:
        n_f = max(int((train_data["gender"] < 0.5).sum()), 1)
        n_m = max(int((train_data["gender"] >= 0.5).sum()), 1)
        w_f = 1.0 / (2.0 * n_f / (n_f + n_m))
        w_m = 1.0 / (2.0 * n_m / (n_f + n_m))
        norm = (w_f + w_m) / 2.0
        gender_class_weights = np.array([w_f / norm, w_m / norm], dtype=np.float64)
        ml_log_params(client, run_id, {
            "loss_gender_reweight": True,
            "loss_gender_weight_F": float(gender_class_weights[0]),
            "loss_gender_weight_M": float(gender_class_weights[1]),
        })
        print(f"Gender reweight: F={gender_class_weights[0]:.3f}, M={gender_class_weights[1]:.3f} (mean=1.0)")

    cell_class_weights = None
    if train_cfg.get("loss_cell_reweight", False) and loss_type == "weighted_mse" and "gender" in train_data.columns:
        label_col = data_cfg.get("label_col", DEFAULT_LABEL_COL)
        cell_class_weights = build_cell_weights(
            train_data[label_col].astype(float).values,
            train_data["gender"].astype(float).values,
        )
        ml_log_params(client, run_id, {
            "loss_cell_reweight": True,
            "loss_cell_weights_max": float(cell_class_weights.max()),
            "loss_cell_weights_min": float(cell_class_weights.min()),
            "loss_cell_weights_F_bin0": float(cell_class_weights[0, 0]),
            "loss_cell_weights_M_bin0": float(cell_class_weights[1, 0]),
        })
        print(f"Cell reweight (2×20, 1/sqrt(count) normalized): max={cell_class_weights.max():.3f}, min={cell_class_weights.min():.3f}")

    n_train_f = int((train_data["gender"] < 0.5).sum())
    n_train_m = int((train_data["gender"] >= 0.5).sum())
    n_val_f = int((val_data["gender"] < 0.5).sum())
    n_val_m = int((val_data["gender"] >= 0.5).sum())
    ml_log_params(client, run_id, {
        "seed": seed,
        "num_train": len(train_dataset),
        "num_val": len(val_data),
        "num_train_female": n_train_f,
        "num_train_male": n_train_m,
        "num_val_female": n_val_f,
        "num_val_male": n_val_m,
        "train_gender_ratio_M_over_F": round(n_train_m / max(n_train_f, 1), 3),
    })
    ml_log_metrics(client, run_id, {
        "data_train_occ_mean": _safe_mean(train_data["FaceOcclusion"]),
        "data_train_occ_std": float(train_data["FaceOcclusion"].std() or 0.0),
        "data_val_occ_mean": _safe_mean(val_data["FaceOcclusion"]),
        "data_val_occ_std": float(val_data["FaceOcclusion"].std() or 0.0),
        "data_train_occ_female_mean": _safe_mean(train_data.loc[train_data["gender"] < 0.5, "FaceOcclusion"]),
        "data_train_occ_male_mean": _safe_mean(train_data.loc[train_data["gender"] >= 0.5, "FaceOcclusion"]),
        "data_val_occ_female_mean": _safe_mean(val_data.loc[val_data["gender"] < 0.5, "FaceOcclusion"]),
        "data_val_occ_male_mean": _safe_mean(val_data.loc[val_data["gender"] >= 0.5, "FaceOcclusion"]),
    })

    model = (
        FaceOccRegressor.load_from_mlflow(resume_from_checkpoint, output_dim=output_dim)
        if resume_from_checkpoint
        else FaceOccRegressor(
            model_name=model_name,
            output_dim=output_dim,
            hidden_dropout_prob=model_cfg.get("hidden_dropout_prob", 0.1),
            pooling=model_cfg.get("pooling", "cls"),
            projection_size=model_cfg.get("projection_size"),
            output_activation=model_cfg.get("output_activation", "sigmoid"),
        )
    )

    init_backbone_from = model_cfg.get("init_backbone_from")
    if init_backbone_from:
        import re
        import mlflow as _ml
        print(f"Loading pretrained backbone from {init_backbone_from}")
        pretrained = _ml.pytorch.load_model(init_backbone_from)
        missing, unexpected = model.backbone.load_state_dict(pretrained.state_dict(), strict=False)
        print(f"  loaded ({len(missing)} missing, {len(unexpected)} unexpected keys)")
        m = re.match(r"runs:/([^/]+)/", init_backbone_from)
        pretrain_run_id = m.group(1) if m else None
        ml_log_params(client, run_id, {
            "init_backbone_from": init_backbone_from,
            "init_backbone_pretrain_run_id": pretrain_run_id,
            "init_backbone_missing_keys": len(missing),
            "init_backbone_unexpected_keys": len(unexpected),
        })
        if pretrain_run_id:
            try:
                pre_run = _ml.tracking.MlflowClient().get_run(pretrain_run_id)
                pre_params = {f"pretrain_{k}": v for k, v in pre_run.data.params.items()}
                ml_log_params(client, run_id, pre_params)
                client.set_tag(run_id, "pretrain_run_id", pretrain_run_id)
                print(f"  logged {len(pre_params)} pretrain_* params from run {pretrain_run_id}")
            except Exception as e:
                print(f"  WARNING: could not fetch pretrain run params: {e}")

    forwarded = {k: v for k, v in train_cfg.items() if k not in _NON_HF_TRAIN_KEYS}
    forwarded.setdefault("fp16", True)
    if not torch.cuda.is_available():
        if forwarded.get("bf16") or forwarded.get("fp16"):
            print(f"WARNING: non-CUDA device — disabling bf16/fp16")
        forwarded["bf16"] = False
        forwarded["fp16"] = False
    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to=["mlflow"] if (use_mlflow and not use_client) else [],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=greater_is_better,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=seed,
        remove_unused_columns=False,
        **forwarded,
    )

    callbacks: List[TrainerCallback] = []
    patience = train_cfg.get("early_stopping_patience", 3)
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
    if use_client and client and run_id:
        callbacks.append(MlflowClientCallback(client, run_id))
    ema_cb = make_ema_callback_from_cfg(train_cfg)
    if ema_cb is not None:
        callbacks.append(ema_cb)
        print(f"EMA enabled: decay={ema_cb.decay}, warmup_steps={ema_cb.warmup_steps}")

    trainer = WeightedMSETrainer(
        loss_type=loss_type,
        focal_gamma=train_cfg.get("loss_focal_gamma", 0.0),
        fairness_lambda=train_cfg.get("loss_fairness_lambda", 0.0),
        group_dro_alpha=train_cfg.get("group_dro_alpha", 0.5),
        importance_pmf_ratio=importance_pmf_ratio,
        gender_class_weights=gender_class_weights,
        cell_class_weights=cell_class_weights,
        patch_mil_alpha=train_cfg.get("loss_patch_mil_alpha", 0.0),
        train_sampler=train_sampler,
        layer_decay=train_cfg.get("layer_decay", 1.0),
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        callbacks=callbacks or None,
    )

    print(f"Training {cfg['name']} (best_metric={best_metric})")
    trainer.train()

    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    eval_score = eval_results.get("eval_score", 0.0)
    err_diff = eval_results.get("eval_err_diff", 0.0)
    err_F = eval_results.get("eval_err_F", 0.0)
    err_M = eval_results.get("eval_err_M", 0.0)
    print(f"score={eval_score:.5f} | err_diff={err_diff:.5f} | loss={eval_loss:.5f}")

    if test_data_csv and trainer.is_world_process_zero():
        try:
            test_df, _ = _load_data(
                test_data_csv,
                image_col=data_cfg.get("image_col", DEFAULT_IMAGE_COL),
                label_col=data_cfg.get("label_col", DEFAULT_LABEL_COL),
                gender_col=data_cfg.get("gender_col", DEFAULT_GENDER_COL),
                split_ratio=0,
            )
            test_ds = FaceOccDataset(
                image_paths=test_df["image_path"].tolist(),
                targets=test_df["FaceOcclusion"].astype(float).tolist(),
                genders=test_df["gender"].astype(float).tolist(),
                processor=processor, image_base_dir=image_base_dir, transform=None,
            )
            test_res = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
            print(f"Test score={test_res.get('test_score', 0):.5f}")
            ml_log_metrics(client, run_id, {
                "test_score": test_res.get("test_score", 0.0),
                "test_err_F": test_res.get("test_err_F", 0.0),
                "test_err_M": test_res.get("test_err_M", 0.0),
            })
        except Exception as e:
            print(f"WARNING test eval: {e}")

    if not trainer.is_world_process_zero():
        return eval_loss, eval_score, err_diff, "", err_F, err_M

    ml_log_metrics(client, run_id, {
        "val_score": eval_score,
        "val_err_F": eval_results.get("eval_err_F", 0.0),
        "val_err_M": eval_results.get("eval_err_M", 0.0),
        "val_err_diff": err_diff,
        "final_eval_loss": eval_loss,
    })
    if use_mlflow:
        artifact = f"configs/architectures/{architecture_name}.yaml"
        if use_client and client and run_id:
            client.log_artifact(run_id, artifact)
        elif mlflow.active_run():
            mlflow.log_artifact(artifact)

    model_uri = ""
    if use_mlflow:
        try:
            model_uri = _save_model_to_mlflow(trainer, processor, run_id, f"{cfg['name']}")
            print(f"Model saved: {model_uri}")
        except Exception as e:
            print(f"ERROR saving model: {e}")
            if mlflow.active_run():
                mlflow.end_run()

    if output_dir and output_dir != "./results":
        shutil.rmtree(output_dir, ignore_errors=True)

    return eval_loss, eval_score, err_diff, model_uri, err_F, err_M


if __name__ == "__main__":
    train(
        architecture_name=CONFIG["architecture"],
        data_csv=CONFIG["data_csv"],
        val_data_csv=CONFIG["val_data_csv"],
        output_dir=CONFIG["output_dir"],
        mlflow_tracking_uri=CONFIG["tracking_uri"],
        use_mlflow=CONFIG["use_mlflow"],
        resume_from_checkpoint=CONFIG["resume_from"],
    )
