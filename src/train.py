import shutil
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

from src.data.dataset import (
    DynamicAugDataset,
    FaceOccDataset,
    _load_data,
    create_balanced_sampler,
)
from src.data.transforms import build_train_transform
from src.models.dinov3_loader import get_image_processor
from src.models.face_occ_classifier import FaceOccClassifier
from src.training.callbacks import (
    AugResamplerCallback,
    MlflowClientCallback,
    make_ema_callback_from_cfg,
)
from src.utils.config import load_architecture_config
from src.utils.environment import setup_environment
from src.utils.losses import FocalLoss, compute_class_weights
from src.utils.metrics import compute_metrics
from src.utils.mlflow_utils import log_metrics as ml_log_metrics
from src.utils.mlflow_utils import log_params as ml_log_params

setup_environment()

CONFIG: Dict[str, Any] = {
    "architecture": "vit-base-face-occ",
    "data_dir": "data/raw/",
    "data_csv": None,
    "val_data_csv": None,
    "output_dir": "./results",
    "tracking_uri": "sqlite:///mlflow.db",
    "num_labels": None,
    "resume_from": None,
}

_NON_HF_TRAIN_KEYS = {
    "label_smoothing_factor", "early_stopping_patience", "aug_rebalance_ratio",
    "use_class_weights", "use_balanced_sampler", "focal_loss_gamma",
    "min_aug_per_class", "metric_for_best_model", "seed",
    "augmentation_level", "ema_decay", "ema_warmup_steps",
}


class WeightedLossTrainer(Trainer):
    def __init__(
        self,
        class_weights: Optional[Any] = None,
        focal_loss_gamma: float = 0.0,
        label_smoothing: float = 0.0,
        train_sampler: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._custom_train_sampler = train_sampler
        self.loss_fct = (
            FocalLoss(class_weights=class_weights, gamma=focal_loss_gamma, label_smoothing=label_smoothing)
            if (class_weights is not None or focal_loss_gamma > 0)
            else None
        )
        if self.loss_fct:
            print(f"FocalLoss: weights={'yes' if class_weights is not None else 'no'}, gamma={focal_loss_gamma}, ls={label_smoothing}")

    def _get_train_sampler(self) -> Any:
        return self._custom_train_sampler if self._custom_train_sampler is not None else super()._get_train_sampler()

    def compute_loss(self, model: Any, inputs: Dict[str, Any], return_outputs: bool = False) -> Any:
        labels = inputs.get("labels")
        if self.loss_fct is not None and model.training:
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            loss = self.loss_fct(logits, labels)
        else:
            outputs = model(**inputs)
            loss = outputs.get("loss") if isinstance(outputs, dict) else getattr(outputs, "loss", None)
        return (loss, outputs) if return_outputs else loss


def _resolve_data_paths(
    data_cfg: Dict[str, Any],
    data_dir: Optional[str],
    data_csv: Optional[str],
) -> str:
    if data_csv:
        return data_csv
    if data_dir:
        return data_dir
    return data_cfg.get("data_csv") or data_cfg.get("data_dir") or ""


def _load_train_val(
    data_cfg: Dict[str, Any],
    data_dir: Optional[str],
    data_csv: Optional[str],
    val_data_csv: Optional[str],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    label_map = data_cfg.get("label_map")
    subset_col = data_cfg.get("subset_col")
    extra_train = data_cfg.get("extra_train_csv")

    train_path = _resolve_data_paths(data_cfg, data_dir, data_csv)
    if not train_path:
        raise ValueError("Provide data_dir or data_csv (via arg or YAML)")

    if val_data_csv:
        train_df, _ = _load_data(train_path, label_map=label_map, subset_col=subset_col,
                                 extra_train_csv=extra_train, split_ratio=0)
        val_df, _ = _load_data(val_data_csv, label_map=label_map, split_ratio=0)
        print(f"Pre-split: train={len(train_df):,} val={len(val_df):,}")
        return train_df, val_df

    return _load_data(train_path, label_map=label_map, subset_col=subset_col,
                      extra_train_csv=extra_train, seed=seed)


def _maybe_load_aug_pool(aug_data_path: Optional[str], rebalance_ratio: Optional[float]) -> Optional[pd.DataFrame]:
    if not (aug_data_path and rebalance_ratio and rebalance_ratio > 0):
        return None
    raw_pool, _ = _load_data(aug_data_path, split_ratio=0)
    return raw_pool if not raw_pool.empty else None


def _build_datasets(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    aug_pool: Optional[pd.DataFrame],
    processor: Any,
    image_base_dir: Optional[str],
    aug_rebalance_ratio: Optional[float],
    min_aug_per_class: int,
    augmentation_level: str,
    seed: int,
) -> Tuple[Union[FaceOccDataset, DynamicAugDataset], FaceOccDataset, bool]:
    transform = build_train_transform(augmentation_level)
    use_dynamic = aug_pool is not None

    if use_dynamic:
        train_ds: Union[DynamicAugDataset, FaceOccDataset] = DynamicAugDataset(
            real_paths=train_data["image_path"].tolist(),
            real_labels=train_data["label"].tolist(),
            aug_pool=aug_pool,
            processor=processor,
            aug_rebalance_ratio=aug_rebalance_ratio or 0.0,
            seed=seed,
            min_aug_per_class=min_aug_per_class,
            image_base_dir=image_base_dir,
            transform=transform,
        )
    else:
        train_ds = FaceOccDataset(
            image_paths=train_data["image_path"].tolist(),
            labels=train_data["label"].tolist(),
            processor=processor,
            image_base_dir=image_base_dir,
            transform=transform,
        )

    val_ds = FaceOccDataset(
        image_paths=val_data["image_path"].tolist(),
        labels=val_data["label"].tolist(),
        processor=processor,
        image_base_dir=image_base_dir,
        transform=None,
    )
    return train_ds, val_ds, use_dynamic


def _start_or_attach_run(
    cfg: Dict[str, Any],
    tracking_uri: str,
    mlflow_run_id: Optional[str],
    mlflow_experiment: str,
    final_num_labels: int,
) -> Tuple[Optional[MlflowClient], Optional[str], bool]:
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow_run_id is not None:
        return MlflowClient(tracking_uri=tracking_uri), mlflow_run_id, True

    mlflow.set_experiment(mlflow_experiment)
    run_name = f"{cfg['name']}_{final_num_labels}classes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.start_run(run_name=run_name)
    return None, mlflow.active_run().info.run_id, False


def _compute_overfit_stats(
    log_history: List[Dict[str, Any]],
    best_metric: str,
) -> Tuple[float, float, Optional[float]]:
    eval_entries = [h for h in log_history if "eval_loss" in h]
    train_entries = [h for h in log_history if "loss" in h and "eval_loss" not in h]

    overfit_gap = 0.0
    best_epoch: Optional[float] = None
    if eval_entries:
        overfit_gap = max(0.0, eval_entries[-1]["eval_loss"] - min(e["eval_loss"] for e in eval_entries))
        best_entry = (
            min(eval_entries, key=lambda h: h["eval_loss"]) if best_metric == "eval_loss"
            else max(eval_entries, key=lambda h: h.get(best_metric, 0.0))
        )
        best_epoch = best_entry.get("epoch")

    train_loss = 0.0
    if train_entries:
        before = [h for h in train_entries if h.get("epoch", float("inf")) <= best_epoch] if best_epoch else []
        train_loss = (before or train_entries)[-1]["loss"]
    return overfit_gap, train_loss, best_epoch


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

    raw = (
        trainer.accelerator.unwrap_model(trainer.model)
        if hasattr(trainer, "accelerator") and trainer.accelerator
        else getattr(trainer.model, "module", trainer.model)
    ).cpu()

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


def train(
    architecture_name: str,
    data_dir: Optional[str] = None,
    data_csv: Optional[str] = None,
    val_data_csv: Optional[str] = None,
    aug_data_path: Optional[str] = None,
    output_dir: str = "./results",
    mlflow_tracking_uri: str = "sqlite:///mlflow.db",
    num_labels: Optional[int] = None,
    mlflow_experiment: str = "face-occlusion-detection",
    resume_from_checkpoint: Optional[str] = None,
    seed: int = 42,
    val_seed: Optional[int] = None,
    mlflow_run_id: Optional[str] = None,
    aug_rebalance_ratio: Optional[float] = None,
    test_data_csv: Optional[str] = None,
) -> Tuple[float, float, float, float, str]:
    cfg = load_architecture_config(architecture_name).to_dict()
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    final_num_labels = num_labels or model_cfg.get("num_labels", 2)
    best_metric = train_cfg.get("metric_for_best_model", "eval_loss")
    image_base_dir = data_cfg.get("image_base_dir")
    use_class_weights = train_cfg.get("use_class_weights", True)
    use_balanced_sampler = train_cfg.get("use_balanced_sampler", False)
    min_aug_per_class = train_cfg.get("min_aug_per_class", 0)
    augmentation_level = train_cfg.get("augmentation_level", "medium")
    if aug_rebalance_ratio is None:
        aug_rebalance_ratio = train_cfg.get("aug_rebalance_ratio")
    if aug_data_path is None:
        aug_data_path = data_cfg.get("aug_data_csv")

    client, run_id, use_client = _start_or_attach_run(
        cfg, mlflow_tracking_uri, mlflow_run_id, mlflow_experiment, final_num_labels,
    )

    model_name = model_cfg.get("model_name", "google/vit-base-patch16-224")
    ml_log_params(client, run_id, {
        "architecture": cfg["name"],
        "num_labels": final_num_labels,
        "model_name": model_name,
        "hidden_dropout_prob": model_cfg.get("hidden_dropout_prob", 0.1),
        "aug_rebalance_ratio": aug_rebalance_ratio,
        "augmentation_level": augmentation_level,
    })

    processor = get_image_processor(model_name)
    train_data, val_data = _load_train_val(data_cfg, data_dir, data_csv, val_data_csv, val_seed or seed)
    aug_pool = _maybe_load_aug_pool(aug_data_path, aug_rebalance_ratio)

    weights_tensor = compute_class_weights(train_data, "label") if use_class_weights else None
    if weights_tensor is not None:
        print(f"Class weights: {weights_tensor.tolist()}")

    train_dataset, val_dataset, use_dynamic = _build_datasets(
        train_data, val_data, aug_pool, processor, image_base_dir,
        aug_rebalance_ratio, min_aug_per_class, augmentation_level, seed,
    )

    train_sampler = None
    if use_balanced_sampler:
        sampler_labels = (
            train_dataset.get_current_labels() if isinstance(train_dataset, DynamicAugDataset)
            else train_data["label"].tolist()
        )
        train_sampler = create_balanced_sampler(sampler_labels, num_classes=final_num_labels)
        print(f"Balanced sampler: {train_sampler.num_samples} samples/epoch")

    ml_log_params(client, run_id, {
        "seed": seed,
        "num_train": len(train_dataset),
        "num_val": len(val_data),
        "use_class_weights": use_class_weights,
        "use_balanced_sampler": use_balanced_sampler,
        "dynamic_aug": use_dynamic,
    })

    model = (
        FaceOccClassifier.load_from_mlflow(resume_from_checkpoint, num_labels=final_num_labels)
        if resume_from_checkpoint
        else FaceOccClassifier(
            model_name=model_name,
            num_labels=final_num_labels,
            hidden_dropout_prob=model_cfg.get("hidden_dropout_prob", 0.1),
            pooling=model_cfg.get("pooling", "cls"),
            projection_size=model_cfg.get("projection_size"),
        )
    )

    if not use_client:
        mlflow.log_params({f"train_{k}": v for k, v in train_cfg.items() if k not in _NON_HF_TRAIN_KEYS})

    forwarded_train_args = {k: v for k, v in train_cfg.items() if k not in _NON_HF_TRAIN_KEYS}
    forwarded_train_args.setdefault("fp16", True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to=[] if use_client else ["mlflow"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=best_metric != "eval_loss",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=seed,
        remove_unused_columns=False,
        **forwarded_train_args,
    )

    callbacks: List[TrainerCallback] = []
    patience = train_cfg.get("early_stopping_patience", 3)
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
    if use_client and client and run_id:
        callbacks.append(MlflowClientCallback(client, run_id))
    if use_dynamic and isinstance(train_dataset, DynamicAugDataset):
        callbacks.append(AugResamplerCallback(train_dataset, base_seed=seed, num_classes=final_num_labels))
    ema_cb = make_ema_callback_from_cfg(train_cfg)
    if ema_cb is not None:
        callbacks.append(ema_cb)
        print(f"EMA enabled: decay={ema_cb.decay}, warmup_steps={ema_cb.warmup_steps}")

    trainer = WeightedLossTrainer(
        class_weights=weights_tensor,
        focal_loss_gamma=train_cfg.get("focal_loss_gamma", 0.0),
        label_smoothing=train_cfg.get("label_smoothing_factor", 0.0),
        train_sampler=train_sampler,
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
    f1_macro = eval_results.get("eval_f1_macro", 0.0)
    overfit_gap, train_loss_at_best, _ = _compute_overfit_stats(trainer.state.log_history, best_metric)

    print(f"F1={f1_macro:.4f} | loss={eval_loss:.4f} | gap={overfit_gap:.4f}")

    if test_data_csv and trainer.is_world_process_zero():
        try:
            test_df, _ = _load_data(test_data_csv, split_ratio=0)
            test_ds = FaceOccDataset(test_df["image_path"].tolist(), test_df["label"].tolist(), processor, image_base_dir)
            test_res = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
            print(f"Test F1={test_res.get('test_f1_macro', 0):.4f}")
            ml_log_metrics(client, run_id, {
                "test_f1_macro": test_res.get("test_f1_macro", 0.0),
                "test_loss": test_res.get("test_loss", 0.0),
            })
        except Exception as e:
            print(f"WARNING test eval: {e}")

    if not trainer.is_world_process_zero():
        return eval_loss, f1_macro, overfit_gap, train_loss_at_best, ""

    ml_log_metrics(client, run_id, {
        "val_f1_macro": f1_macro,
        "val_f1_class_diff": abs(eval_results.get("eval_f1_class0", 0.0) - eval_results.get("eval_f1_class1", 0.0)),
        "overfit_gap": overfit_gap,
        "final_eval_loss": eval_loss,
        "final_train_loss": train_loss_at_best,
    })
    artifact = f"configs/architectures/{architecture_name}.yaml"
    if use_client and client and run_id:
        client.log_artifact(run_id, artifact)
    else:
        mlflow.log_artifact(artifact)

    model_uri = ""
    try:
        model_uri = _save_model_to_mlflow(
            trainer, processor, run_id, f"{cfg['name']}-{final_num_labels}classes",
        )
        print(f"Model saved: {model_uri}")
    except Exception as e:
        print(f"ERROR saving model: {e}")
        if mlflow.active_run():
            mlflow.end_run()

    if output_dir and output_dir != "./results":
        shutil.rmtree(output_dir, ignore_errors=True)

    return eval_loss, f1_macro, overfit_gap, train_loss_at_best, model_uri


if __name__ == "__main__":
    train(
        architecture_name=CONFIG["architecture"],
        data_dir=CONFIG["data_dir"],
        data_csv=CONFIG["data_csv"],
        val_data_csv=CONFIG["val_data_csv"],
        output_dir=CONFIG["output_dir"],
        mlflow_tracking_uri=CONFIG["tracking_uri"],
        num_labels=CONFIG["num_labels"],
        resume_from_checkpoint=CONFIG["resume_from"],
    )
