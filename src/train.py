import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from transformers import (
    AutoImageProcessor,
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
from src.models.face_occ_classifier import FaceOccClassifier
from src.utils.config import load_architecture_config
from src.utils.environment import setup_environment
from src.utils.losses import FocalLoss, compute_class_weights
from src.utils.metrics import compute_metrics

setup_environment()

CONFIG: Dict[str, Any] = {
    "architecture": "vit-base-face-occ",
    "data_dir": "data/raw/",   # ImageFolder: data/raw/class0/, data/raw/class1/
    "data_csv": None,           # alternative: CSV with image_path + label columns
    "output_dir": "./results",
    "tracking_uri": "sqlite:///mlflow.db",
    "num_labels": None,
    "resume_from": None,
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
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            loss = outputs.get("loss") if isinstance(outputs, dict) else getattr(outputs, "loss", None)
        return (loss, outputs) if return_outputs else loss


class MlflowClientCallback(TrainerCallback):
    def __init__(self, client: MlflowClient, run_id: str):
        self.client = client
        self.run_id = run_id

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict] = None, **kwargs: Any) -> None:
        for key, value in (logs or {}).items():
            if isinstance(value, (int, float)):
                try:
                    self.client.log_metric(self.run_id, key, value, step=state.global_step)
                except Exception:
                    pass


class AugResamplerCallback(TrainerCallback):
    def __init__(self, dataset: DynamicAugDataset, base_seed: int = 42, num_classes: int = 2):
        self.dataset = dataset
        self.base_seed = base_seed
        self.num_classes = num_classes

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.dataset.resample(seed=self.base_seed + int(state.epoch or 0))
        trainer = kwargs.get("trainer")
        if trainer and getattr(trainer, "_custom_train_sampler", None):
            trainer._custom_train_sampler = create_balanced_sampler(
                self.dataset.get_current_labels(), self.num_classes
            )


def _mlflow_log_params(client: Optional[MlflowClient], run_id: Optional[str], params: Dict[str, Any]) -> None:
    if client and run_id:
        for k, v in params.items():
            client.log_param(run_id, k, v)
    elif mlflow.active_run():
        mlflow.log_params(params)


def _mlflow_log_metrics(client: Optional[MlflowClient], run_id: Optional[str], metrics: Dict[str, float]) -> None:
    if client and run_id:
        for k, v in metrics.items():
            client.log_metric(run_id, k, v)
    elif mlflow.active_run():
        mlflow.log_metrics(metrics)


def train(
    architecture_name: str,
    data_dir: Optional[str] = None,
    data_csv: Optional[str] = None,
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
    final_num_labels = num_labels or cfg["model"]["num_labels"]
    best_metric = cfg["training"].get("metric_for_best_model", "eval_loss")
    data_cfg = cfg.get("data", {})
    image_base_dir = data_cfg.get("image_base_dir", None)
    use_class_weights = cfg["training"].get("use_class_weights", True)
    use_balanced_sampler = cfg["training"].get("use_balanced_sampler", False)
    min_aug_per_class = cfg["training"].get("min_aug_per_class", 0)

    if aug_rebalance_ratio is None:
        aug_rebalance_ratio = cfg["training"].get("aug_rebalance_ratio", None)
    if aug_data_path is None:
        aug_data_path = data_cfg.get("aug_data_csv", None)

    # Fall back to data paths from YAML if not provided as args
    if data_dir is None and data_csv is None:
        data_dir = data_cfg.get("data_dir", None)
        data_csv = data_cfg.get("data_csv", None)
    if data_dir is None and data_csv is None:
        raise ValueError("Provide data_dir or data_csv (via arg or YAML data.data_dir / data.data_csv)")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    use_client = mlflow_run_id is not None
    client = MlflowClient(tracking_uri=mlflow_tracking_uri) if use_client else None

    if not use_client:
        mlflow.set_experiment(mlflow_experiment)
        run_name = f"{cfg['name']}_{final_num_labels}classes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        mlflow_run_id = mlflow.active_run().info.run_id

    model_name = cfg["model"].get("model_name", "google/vit-base-patch16-224")

    _mlflow_log_params(client, mlflow_run_id, {
        "architecture": cfg["name"],
        "num_labels": final_num_labels,
        "model_name": model_name,
        "hidden_dropout_prob": cfg["model"].get("hidden_dropout_prob", 0.1),
        "aug_rebalance_ratio": aug_rebalance_ratio,
    })

    processor = AutoImageProcessor.from_pretrained(model_name)

    # Load data — _load_data dispatches on .csv extension vs folder
    data_path = data_csv or data_dir
    label_map = data_cfg.get("label_map", None)
    subset_col = data_cfg.get("subset_col", None)
    extra_train = data_cfg.get("extra_train_csv", None)

    train_data, val_data = _load_data(
        data_path,
        label_map=label_map,        # used only by load_image_folder
        subset_col=subset_col,      # used only by load_csv_data
        extra_train_csv=extra_train,
        seed=val_seed or seed,
    )

    # After loading, label column is always "label"
    aug_pool: Optional[pd.DataFrame] = None
    use_dynamic = False
    if aug_data_path and aug_rebalance_ratio and aug_rebalance_ratio > 0:
        raw_pool, _ = _load_data(aug_data_path, split_ratio=0)
        if not raw_pool.empty:
            aug_pool, use_dynamic = raw_pool, True

    weights_tensor = compute_class_weights(train_data, "label") if use_class_weights else None
    if weights_tensor is not None:
        print(f"Class weights: {weights_tensor.tolist()}")

    train_dataset: Union[DynamicAugDataset, FaceOccDataset]
    if use_dynamic and aug_pool is not None:
        train_dataset = DynamicAugDataset(
            real_paths=train_data["image_path"].tolist(),
            real_labels=train_data["label"].tolist(),
            aug_pool=aug_pool,
            processor=processor,
            aug_rebalance_ratio=aug_rebalance_ratio,
            seed=seed,
            min_aug_per_class=min_aug_per_class,
            image_base_dir=image_base_dir,
        )
    else:
        train_dataset = FaceOccDataset(
            image_paths=train_data["image_path"].tolist(),
            labels=train_data["label"].tolist(),
            processor=processor,
            image_base_dir=image_base_dir,
        )

    val_dataset = FaceOccDataset(
        image_paths=val_data["image_path"].tolist(),
        labels=val_data["label"].tolist(),
        processor=processor,
        image_base_dir=image_base_dir,
    )

    train_sampler = None
    if use_balanced_sampler:
        sampler_labels = (
            train_dataset.get_current_labels() if isinstance(train_dataset, DynamicAugDataset)
            else train_data["label"].tolist()
        )
        train_sampler = create_balanced_sampler(sampler_labels, num_classes=final_num_labels)
        print(f"Balanced sampler: {train_sampler.num_samples} samples/epoch")

    _mlflow_log_params(client, mlflow_run_id, {
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
            hidden_dropout_prob=cfg["model"].get("hidden_dropout_prob", 0.1),
            pooling=cfg["model"].get("pooling", "cls"),
            projection_size=cfg["model"].get("projection_size", None),
        )
    )

    _excluded = {
        "label_smoothing_factor", "early_stopping_patience", "aug_rebalance_ratio",
        "use_class_weights", "use_balanced_sampler", "focal_loss_gamma",
        "min_aug_per_class", "metric_for_best_model", "seed",
    }

    if not use_client:
        mlflow.log_params({f"train_{k}": v for k, v in cfg["training"].items() if k not in _excluded})

    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to=[] if use_client else ["mlflow"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=best_metric != "eval_loss",
        fp16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=seed,
        remove_unused_columns=False,
        **{k: v for k, v in cfg["training"].items() if k not in _excluded},
    )

    callbacks: List[TrainerCallback] = []
    patience = cfg["training"].get("early_stopping_patience", 3)
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
    if use_client and client:
        callbacks.append(MlflowClientCallback(client, mlflow_run_id))
    if use_dynamic and isinstance(train_dataset, DynamicAugDataset):
        callbacks.append(AugResamplerCallback(train_dataset, base_seed=seed, num_classes=final_num_labels))

    trainer = WeightedLossTrainer(
        class_weights=weights_tensor,
        focal_loss_gamma=cfg["training"].get("focal_loss_gamma", 0.0),
        label_smoothing=cfg["training"].get("label_smoothing_factor", 0.0),
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

    eval_entries = [h for h in trainer.state.log_history if "eval_loss" in h]
    train_entries = [h for h in trainer.state.log_history if "loss" in h and "eval_loss" not in h]

    overfit_gap = 0.0
    best_epoch = None
    if eval_entries:
        overfit_gap = max(0.0, eval_entries[-1]["eval_loss"] - min(e["eval_loss"] for e in eval_entries))
        best_entry = (
            min(eval_entries, key=lambda h: h["eval_loss"]) if best_metric == "eval_loss"
            else max(eval_entries, key=lambda h: h.get(best_metric, 0.0))
        )
        best_epoch = best_entry.get("epoch")

    train_loss_at_best = 0.0
    if train_entries:
        before = [h for h in train_entries if h.get("epoch", float("inf")) <= best_epoch] if best_epoch else []
        train_loss_at_best = (before or train_entries)[-1]["loss"]

    print(f"F1={f1_macro:.4f} | loss={eval_loss:.4f} | gap={overfit_gap:.4f}")

    if test_data_csv and trainer.is_world_process_zero():
        try:
            test_df, _ = _load_data(test_data_csv, split_ratio=0)
            test_ds = FaceOccDataset(test_df["image_path"].tolist(), test_df["label"].tolist(), processor, image_base_dir)
            test_res = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
            print(f"Test F1={test_res.get('test_f1_macro', 0):.4f}")
            _mlflow_log_metrics(client, mlflow_run_id, {
                "test_f1_macro": test_res.get("test_f1_macro", 0.0),
                "test_loss": test_res.get("test_loss", 0.0),
            })
        except Exception as e:
            print(f"WARNING test eval: {e}")

    if not trainer.is_world_process_zero():
        return eval_loss, f1_macro, overfit_gap, train_loss_at_best, ""

    _mlflow_log_metrics(client, mlflow_run_id, {
        "val_f1_macro": f1_macro,
        "val_f1_class_diff": abs(eval_results.get("eval_f1_class0", 0.0) - eval_results.get("eval_f1_class1", 0.0)),
        "overfit_gap": overfit_gap,
        "final_eval_loss": eval_loss,
        "final_train_loss": train_loss_at_best,
    })
    if use_client and client:
        client.log_artifact(mlflow_run_id, f"configs/architectures/{architecture_name}.yaml")
    else:
        mlflow.log_artifact(f"configs/architectures/{architecture_name}.yaml")

    model_uri = ""
    try:
        while mlflow.active_run():
            mlflow.end_run()
        mlflow.start_run(run_id=mlflow_run_id)

        raw = (
            trainer.accelerator.unwrap_model(trainer.model)
            if hasattr(trainer, "accelerator") and trainer.accelerator
            else getattr(trainer.model, "module", trainer.model)
        ).cpu()

        import torch.nn as _nn
        _orig = _nn.Module.__dict__.get("__getstate__")
        _nn.Module.__getstate__ = lambda self: {k: v for k, v in self.__dict__.items()}
        try:
            info = mlflow.pytorch.log_model(
                pytorch_model=raw,
                artifact_path="model",
                registered_model_name=f"{cfg['name']}-{final_num_labels}classes",
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
        print(f"Model saved: {model_uri}")
    except Exception as e:
        print(f"ERROR saving model: {e}")
        if mlflow.active_run():
            mlflow.end_run()

    import shutil
    if output_dir and output_dir != "./results":
        shutil.rmtree(output_dir, ignore_errors=True)

    return eval_loss, f1_macro, overfit_gap, train_loss_at_best, model_uri


if __name__ == "__main__":
    train(
        architecture_name=CONFIG["architecture"],
        data_dir=CONFIG["data_dir"],
        data_csv=CONFIG["data_csv"],
        output_dir=CONFIG["output_dir"],
        mlflow_tracking_uri=CONFIG["tracking_uri"],
        num_labels=CONFIG["num_labels"],
        resume_from_checkpoint=CONFIG["resume_from"],
    )
