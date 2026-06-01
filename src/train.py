import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as torch_mp
from mlflow.tracking import MlflowClient

# Use file_system sharing instead of file_descriptor. Across Optuna trials, DataLoader
# workers spawn shared-memory handles that don't get cleaned up properly between
# trials → exhausts FD limit after ~10 trials → "Too many open files" hang.
# file_system uses named files (unbounded), bypassing FD limit entirely.
try:
    torch_mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.data.dataset import (
    DEFAULT_GENDER_COL,
    DEFAULT_IMAGE_COL,
    DEFAULT_LABEL_COL,
    FaceOccDataset,
    _load_data,
)
from src.data.transforms import build_train_transform
from src.models.dinov3_loader import get_image_processor
from src.models.face_occ_regressor import FaceOccRegressor
from src.training.callbacks import MlflowClientCallback
from src.utils.config import load_architecture_config
from src.utils.environment import setup_environment
from src.utils.losses import WeightedMSELoss
from src.utils.metrics import make_compute_metrics
from src.utils.mlflow_utils import log_metrics as ml_log_metrics
from src.utils.mlflow_utils import log_params as ml_log_params

setup_environment()

CONFIG: Dict[str, Any] = {
    "architecture": os.environ.get("FACE_OCC_ARCH", "dinov3-vitb16-3090-v11"),
    "data_csv": "data/raw/train.csv",
    "val_data_csv": None,
    "output_dir": "./results",
    "tracking_uri": "sqlite:///mlflow.db",
    "use_mlflow": True,
    "resume_from": None,
}

_NON_HF_TRAIN_KEYS = {
    "early_stopping_patience", "metric_for_best_model", "greater_is_better", "seed",
    "augmentation_level",
    "loss_focal_gamma",
    "loss_lambda_init", "loss_lambda_lr", "loss_lambda_max", "loss_lambda_min",
    "loss_lambda_ema", "loss_lambda_threshold",
    "correction_strength", "axis1_power", "axis2_power", "sampler_participation",
    "feature_fairness", "mmd_lambda", "adv_lambda", "ot_lambda", "ot_method", "sinkhorn_eps",
    "loss_query_diversity_lambda",
    "save_qualitative_k",
    "layer_decay",
    "ema_decay",
    "min_lr_rate",   # v12: top-level HPO param injected into lr_scheduler_kwargs below
}


def _unwrap(module: Any) -> Any:
    return getattr(module, "module", module)


def _custom_collator(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in batch[0]:
        if key == "loss_weight":
            out[key] = torch.stack([torch.as_tensor(b[key]) for b in batch]).float()
        else:
            out[key] = torch.stack([b[key] for b in batch])
    return out


class EMAWeightCallback(TrainerCallback):
    """Maintain an EMA copy of model weights, updated every training step.

    Eval helpers:
        _swap_to_ema(model)  → temporarily put EMA weights into the live model
        _swap_to_live(model) → restore the live weights
    Both are always paired (no leftover EMA in the live model after eval).
    snapshot() → return a cpu copy of the current EMA state (for best-epoch tracking).
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.ema_state: Dict[str, torch.Tensor] = {}
        self.backup_state: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.ema_state[name] = p.detach().clone()

    @staticmethod
    def _model(kwargs: Dict[str, Any]) -> Optional[torch.nn.Module]:
        m = kwargs.get("model")
        return _unwrap(m) if m is not None else None

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        model = self._model(kwargs)
        if model is None:
            return
        with torch.no_grad():
            for name, p in model.named_parameters():
                buf = self.ema_state.get(name)
                if buf is None or not p.requires_grad:
                    continue
                if buf.device != p.device:
                    buf = buf.to(p.device)
                    self.ema_state[name] = buf
                buf.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def _swap_to_ema(self, model: torch.nn.Module) -> None:
        if self.backup_state:
            return
        with torch.no_grad():
            for name, p in model.named_parameters():
                buf = self.ema_state.get(name)
                if buf is None:
                    continue
                if buf.device != p.device:
                    buf = buf.to(p.device)
                    self.ema_state[name] = buf
                self.backup_state[name] = p.detach().clone()
                p.copy_(buf)

    def _swap_to_live(self, model: torch.nn.Module) -> None:
        if not self.backup_state:
            return
        with torch.no_grad():
            for name, p in model.named_parameters():
                live = self.backup_state.get(name)
                if live is not None:
                    p.copy_(live.to(p.device) if live.device != p.device else live)
        self.backup_state.clear()

    def snapshot(self) -> Dict[str, torch.Tensor]:
        return {name: buf.detach().cpu().clone() for name, buf in self.ema_state.items()}

    def load_snapshot(self, snapshot: Dict[str, torch.Tensor], model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, p in model.named_parameters():
                buf = snapshot.get(name)
                if buf is None:
                    continue
                p.copy_(buf.to(p.device))


class EMABestTracker(TrainerCallback):
    """Track the best EMA epoch independently of HF Trainer's best-model logic.

    HF tracks the best `eval_*` epoch (= live model). We mirror that on `ema_*`,
    saving the EMA state to local disk (TMPDIR) at the best EMA epoch — keeping
    it in CPU RAM would cost ~ model_size × num_ranks (≈ 1GB for sapiens × 2 ranks)
    on top of dataloader buffers, which OOM'd 60G allocations. Disk is local SSD,
    cost is one write per "new best" event.

    After training, the caller compares best_live vs best_ema and loads from disk.
    """
    def __init__(self, ema_cb: "EMAWeightCallback", metric_key: str, greater_is_better: bool,
                 snapshot_dir: str) -> None:
        self.ema_cb = ema_cb
        self.metric_key = metric_key
        self.greater_is_better = greater_is_better
        self.snapshot_path = Path(snapshot_dir) / f"ema_best_rank{os.environ.get('LOCAL_RANK', '0')}.pt"
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_score: float = float("-inf") if greater_is_better else float("inf")
        self.best_epoch: Optional[float] = None
        self._snapshot_saved: bool = False

    def _improved(self, score: float) -> bool:
        return score > self.best_score if self.greater_is_better else score < self.best_score

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Optional[Dict[str, float]] = None, **kwargs: Any) -> None:
        # All ranks update — metrics are collective-gathered, EMA weights are DDP-synced,
        # so each rank's snapshot is identical. Gating on rank 0 would desync the model
        # in the post-train swap and corrupt the next predict() collective call.
        if metrics is None:
            return
        score = metrics.get(self.metric_key)
        if score is None or not isinstance(score, (int, float)):
            return
        if self._improved(float(score)):
            self.best_score = float(score)
            self.best_epoch = float(state.epoch or 0.0)
            torch.save(self.ema_cb.snapshot(), self.snapshot_path)
            self._snapshot_saved = True

    def load_best_into(self, model: torch.nn.Module) -> bool:
        if not self._snapshot_saved or not self.snapshot_path.exists():
            return False
        state = torch.load(self.snapshot_path, map_location="cpu")
        self.ema_cb.load_snapshot(state, model)
        return True


class OptunaPruningCallback(TrainerCallback):
    """Report eval_challenge_score to Optuna trial + check should_prune at each eval.

    Saves compute by stopping clearly-mediocre trials early (~30% gain on full sweep).
    Triggered after each epoch eval via on_evaluate hook.
    """
    def __init__(self, trial: Any, metric_key: str = "eval_challenge_score") -> None:
        self.trial = trial
        self.metric_key = metric_key

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Optional[Dict[str, float]] = None, **kwargs: Any) -> None:
        if not getattr(state, "is_world_process_zero", True):
            return
        if metrics is None:
            return
        value = metrics.get(self.metric_key)
        if value is None or not isinstance(value, (int, float)):
            return
        try:
            self.trial.report(float(value), int(state.epoch or 0))
            if self.trial.should_prune():
                import optuna
                raise optuna.exceptions.TrialPruned()
        except Exception as e:
            # Re-raise TrialPruned, swallow others (Optuna unavailable, etc.)
            from optuna.exceptions import TrialPruned as _TP
            if isinstance(e, _TP):
                raise
            print(f"[OptunaPruningCallback] WARNING: {e}", flush=True)


class LambdaLogCallback(TrainerCallback):
    """v16.5: drives the adaptive Lagrangian λ from the CLEAN val signal (eval_err_diff
    over 15k samples), updating once per epoch on on_evaluate. Logs λ to MLflow.

    Previous version used per-batch err_diff_ema (training, noisy) which systematically
    over-estimated err_diff (2-18× vs val) → λ stuck at cap.
    """
    def __init__(self, trainer_ref: List[Any], client: Optional[Any], run_id: Optional[str]) -> None:
        self._trainer_ref = trainer_ref
        self._client = client
        self._run_id = run_id

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Optional[Dict[str, float]] = None, **kwargs: Any) -> None:
        if metrics is None or not self._trainer_ref:
            return
        loss_fct = getattr(self._trainer_ref[0], "loss_fct", None)
        if loss_fct is None or not hasattr(loss_fct, "update_lambda"):
            return
        val_err_diff = metrics.get("eval_err_diff")
        if val_err_diff is None or not isinstance(val_err_diff, (int, float)):
            return
        # update_lambda is rank-safe: all ranks recompute the same lambda from same val signal
        new_lambda = loss_fct.update_lambda(float(val_err_diff))
        if getattr(state, "is_world_process_zero", True):
            epoch = int(state.epoch or 0)
            print(f"  Lagrangien epoch {epoch}: λ_adapt={new_lambda:.4f}  val_err_diff={float(val_err_diff):.6f}", flush=True)
            ml_log_metrics(self._client, self._run_id,
                            {"lambda_adapt": new_lambda, "val_err_diff_used": float(val_err_diff)},
                            step=epoch)


class WeightedMSETrainer(Trainer):
    def __init__(
        self,
        focal_gamma: float = 0.0,
        lambda_init: float = 1.0,
        lambda_lr: float = 1.0,
        lambda_max: float = 3.0,
        lambda_min: float = 0.5,
        lambda_threshold: float = 0.001,
        query_diversity_lambda: float = 0.0,
        adv_lambda: float = 0.0,
        mmd_lambda: float = 0.0,
        ot_lambda: float = 0.0,
        sinkhorn_lambda: float = 0.0,
        sinkhorn_eps: float = 0.1,
        layer_decay: float = 1.0,
        gender_sampler: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._query_diversity_lambda = float(query_diversity_lambda)
        self._adv_lambda = float(adv_lambda)
        self._mmd_lambda = float(mmd_lambda)
        self._ot_lambda = float(ot_lambda)
        self._sinkhorn_lambda = float(sinkhorn_lambda)
        self._sinkhorn_eps = float(sinkhorn_eps)
        self._layer_decay = float(layer_decay)
        self._gender_sampler = gender_sampler
        self.loss_fct = WeightedMSELoss(
            focal_gamma=focal_gamma,
            lambda_init=lambda_init,
            lambda_lr=lambda_lr,
            lambda_max=lambda_max,
            lambda_min=lambda_min,
            lambda_threshold=lambda_threshold,
        )
        print(f"WeightedMSETrainer: focal_gamma={focal_gamma}, "
              f"λ_init={lambda_init}, λ_lr={lambda_lr}, λ∈[{lambda_min}, {lambda_max}], "
              f"threshold={lambda_threshold}, "
              f"query_div={query_diversity_lambda}, adv={adv_lambda}, "
              f"mmd={mmd_lambda}, ot={ot_lambda}, sinkhorn={sinkhorn_lambda}, "
              f"layer_decay={layer_decay}")

    def _get_train_sampler(self, *args: Any, **kwargs: Any) -> Any:
        if self._gender_sampler is not None:
            return self._gender_sampler
        return super()._get_train_sampler(*args, **kwargs)

    def _find_ema_cb(self) -> Optional["EMAWeightCallback"]:
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, EMAWeightCallback):
                return cb
        return None

    def evaluate(self, eval_dataset: Any = None, ignore_keys: Any = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        metrics_live = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        ema_cb = self._find_ema_cb()
        # Skip the EMA shadow eval when explicitly requested (e.g. post-train evaluate where
        # we already swapped the winning weights into the model and the EMA buffer no longer
        # corresponds to a best epoch).
        if ema_cb is None or getattr(self, "_skip_ema_double_eval", False):
            return metrics_live

        model_inner = _unwrap(self.model)
        ema_cb._swap_to_ema(model_inner)
        try:
            # CRITICAL: HF Trainer.evaluate() calls self.log(metrics) INTERNALLY before
            # returning. If we reuse metric_key_prefix="eval", both the live and EMA evals
            # get logged under eval_* (last write wins per step in some MLflow backends,
            # but otherwise two values per step → zigzag). Use prefix "ema" so HF logs the
            # second call's metrics directly under ema_* — clean separate MLflow chart.
            metrics_ema = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix="ema")
        finally:
            ema_cb._swap_to_live(model_inner)

        return {**metrics_live, **metrics_ema}

    def compute_loss(self, model: Any, inputs: Dict[str, Any], return_outputs: bool = False, num_items_in_batch: Any = None) -> Any:
        labels = inputs["labels"]
        sample_loss_weight = inputs.pop("loss_weight", None)
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        preds = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        self.loss_fct = self.loss_fct.to(preds.device)
        # CRITICAL: HF doesn't propagate model.train/eval to loss_fct, so we sync here.
        # Without this, loss_fct.training stays True during eval → triggers the all_reduce
        # in WeightedMSELoss.forward() and pollutes lambda_adapt with val data. Worse: DDP
        # collective ops during a torch.no_grad() prediction step can deadlock.
        self.loss_fct.train(model.training)
        loss = self.loss_fct(preds, labels, sample_loss_weight=sample_loss_weight)

        if self._query_diversity_lambda > 0 and isinstance(outputs, dict) and "attn_weights" in outputs:
            from src.models.face_occ_regressor import _query_diversity_penalty
            div = _query_diversity_penalty(outputs["attn_weights"])
            loss = loss + self._query_diversity_lambda * div

        if self._adv_lambda > 0 and isinstance(outputs, dict) and "adv_logits" in outputs:
            if labels.dim() == 2 and labels.size(1) >= 2:
                g_tgt = (labels[:, 1] >= 0.5).long()
                adv = torch.nn.functional.cross_entropy(outputs["adv_logits"], g_tgt)
                loss = loss + self._adv_lambda * adv

        if self._mmd_lambda > 0 and isinstance(outputs, dict) and "features" in outputs:
            if labels.dim() == 2 and labels.size(1) >= 2:
                from src.utils.losses import mmd_rbf
                feats = outputs["features"]
                g = labels[:, 1]
                f_mask = g < 0.5
                m_mask = g >= 0.5
                mmd = mmd_rbf(feats[f_mask], feats[m_mask])
                loss = loss + self._mmd_lambda * mmd

        if self._ot_lambda > 0 and isinstance(outputs, dict) and "features" in outputs:
            if labels.dim() == 2 and labels.size(1) >= 2:
                from src.utils.losses import sliced_wasserstein
                feats = outputs["features"]
                g = labels[:, 1]
                f_mask = g < 0.5
                m_mask = g >= 0.5
                sw = sliced_wasserstein(feats[f_mask], feats[m_mask])
                loss = loss + self._ot_lambda * sw

        if self._sinkhorn_lambda > 0 and isinstance(outputs, dict) and "features" in outputs:
            if labels.dim() == 2 and labels.size(1) >= 2:
                from src.utils.losses import sinkhorn_distance
                feats = outputs["features"]
                g = labels[:, 1]
                f_mask = g < 0.5
                m_mask = g >= 0.5
                sk = sinkhorn_distance(feats[f_mask], feats[m_mask], eps=self._sinkhorn_eps)
                loss = loss + self._sinkhorn_lambda * sk

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
            print(f"WARNING: backbone has no .blocks/.encoder.layer — LLRD disabled")
            return super().create_optimizer()

        decay = self._layer_decay
        groups: List[Dict[str, Any]] = []
        assigned: set = set()

        def _take(predicate, lr: float) -> None:
            picked: List[torch.nn.Parameter] = []
            for n, p in self.model.named_parameters():
                if id(p) in assigned or not p.requires_grad:
                    continue
                if predicate(n):
                    picked.append(p)
                    assigned.add(id(p))
            if picked:
                groups.append({"params": picked, "lr": lr, "weight_decay": wd})

        _take(
            lambda n: "backbone" in n and any(
                k in n for k in ("embed", "cls_token", "mask_token", "storage_tokens", "register_tokens")
            ),
            base_lr * decay ** (n_layers + 1),
        )
        for i in range(n_layers):
            pattern = f"backbone.{block_attr}.{i}."
            _take(lambda n, p=pattern: p in n, base_lr * decay ** (n_layers - i))
        _take(lambda n: "backbone" in n, base_lr)
        _take(lambda n: True, base_lr)

        leftover = [n for n, p in self.model.named_parameters()
                    if p.requires_grad and id(p) not in assigned]
        if leftover:
            raise RuntimeError(f"LLRD did not cover {len(leftover)} trainable params: {leftover[:5]}")

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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (train, val, test_holdout). test_holdout is empty unless
    test_split_ratio > 0 in the data yaml section.
    """
    image_col = data_cfg.get("image_col", DEFAULT_IMAGE_COL)
    label_col = data_cfg.get("label_col", DEFAULT_LABEL_COL)
    gender_col = data_cfg.get("gender_col", DEFAULT_GENDER_COL)
    extra_train = data_cfg.get("extra_train_csv")
    val_split_ratio = float(data_cfg.get("val_split_ratio", 0.15))
    test_split_ratio = float(data_cfg.get("test_split_ratio", 0.0))

    train_path = _resolve_data_path(data_cfg, data_csv)
    if not train_path:
        raise ValueError("Provide data_csv via arg or YAML data.data_csv")

    common = dict(image_col=image_col, label_col=label_col, gender_col=gender_col)
    if val_data_csv:
        train_df, _, _ = _load_data(train_path, **common, extra_train_csv=extra_train, split_ratio=0)
        val_df, _, _ = _load_data(val_data_csv, **common, split_ratio=0)
        print(f"Pre-split: train={len(train_df):,} val={len(val_df):,}")
        return train_df, val_df, pd.DataFrame()

    return _load_data(train_path, **common, extra_train_csv=extra_train, seed=seed,
                       split_ratio=val_split_ratio, test_split_ratio=test_split_ratio)


def _build_datasets(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    processor: Any,
    image_base_dir: Optional[str],
    augmentation_level: str,
    correction_strength: float,
) -> Tuple[FaceOccDataset, FaceOccDataset, Dict[str, float]]:
    transform = build_train_transform(augmentation_level)
    targets_arr = train_data["FaceOcclusion"].astype(float).values
    gender_arr = train_data["gender"].astype(float).values

    from src.utils.distribution import compute_balancing_weights
    loss_weights = compute_balancing_weights(
        targets=targets_arr,
        gender=gender_arr,
        correction_strength=correction_strength,
    )

    train_ds = FaceOccDataset(
        image_paths=train_data["image_path"].tolist(),
        targets=targets_arr.tolist(),
        genders=gender_arr.tolist(),
        processor=processor,
        image_base_dir=image_base_dir,
        transform=transform,
        loss_weights=loss_weights,
    )
    val_ds = FaceOccDataset(
        image_paths=val_data["image_path"].tolist(),
        targets=val_data["FaceOcclusion"].astype(float).tolist(),
        genders=val_data["gender"].astype(float).tolist(),
        processor=processor,
        image_base_dir=image_base_dir,
        transform=None,
    )
    # Target mean weighted by sample_weights — used to init head bias optimally for the
    # TARGET distribution of this trial (depends on axis1_power, axis2_power, not just train).
    # axis1=0 → equals E[Y_train]; axis1=1 → equals E[Y_test] ≈ 2 × E[Y_train] for our task.
    target_mean_weighted = float((targets_arr * loss_weights).sum() / max(loss_weights.sum(), 1e-9))

    summary = {
        "loss_weight_min": float(loss_weights.min()),
        "loss_weight_max": float(loss_weights.max()),
        "loss_weight_mean": float(loss_weights.mean()),
        "loss_weight_std": float(loss_weights.std()),
        "target_mean_weighted": target_mean_weighted,
        "target_mean_unweighted": float(targets_arr.mean()),
    }
    return train_ds, val_ds, summary


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

    if hasattr(trainer, "accelerator") and trainer.accelerator:
        try:
            raw = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
        except TypeError:
            raw = trainer.accelerator.unwrap_model(trainer.model)
    else:
        raw = _unwrap(trainer.model)
    raw = raw.to(torch.float32).cpu()

    mlflow.start_run(run_id=run_id)
    _orig = _nn.Module.__dict__.get("__getstate__")
    _nn.Module.__getstate__ = lambda self: {k: v for k, v in self.__dict__.items()}
    try:
        info = mlflow.pytorch.log_model(
            pytorch_model=raw,
            artifact_path="model",
            registered_model_name=model_register_name,
        )
        model_uri = info.model_uri
        with tempfile.TemporaryDirectory() as tmp:
            processor.save_pretrained(tmp)
            mlflow.log_artifacts(tmp, "processor")
    finally:
        if _orig is not None:
            _nn.Module.__getstate__ = _orig
        else:
            try:
                delattr(_nn.Module, "__getstate__")
            except AttributeError:
                pass

    return model_uri


def _safe_mean(series) -> float:
    v = float(series.mean()) if len(series) > 0 else 0.0
    return 0.0 if v != v else v


def _save_diagnostic_charts(
    qual_root: Path,
    preds: np.ndarray,
    gt: np.ndarray,
    gender: np.ndarray,
    bin_width: float = 0.05,
) -> None:
    """Diagnostic chart : MAE per bin × gender + density."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = qual_root / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)

    abs_err = np.abs(preds - gt)
    sq_err = (preds - gt) ** 2
    weight_offset = 1.0 / 30.0
    w_sample = weight_offset + gt

    mask_f = gender < 0.5
    mask_m = gender >= 0.5

    n_bins = 10
    edges = np.linspace(0.0, n_bins * bin_width, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:]) * 100.0

    def _binned_stats(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        means = np.full(n_bins, np.nan)
        ses = np.full(n_bins, np.nan)
        counts = np.zeros(n_bins)
        contrib = np.zeros(n_bins)
        if not mask.any():
            return means, ses, counts, contrib
        gt_sub = gt[mask]
        err_sub = abs_err[mask]
        sq_sub = sq_err[mask]
        w_sub = w_sample[mask]
        total_w = float(w_sub.sum())
        bin_idx = np.clip((gt_sub / bin_width).astype(int), 0, n_bins - 1)
        for b in range(n_bins):
            sel = bin_idx == b
            n = int(sel.sum())
            counts[b] = n
            if n > 0:
                means[b] = err_sub[sel].mean()
                ses[b] = err_sub[sel].std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
                contrib[b] = float((w_sub[sel] * sq_sub[sel]).sum()) / total_w if total_w > 0 else 0.0
        return means, ses, counts, contrib

    f_means, f_se, f_counts, f_contrib = _binned_stats(mask_f)
    m_means, m_se, m_counts, m_contrib = _binned_stats(mask_m)
    overall_means, _, overall_counts, _ = _binned_stats(np.ones_like(gt, dtype=bool))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axA, axB = axes[0]
    axC, axD = axes[1]

    Z = 1.96
    axA.plot(centers, overall_means * 100.0, "-", lw=1.5, color="black", alpha=0.5, label="Overall")
    axA.errorbar(centers, f_means * 100.0, yerr=f_se * Z * 100.0, fmt="-s", color="tab:red",
                 alpha=0.85, capsize=3, label=f"F (n={int(mask_f.sum())})")
    axA.errorbar(centers, m_means * 100.0, yerr=m_se * Z * 100.0, fmt="-^", color="tab:blue",
                 alpha=0.85, capsize=3, label=f"M (n={int(mask_m.sum())})")
    axA.set_xlabel("True occlusion Y (% points)")
    axA.set_ylabel("MAE (% points), ±95% CI")
    axA.set_title("A — MAE per Y bin × gender")
    axA.grid(alpha=0.3); axA.legend(loc="upper left")

    bw_x = (centers[1] - centers[0]) * 0.4
    axB.bar(centers - bw_x/2, f_contrib, width=bw_x, color="tab:red", alpha=0.85,
            label=f"F (Σ={f_contrib.sum():.5f})")
    axB.bar(centers + bw_x/2, m_contrib, width=bw_x, color="tab:blue", alpha=0.85,
            label=f"M (Σ={m_contrib.sum():.5f})")
    axB.set_xlabel("True occlusion Y (%)")
    axB.set_ylabel("Σwᵢ·(p-y)² / Σw_g")
    axB.set_title("B — Contribution to err_g per bin")
    axB.grid(alpha=0.3, axis="y"); axB.legend(loc="upper right")

    axC.bar(centers - bw_x/2, f_counts, width=bw_x, color="tab:red", alpha=0.75,
            label=f"F (Σ={int(f_counts.sum())})")
    axC.bar(centers + bw_x/2, m_counts, width=bw_x, color="tab:blue", alpha=0.75,
            label=f"M (Σ={int(m_counts.sum())})")
    axC.set_xlabel("True occlusion Y (%)")
    axC.set_ylabel("Sample count")
    axC.set_title("C — Sample density per bin × gender")
    axC.grid(alpha=0.3, axis="y"); axC.legend(loc="upper right")

    bins = np.linspace(0.0, max(0.3, float(abs_err.max() + 0.01)), 60)
    axD.hist(abs_err[mask_f], bins=bins, density=True, alpha=0.6, color="tab:red", label="F")
    axD.hist(abs_err[mask_m], bins=bins, density=True, alpha=0.6, color="tab:blue", label="M")
    axD.axvline(abs_err[mask_f].mean(), color="tab:red", linestyle="--", lw=1.5,
                label=f"MAE_F = {abs_err[mask_f].mean()*100:.2f}%")
    axD.axvline(abs_err[mask_m].mean(), color="tab:blue", linestyle="--", lw=1.5,
                label=f"MAE_M = {abs_err[mask_m].mean()*100:.2f}%")
    axD.set_xlabel("|pred - gt|")
    axD.set_ylabel("Density")
    axD.set_title("D — Error distribution by gender")
    axD.grid(alpha=0.3); axD.legend(loc="upper right")

    plt.tight_layout()
    chart_path = out / "error_vs_occlusion_and_density.png"
    plt.savefig(chart_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic chart: {chart_path}")


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
    min_score_to_save: Optional[float] = None,
    optuna_trial: Any = None,
) -> Tuple[float, float, float, str, float, float]:
    # DDP coordination is handled at the trial boundary by the caller (optimize.py):
    # SYNC 1 = broadcast(trial_data) at trial start, SYNC 2 = barrier() at trial end.
    # Inside train(), HF Trainer + accelerator handle DDP internally (model wrap,
    # gradient all-reduce, etc.) so no extra barriers are needed here.
    cfg = load_architecture_config(architecture_name).to_dict()
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    output_dim = model_cfg.get("output_dim", 1)
    best_metric = train_cfg.get("metric_for_best_model", "eval_challenge_score")
    greater_is_better = bool(train_cfg.get("greater_is_better", False))
    image_base_dir = data_cfg.get("image_base_dir")
    augmentation_level = train_cfg.get("augmentation_level", "light")

    # v16: 1 single axis (under H_C). axis1_power kept as fallback name for backward compat.
    correction_strength = float(train_cfg.get("correction_strength",
                                                train_cfg.get("axis1_power", 0.0)))
    layer_decay = float(train_cfg.get("layer_decay", 1.0))

    feature_fairness = str(train_cfg.get("feature_fairness", "none"))
    mmd_active = feature_fairness == "mmd"
    dann_active = feature_fairness == "dann"
    ot_active = feature_fairness == "ot"
    ot_method = str(train_cfg.get("ot_method", "sliced"))   # 'sliced' or 'sinkhorn'
    mmd_lambda = float(train_cfg.get("mmd_lambda", 0.0)) if mmd_active else 0.0
    adv_lambda = float(train_cfg.get("adv_lambda", 0.01)) if dann_active else 0.0
    # Single ot_lambda dispatched to either Sliced-W (default, fast) or Sinkhorn (precise).
    ot_lambda = float(train_cfg.get("ot_lambda", 0.0)) if ot_active else 0.0
    sinkhorn_eps = float(train_cfg.get("sinkhorn_eps", 0.1))
    sliced_lambda = ot_lambda if (ot_active and ot_method == "sliced") else 0.0
    sinkhorn_lambda = ot_lambda if (ot_active and ot_method == "sinkhorn") else 0.0

    client, run_id, use_client = _start_or_attach_run(
        cfg, mlflow_tracking_uri, mlflow_run_id, mlflow_experiment, use_mlflow,
    )

    model_name = model_cfg.get("model_name", "dinov3_vits16")
    ml_log_params(client, run_id, {"architecture": cfg["name"]})
    ml_log_params(client, run_id, dict(model_cfg))
    # Log RESOLVED values for conditional knobs so MLflow doesn't show stale yaml defaults
    # (e.g. adv_lambda=0.01 from yaml when feature_fairness=none → actual adv_lambda=0).
    train_cfg_logged = dict(train_cfg)
    train_cfg_logged["correction_strength"] = correction_strength
    train_cfg_logged["adv_lambda"] = adv_lambda
    train_cfg_logged["mmd_lambda"] = mmd_lambda
    train_cfg_logged["ot_lambda"] = ot_lambda
    train_cfg_logged["ot_method"] = ot_method if ot_active else "none"
    train_cfg_logged["sinkhorn_eps"] = sinkhorn_eps if (ot_active and ot_method == "sinkhorn") else 0.0
    ml_log_params(client, run_id, train_cfg_logged)
    ml_log_params(client, run_id, dict(data_cfg))

    image_size = model_cfg.get("image_size")
    processor = get_image_processor(model_name, image_size=image_size)
    train_data, val_data, test_data = _load_train_val(data_cfg, data_csv, val_data_csv, val_seed or seed)
    train_dataset, val_dataset, weight_summary = _build_datasets(
        train_data, val_data, processor, image_base_dir, augmentation_level,
        correction_strength=correction_strength,
    )
    test_holdout_dataset = None
    if not test_data.empty:
        test_holdout_dataset = FaceOccDataset(
            image_paths=test_data["image_path"].tolist(),
            targets=test_data["FaceOcclusion"].astype(float).tolist(),
            genders=test_data["gender"].astype(float).tolist(),
            processor=processor,
            image_base_dir=image_base_dir,
            transform=None,
        )
    print(f"Loss weights summary: {weight_summary}")
    ml_log_metrics(client, run_id, weight_summary)

    label_col = data_cfg.get("label_col", DEFAULT_LABEL_COL)
    n_train_f = int((train_data["gender"] < 0.5).sum())
    n_train_m = int((train_data["gender"] >= 0.5).sum())
    n_val_f = int((val_data["gender"] < 0.5).sum())
    n_val_m = int((val_data["gender"] >= 0.5).sum())
    ml_log_params(client, run_id, {
        "seed": seed,
        "num_train": len(train_data),
        "num_val": len(val_data),
        "num_train_female": n_train_f,
        "num_train_male": n_train_m,
        "num_val_female": n_val_f,
        "num_val_male": n_val_m,
    })
    ml_log_metrics(client, run_id, {
        "data_train_occ_mean": _safe_mean(train_data["FaceOcclusion"]),
        "data_val_occ_mean": _safe_mean(val_data["FaceOcclusion"]),
        "data_train_occ_female_mean": _safe_mean(train_data.loc[train_data["gender"] < 0.5, "FaceOcclusion"]),
        "data_train_occ_male_mean": _safe_mean(train_data.loc[train_data["gender"] >= 0.5, "FaceOcclusion"]),
    })

    pretrained = bool(model_cfg.get("pretrained", True))
    # v12: head bias init via WEIGHTED target mean (adapts to axis1/axis2 of this trial).
    # axis1=0 → E[Y_train]≈0.085 (init bias≈-2.40). axis1=1 → E[Y_test]≈0.165 (init bias≈-1.62).
    # ~2× difference → init was previously biased toward train, gaspillait 1-2 epochs.
    target_mean = float(weight_summary["target_mean_weighted"])
    print(f"Head bias init: target_mean_weighted={target_mean:.4f}  "
          f"(unweighted={weight_summary['target_mean_unweighted']:.4f})  ->  "
          f"bias=logit(mean)≈{np.log(max(target_mean,1e-6)/max(1-target_mean,1e-6)):.3f}")
    model = (
        FaceOccRegressor.load_from_mlflow(resume_from_checkpoint, output_dim=output_dim)
        if resume_from_checkpoint
        else FaceOccRegressor(
            model_name=model_name,
            output_dim=output_dim,
            head_dropout=float(model_cfg.get("head_dropout", 0.1)),
            projection_size=model_cfg.get("projection_size"),
            output_activation=model_cfg.get("output_activation", "sigmoid"),
            backbone_drop_path_rate=float(model_cfg.get("backbone_drop_path_rate", 0.0)),
            pretrained=pretrained,
            pooling_type=str(model_cfg.get("pooling_type", "attention_k_query")),
            n_focal=int(model_cfg.get("n_focal", 2)),
            n_diffuse=int(model_cfg.get("n_diffuse", 2)),
            n_free=int(model_cfg.get("n_free", 2)),
            tau_focal_init=float(model_cfg.get("tau_focal_init", 0.1)),
            tau_diffuse_init=float(model_cfg.get("tau_diffuse_init", 1.5)),
            tau_free_init=float(model_cfg.get("tau_free_init", 1.0)),
            learnable_tau=bool(model_cfg.get("learnable_tau", True)),
            mil_agg=str(model_cfg.get("mil_agg", "multi")),
            mil_hidden=int(model_cfg.get("mil_hidden", 128)),
            mil_k_top=int(model_cfg.get("mil_k_top", 30)),
            grid_size=int(model_cfg.get("grid_size", 2)),
            pool_attn_dropout=float(model_cfg.get("pool_attn_dropout", 0.0)),
            pool_proj_dropout=float(model_cfg.get("pool_proj_dropout", 0.0)),
            enable_adv_disc=dann_active,
            target_mean=target_mean,
        )
    )
    # Rank-aware diagnostic — helps detect silent rank crashes in the model construction
    # path (root cause of mysterious DDP "_verify_param_shape" timeouts across trials).
    _rank = int(os.environ.get("LOCAL_RANK", "0"))
    _n_params = sum(p.numel() for p in model.parameters())
    print(f"[Rank {_rank}] Model built: {model_name}, pooling={model_cfg.get('pooling_type')}, "
          f"params={_n_params/1e6:.1f}M", flush=True)

    init_backbone_from = model_cfg.get("init_backbone_from") if pretrained else None
    if init_backbone_from:
        import re
        import mlflow as _ml
        print(f"Loading pretrained backbone from {init_backbone_from}")
        pretrained_model = _ml.pytorch.load_model(init_backbone_from)
        missing, unexpected = model.backbone.load_state_dict(pretrained_model.state_dict(), strict=False)
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
            except Exception as e:
                print(f"  WARNING: could not fetch pretrain run params: {e}")

    forwarded = {k: v for k, v in train_cfg.items() if k not in _NON_HF_TRAIN_KEYS}
    forwarded.setdefault("fp16", False)

    # v12: inject HPO-tuned min_lr_rate into lr_scheduler_kwargs (yaml is nested,
    # HPO param is top-level, so we merge here).
    if "min_lr_rate" in train_cfg:
        existing_kwargs = dict(forwarded.get("lr_scheduler_kwargs") or {})
        existing_kwargs["min_lr_rate"] = float(train_cfg["min_lr_rate"])
        forwarded["lr_scheduler_kwargs"] = existing_kwargs
    if not torch.cuda.is_available():
        if forwarded.get("bf16") or forwarded.get("fp16"):
            print(f"WARNING: non-CUDA — disabling bf16/fp16")
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
        dataloader_persistent_workers=True,
        # adv_disc head only exists when dann_active=True, and when present it always
        # receives gradient (adv_lambda > 0 enforced upstream) → no unused params either way.
        ddp_find_unused_parameters=False,
        # DDP timeout bumped to 1h (default 10min trop court si NFS lent ou save MLflow long)
        ddp_timeout=3600,
        seed=seed,
        remove_unused_columns=False,
        prediction_loss_only=False,
        label_names=["labels"],
        **forwarded,
    )

    callbacks: List[TrainerCallback] = []
    patience = train_cfg.get("early_stopping_patience", 3)
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
    if use_client and client and run_id:
        callbacks.append(MlflowClientCallback(client, run_id))

    test_pmf_joint_val = None
    if not test_data.empty:
        from src.utils.distribution import estimate_test_pmf_joint, N_BINS, BIN_WIDTH
        test_pmf_joint_val = estimate_test_pmf_joint(
            targets=val_data["FaceOcclusion"].astype(float).values,
            gender=val_data["gender"].astype(float).values,
        )
        print(f"Val: iid P_train. Metric uses stratified IS with test_pmf_joint built via H_C "
              f"(see docs/v12_theory.md §3 — estimate_test_pmf_joint).")
        compute_metrics = make_compute_metrics(test_pmf_joint=test_pmf_joint_val, bin_width=BIN_WIDTH, n_bins=N_BINS)
    else:
        compute_metrics = make_compute_metrics()

    # v16: no sampler. With P_train(F)=0.324 and batch≥32, ~10-41 F per batch is
    # plenty for MMD/DANN/OT (need ≥2) and for fairness gradient variance. The full
    # P_target correction is carried by the per-sample loss weight (IS unbiased).
    gender_sampler = None
    print(f"correction_strength={correction_strength:.2f} (loss-only, no sampler — v16)")

    ema_cb: Optional[EMAWeightCallback] = None
    ema_best_tracker: Optional[EMABestTracker] = None
    ema_decay = float(train_cfg.get("ema_decay", 0.0))
    if ema_decay > 0:
        ema_cb = EMAWeightCallback(model=model, decay=ema_decay)
        callbacks.append(ema_cb)
        # Mirror the best-model logic on the EMA series: same metric key, same direction.
        best_metric_key = str(train_cfg.get("metric_for_best_model", "eval_challenge_score"))
        if not best_metric_key.startswith("eval_"):
            best_metric_key = f"eval_{best_metric_key}"
        ema_metric_key = "ema_" + best_metric_key[len("eval_"):]
        ema_best_tracker = EMABestTracker(
            ema_cb=ema_cb,
            metric_key=ema_metric_key,
            greater_is_better=bool(train_cfg.get("greater_is_better", False)),
            snapshot_dir=output_dir,
        )
        callbacks.append(ema_best_tracker)
        print(f"EMA weights enabled with decay={ema_decay}, tracking best on {ema_metric_key}")

    trainer_ref: List[Any] = []
    callbacks.append(LambdaLogCallback(trainer_ref, client if use_client else None, run_id if use_client else None))

    if optuna_trial is not None:
        callbacks.append(OptunaPruningCallback(optuna_trial, metric_key="eval_challenge_score"))

    trainer = WeightedMSETrainer(
        focal_gamma=float(train_cfg.get("loss_focal_gamma", 0.0)),
        lambda_init=float(train_cfg.get("loss_lambda_init", 1.0)),
        lambda_lr=float(train_cfg.get("loss_lambda_lr", 1.0)),
        lambda_max=float(train_cfg.get("loss_lambda_max", 3.0)),
        lambda_min=float(train_cfg.get("loss_lambda_min", 0.5)),
        lambda_threshold=float(train_cfg.get("loss_lambda_threshold", 0.001)),
        query_diversity_lambda=float(train_cfg.get("loss_query_diversity_lambda", 0.0)),
        adv_lambda=adv_lambda,
        mmd_lambda=mmd_lambda,
        ot_lambda=sliced_lambda,
        sinkhorn_lambda=sinkhorn_lambda,
        sinkhorn_eps=sinkhorn_eps,
        layer_decay=layer_decay,
        gender_sampler=gender_sampler,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=_custom_collator,
        callbacks=callbacks or None,
    )
    trainer_ref.append(trainer)

    _rank = int(os.environ.get("LOCAL_RANK", "0"))
    print(f"[Rank {_rank}] Training {cfg['name']} (best_metric={best_metric}) — about to call accelerator.prepare via trainer.train()", flush=True)
    trainer.train()

    # HF has loaded the best-live epoch (via load_best_model_at_end). If the EMA shadow
    # reached a better score at some epoch, swap those EMA weights (loaded from disk
    # snapshot) into the model.
    best_live_score = float(getattr(trainer.state, "best_metric", float("inf")) or float("inf"))
    used_ema = False
    if ema_best_tracker is not None and ema_best_tracker._snapshot_saved:
        better = (
            ema_best_tracker.best_score > best_live_score
            if bool(train_cfg.get("greater_is_better", False))
            else ema_best_tracker.best_score < best_live_score
        )
        print(f"  best_live_score={best_live_score:.5f}  "
              f"best_ema_score={ema_best_tracker.best_score:.5f} (epoch {ema_best_tracker.best_epoch})  "
              f"→ {'EMA WINS, swapping weights' if better else 'LIVE wins, keeping weights'}")
        if better:
            used_ema = ema_best_tracker.load_best_into(_unwrap(trainer.model))
        if use_client and client and run_id:
            ml_log_metrics(client, run_id, {
                "best_live_score": best_live_score,
                "best_ema_score": float(ema_best_tracker.best_score),
                "used_ema_weights": float(used_ema),
            })

    trainer._skip_ema_double_eval = True
    eval_results = trainer.evaluate()
    trainer._skip_ema_double_eval = False
    eval_loss = eval_results.get("eval_loss", float("inf"))
    eval_score = eval_results.get("eval_challenge_score", 0.0)
    err_diff = eval_results.get("eval_err_diff", 0.0)
    err_F = eval_results.get("eval_err_F", 0.0)
    err_M = eval_results.get("eval_err_M", 0.0)
    print(f"loss={eval_loss:.5f}  score={eval_score:.5f}  err_F={err_F:.5f}  err_M={err_M:.5f}  err_diff={err_diff:.5f}  "
          f"({'EMA' if used_ema else 'LIVE'} weights)")
    mae_pct = eval_results.get("eval_mae_pct", 0.0)
    r2 = eval_results.get("eval_r2", 0.0)
    print(f"  human-readable : MAE_pct={mae_pct:.2f}%  R²={r2:.3f}")

    # IMPORTANT: trainer.predict() is a DDP collective — ALL ranks must call it.
    # Only AFTER predict completes, we can isolate post-processing on rank 0.
    try:
        pred_out = trainer.predict(val_dataset)
    except Exception as e:
        print(f"WARNING: trainer.predict(val) failed: {e}")
        pred_out = None

    test_pred_out = None
    if test_holdout_dataset is not None:
        try:
            test_pred_out = trainer.predict(test_holdout_dataset)
        except Exception as e_test:
            print(f"WARNING: trainer.predict(test_holdout) failed: {e_test}")

    preds = gt = gender = None
    if pred_out is not None and trainer.is_world_process_zero():
        preds_raw = pred_out.predictions
        if isinstance(preds_raw, (tuple, list)):
            preds_raw = preds_raw[0]
        preds = np.asarray(preds_raw).astype(np.float64).flatten()
        labels = np.asarray(pred_out.label_ids).astype(np.float64)
        gt = labels[:, 0] if labels.ndim == 2 else labels.flatten()
        gender = labels[:, 1] if (labels.ndim == 2 and labels.shape[1] >= 2) else np.zeros_like(gt)

    if preds is not None and trainer.is_world_process_zero():
        from src.inference.calibrators import fit_all_calibrators
        from src.utils.metrics import compute_score, compute_score_stratified_is
        from src.utils.distribution import N_BINS, BIN_WIDTH

        # Fit 3 calibrators on val (per-gender, IS-weighted):
        #   isotonic (PAV), linear (Platt-like), pchip (monotone cubic spline)
        cals = fit_all_calibrators(preds, gt, gender, use_is_weight=True)

        # SELECTION via IS-stratified eval on val (estimates P_test perf, unbiased target).
        # Scan α ∈ [0, 1.5] for each calibrator: pred_blend = α·cal + (1-α)·raw = raw + α·(cal-raw).
        #   α=0    → raw (no correction)
        #   α=1    → full calibration
        #   α>1    → over-correct (extrapolate beyond cal); helps if cal is conservative on
        #            rare high-y bins where the model under-predicts and isotonic can't fully
        #            stretch due to sparse val support. Best (cal,α) selected on val IS-strat.
        alphas = np.linspace(0.0, 1.5, 16)  # 0.0, 0.1, ..., 1.5
        val_is_eval_scores: Dict[str, float] = {}
        val_combo_scores: Dict[Tuple[str, float], float] = {}
        for cal_name, cal in cals.items():
            preds_cal = cal.transform(preds, gender)
            scores_cal_raw = compute_score(preds_cal, gt, gender)
            if test_pmf_joint_val is not None:
                scores_cal_is = compute_score_stratified_is(
                    preds_cal, gt, gender, test_pmf_joint_val, BIN_WIDTH, N_BINS,
                )
            else:
                scores_cal_is = scores_cal_raw
            val_is_eval_scores[cal_name] = scores_cal_is["challenge_score"]
            ml_log_metrics(client, run_id, {
                f"val_score_{cal_name}_is_eval": float(scores_cal_is["challenge_score"]),
                f"val_err_F_{cal_name}_is_eval": float(scores_cal_is["err_F"]),
                f"val_err_M_{cal_name}_is_eval": float(scores_cal_is["err_M"]),
                f"val_score_{cal_name}_raw_eval": float(scores_cal_raw["challenge_score"]),
            })
            # Alpha-blend scan
            for alpha in alphas:
                preds_blend = alpha * preds_cal + (1.0 - alpha) * preds
                preds_blend = np.clip(preds_blend, 0.0, 1.0)
                if test_pmf_joint_val is not None:
                    s = compute_score_stratified_is(preds_blend, gt, gender, test_pmf_joint_val, BIN_WIDTH, N_BINS)
                else:
                    s = compute_score(preds_blend, gt, gender)
                val_combo_scores[(cal_name, round(float(alpha), 2))] = s["challenge_score"]

        # Best combo = argmin over (cal_name, alpha)
        best_combo = min(val_combo_scores, key=val_combo_scores.get)
        best_cal_for_submission, best_alpha_for_submission = best_combo
        # Also compute best α PER calibrator (for UI viz: show each cal at its own optimal α)
        best_alpha_per_cal: Dict[str, float] = {}
        for cal_name in cals:
            alphas_for_cal = [(a, s) for (cn, a), s in val_combo_scores.items() if cn == cal_name]
            if alphas_for_cal:
                best_alpha_per_cal[cal_name] = float(min(alphas_for_cal, key=lambda x: x[1])[0])
        ml_log_params(client, run_id, {
            "best_cal_for_submission": best_cal_for_submission,
            "best_alpha_for_submission": str(best_alpha_for_submission),
        })
        ml_log_metrics(client, run_id, {
            "val_score_best_combo_is_eval": float(val_combo_scores[best_combo]),
            "best_alpha_for_submission_value": float(best_alpha_for_submission),
            **{f"best_alpha_{cn}": a for cn, a in best_alpha_per_cal.items()},
        })
        print(f"  Best (cal, α) (val IS-strat): {best_cal_for_submission}, α={best_alpha_for_submission}  "
              f"→ val_score={val_combo_scores[best_combo]:.5f}")
        print(f"  Best α per cal: {', '.join(f'{c}=α{a:.1f}' for c, a in best_alpha_per_cal.items())}")

        # === True post-hoc validation on test holdout (predict already done above) ===
        if test_pred_out is not None:
            try:
                test_preds_raw = test_pred_out.predictions
                if isinstance(test_preds_raw, (tuple, list)):
                    test_preds_raw = test_preds_raw[0]
                test_preds = np.asarray(test_preds_raw).astype(np.float64).flatten()
                test_labels = np.asarray(test_pred_out.label_ids).astype(np.float64)
                test_gt = test_labels[:, 0] if test_labels.ndim == 2 else test_labels.flatten()
                test_gender = test_labels[:, 1] if (test_labels.ndim == 2 and test_labels.shape[1] >= 2) else np.zeros_like(test_gt)

                # Test holdout already matches P_test (H_C) → use standard compute_score
                test_scores_raw = compute_score(test_preds, test_gt, test_gender)

                # Apply EACH calibrator (α=1 pure) on test holdout
                test_scores_per_cal: Dict[str, Dict[str, float]] = {}
                test_preds_per_cal: Dict[str, np.ndarray] = {}
                for cal_name, cal in cals.items():
                    test_preds_cal = cal.transform(test_preds, test_gender)
                    test_scores_per_cal[cal_name] = compute_score(test_preds_cal, test_gt, test_gender)
                    test_preds_per_cal[cal_name] = test_preds_cal

                # Apply best (cal, α) blend selected on val IS-stratified → submission proxy
                test_cal_best = cals[best_cal_for_submission].transform(test_preds, test_gender)
                test_preds_blend = best_alpha_for_submission * test_cal_best + (1.0 - best_alpha_for_submission) * test_preds
                test_preds_blend = np.clip(test_preds_blend, 0.0, 1.0)
                best_scores = compute_score(test_preds_blend, test_gt, test_gender)
                best_gain = test_scores_raw["challenge_score"] - best_scores["challenge_score"]
                # Oracle = post-hoc selection on test holdout (biased, diagnostic only)
                oracle_cal_name = min(test_scores_per_cal,
                                       key=lambda k: test_scores_per_cal[k]["challenge_score"])
                oracle_scores = test_scores_per_cal[oracle_cal_name]
                best_cal_name = best_cal_for_submission

                print(f"  Test holdout (n={len(test_preds)}, P_test dist):")
                print(f"    raw            : score={test_scores_raw['challenge_score']:.5f}  "
                      f"err_F={test_scores_raw['err_F']:.5f}  err_M={test_scores_raw['err_M']:.5f}  "
                      f"diff={test_scores_raw['err_diff']:.5f}")
                for cal_name in ("isotonic", "linear", "pchip"):
                    if cal_name not in test_scores_per_cal:
                        continue
                    s = test_scores_per_cal[cal_name]
                    gain_s = test_scores_raw["challenge_score"] - s["challenge_score"]
                    markers = ""
                    if cal_name == best_cal_name:
                        markers += " ← SELECTED (val)"
                    if cal_name == oracle_cal_name and oracle_cal_name != best_cal_name:
                        markers += " ← ORACLE (test, biased)"
                    print(f"    {cal_name:14s}: score={s['challenge_score']:.5f}  "
                          f"err_F={s['err_F']:.5f}  err_M={s['err_M']:.5f}  "
                          f"diff={s['err_diff']:.5f}  (gain {gain_s:+.5f}){markers}")

                # Log all per-calibrator metrics + best
                log_metrics = {
                    "test_holdout_n": float(len(test_preds)),
                    "test_holdout_score_raw": float(test_scores_raw["challenge_score"]),
                    "test_holdout_err_F_raw": float(test_scores_raw["err_F"]),
                    "test_holdout_err_M_raw": float(test_scores_raw["err_M"]),
                    "test_holdout_err_diff_raw": float(test_scores_raw["err_diff"]),
                    "test_holdout_mae_pct_raw": float(test_scores_raw["mae_pct"]),
                    "test_holdout_r2_raw": float(test_scores_raw["r2"]),
                    # SELECTED = method chosen on val (what we'd submit) — unbiased
                    "test_holdout_score_selected_cal": float(best_scores["challenge_score"]),
                    "test_holdout_err_F_selected_cal": float(best_scores["err_F"]),
                    "test_holdout_err_M_selected_cal": float(best_scores["err_M"]),
                    "test_holdout_selected_cal_gain": float(best_gain),
                    # ORACLE = method picked POST-HOC on test holdout (biased, upper bound)
                    "test_holdout_score_oracle_cal": float(oracle_scores["challenge_score"]),
                    "test_holdout_oracle_cal_gain": float(test_scores_raw["challenge_score"] - oracle_scores["challenge_score"]),
                }
                for cal_name, s in test_scores_per_cal.items():
                    log_metrics[f"test_holdout_score_{cal_name}"] = float(s["challenge_score"])
                    log_metrics[f"test_holdout_err_F_{cal_name}"] = float(s["err_F"])
                    log_metrics[f"test_holdout_err_M_{cal_name}"] = float(s["err_M"])
                    log_metrics[f"test_holdout_err_diff_{cal_name}"] = float(s["err_diff"])
                ml_log_metrics(client, run_id, log_metrics)

                # Save per-sample predictions + calibrator mappings for UI viz
                try:
                    qual_root = Path(output_dir) / "qualitative"
                    qual_root.mkdir(parents=True, exist_ok=True)

                    # Per-sample preds (all calibrators)
                    test_df_dict = {"gt": test_gt, "pred_raw": test_preds, "gender": test_gender}
                    for cal_name, p_cal in test_preds_per_cal.items():
                        test_df_dict[f"pred_{cal_name}"] = p_cal
                    test_df_dict["best_cal"] = [best_cal_name] * len(test_preds)
                    test_csv = qual_root / "test_holdout_predictions.csv"
                    pd.DataFrame(test_df_dict).to_csv(test_csv, index=False)

                    # Calibrator mappings (apply each to a grid for plotting)
                    grid = np.linspace(0.0, 1.0, 200)
                    grid_gender_F = np.zeros_like(grid)
                    grid_gender_M = np.ones_like(grid)
                    mapping_df = {"x": grid}
                    for cal_name, cal in cals.items():
                        mapping_df[f"{cal_name}_F"] = cal.transform(grid, grid_gender_F)
                        mapping_df[f"{cal_name}_M"] = cal.transform(grid, grid_gender_M)
                    cal_curve_csv = qual_root / "calibrator_mappings.csv"
                    pd.DataFrame(mapping_df).to_csv(cal_curve_csv, index=False)

                    # Legacy isotonic_mapping.csv kept for UI backward-compat
                    iso_curve_csv = qual_root / "isotonic_mapping.csv"
                    if "isotonic" in cals:
                        pd.DataFrame({"x": grid,
                                       "iso_F": mapping_df["isotonic_F"],
                                       "iso_M": mapping_df["isotonic_M"]}).to_csv(iso_curve_csv, index=False)

                    if use_mlflow:
                        if use_client and client and run_id:
                            client.log_artifact(run_id, str(test_csv), "qualitative")
                            client.log_artifact(run_id, str(cal_curve_csv), "qualitative")
                            if iso_curve_csv.exists():
                                client.log_artifact(run_id, str(iso_curve_csv), "qualitative")
                        elif mlflow.active_run():
                            mlflow.log_artifact(str(test_csv), "qualitative")
                            mlflow.log_artifact(str(cal_curve_csv), "qualitative")
                            if iso_curve_csv.exists():
                                mlflow.log_artifact(str(iso_curve_csv), "qualitative")
                    print(f"  Saved test holdout preds ({len(test_preds)} rows, {len(cals)} calibrators) + mappings to MLflow artifact")
                except Exception as e_save:
                    print(f"  WARNING: could not save test holdout artifacts: {e_save}")
            except Exception as e_test:
                print(f"WARNING: test holdout eval failed: {e_test}")

    # Qualitatives, diagnostic charts and model saving (Rank 0 only)
    model_uri = ""
    if trainer.is_world_process_zero():
        save_qualitative_k = int(train_cfg.get("save_qualitative_k", 0))
        if save_qualitative_k > 0 and pred_out is not None:
            try:
                w = 1.0 / 30.0 + gt
                per_sample_err = w * (preds - gt) ** 2
                worst_order = np.argsort(-per_sample_err)[:save_qualitative_k]
                best_order = np.argsort(per_sample_err)[:save_qualitative_k]
                paths_all = val_data["image_path"].values if "image_path" in val_data.columns else None
                qual_root = Path(output_dir) / "qualitative"
                base = Path(image_base_dir) if image_base_dir else None

                def _dump(order: Any, label: str) -> None:
                    paths = paths_all[order] if paths_all is not None else None
                    df = pd.DataFrame({
                        "rank": np.arange(1, len(order) + 1),
                        "filename": paths if paths is not None else order,
                        "gt": gt[order],
                        "pred": preds[order],
                        "abs_err": np.abs(preds[order] - gt[order]),
                        "weighted_err": per_sample_err[order],
                        "gender": gender[order],
                    })
                    sub = qual_root / label
                    sub.mkdir(parents=True, exist_ok=True)
                    df.to_csv(sub / f"{label}.csv", index=False)
                    img_dir = sub / "images"
                    img_dir.mkdir(exist_ok=True)
                    if paths is None:
                        return
                    for rank, row in enumerate(df.itertuples(index=False), start=1):
                        src = Path(row.filename)
                        if base and not src.is_absolute():
                            src = base / row.filename
                        if not src.exists():
                            continue
                        dst = img_dir / f"{rank:03d}_gt{row.gt:.3f}_pred{row.pred:.3f}_g{int(row.gender)}_{src.name}"
                        if dst.exists():
                            dst.unlink()
                        shutil.copy(src, dst)
                    print(f"Saved {len(df)} {label} : {sub}/{label}.csv + {img_dir}")

                _dump(worst_order, "worst")
                _dump(best_order, "best")

                try:
                    _save_diagnostic_charts(qual_root, preds, gt, gender)
                except Exception as e_chart:
                    print(f"WARNING: could not save diagnostic charts: {e_chart}")

                if use_mlflow:
                    if use_client and client and run_id:
                        client.log_artifacts(run_id, str(qual_root), "qualitative")
                    elif mlflow.active_run():
                        mlflow.log_artifacts(str(qual_root), "qualitative")
            except Exception as e:
                print(f"WARNING: could not save qualitative-K: {e}")

        ml_log_metrics(client, run_id, {
            "val_score": eval_score,
            "val_err_F": err_F,
            "val_err_M": err_M,
            "val_err_diff": err_diff,
            "final_eval_loss": eval_loss,
        })
        if use_mlflow:
            artifact = f"configs/architectures/{architecture_name}.yaml"
            if use_client and client and run_id:
                client.log_artifact(run_id, artifact)
            elif mlflow.active_run():
                mlflow.log_artifact(artifact)

        should_save = use_mlflow and (
            min_score_to_save is None or float(eval_score) < float(min_score_to_save)
        )
        if use_mlflow and not should_save:
            print(f"Skipping model save: eval_score={eval_score:.5f} ≥ best={min_score_to_save:.5f}")
        if should_save:
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
