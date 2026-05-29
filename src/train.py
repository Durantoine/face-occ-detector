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
from mlflow.tracking import MlflowClient
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
    TargetedAugDataset,
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
    "architecture": os.environ.get("FACE_OCC_ARCH", "dinov3-vitb16-3090-v10"),
    "data_csv": "data/raw/train.csv",
    "val_data_csv": None,
    "output_dir": "./results",
    "tracking_uri": "sqlite:///mlflow.db",
    "use_mlflow": True,
    "resume_from": None,
}

# Keys consumed by our custom logic (not passed to HF TrainingArguments)
_NON_HF_TRAIN_KEYS = {
    "early_stopping_patience", "metric_for_best_model", "greater_is_better", "seed",
    "augmentation_level",
    "loss_type", "loss_focal_gamma", "loss_fairness_lambda",
    # v10 axes (rebalancing target)
    "axis1_power", "axis2_power", "aug_share", "aug_repli_max",
    # v10 feature fairness (lambdas HPO, activation via feature_fairness)
    "feature_fairness", "mmd_lambda", "adv_lambda",
    # pool diversity penalty
    "loss_query_diversity_lambda",
    "save_qualitative_k",
}


def _unwrap(module: Any) -> Any:
    return getattr(module, "module", module)


def _custom_collator(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Default collator + stack loss_weight if present (from TargetedAugDataset)."""
    out: Dict[str, torch.Tensor] = {}
    for key in batch[0]:
        if key == "loss_weight":
            out[key] = torch.stack([torch.as_tensor(b[key]) for b in batch]).float()
        else:
            out[key] = torch.stack([b[key] for b in batch])
    return out


class WeightedMSETrainer(Trainer):
    """v10 — simplifie l'héritage HF Trainer.
    Loss = WeightedMSELoss avec sample_loss_weight passé via le batch.
    Optional extras : query_diversity, DANN adv, MMD.
    LLRD, EMA, custom sampler, group_dro retirés.
    """

    def __init__(
        self,
        focal_gamma: float = 0.0,
        fairness_lambda: float = 1.0,
        query_diversity_lambda: float = 0.0,
        adv_lambda: float = 0.0,
        mmd_lambda: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._query_diversity_lambda = float(query_diversity_lambda)
        self._adv_lambda = float(adv_lambda)
        self._mmd_lambda = float(mmd_lambda)
        self.loss_fct = WeightedMSELoss(
            focal_gamma=focal_gamma,
            fairness_lambda=fairness_lambda,
        )
        print(f"WeightedMSELoss : focal_gamma={focal_gamma}, fairness_lambda={fairness_lambda}, "
              f"query_div_lambda={query_diversity_lambda}, adv_lambda={adv_lambda}, mmd_lambda={mmd_lambda}")

    def compute_loss(self, model: Any, inputs: Dict[str, Any], return_outputs: bool = False, num_items_in_batch: Any = None) -> Any:
        labels = inputs["labels"]
        sample_loss_weight = inputs.pop("loss_weight", None)
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        preds = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        self.loss_fct = self.loss_fct.to(preds.device)
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

        return (loss, outputs) if return_outputs else loss


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
    val_split_ratio = float(data_cfg.get("val_split_ratio", 0.12))

    train_path = _resolve_data_path(data_cfg, data_csv)
    if not train_path:
        raise ValueError("Provide data_csv via arg or YAML data.data_csv")

    common = dict(image_col=image_col, label_col=label_col, gender_col=gender_col)
    if val_data_csv:
        train_df, _ = _load_data(train_path, **common, extra_train_csv=extra_train, split_ratio=0)
        val_df, _ = _load_data(val_data_csv, **common, split_ratio=0)
        print(f"Pre-split: train={len(train_df):,} val={len(val_df):,}")
        return train_df, val_df

    return _load_data(train_path, **common, extra_train_csv=extra_train, seed=seed, split_ratio=val_split_ratio)


def _build_datasets(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    processor: Any,
    image_base_dir: Optional[str],
    augmentation_level: str,
    axis1_power: float,
    axis2_power: float,
    aug_share: float,
    aug_repli_max: int,
    seed: int,
) -> Tuple[Any, FaceOccDataset, Dict[str, float]]:
    # v10 : base dataset SANS transform (image raw). Aug appliqué SEULEMENT sur replicas
    # par TargetedAugDataset.get_with_transform() → originaux jamais augmentés.
    transform = build_train_transform(augmentation_level)
    train_base = FaceOccDataset(
        image_paths=train_data["image_path"].tolist(),
        targets=train_data["FaceOcclusion"].astype(float).tolist(),
        genders=train_data["gender"].astype(float).tolist(),
        processor=processor,
        image_base_dir=image_base_dir,
        transform=None,   # ← v10 : pas d'aug sur base. Replicas only.
    )
    targeted = TargetedAugDataset(
        base_dataset=train_base,
        targets=train_data["FaceOcclusion"].astype(float).values,
        gender=train_data["gender"].astype(float).values,
        axis1_power=axis1_power,
        axis2_power=axis2_power,
        aug_share=aug_share,
        k_max=aug_repli_max,
        seed=seed,
        transform=transform,   # ← appliqué uniquement sur replicas
    )
    val_ds = FaceOccDataset(
        image_paths=val_data["image_path"].tolist(),
        targets=val_data["FaceOcclusion"].astype(float).tolist(),
        genders=val_data["gender"].astype(float).tolist(),
        processor=processor,
        image_base_dir=image_base_dir,
        transform=None,
    )
    return targeted, val_ds, targeted.summary()


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
) -> Tuple[float, float, float, str, float, float]:
    cfg = load_architecture_config(architecture_name).to_dict()
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    output_dim = model_cfg.get("output_dim", 1)
    best_metric = train_cfg.get("metric_for_best_model", "eval_challenge_score")
    greater_is_better = bool(train_cfg.get("greater_is_better", False))
    image_base_dir = data_cfg.get("image_base_dir")
    augmentation_level = train_cfg.get("augmentation_level", "light_v10")

    # v10 axes
    axis1_power = float(train_cfg.get("axis1_power", 0.0))
    axis2_power = float(train_cfg.get("axis2_power", 0.0))
    aug_share = float(train_cfg.get("aug_share", 0.0))
    aug_repli_max = int(train_cfg.get("aug_repli_max", 3))

    # v10 fairness features
    feature_fairness = str(train_cfg.get("feature_fairness", "none"))
    mmd_active = feature_fairness in ("mmd", "both")
    dann_active = feature_fairness in ("dann", "both")
    mmd_lambda = float(train_cfg.get("mmd_lambda", 0.0)) if mmd_active else 0.0
    adv_lambda = float(train_cfg.get("adv_lambda", 0.01)) if dann_active else 0.0

    client, run_id, use_client = _start_or_attach_run(
        cfg, mlflow_tracking_uri, mlflow_run_id, mlflow_experiment, use_mlflow,
    )

    model_name = model_cfg.get("model_name", "dinov3_vits16")
    ml_log_params(client, run_id, {"architecture": cfg["name"]})
    ml_log_params(client, run_id, dict(model_cfg))
    ml_log_params(client, run_id, dict(train_cfg))
    ml_log_params(client, run_id, dict(data_cfg))

    processor = get_image_processor(model_name)
    train_data, val_data = _load_train_val(data_cfg, data_csv, val_data_csv, val_seed or seed)
    train_dataset, val_dataset, aug_summary = _build_datasets(
        train_data, val_data, processor, image_base_dir, augmentation_level,
        axis1_power=axis1_power, axis2_power=axis2_power, aug_share=aug_share,
        aug_repli_max=aug_repli_max, seed=val_seed or seed,
    )
    print(f"TargetedAugDataset summary: {aug_summary}")
    ml_log_metrics(client, run_id, {f"aug_{k}": v for k, v in aug_summary.items() if isinstance(v, (int, float))})

    label_col = data_cfg.get("label_col", DEFAULT_LABEL_COL)
    n_train_f = int((train_data["gender"] < 0.5).sum())
    n_train_m = int((train_data["gender"] >= 0.5).sum())
    n_val_f = int((val_data["gender"] < 0.5).sum())
    n_val_m = int((val_data["gender"] >= 0.5).sum())
    ml_log_params(client, run_id, {
        "seed": seed,
        "num_train_base": len(train_data),
        "num_train_virtual": len(train_dataset),
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
            num_heads=int(model_cfg.get("num_heads", 4)),
            pool_attn_dropout=float(model_cfg.get("pool_attn_dropout", 0.0)),
            pool_proj_dropout=float(model_cfg.get("pool_proj_dropout", 0.0)),
            enable_adv_disc=dann_active,
        )
    )

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

    compute_metrics = make_compute_metrics()   # v10 : pas de eval_pmf_ratio (val ≡ P_test direct)

    trainer = WeightedMSETrainer(
        focal_gamma=float(train_cfg.get("loss_focal_gamma", 0.0)),
        fairness_lambda=float(train_cfg.get("loss_fairness_lambda", 1.0)),
        query_diversity_lambda=float(train_cfg.get("loss_query_diversity_lambda", 0.0)),
        adv_lambda=adv_lambda,
        mmd_lambda=mmd_lambda,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=_custom_collator,
        callbacks=callbacks or None,
    )

    print(f"Training {cfg['name']} (best_metric={best_metric})")
    trainer.train()

    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss", float("inf"))
    eval_score = eval_results.get("eval_challenge_score", 0.0)
    err_diff = eval_results.get("eval_err_diff", 0.0)
    err_F = eval_results.get("eval_err_F", 0.0)
    err_M = eval_results.get("eval_err_M", 0.0)
    print(f"loss={eval_loss:.5f}  score={eval_score:.5f}  err_F={err_F:.5f}  err_M={err_M:.5f}  err_diff={err_diff:.5f}")
    mae_pct = eval_results.get("eval_mae_pct", 0.0)
    r2 = eval_results.get("eval_r2", 0.0)
    print(f"  human-readable : MAE_pct={mae_pct:.2f}%  R²={r2:.3f}")

    try:
        pred_out = trainer.predict(val_dataset)
    except Exception as e:
        print(f"WARNING: trainer.predict failed: {e}")
        pred_out = None

    preds = gt = gender = None
    if pred_out is not None and trainer.is_world_process_zero():
        preds_raw = pred_out.predictions
        if isinstance(preds_raw, (tuple, list)):
            preds_raw = preds_raw[0]
        preds = np.asarray(preds_raw).astype(np.float64).flatten()
        labels = np.asarray(pred_out.label_ids).astype(np.float64)
        gt = labels[:, 0] if labels.ndim == 2 else labels.flatten()
        gender = labels[:, 1] if (labels.ndim == 2 and labels.shape[1] >= 2) else np.zeros_like(gt)

    save_qualitative_k = int(train_cfg.get("save_qualitative_k", 0))
    if save_qualitative_k > 0 and pred_out is not None and trainer.is_world_process_zero():
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

    if not trainer.is_world_process_zero():
        return eval_loss, eval_score, err_diff, "", err_F, err_M

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

    model_uri = ""
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
