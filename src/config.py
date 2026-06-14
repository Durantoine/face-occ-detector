from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "architectures"


@dataclass
class Config:
    name: str = "default"
    description: str = ""
    # data
    data_csv: str = "data/raw/train.csv"  # full dataset; v35 anchor-removal apparatus retired
    remove_csv: str = ""  # optional CSV of filenames (vetted render glitches) to drop from train+val
    image_dir: str = "data/raw"
    test_csv: str = "data/raw/test_students.csv"
    image_col: str = "filename"
    label_col: str = "FaceOcclusion"
    gender_col: str = "gender"
    val_ratio: float = 0.15
    val_mode: str = "ptrain"
    # augmentation: pre-rendered images appended to the TRAIN pool only; proportion HPO-searchable.
    # aug_csv = one or more CSVs (comma-separated) with columns filename, FaceOcclusion, gender
    # [, source_filename]. Image files live flat in aug_image_dir (matched by basename), so the
    # CSV's own path column may differ. source_filename (the original train frame) enables the
    # val-leakage drop.
    aug_csv: str = ""
    aug_image_dir: str = "data/aug"
    aug_proportion: float = 0.0  # fraction of the aug pool added to train (0 = off)
    aug_level: str = "off"  # on-the-fly label-preserving aug: off|light|medium (per-sample intensity scaled by importance w)
    # model
    backbone: str = "vit_small_patch16_224.augreg_in21k"
    pooling_type: str = "mean"
    grid_size: int = 4
    attn_queries: int = 4
    drop_path: float = 0.1
    pooling_dropout: float = 0.0
    head_dropout: float = 0.1
    head_mlp_ratio: float = 0.0  # 0 = linear head; else GELU MLP with hidden = round(ratio * pooled_dim)
    init_backbone_from: str | None = None
    pretrained_source: str | None = None
    # training
    lambda_gap: float = 1.0
    lambda_adaptive: bool = True
    lambda_lr: float = 0.2
    lambda_min: float = 1.0
    lambda_max: float = 2.0
    lambda_threshold: float = 3e-4
    sampler_participation: float = 1.0 # 1.0=pure sampler, 0.0=pure loss
    is_target: str = "hc" # "hc" or "joint"
    is_lambda: float = 0.10 # regularization floor on P_train in w = P_test/(P_train+is_lambda)
    # alpha of the IS-stratified EVAL/selection weight (linear blend P_train<->P_test): 0=raw,
    # 1=full P_test, 0.5=intermediate (matches the interim leaderboard, between train and test).
    eval_alpha: float = 0.5
    epochs: int = 10
    batch_size: int = 64
    lr: float = 3e-5
    weight_decay: float = 0.1
    layer_decay: float = 1.0
    warmup_ratio: float = 0.1
    early_stop_patience: int = 5
    seed: int = 42
    bf16: bool = True
    num_workers: int = 8
    save_qualitative_k: int = 10
    log_test_pred: bool = True  # log test-set prediction distribution (new-best trials only)
    compile_model: bool = True  # torch.compile the model (helps both CNN and ViT)
    out_dir: str = "results"


_FIELD_NAMES = {f.name for f in fields(Config)}


def _resolve_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    cand = CONFIG_DIR / f"{name_or_path}.yaml"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"Config not found: {name_or_path} (looked in {CONFIG_DIR})")


def load_config(name_or_path: str) -> tuple[Config, dict[str, Any]]:
    
    if name_or_path == "default":
        name_or_path = "efficientnet-mini-local"
    path = _resolve_path(name_or_path)
    raw = yaml.safe_load(path.read_text()) or {}
    flat: dict[str, Any] = {"name": raw.get("name", path.stem),
                            "description": raw.get("description", "")}
    for section in ("data", "model", "training"):
        for k, v in (raw.get(section) or {}).items():
            if k in _FIELD_NAMES:
                # v36: Cast to annotated type to fix YAML scientific notation parsing as strings (e.g. 4e-05)
                # and ensure consistency across platforms (MPS/Linux).
                ftype = Config.__annotations__.get(k)
                if ftype in (float, int, str) and v is not None:
                    try:
                        v = ftype(v)
                    except (ValueError, TypeError):
                        pass
                flat[k] = v
    optuna = raw.get("optuna", {}) or {}
    return Config(**flat), optuna
