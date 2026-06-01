from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler


# v16: sampler classes retirées. Le rééquilibrage P_train→P_test est porté entièrement
# par les poids de loss (compute_balancing_weights returns loss_weights). Avec
# P_train(F)=0.324 et batch≥32, ~10-41 F par batch — assez pour les fairness mechs
# (MMD/DANN/OT need ≥2) et pour la variance des gradients sur le minoritaire.
# Voir docs/v16_theory.md §4.

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

DEFAULT_IMAGE_COL = "filename"
DEFAULT_LABEL_COL = "FaceOcclusion"
DEFAULT_GENDER_COL = "gender"


def _normalize_df(
    df: pd.DataFrame,
    image_col: str,
    label_col: Optional[str],
    gender_col: Optional[str],
) -> pd.DataFrame:
    df = df.copy()
    required = [image_col]
    if label_col:
        required.append(label_col)
    df = df.dropna(subset=required)
    df = df[df[image_col].astype(str).str.strip() != ""]
    rename = {image_col: "image_path"}
    if label_col:
        rename[label_col] = "FaceOcclusion"
    if gender_col and gender_col in df.columns:
        rename[gender_col] = "gender"
    df = df.rename(columns=rename)
    if "FaceOcclusion" in df.columns:
        df["FaceOcclusion"] = df["FaceOcclusion"].astype(np.float32)
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(np.float32)
    return df.reset_index(drop=True)


def load_csv_data(
    data_csv: str,
    extra_train_csv: Optional[str] = None,
    image_col: str = DEFAULT_IMAGE_COL,
    label_col: str = DEFAULT_LABEL_COL,
    gender_col: Optional[str] = DEFAULT_GENDER_COL,
    split_ratio: float = 0.15,
    test_split_ratio: float = 0.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into (train, val, test).

    Two modes:
      - test_split_ratio = 0 (legacy): val matches P_test via H_C; test = empty.
      - test_split_ratio > 0 (v12+): val iid from P_train (keeps rare-bin samples in
        train), test holdout matches P_test via H_C. Val metric estimated via
        stratified importance sampling (see metrics.compute_score_stratified_is).
    """
    df = pd.read_csv(data_csv)
    df = _normalize_df(df, image_col, label_col, gender_col)
    if extra_train_csv:
        extra = _normalize_df(pd.read_csv(extra_train_csv), image_col, label_col, gender_col)
        df = pd.concat([df, extra], ignore_index=True)

    print(f"Loaded {len(df):,} samples from {data_csv}")
    if split_ratio == 0 or len(df) < 2:
        return df.reset_index(drop=True), pd.DataFrame(), pd.DataFrame()

    if test_split_ratio > 0 and "FaceOcclusion" in df.columns:
        train_df, val_df, test_df = _split_train_val_test(df, split_ratio, test_split_ratio, seed)
        print(f"Train: {len(train_df):,} (iid P_train) | "
              f"Val: {len(val_df):,} (iid P_train, IS-stratified eval) | "
              f"Test holdout: {len(test_df):,} (match P_test via H_C)")
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    if "FaceOcclusion" in df.columns:
        train_df, val_df = _split_val_to_match_test_pmf(df, split_ratio, seed)
    else:
        train_df, val_df = train_test_split(df, test_size=split_ratio, random_state=seed)

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), pd.DataFrame()


def _split_train_val_test(
    df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wrapper around distribution.split_train_val_test for DataFrame I/O."""
    from src.utils.distribution import split_train_val_test
    y = df["FaceOcclusion"].astype(float).values
    g = df["gender"].astype(float).values if "gender" in df.columns else np.zeros(len(df))
    train_idx, val_idx, test_idx = split_train_val_test(y, g, val_ratio, test_ratio, seed=seed)
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy()


def _split_val_to_match_test_pmf(
    df: pd.DataFrame, split_ratio: float, seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy v11 two-way split: val resampled to match P_test via H_C.
    Wrapper around distribution.split_val_matching_test_pmf.
    """
    from src.utils.distribution import split_val_matching_test_pmf
    y = df["FaceOcclusion"].astype(float).values
    g = df["gender"].astype(float).values if "gender" in df.columns else np.zeros(len(df))
    train_idx, val_idx = split_val_matching_test_pmf(y, g, split_ratio, seed=seed)
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def _load_data(path: str, **kwargs: Any) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_csv_data(path, **kwargs)


def _open_rgb(path: str, base: Optional[Path]) -> Image.Image:
    p = Path(path)
    full = base / p if (base and not p.is_absolute()) else p
    return Image.open(full).convert("RGB")


def _encode(
    processor: Any,
    image: Image.Image,
    target: float,
    gender: float,
    transform: Optional[Any] = None,
) -> Dict[str, Any]:
    if transform is not None:
        image = transform(image)
    enc = processor(images=image, return_tensors="pt")
    item = {k: v.squeeze(0) for k, v in enc.items()}
    item["labels"] = torch.tensor([target, gender], dtype=torch.float32)
    return item


class FaceOccDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        targets: List[float],
        genders: List[float],
        processor: Any,
        image_base_dir: Optional[str] = None,
        transform: Optional[Any] = None,
        loss_weights: Optional[np.ndarray] = None,
    ) -> None:
        self.image_paths = image_paths
        self.targets = targets
        self.genders = genders
        self.processor = processor
        self.transform = transform
        self.base = Path(image_base_dir) if image_base_dir else None
        self.loss_weights = (
            np.asarray(loss_weights, dtype=np.float32) if loss_weights is not None else None
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img = _open_rgb(self.image_paths[idx], self.base)
        item = _encode(self.processor, img, self.targets[idx], self.genders[idx], self.transform)
        if self.loss_weights is not None:
            item["loss_weight"] = torch.tensor(float(self.loss_weights[idx]), dtype=torch.float32)
        return item
