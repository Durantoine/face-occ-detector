from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

DEFAULT_IMAGE_COL = "filename"
DEFAULT_LABEL_COL = "FaceOcclusion"
DEFAULT_GENDER_COL = "gender"


def create_balanced_sampler(group_keys: List[int], num_groups: int = 2) -> WeightedRandomSampler:
    counts = np.bincount(np.asarray(group_keys).astype(int), minlength=num_groups)
    weights = np.zeros(len(group_keys), dtype=np.float64)
    for g in range(num_groups):
        if counts[g] > 0:
            weights[np.asarray(group_keys).astype(int) == g] = 1.0 / counts[g]
    n_samples = int(counts[counts > 0].min() * num_groups) if (counts > 0).any() else len(group_keys)
    return WeightedRandomSampler(weights=weights.tolist(), num_samples=n_samples, replacement=True)


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
    split_ratio: float = 0.2,
    seed: int = 42,
    n_buckets: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(data_csv)
    df = _normalize_df(df, image_col, label_col, gender_col)
    if extra_train_csv:
        extra = _normalize_df(pd.read_csv(extra_train_csv), image_col, label_col, gender_col)
        df = pd.concat([df, extra], ignore_index=True)

    print(f"Loaded {len(df):,} samples from {data_csv}")
    if split_ratio == 0 or len(df) < 2:
        return df.reset_index(drop=True), pd.DataFrame()

    from src.utils.losses import stratify_key
    stratify = None
    if "gender" in df.columns and "FaceOcclusion" in df.columns:
        try:
            stratify = stratify_key(df["gender"], df["FaceOcclusion"], n_buckets=n_buckets)
        except Exception:
            stratify = None

    try:
        train_df, val_df = train_test_split(df, test_size=split_ratio, random_state=seed, stratify=stratify)
    except ValueError:
        train_df, val_df = train_test_split(df, test_size=split_ratio, random_state=seed)

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _load_data(path: str, **kwargs: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    ) -> None:
        self.image_paths = image_paths
        self.targets = targets
        self.genders = genders
        self.processor = processor
        self.transform = transform
        self.base = Path(image_base_dir) if image_base_dir else None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img = _open_rgb(self.image_paths[idx], self.base)
        return _encode(self.processor, img, self.targets[idx], self.genders[idx], self.transform)
