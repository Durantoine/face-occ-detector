from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def create_balanced_sampler(labels: List[int], num_classes: int = 2) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=num_classes)
    weights = np.zeros(len(labels))
    for cls in range(num_classes):
        if counts[cls] > 0:
            weights[np.array(labels) == cls] = 1.0 / counts[cls]
    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=int(counts[counts > 0].min() * num_classes),
        replacement=True,
    )


def _scan_class_dirs(folder: Path, label_map: Optional[Dict[str, int]]) -> pd.DataFrame:
    class_dirs = sorted(d for d in folder.iterdir() if d.is_dir())
    if not class_dirs:
        raise ValueError(f"No class subdirectories in {folder}")
    if label_map is None:
        label_map = {d.name: i for i, d in enumerate(class_dirs)}
    dfs = []
    for d in class_dirs:
        if d.name not in label_map:
            continue
        paths = [p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        print(f"  {d.name} → label {label_map[d.name]}: {len(paths):,} images")
        dfs.append(pd.DataFrame({"image_path": [str(p) for p in paths], "label": label_map[d.name]}))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_image_folder(
    data_dir: str,
    label_map: Optional[Dict[str, int]] = None,
    split_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load images from a folder hierarchy.

    Layout A (pre-split):
        data_dir/train/<class>/*.jpg
        data_dir/val/<class>/*.jpg   (or valid / validation / test)

    Layout B (single folder, will be split):
        data_dir/<class>/*.jpg

    label_map: {"class_name": int_label}. Auto-assigned alphabetically if None.
    Returns (train_df, val_df) with columns [image_path, label].
    """
    root = Path(data_dir)
    train_sub = root / "train"
    val_sub = next((root / d for d in ("val", "valid", "validation", "test") if (root / d).is_dir()), None)

    if train_sub.is_dir():
        train_df = _scan_class_dirs(train_sub, label_map)
        val_df = _scan_class_dirs(val_sub, label_map) if val_sub else pd.DataFrame()
        print(f"Pre-split: train={len(train_df):,} val={len(val_df):,}")
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    df = _scan_class_dirs(root, label_map)
    print(f"Loaded {len(df):,} images from {data_dir}")

    if split_ratio == 0:
        return df.reset_index(drop=True), pd.DataFrame()

    try:
        train_df, val_df = train_test_split(df, test_size=split_ratio, random_state=seed, stratify=df["label"])
    except ValueError:
        train_df, val_df = train_test_split(df, test_size=split_ratio, random_state=seed)

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_csv_data(
    data_csv: str,
    extra_train_csv: Optional[str] = None,
    label_col: str = "label",
    subset_col: Optional[str] = None,
    split_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load from a CSV annotation file.

    CSV must have: image_path, <label_col>.
    The label column is always renamed to 'label' in the returned DataFrames.
    Returns (train_df, val_df) with columns [image_path, label].
    """
    df = pd.read_csv(data_csv)
    df = df.dropna(subset=["image_path", label_col])
    df = df[df["image_path"].astype(str).str.strip() != ""]
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})

    if extra_train_csv:
        extra = pd.read_csv(extra_train_csv).dropna(subset=["image_path", label_col])
        if label_col != "label":
            extra = extra.rename(columns={label_col: "label"})
        df = pd.concat([df, extra], ignore_index=True)

    print(f"Loaded {len(df):,} samples from {data_csv}")

    if split_ratio == 0:
        return df.reset_index(drop=True), pd.DataFrame()

    stratify = (
        df[subset_col].astype(str) + "_" + df["label"].astype(str)
        if subset_col and subset_col in df.columns
        else df["label"]
    )
    try:
        train_df, val_df = train_test_split(df, test_size=split_ratio, random_state=seed, stratify=stratify)
    except ValueError:
        train_df, val_df = train_test_split(df, test_size=split_ratio, random_state=seed)

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _load_data(path: str, **kwargs: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return load_csv_data(path, **kwargs) if path.endswith(".csv") else load_image_folder(path, **kwargs)


class FaceOccDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        processor: Any,
        image_base_dir: Optional[str] = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.base = Path(image_base_dir) if image_base_dir else None

    def _open(self, path: str) -> Image.Image:
        p = Path(path)
        return Image.open(self.base / p if (self.base and not p.is_absolute()) else p).convert("RGB")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        enc = self.processor(images=self._open(self.image_paths[idx]), return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class DynamicAugDataset(Dataset):
    """Real images + augmented pool, resampled each epoch.

    aug_pool must be a DataFrame with columns [image_path, label] (already normalized).
    """

    def __init__(
        self,
        real_paths: List[str],
        real_labels: List[int],
        aug_pool: pd.DataFrame,
        processor: Any,
        aug_rebalance_ratio: float = 1.0,
        seed: int = 42,
        min_aug_per_class: int = 0,
        image_base_dir: Optional[str] = None,
    ) -> None:
        self.real_paths = real_paths
        self.real_labels = real_labels
        self.processor = processor
        self.base = Path(image_base_dir) if image_base_dir else None

        self._aug_paths = aug_pool["image_path"].astype(str).tolist()
        self._aug_labels = aug_pool["label"].astype(int).tolist()
        self._by_class: Dict[int, List[int]] = {}
        for i, lbl in enumerate(self._aug_labels):
            self._by_class.setdefault(lbl, []).append(i)

        counts: Dict[int, int] = {}
        for lbl in real_labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        majority = max(counts.values())
        total = sum(counts.values())

        self._targets: Dict[int, int] = {}
        for cls in range(max(counts) + 1):
            count = counts.get(cls, 0)
            imb = abs(majority - count) / total if total > 0 else 0
            mpc = min_aug_per_class if imb <= 0.15 else 0
            deficit = int((majority - count) * aug_rebalance_ratio)
            pool = self._by_class.get(cls, [])
            self._targets[cls] = min(max(deficit, mpc), len(pool))

        self._current: List[int] = []
        self.resample(seed)

    def resample(self, seed: Optional[int] = None) -> None:
        rng = np.random.RandomState(seed)
        selected: List[int] = []
        for cls, target in self._targets.items():
            pool = self._by_class.get(cls, [])
            if target > 0 and pool:
                selected.extend(rng.choice(pool, size=target, replace=False).tolist())
        rng.shuffle(selected)
        self._current = selected
        print(f"[DynamicAugDataset] {len(selected):,} aug samples this epoch")

    def get_current_labels(self) -> List[int]:
        return list(self.real_labels) + [self._aug_labels[i] for i in self._current]

    def __len__(self) -> int:
        return len(self.real_paths) + len(self._current)

    def _open(self, path: str) -> Image.Image:
        p = Path(path)
        return Image.open(self.base / p if (self.base and not p.is_absolute()) else p).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < len(self.real_paths):
            path, label = self.real_paths[idx], self.real_labels[idx]
        else:
            i = self._current[idx - len(self.real_paths)]
            path, label = self._aug_paths[i], self._aug_labels[i]
        enc = self.processor(images=self._open(path), return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item
