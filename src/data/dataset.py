from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler


class GenderBalancedSampler(Sampler[int]):
    """Per-epoch sampler with configurable F/M target ratio.

    target_F ∈ [P_train(F), P_test(F)] is the target Female fraction in the batches.
    target_F = P_train(F)  → equivalent to no sampler (preserves natural ratio)
    target_F = 0.5         → fully balanced 50/50
    Any value in between   → partial balancing

    Construction (per epoch, draws stratified without replacement, may repeat within
    each gender pool to reach the target count):

      - epoch_len = max(n_F / target_F, n_M / (1 - target_F))  (so the rarer class is fully visited)
      - n_F_to_draw = round(epoch_len · target_F)
      - n_M_to_draw = epoch_len - n_F_to_draw
      - draw n_F_to_draw F samples and n_M_to_draw M samples (each without replacement
        within full-permutation passes; extras drawn without replacement)
      - shuffle the union

    With target_F = P_train(F), epoch_len ≈ n_total and each sample is seen ~once.
    With target_F = 0.5, epoch_len = 2 · max(n_F, n_M).
    """

    def __init__(self, gender: np.ndarray, target_F: float = 0.5, seed: int = 42) -> None:
        g = (np.asarray(gender) >= 0.5).astype(int)
        self.f_idx = np.where(g == 0)[0].astype(np.int64)
        self.m_idx = np.where(g == 1)[0].astype(np.int64)
        if len(self.f_idx) == 0 or len(self.m_idx) == 0:
            raise ValueError(f"Cannot balance: F={len(self.f_idx)}, M={len(self.m_idx)}")
        target_F = float(target_F)
        if not (0.0 < target_F < 1.0):
            raise ValueError(f"target_F must be in (0, 1), got {target_F}")
        self.target_F = target_F
        # Epoch length = max so the rarer class (after weighting) is fully visited
        n_F, n_M = len(self.f_idx), len(self.m_idx)
        epoch_len = max(int(round(n_F / target_F)), int(round(n_M / (1.0 - target_F))))
        self.n_F_draw = int(round(epoch_len * target_F))
        self.n_M_draw = epoch_len - self.n_F_draw
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _draw_without_replacement(self, idx: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
        if n <= len(idx):
            return rng.choice(idx, n, replace=False)
        n_full = n // len(idx)
        n_extra = n % len(idx)
        parts = [rng.permutation(idx) for _ in range(n_full)]
        if n_extra > 0:
            parts.append(rng.choice(idx, n_extra, replace=False))
        return np.concatenate(parts)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.RandomState(self.seed + self.epoch)
        f = self._draw_without_replacement(self.f_idx, self.n_F_draw, rng)
        m = self._draw_without_replacement(self.m_idx, self.n_M_draw, rng)
        combined = np.concatenate([f, m])
        rng.shuffle(combined)
        return iter(combined.tolist())

    def __len__(self) -> int:
        return self.n_F_draw + self.n_M_draw


def create_gender_balanced_sampler(gender: np.ndarray, target_F: float = 0.5, seed: int = 42) -> GenderBalancedSampler:
    return GenderBalancedSampler(gender=gender, target_F=target_F, seed=seed)


class DistributedWeightedSampler(Sampler[int]):
    """DDP-aware weighted sampler with replacement.

    PyTorch's WeightedRandomSampler is NOT distributed-aware: in DDP, each rank
    would draw indices independently → duplicate / missed samples across ranks.

    This sampler:
      - Generates the SAME full draw sequence on all ranks (seeded by seed+epoch)
      - Each rank takes every num_replicas-th index starting at `rank` → disjoint shards
      - Total batched samples per epoch = num_samples (≈ len(dataset) by default)
      - Per rank length = num_samples // num_replicas

    Call `set_epoch(e)` between epochs for the seed to advance (HF Trainer does this).
    """

    def __init__(
        self,
        weights: np.ndarray,
        num_samples: Optional[int] = None,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
        replacement: bool = True,
    ) -> None:
        import torch
        self.weights = torch.as_tensor(np.asarray(weights, dtype=np.float64), dtype=torch.double)
        self.num_samples = int(num_samples) if num_samples is not None else len(weights)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.replacement = bool(replacement)
        self.epoch = 0
        # Truncate so total_size is divisible by num_replicas (no orphan sample)
        self.total_size = (self.num_samples // self.num_replicas) * self.num_replicas
        self.per_rank = self.total_size // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        import torch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, replacement=self.replacement, generator=g)
        return iter(indices[self.rank::self.num_replicas].tolist())

    def __len__(self) -> int:
        return self.per_rank

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
