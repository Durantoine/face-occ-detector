from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler


class GenderBalancedSampler(Sampler[int]):
    """Per-epoch: every Male sample seen exactly once + oversampled Females (also without
    replacement within each pass) so the batch composition is ≈ 50/50 F/M.

    Concretely, per epoch:
      - All n_M Males are yielded (each exactly once, shuffled)
      - n_M Females are yielded, drawn as (n_M // n_F) full permutations of the F set
        + (n_M % n_F) extras drawn without replacement from F.
      - Final order is one big shuffle of all 2·n_M indices.

    Each epoch sees all Males and oversamples Females by a factor ≈ n_M / n_F.
    Stateless except for epoch counter — call `set_epoch(e)` for reproducibility under DDP.
    """

    def __init__(self, gender: np.ndarray, seed: int = 42) -> None:
        g = (np.asarray(gender) >= 0.5).astype(int)
        self.f_idx = np.where(g == 0)[0].astype(np.int64)
        self.m_idx = np.where(g == 1)[0].astype(np.int64)
        if len(self.f_idx) == 0 or len(self.m_idx) == 0:
            raise ValueError(f"Cannot balance: F={len(self.f_idx)}, M={len(self.m_idx)}")
        self.n_per_class = max(len(self.f_idx), len(self.m_idx))
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
        f = self._draw_without_replacement(self.f_idx, self.n_per_class, rng)
        m = self._draw_without_replacement(self.m_idx, self.n_per_class, rng)
        combined = np.concatenate([f, m])
        rng.shuffle(combined)
        return iter(combined.tolist())

    def __len__(self) -> int:
        return 2 * self.n_per_class


def create_gender_balanced_sampler(gender: np.ndarray, seed: int = 42) -> GenderBalancedSampler:
    return GenderBalancedSampler(gender=gender, seed=seed)

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
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Val split with stratification matching P_test marginal Y."""
    df = pd.read_csv(data_csv)
    df = _normalize_df(df, image_col, label_col, gender_col)
    if extra_train_csv:
        extra = _normalize_df(pd.read_csv(extra_train_csv), image_col, label_col, gender_col)
        df = pd.concat([df, extra], ignore_index=True)

    print(f"Loaded {len(df):,} samples from {data_csv}")
    if split_ratio == 0 or len(df) < 2:
        return df.reset_index(drop=True), pd.DataFrame()

    if "FaceOcclusion" in df.columns:
        train_df, val_df = _split_val_to_match_test_pmf(df, split_ratio, seed)
    else:
        train_df, val_df = train_test_split(df, test_size=split_ratio, random_state=seed)

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _split_val_to_match_test_pmf(
    df: pd.DataFrame,
    split_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Allocate val samples per (gender, Y bin) cell.

    P_val_target(g, y) = P_train(g | y) × P_test(y)
    (test gender unknown — we assume P_test(g|y) ≈ P_train(g|y), only Y marginal shifts)
    """
    from src.utils.losses import _TEST_PMF, N_BINS, BIN_WIDTH
    test_pmf_y = np.asarray(_TEST_PMF, dtype=np.float64).flatten()
    n_bins = N_BINS
    bin_width = BIN_WIDTH
    y = df["FaceOcclusion"].astype(float).values
    g = (df["gender"].astype(float).values >= 0.5).astype(int) if "gender" in df.columns else np.zeros(len(df), dtype=int)
    bin_idx = np.clip((y / bin_width).astype(int), 0, n_bins - 1)

    counts = np.zeros((2, n_bins), dtype=np.float64)
    for gi, bi in zip(g, bin_idx):
        counts[gi, bi] += 1
    p_train_y = counts.sum(axis=0) / max(counts.sum(), 1)
    safe_y = np.maximum(p_train_y, 1e-12)
    p_train_g_given_y = counts / (counts.sum(axis=0, keepdims=True) + 1e-12)
    p_val_target_joint = test_pmf_y[None, :] * p_train_g_given_y

    val_size_target = max(int(len(df) * split_ratio), 1)
    target_per_cell = (p_val_target_joint * val_size_target).astype(int)

    rng = np.random.RandomState(seed)
    val_indices: List[int] = []
    skipped: List[Tuple[int, int, int, int]] = []
    for gi in range(2):
        for b in range(n_bins):
            in_cell = np.where((g == gi) & (bin_idx == b))[0]
            need = int(target_per_cell[gi, b])
            if need == 0:
                continue
            if len(in_cell) >= need:
                val_indices.extend(rng.choice(in_cell, need, replace=False).tolist())
            else:
                val_indices.extend(in_cell.tolist())
                skipped.append((gi, b, need, len(in_cell)))

    if skipped:
        print(f"  WARNING: {len(skipped)} cells under-sampled for val:")
        for gi, b, need, got in skipped[:5]:
            gname = "F" if gi == 0 else "M"
            y_lo = b * bin_width
            y_hi = y_lo + bin_width
            print(f"    {gname} bin {b:2d} [Y∈{y_lo:.3f}-{y_hi:.3f}]: needed {need}, got {got}")

    val_idx_arr = np.array(val_indices, dtype=int)
    train_idx_arr = np.setdiff1d(np.arange(len(df)), val_idx_arr)
    return df.iloc[train_idx_arr].copy(), df.iloc[val_idx_arr].copy()


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
