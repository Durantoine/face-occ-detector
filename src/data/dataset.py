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


def create_test_pmf_sampler(
    y: np.ndarray,
    test_pmf: np.ndarray,
    train_pmf: Optional[np.ndarray] = None,
    bin_width: float = 0.025,
    clip: float = 20.0,
    num_samples: Optional[int] = None,
    power: float = 1.0,
) -> WeightedRandomSampler:
    """Sampler that resamples training data so the effective Y-distribution matches `test_pmf`.

    Per-sample weight = (P_test[b] / P_train[b])^power where b is the Y-bin.
    v9 : si `train_pmf=None`, on utilise _TRAIN_PMF_0025 (smoothed Mix(spike+Beta))
    plutôt que l'empirique du subset — ratios stables et reproductibles entre trials.
    """
    from src.utils.losses import _TRAIN_PMF_0025
    y_arr = np.asarray(y, dtype=np.float64).flatten()
    test = np.asarray(test_pmf, dtype=np.float64).flatten()
    n_bins = len(test)
    bin_idx = np.clip((y_arr / bin_width).astype(int), 0, n_bins - 1)
    train = train_pmf if train_pmf is not None else _TRAIN_PMF_0025
    ratio = test / np.maximum(train, 1e-6)
    if power != 1.0:
        ratio = np.power(ratio, power)
    ratio = np.clip(ratio, 1.0 / clip, clip)
    weights = ratio[bin_idx]
    n = int(num_samples if num_samples is not None else len(y_arr))
    return WeightedRandomSampler(weights=weights.tolist(), num_samples=n, replacement=True)


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
    val_split_strategy: str = "stratified_yg",
    val_split_alpha: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(data_csv)
    df = _normalize_df(df, image_col, label_col, gender_col)
    if extra_train_csv:
        extra = _normalize_df(pd.read_csv(extra_train_csv), image_col, label_col, gender_col)
        df = pd.concat([df, extra], ignore_index=True)

    print(f"Loaded {len(df):,} samples from {data_csv}")
    if split_ratio == 0 or len(df) < 2:
        return df.reset_index(drop=True), pd.DataFrame()

    if val_split_strategy == "test_pmf" and "FaceOcclusion" in df.columns:
        train_df, val_df = _split_val_to_match_test_pmf(df, split_ratio, seed, val_split_alpha=val_split_alpha)
        print(f"Val split = test_pmf (α={val_split_alpha:.2f} interpolation P_test ↔ P_train)")
    else:
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


def _split_val_to_match_test_pmf(
    df: pd.DataFrame,
    split_ratio: float,
    seed: int,
    bin_width: float = 0.025,
    val_split_alpha: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build val with target distribution P_val = α·P_test + (1−α)·P_train (v9).

    val_split_alpha (∈ [0, 1]) controls the trade-off train/eval :
      * α=1 : val matches P_test exact (v8 B' behavior). Eval lisible, train perd
              une grosse part proportionnelle des bins haut-Y rares.
      * α=0 : val matches P_train marginal. Train préserve les bins rares mais eval
              nécessite un reweight ×10 sur les bins haut-Y (variance amplifiée).
      * α=0.5 : mid-ground. Train récupère ~50% des high-Y rares vs α=1, eval reweight
                modéré (×~1.7 sur bin 18 vs ×5 pour α=0).

    Eval reweight = P_test / P_val_target compensera dans train.py pour rester
    estimateur non-biaisé du risque test sous H1 (covariate shift Y-only).

    Falls back to a smaller val when bins rares have too few samples on train.
    """
    from src.utils.losses import _TEST_PMF_0025
    test_pmf = np.asarray(_TEST_PMF_0025, dtype=np.float64).flatten()
    n_bins = len(test_pmf)
    y = df["FaceOcclusion"].astype(float).values
    bin_idx = np.clip((y / bin_width).astype(int), 0, n_bins - 1)

    val_size_target = max(int(len(df) * split_ratio), 1)

    # v9 : interpolation P_test ↔ P_train sur le target val
    alpha = float(np.clip(val_split_alpha, 0.0, 1.0))
    if alpha < 1.0:
        train_counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
        train_pmf = train_counts / max(train_counts.sum(), 1e-9)
        target_pmf = alpha * test_pmf + (1.0 - alpha) * train_pmf
        target_pmf = target_pmf / max(target_pmf.sum(), 1e-9)
        print(f"  val_split_alpha={alpha:.2f} → interpolation P_test ({alpha*100:.0f}%) + P_train ({(1-alpha)*100:.0f}%)")
    else:
        target_pmf = test_pmf
        print(f"  val_split_alpha=1.0 → val matches P_test exact (B' behavior)")

    target_per_bin = (target_pmf * val_size_target).astype(int)
    rng = np.random.RandomState(seed)
    val_indices: List[int] = []
    skipped: List[Tuple[int, int, int]] = []
    for b in range(n_bins):
        in_bin = np.where(bin_idx == b)[0]
        need = int(target_per_bin[b])
        if need == 0:
            continue
        if len(in_bin) >= need:
            val_indices.extend(rng.choice(in_bin, need, replace=False).tolist())
        else:
            val_indices.extend(in_bin.tolist())
            skipped.append((b, need, len(in_bin)))

    if skipped:
        print(f"  WARNING: {len(skipped)} bins under-sampled for val (val will be slightly smaller):")
        for b, need, got in skipped[:5]:
            y_lo = b * bin_width
            y_hi = y_lo + bin_width
            print(f"    bin {b:2d} [Y∈{y_lo:.3f}-{y_hi:.3f}]: needed {need}, got {got}")

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


