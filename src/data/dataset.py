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


def create_test_pmf_sampler(
    y: np.ndarray,
    test_pmf: np.ndarray,
    bin_width: float = 0.025,
    clip: float = 10.0,
    num_samples: Optional[int] = None,
    power: float = 1.0,
) -> WeightedRandomSampler:
    """Sampler that resamples the training data so the effective distribution of Y
    in each epoch matches `test_pmf`. Per-sample weight = (P_test[b] / P_train[b])^power
    where b is the Y-bin index. Clipped to [1/clip, clip] for stability.

    `power` (paired-α design v6.5) :
      * 1.0 → full sampler-side correction (legacy)
      * 0.5 → √-strength : combined with √-strength loss reweight = full correction
      * 0.0 → uniform → no sampler effect (loss does 100%)
    """
    y_arr = np.asarray(y, dtype=np.float64).flatten()
    test = np.asarray(test_pmf, dtype=np.float64).flatten()
    n_bins = len(test)
    bin_idx = np.clip((y_arr / bin_width).astype(int), 0, n_bins - 1)
    train_pmf = np.bincount(bin_idx, minlength=n_bins).astype(np.float64) / max(len(y_arr), 1)
    ratio = test / np.maximum(train_pmf, 1e-6)
    if power != 1.0:
        ratio = np.power(ratio, power)
    ratio = np.clip(ratio, 1.0 / clip, clip)
    weights = ratio[bin_idx]
    n = int(num_samples if num_samples is not None else len(y_arr))
    return WeightedRandomSampler(weights=weights.tolist(), num_samples=n, replacement=True)


def create_gender_within_bin_sampler(
    y: np.ndarray,
    gender: np.ndarray,
    target_pmf: Optional[np.ndarray] = None,
    bin_width: float = 0.025,
    n_bins: int = 20,
    clip: float = 10.0,
    power: float = 1.0,
) -> WeightedRandomSampler:
    """Sampler égalisant F/M *intra-bin* tout en suivant `target_pmf` sur Y.

    Pour chaque sample i de bin b(i) et genre g(i) ∈ {0=F, 1=M} :
        weight_i = (0.5 × target_pmf[b(i)] / count(g(i), b(i)))^power

    `power` (paired-α design v6.5) :
      * 1.0 → full sampler-side correction (legacy)
      * 0.5 → √-strength : combined with √-strength loss = full correction
      * 0.0 → uniform → no sampler effect (loss does 100%)

    Si `target_pmf=None` → utilise la PMF empirique de Y dans le train (préserve P_train).
    Si `target_pmf=_TEST_PMF_0025` → matche P_test (corrige aussi le shift Y).
    """
    y_arr = np.asarray(y, dtype=np.float64).flatten()
    g_arr = (np.asarray(gender, dtype=np.float64).flatten() >= 0.5).astype(int)
    bin_idx = np.clip((y_arr / bin_width).astype(int), 0, n_bins - 1)

    if target_pmf is None:
        bin_counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
        target = bin_counts / max(bin_counts.sum(), 1)
    else:
        target = np.asarray(target_pmf, dtype=np.float64).flatten()
        if len(target) != n_bins:
            raise ValueError(f"target_pmf has len {len(target)}, expected {n_bins}")
        target = target / max(target.sum(), 1e-9)

    cell_counts = np.zeros((2, n_bins), dtype=np.float64)
    for g, b in zip(g_arr, bin_idx):
        cell_counts[g, b] += 1

    n_total = len(y_arr)
    weights = np.zeros(n_total, dtype=np.float64)
    for i in range(n_total):
        g, b = g_arr[i], bin_idx[i]
        cnt = cell_counts[g, b]
        if cnt > 0:
            weights[i] = 0.5 * target[b] / cnt

    if power != 1.0:
        pos_mask = weights > 0
        weights[pos_mask] = np.power(weights[pos_mask], power)

    # Clip pour éviter qu'un sample dans une cellule ultra-rare ait un poids absurde
    pos = weights[weights > 0]
    if len(pos) > 0:
        median_w = float(np.median(pos))
        weights = np.clip(weights, 0.0, median_w * clip)

    if weights.sum() == 0:
        weights = np.ones(n_total) / n_total
    return WeightedRandomSampler(weights=weights.tolist(), num_samples=n_total, replacement=True)


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
        train_df, val_df = _split_val_to_match_test_pmf(df, split_ratio, seed)
        print(f"Val split = test_pmf (val matches P_test marginal on Y)")
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build val by sampling per-Y-bin so that the val marginal P_val(Y) = P_test(Y).

    Falls back to a smaller val when high-Y bins have too few samples on train.
    """
    from src.utils.losses import _TEST_PMF_0025
    test_pmf = np.asarray(_TEST_PMF_0025, dtype=np.float64).flatten()
    n_bins = len(test_pmf)
    y = df["FaceOcclusion"].astype(float).values
    bin_idx = np.clip((y / bin_width).astype(int), 0, n_bins - 1)

    val_size_target = max(int(len(df) * split_ratio), 1)
    target_per_bin = (test_pmf * val_size_target).astype(int)
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


class YConditionalAugDataset(Dataset):
    """Expansion virtuelle d'un dataset où chaque sample est répliqué selon le bin Y.

    Pour chaque sample i dans le bin b_i, l'espérance du nombre de copies virtuelles est :
        k_i = (P_test(b_i) / P_train(b_i))^aug_share, clippé à [1/clip, clip]

    Stochastic Bernoulli rounding préserve E[copies] = k_i exact :
        copies = floor(k_i) + 1{Bernoulli(k_i - floor(k_i))}

    Chaque accès __getitem__ ré-applique la pipeline d'augmentation (côté base dataset
    via FaceOccDataset.transform stochastique) → vue différente pour chaque copie
    virtuelle d'un même sample base. Combiné avec un sampler test_pmf ou un loss
    reweight, l'effet total sur le gradient = r^(sampler_power + loss_power + aug_share).

    **DESIGN NOTE** : virtual_to_base est fixé au __init__ et NE CHANGE PAS pendant
    le training. Sinon le sampler (qui prend des poids alignés sur virtual_to_base à
    sa création) deviendrait incohérent à chaque re-roll. La diversité epoch-à-epoch
    vient de :
      (a) la pipeline d'augmentation stochastique sur chaque __getitem__ call
      (b) le sampler (with replacement) qui pioche différentes virtual_idx par epoch
      (c) le DataLoader random shuffle
    Largement suffisant pour éviter la memorization.

    Stochastic Bernoulli rounding (au __init__) préserve E[copies] = k_float exact :
        copies = floor(k_float) + 1{Bernoulli(k_float - floor(k_float))}
    """

    def __init__(
        self,
        base_dataset: Dataset,
        y_array: np.ndarray,
        aug_share: float,
        test_pmf: np.ndarray,
        bin_width: float = 0.025,
        clip: float = 10.0,
        seed: int = 42,
    ) -> None:
        self.base = base_dataset
        self.y_array = np.asarray(y_array, dtype=np.float64)
        self.aug_share = float(aug_share)
        self.bin_width = bin_width
        self.clip = clip

        n_bins = len(test_pmf)
        self.bin_idx = np.clip((self.y_array / bin_width).astype(int), 0, n_bins - 1)
        train_pmf = np.bincount(self.bin_idx, minlength=n_bins).astype(np.float64) / max(len(self.y_array), 1)
        ratio = test_pmf / np.maximum(train_pmf, 1e-6)
        if self.aug_share > 0:
            ratio = np.power(ratio, self.aug_share)
        else:
            ratio = np.ones_like(ratio)
        ratio = np.clip(ratio, 1.0 / clip, clip)
        self.k_float = ratio[self.bin_idx]   # shape (N,) — espérance copies par sample

        # Bernoulli stochastic rounding (deterministic seed → reproducible across runs).
        # Fixed mapping after __init__ (no per-epoch reroll, see DESIGN NOTE).
        rng = np.random.RandomState(seed)
        floor = np.floor(self.k_float).astype(int)
        frac = self.k_float - floor
        extra = (rng.uniform(size=len(self.k_float)) < frac).astype(int)
        k = np.maximum(floor + extra, 0)
        self.virtual_to_base = np.repeat(np.arange(len(self.k_float)), k)

    def __len__(self) -> int:
        return len(self.virtual_to_base)

    def __getitem__(self, virtual_idx: int) -> Dict[str, Any]:
        base_idx = int(self.virtual_to_base[virtual_idx])
        return self.base[base_idx]


def create_sampler_weights_for_virtual(
    y_base: np.ndarray,
    test_pmf: np.ndarray,
    virtual_to_base: np.ndarray,
    sampler_power: float,
    bin_width: float = 0.025,
    clip: float = 10.0,
) -> WeightedRandomSampler:
    """Poids sampler pour un dataset virtuel (expansé via YConditionalAugDataset).

    weight[virtual_idx] = (P_test(b) / P_train(b))^sampler_power
    où b est le bin de y_base[virtual_to_base[virtual_idx]]. Si sampler_power=0, poids uniformes.
    """
    n_bins = len(test_pmf)
    bin_idx_base = np.clip((np.asarray(y_base) / bin_width).astype(int), 0, n_bins - 1)
    train_pmf = np.bincount(bin_idx_base, minlength=n_bins).astype(np.float64) / max(len(y_base), 1)
    ratio = test_pmf / np.maximum(train_pmf, 1e-6)
    if sampler_power != 1.0:
        ratio = np.power(ratio, sampler_power)
    ratio = np.clip(ratio, 1.0 / clip, clip)
    weights_virtual = ratio[bin_idx_base][virtual_to_base]
    return WeightedRandomSampler(
        weights=weights_virtual.tolist(),
        num_samples=len(weights_virtual),
        replacement=True,
    )
