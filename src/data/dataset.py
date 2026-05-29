from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

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
    split_ratio: float = 0.12,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """v10 — val split toujours = match P_test marginal Y (test_pmf strategy).
    Pas d'interpolation alpha, pas de stratified_yg fallback.
    """
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
    bin_width: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """v10 — val matches P_test marginal Y exactly (10 bins × 0.05).

    Bins rares (high-Y) où on a moins de samples que ce que P_test demande sont
    skippés (val plus petit que target, warning visible).
    """
    from src.utils.losses import _TEST_PMF, N_BINS, BIN_WIDTH
    test_pmf = np.asarray(_TEST_PMF, dtype=np.float64).flatten()
    n_bins = N_BINS
    bin_width = BIN_WIDTH
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
        print(f"  WARNING: {len(skipped)} bins under-sampled for val:")
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
    """Base dataset : 1 sample = 1 forward.

    v10 : transform PAR DÉFAUT None (= image raw). TargetedAugDataset applique
    le transform UNIQUEMENT sur les replicas via `get_with_transform()`.
    Pour val/test : transform=None toujours.
    """

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
        return self.get_with_transform(idx, self.transform)

    def get_with_transform(self, idx: int, transform: Optional[Any]) -> Dict[str, Any]:
        img = _open_rgb(self.image_paths[idx], self.base)
        return _encode(self.processor, img, self.targets[idx], self.genders[idx], transform)


class TargetedAugDataset(Dataset):
    """v10 — Unified replication dataset combining axe 1 (Y-shift) + axe 2 (G-balance).

    Pour chaque sample i, on calcule un `aug_repli_i` ∈ [1, K_max] via la formule unifiée :
      target_weight_i = P_target(g_i, y_i) / P_train(g_i, y_i)
                      = mix_y(α1) × mix_g(α2) / P_train
      aug_repli_i = bernoulli_round(target_weight_i ^ aug_share), capé à K_max

    AUGMENTATION CIBLÉE UNIQUEMENT :
      * 1ère occurrence d'un base sample (= original) → IMAGE RAW, pas d'aug
      * occurrences suivantes (= replicas) → aug stochastique appliqué pour les distinguer

    Logique : on n'augmente pas inutilement les samples sur-représentés (qui n'ont aucune
    réplica, donc 1 occurrence = original = pas d'aug). Aug seulement pour casser la
    mémorisation sur les cells rares (qui ont 2-3 copies virtuelles).

    Note : virtual_to_base est FIXE après __init__. Diversité epoch-à-epoch via shuffle
    DataLoader + l'aug stochastique sur les replicas.
    """

    def __init__(
        self,
        base_dataset: FaceOccDataset,
        targets: np.ndarray,
        gender: np.ndarray,
        axis1_power: float,
        axis2_power: float,
        aug_share: float,
        k_max: int = 3,
        seed: int = 42,
        transform: Optional[Any] = None,
    ) -> None:
        from src.utils.losses import compute_target_weights, split_loss_aug
        self.base = base_dataset
        self.transform = transform   # appliqué SEULEMENT sur replicas (cf __getitem__)
        self.targets = np.asarray(targets, dtype=np.float64)
        self.gender = np.asarray(gender, dtype=np.float64)
        self.axis1_power = float(axis1_power)
        self.axis2_power = float(axis2_power)
        self.aug_share = float(aug_share)
        self.k_max = int(k_max)

        # Compute target weight per sample
        self.target_weight = compute_target_weights(
            self.targets, self.gender,
            axis1_power=self.axis1_power, axis2_power=self.axis2_power,
        )

        # Split into loss + aug
        self.loss_weight, self.aug_repli = split_loss_aug(
            self.target_weight, aug_share=self.aug_share, k_max=self.k_max, seed=seed,
        )

        # Build virtual_to_base mapping (each base sample replicated aug_repli_i times)
        self.virtual_to_base = np.repeat(np.arange(len(self.target_weight)), self.aug_repli)
        self.virtual_loss_weight = self.loss_weight[self.virtual_to_base]

        # is_replica[v] = True si v est une COPIE virtuelle (pas la 1ère occurrence du base sample)
        # virtual_to_base est sorted by construction (np.repeat) donc detection vectorisée :
        self.is_replica = np.zeros(len(self.virtual_to_base), dtype=bool)
        if len(self.virtual_to_base) > 1:
            self.is_replica[1:] = self.virtual_to_base[1:] == self.virtual_to_base[:-1]

    def __len__(self) -> int:
        return len(self.virtual_to_base)

    def __getitem__(self, virtual_idx: int) -> Dict[str, Any]:
        base_idx = int(self.virtual_to_base[virtual_idx])
        # Apply transform ONLY on replicas (originals = raw image)
        active_transform = self.transform if (self.is_replica[virtual_idx] and self.transform is not None) else None
        item = self.base.get_with_transform(base_idx, active_transform)
        item["loss_weight"] = torch.tensor(
            float(self.virtual_loss_weight[virtual_idx]), dtype=torch.float32,
        )
        return item

    def summary(self) -> Dict[str, float]:
        return {
            "base_size": int(len(self.target_weight)),
            "virtual_size": int(len(self.virtual_to_base)),
            "n_replicas": int(self.is_replica.sum()),
            "replica_pct": float(100.0 * self.is_replica.mean()),
            "expansion_ratio": float(len(self.virtual_to_base) / max(len(self.target_weight), 1)),
            "target_weight_min": float(self.target_weight.min()),
            "target_weight_max": float(self.target_weight.max()),
            "target_weight_mean": float(self.target_weight.mean()),
            "loss_weight_min": float(self.loss_weight.min()),
            "loss_weight_max": float(self.loss_weight.max()),
            "aug_repli_max": int(self.aug_repli.max()),
            "aug_repli_mean": float(self.aug_repli.mean()),
        }
