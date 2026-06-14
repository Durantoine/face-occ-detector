from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from src.config import Config
from src.utils.distribution import p_train_val_indices

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
_MEAN, _STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# On-the-fly augmentation levels (HPO categorical). Each transform = (probability, magnitude); the
# probability is scaled per-sample by the importance weight so the interesting (high-w) ranges get
# augmented more. All are occlusion-LABEL-PRESERVING -- occlusion-ADDING aug (cutout/erasing) would
# change the regressed target and stays in the pre-rendered aug_csv pipeline (with adjusted labels).
_AUG_LEVELS: dict = {
    "off": None,
    "light":  {"flip": 0.5, "color": (0.7, 0.10), "blur": (0.0, 0.0), "noise": (0.0, 0.00), "affine": (0.2, 5.0)},
    "medium": {"flip": 0.5, "color": (0.8, 0.20), "blur": (0.25, 1.2), "noise": (0.3, 0.02), "affine": (0.3, 8.0)},
}
AUG_S_MIN = 0.25  # floor on the per-sample aug-intensity scale (over-represented bulk still gets some aug)


class FaceOccDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: Config, train: bool,
                 is_w: np.ndarray | None = None, aug_w: np.ndarray | None = None) -> None:
        self.paths = df[cfg.image_col].astype(str).tolist()
        self.y = df[cfg.label_col].astype(float).values
        self.g = (df[cfg.gender_col].astype(float).values if cfg.gender_col in df.columns
                  else np.zeros(len(df), dtype=float))
        self.is_w = is_w
        self.aug_w = aug_w
        self.base = Path(cfg.image_dir)
        # v30: Minimal transforms, source is already 224x224
        self.t = T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)])
        # On-the-fly label-preserving aug, train only; None for eval/predict. Per-sample intensity is
        # scaled by aug_w so the interesting (high importance-weight) ranges get augmented more.
        self.aug = _AUG_LEVELS.get(cfg.aug_level) if train else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        img = Image.open(self.base / self.paths[i]).convert("RGB")
        x = self._augment(img, i) if self.aug is not None else self.t(img)
        item = {"x": x,
                "y": torch.tensor(self.y[i], dtype=torch.float32),
                "g": torch.tensor(self.g[i], dtype=torch.float32)}
        if self.is_w is not None:
            # v30: Residual weights for the loss
            item["w"] = torch.tensor(self.is_w[i], dtype=torch.float32)
        return item

    def _augment(self, img: Image.Image, i: int) -> torch.Tensor:
        a = self.aug
        s = float(self.aug_w[i]) if self.aug_w is not None else 1.0  # higher importance w -> more aug
        if random.random() < a["flip"] * s:
            img = TF.hflip(img)
        pc, mc = a["color"]
        if mc > 0 and random.random() < pc * s:
            img = TF.adjust_brightness(img, 1.0 + random.uniform(-mc, mc))
            img = TF.adjust_contrast(img, 1.0 + random.uniform(-mc, mc))
            img = TF.adjust_saturation(img, 1.0 + random.uniform(-mc, mc))
            img = TF.adjust_hue(img, random.uniform(-mc / 2, mc / 2))
        pb, sb = a["blur"]
        if sb > 0 and random.random() < pb * s:
            img = TF.gaussian_blur(img, kernel_size=5, sigma=random.uniform(0.1, sb))
        pa, da = a["affine"]
        if da > 0 and random.random() < pa * s:
            img = TF.affine(img, angle=random.uniform(-da, da), translate=[0, 0],
                            scale=random.uniform(0.95, 1.05), shear=[0.0, 0.0])
        x = self.t(img)
        pn, sn = a["noise"]
        if sn > 0 and random.random() < pn * s:
            x = x + torch.randn_like(x) * sn
        return x


_VAL_SAMPLERS = {"ptrain_strat": p_train_val_indices}


def _log_split(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: Config) -> None:
    # F/M counts per occlusion bin -- the fairness gap is dominated by the sparse high-occ tail.
    print(f"\n[data] Split generated using mode: {cfg.val_mode}")
    print(f"{'Occlusion Range':<16} | {'Train F/M':<15} | {'Val F/M':<15}\n" + "-" * 54)
    y_tr = train_df[cfg.label_col].values; g_tr = train_df[cfg.gender_col].values >= 0.5
    y_vl = val_df[cfg.label_col].values; g_vl = val_df[cfg.gender_col].values >= 0.5
    def fm(y, g, lo, hi):
        m = ((y >= lo) & (y < hi)) if hi < 1.0 else ((y >= lo) & (y <= hi))
        return int((m & ~g).sum()), int((m & g).sum())
    for low, high in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        tf, tm = fm(y_tr, g_tr, low, high); vf, vm = fm(y_vl, g_vl, low, high)
        print(f"{int(low*100):2d}% - {int(high*100):2d}% {' ':<4} | {tf:>6,} / {tm:<6,} | {vf:>6,} / {vm:<6,}")
    print("-" * 54)
    print(f"{'TOTAL':<16} | {(~g_tr).sum():>6,} / {g_tr.sum():<6,} | {(~g_vl).sum():>6,} / {g_vl.sum():<6,}")
    print(f"Mean Occlusion   | {'F %.3f M %.3f' % (y_tr[~g_tr].mean(), y_tr[g_tr].mean()):<15} | "
          f"{'F %.3f M %.3f' % (y_vl[~g_vl].mean(), y_vl[g_vl].mean()):<15}\n", flush=True)


def make_splits(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    # v36: val = held-out frames from ALL identities (no test-MID restriction). Every train identity
    # is "seen", so a held-out frame of any of them is an equally valid transductive probe; the old
    # restriction only shrank/biased val (excluded 28% of data, lower-occ) without adding test signal.
    df = pd.read_csv(cfg.data_csv)
    rm_path = getattr(cfg, "remove_csv", "") or ""
    if rm_path and Path(rm_path).exists():
        drop = set(pd.read_csv(rm_path)[cfg.image_col])
        n0 = len(df); df = df[~df[cfg.image_col].isin(drop)].reset_index(drop=True)
        print(f"[data] removed {n0 - len(df)} vetted glitches ({rm_path})", flush=True)
    if getattr(cfg, "full_data", False):
        df = df.reset_index(drop=True)
        val_df = df.sample(n=min(2000, len(df)), random_state=cfg.seed).reset_index(drop=True)  # in-sample MONITORING only (NOT held out)
        print(f"[data] FULL-DATA: train on ALL {len(df):,} frames; val={len(val_df):,} in-sample (monitoring, not held out)", flush=True)
        _log_split(df, val_df, cfg)
        return df, val_df
    n_val = max(int(len(df) * cfg.val_ratio), 1)
    y, g = df[cfg.label_col].values, df[cfg.gender_col].values

    sampler = _VAL_SAMPLERS.get(cfg.val_mode)
    if sampler is not None:
        val_idx = sampler(y, g, n_val, seed=cfg.seed)
    else:  # "ptrain" / default: pure random on the natural distribution
        val_idx = np.random.RandomState(cfg.seed).choice(len(df), n_val, replace=False)

    mask = np.ones(len(df), dtype=bool); mask[val_idx] = False
    train_df = df.iloc[mask].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    print(f"[data] val: {len(val_df):,} held-out frames from all identities (no test-MID restriction)", flush=True)
    _log_split(train_df, val_df, cfg)
    return train_df, val_df


class StratifiedNoReplacementSampler(Sampler):
    """
    v30: Optimized Stratified No-Replacement Sampler.
    Keeps all rare samples (W > 1) and sub-samples common ones.
    """
    def __init__(self, weights: np.ndarray, rho: float = 1.0, 
                 num_replicas: int = 1, rank: int = 0, seed: int = 42):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.rho = rho
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        
        # Inclusion probability: p = min(1, w^rho)
        # Transition point is exactly at w=1 (P_target = P_train)
        self.p = torch.clamp(self.weights ** rho, 0, 1)
        
        # Expected total size (across all ranks)
        self.expected_total_size = int(torch.sum(self.p).item())
        # Size per rank (approximate)
        self.num_samples = math.ceil(self.expected_total_size / self.num_replicas)

    def __iter__(self):
        # Deterministic across ranks for same epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # 1. Selection
        # Samples with p=1 are ALWAYS selected. 
        # Samples with p<1 are selected via Bernoulli.
        mask = torch.bernoulli(self.p, generator=g).bool()
        # Safety: Force p=1 to be True (Bernoulli(1.0) is 1, but this is explicit)
        mask = mask | (self.p >= 0.999)
        
        indices = torch.where(mask)[0]
        
        # 2. Global shuffle
        perm = torch.randperm(len(indices), generator=g)
        indices = indices[perm].tolist()
        
        # 3. Shard the subset
        # To ensure all ranks have the same length for DDP, we pad/crop.
        # Crucially, we pad/crop from a SHUFFLED list, so even if we crop 1 or 2 samples
        # for alignment, it's statistically neutral.
        total_size = self.num_samples * self.num_replicas
        if len(indices) < total_size:
            # Pad by repeating
            indices += indices[:(total_size - len(indices))]
        else:
            # Crop to align
            indices = indices[:total_size]
            
        return iter(indices[self.rank:total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class GenderBalancedBatchSampler(Sampler):
    """v38: no-replacement BATCH sampler that puts BOTH genders in every batch (in their natural
    ratio) -> the per-batch fairness gap (err_F, err_M) is a low-variance estimate -> stable gap
    gradient (far fewer spurious sign-flips). Same importance sub-sampling as the stratified sampler
    (keep w>=1, Bernoulli-drop the w<1 surplus), then interleaves the kept F and M streams. NO example
    is drawn twice in an epoch -> no rare-tail over-sampling / overfit. DDP-safe (batches sharded by rank)."""
    def __init__(self, weights: np.ndarray, gender: np.ndarray, batch_size: int, rho: float = 1.0,
                 num_replicas: int = 1, rank: int = 0, seed: int = 42, remainder: str = "redistribute"):
        self.p = torch.clamp(torch.as_tensor(weights, dtype=torch.double) ** rho, 0, 1)
        self.is_male = np.asarray(gender) >= 0.5
        self.batch_size = int(batch_size)
        self.remainder = remainder  # "redistribute" (round-robin into existing batches) | "drop" (drop_last)
        self.num_replicas, self.rank, self.seed, self.epoch = num_replicas, rank, seed, 0
        self._n_batches = max(1, (int(self.p.sum().item()) // self.batch_size) // num_replicas)

    def _build_batches(self):
        g = torch.Generator(); g.manual_seed(self.seed + self.epoch)
        mask = torch.bernoulli(self.p, generator=g).bool() | (self.p >= 0.999)
        kept = torch.where(mask)[0].numpy()
        rng = np.random.RandomState(self.seed + self.epoch)
        gk = self.is_male[kept]
        F = kept[~gk].copy(); M = kept[gk].copy(); rng.shuffle(F); rng.shuffle(M)
        nF, nM = len(F), len(M)
        if nF == 0 or nM == 0:                       # degenerate -> plain shuffled batches
            allk = kept.copy(); rng.shuffle(allk)
            return [allk[i:i + self.batch_size].tolist()
                    for i in range(0, len(allk) - self.batch_size + 1, self.batch_size)]
        nfb = min(max(int(round(self.batch_size * nF / (nF + nM))), 1), self.batch_size - 1)  # >=1 each gender
        nmb = self.batch_size - nfb
        n_batches = min(nF // nfb, nM // nmb)
        if n_batches == 0:                            # extreme gender imbalance -> one shuffled batch
            allk = kept.copy(); rng.shuffle(allk); return [allk.tolist()]
        batches = []
        for b in range(n_batches):
            idx = np.concatenate([F[b * nfb:(b + 1) * nfb], M[b * nmb:(b + 1) * nmb]])
            rng.shuffle(idx); batches.append(idx.tolist())
        # v40: the leftover frames (not enough to form a balanced batch). "drop" = drop_last (random,
        # reshuffled each epoch, no occ bias). "redistribute" = round-robin into the existing balanced
        # batches -> exact coverage every epoch, no skewed final batch, no high-occ loss. Searched in HPO.
        if self.remainder != "drop":
            left = np.concatenate([F[n_batches * nfb:], M[n_batches * nmb:]])
            rng.shuffle(left)
            for j, ex in enumerate(left):
                batches[j % n_batches].append(int(ex))
        return batches

    def __iter__(self):
        return iter(self._build_batches()[self.rank::self.num_replicas])

    def __len__(self):
        return self._n_batches

    def set_epoch(self, epoch):
        self.epoch = epoch


def append_augmented(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Append a sampled fraction (cfg.aug_proportion) of pre-rendered augmented images to the TRAINING
    pool only -- augmented rows never enter val. cfg.aug_csv is one or more comma-separated CSVs;
    each row's image is matched by basename in cfg.aug_image_dir and rewritten to an absolute path
    (so it resolves regardless of cfg.image_dir). Rows whose 'source_filename' is a val frame are
    dropped (no leakage). The caller fits the sampler/loss weight on the RETURNED pool, so the
    P_test/P_train ratio accounts for the added images and does not double-count the augmented tail."""
    paths = [p.strip() for p in (getattr(cfg, "aug_csv", "") or "").split(",") if p.strip()]
    prop = float(getattr(cfg, "aug_proportion", 0.0) or 0.0)
    paths = [p for p in paths if Path(p).exists()]
    if not paths or prop <= 0:
        return train_df
    aug = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    leak = 0
    # Drop any aug row whose ORIGINAL train source(s) fell in val: covers single-source (source_filename)
    # AND two-source blends (source_a/source_b). source_blend is an intermediate image, not a train frame.
    src_cols = [c for c in aug.columns if c.startswith("source") and c != "source_blend"]
    if src_cols:
        valset = set(val_df[cfg.image_col])
        keep = ~aug[src_cols].isin(valset).any(axis=1)
        leak = int((~keep).sum()); aug = aug[keep].copy()
    # Resolve images by BASENAME across one or more comma-separated aug_image_dir roots (rglob, subdirs ok).
    name2path: dict = {}
    for root in [Path(p.strip()) for p in str(getattr(cfg, "aug_image_dir", "data/aug")).split(",") if p.strip()]:
        if root.exists():
            for p in root.rglob("*"):
                if p.suffix.lower() in IMAGE_EXTS:
                    name2path.setdefault(p.name, str(p.resolve()))
    aug[cfg.image_col] = aug[cfg.image_col].map(lambda f: name2path.get(Path(str(f)).name, str(f)))
    k = min(int(round(len(aug) * prop)), len(aug))
    if k <= 0:
        return train_df
    sample = aug.sample(n=k, random_state=cfg.seed)[[cfg.image_col, cfg.label_col, cfg.gender_col]]
    pool = pd.concat([train_df, sample], ignore_index=True)
    print(f"[data] augmented pool: +{k}/{len(aug)} imgs from {len(paths)} csv(s) (aug_proportion={prop:.2f})"
          + (f" | dropped {leak} val-derived (no leak)" if leak else ""), flush=True)
    return pool


def make_loaders(cfg: Config, rank: int = 0, world_size: int = 1, pin_memory: bool = True):
    train_df, val_df = make_splits(cfg)                 # train_df = REAL (returned for the eval metric)
    train_df = train_df.reset_index(drop=True)
    pool_df = append_augmented(train_df, val_df, cfg)   # real + sampled augmented (training pool)

    # Sampler/loss weight W = P_test/(P_pool+lambda), fit on the POOL we actually sample from. Fitting
    # on P_pool (not the real P_train) is what makes the reweighting correct AFTER augmentation: the
    # augmented tail is already less scarce, so it is not double-counted. The EVAL metric keeps the
    # REAL P_train (val is drawn from real data) -- it is fit in train.py from the returned train_df.
    from src.utils.distribution import get_joint_weight_fn
    y_pool = pool_df[cfg.label_col].values
    g_pool = pool_df[cfg.gender_col].values
    # v40: TRAINING loss + sampler weights use train_alpha (decoupled from eval_alpha, which now only
    # drives selection). train_alpha=0 -> uniform IS weight -> natural training (the metric weight
    # 1/30+y stays in the loss) and the sampler keeps only gender balancing, no IS oversampling.
    weight_fn = get_joint_weight_fn(y_pool, g_pool, target=cfg.is_target,
                                    alpha=getattr(cfg, "train_alpha", cfg.eval_alpha), lam=cfg.is_lambda)
    total_weights_raw = weight_fn(y_pool, g_pool)

    # Loss residual = W / min(1, W**rho) keeps importance sampling exact for any rho. At rho=1 this
    # is 1.0 on the surplus (W<1, fully handled by the sampler) and W on the scarcity tail (W>1).
    rho = cfg.sampler_participation
    p_inclusion = np.clip(total_weights_raw ** rho, 0, 1)
    loss_weights = total_weights_raw / np.maximum(p_inclusion, 1e-9)

    aug_w = np.clip(total_weights_raw, AUG_S_MIN, 1.0)  # follows cfg.is_target (coupled): full aug on w>=1, less on surplus
    train_ds = FaceOccDataset(pool_df, cfg, train=True, is_w=loss_weights, aug_w=aug_w)
    val_ds = FaceOccDataset(val_df, cfg, train=False)
    # Keep workers alive across epochs (avoids ~1000 pool restarts over an HPO sweep). The FD leak
    # [Errno 24] that motivated disabling this is fixed by the file_system sharing strategy + a high
    # ulimit (the sbatch sets `ulimit -n 65536`), not by tearing the pool down every epoch.
    if cfg.num_workers > 0:
        try:
            torch.multiprocessing.set_sharing_strategy("file_system")
        except Exception:
            pass
    dl = dict(num_workers=cfg.num_workers, pin_memory=pin_memory,
              persistent_workers=False)  # v38: workers reaped each epoch -> no FD accumulation across HPO trials ([Errno 24])

    # v30 per-sample stratified sampler; v38 optional gender-balanced BATCH sampler (stable gap gradient).
    if getattr(cfg, "sampler_mode", "stratified") == "group_batch":
        sampler = GenderBalancedBatchSampler(total_weights_raw, g_pool, cfg.batch_size, rho=rho,
                                             num_replicas=world_size, rank=rank, seed=cfg.seed,
                                             remainder=getattr(cfg, "batch_remainder", "redistribute"))
        train_loader = DataLoader(train_ds, batch_sampler=sampler, **dl)
    else:
        sampler = StratifiedNoReplacementSampler(total_weights_raw, rho=rho, num_replicas=world_size, rank=rank, seed=cfg.seed)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=True, **dl)
    
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False, **dl)
    return train_loader, val_loader, train_df, val_df, sampler

