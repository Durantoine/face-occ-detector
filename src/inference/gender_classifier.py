"""Test-time gender inference via hybrid MID lookup + Sapiens linear probe.

The challenge test_students.csv contains only filenames (no gender label). To apply
per-gender post-hoc calibration at inference, we need to estimate the gender for each
test sample. Two-stage strategy:

  1. **MID lookup** (~93.5% coverage on test_students.csv): the filename contains a
     Freebase Machine ID (m.XXXXX) identifying the person. If this MID also appears
     in train.csv, we lookup the gender (deterministic, 100% accurate for known MIDs).

  2. **Sapiens linear probe** (~6.5% remaining): for test samples whose MID is not in
     train.csv, we infer gender via a logistic regression trained on Sapiens features
     (extracted from train images with known gender labels).

Usage:
    classifier = GenderClassifier.fit_or_load(train_csv="data/raw/train.csv",
                                                 probe_cache="cache/sapiens_probe.pkl")
    genders = classifier.predict("data/raw/test_students.csv")  # → np.ndarray of {0, 1}
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# MID extraction + lookup
# ============================================================================

def extract_mid(filename: str) -> str:
    """Parse 'm.XXXXX' from filename path. Returns '' if not found."""
    for part in Path(filename).parts:
        if part.startswith("m.") and len(part) > 2:
            return part
    return ""


def build_mid_gender_mapping(train_csv: str) -> Dict[str, int]:
    """Mapping {MID: gender_int} from train.csv. Uses mode per MID (majority gender)
    in case of inconsistency (same person tagged differently across rows)."""
    df = pd.read_csv(train_csv)
    df["mid"] = df["filename"].apply(extract_mid)
    df = df[df["mid"] != ""]
    df["gender_int"] = (df["gender"].astype(float) >= 0.5).astype(int)
    mode_per_mid = df.groupby("mid")["gender_int"].agg(lambda s: int(s.mode().iloc[0]))
    return mode_per_mid.to_dict()


# ============================================================================
# Sapiens linear probe
# ============================================================================

def extract_sapiens_features(
    image_paths: List[str],
    image_base_dir: Optional[str] = None,
    model_name: str = "sapiens2_0.1b",
    batch_size: int = 64,
    device: str = "auto",
) -> np.ndarray:
    """Forward Sapiens backbone on a list of images, return CLS features (N, D)."""
    import torch
    from PIL import Image
    from src.models.face_occ_regressor import _build_backbone, _forward_backbone
    from src.models.dinov3_loader import get_image_processor

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading {model_name} backbone for feature extraction (device={device})...")
    backbone, hidden_dim = _build_backbone(model_name, drop_path_rate=0.0, pretrained=True)
    backbone = backbone.to(device).eval()
    processor = get_image_processor(model_name)
    base = Path(image_base_dir) if image_base_dir else None

    feats: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            imgs = []
            for p in batch_paths:
                fp = base / p if base and not Path(p).is_absolute() else Path(p)
                imgs.append(Image.open(fp).convert("RGB"))
            enc = processor(images=imgs, return_tensors="pt")
            pixel_values = enc["pixel_values"].to(device)
            out = _forward_backbone(backbone, pixel_values, model_name)  # (B, N, D)
            # Use CLS token if available (token 0) else mean pool
            cls_or_mean = out[:, 0, :] if out.size(1) > 1 else out.squeeze(1)
            feats.append(cls_or_mean.cpu().numpy())
            if (i // batch_size) % 20 == 0:
                print(f"  features: {i + len(batch_paths)}/{len(image_paths)}")
    return np.concatenate(feats, axis=0)


def train_sapiens_probe(
    train_csv: str,
    image_base_dir: str,
    sample_size: Optional[int] = None,
    model_name: str = "sapiens2_0.1b",
    seed: int = 42,
    n_cv_folds: int = 5,
    c_grid: Tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
) -> Tuple[object, np.ndarray, Dict[str, Any]]:
    """Train logistic regression on Sapiens features → gender.

    Pipeline :
    1. Extract features from `sample_size` (or full train if None) labeled samples
    2. Grid search C across {0.01, 0.1, 1, 10, 100, 1000} via StratifiedKFold CV
    3. Pick best C by OOF accuracy mean; report per-fold + per-class + confusion
    4. Refit on full data with best C (= production model)

    Returns:
        (clf, norm_params, metrics) where:
          - clf : sklearn LogisticRegression refitted on full data with best C
          - norm_params : (2, D) array [mean, std]
          - metrics : {cv_accuracy_mean, cv_accuracy_std, best_C, train_acc, all_c_results, ...}
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    df = pd.read_csv(train_csv)
    df = df[df["filename"].notna() & df["gender"].notna()].reset_index(drop=True)
    if sample_size is not None and len(df) > sample_size:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(df), sample_size, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    n = len(df)
    print(f"Training Sapiens probe on {'FULL' if sample_size is None else 'subsample'} "
          f"{n:,} labeled train samples ({model_name})...")
    features = extract_sapiens_features(df["filename"].tolist(), image_base_dir, model_name=model_name)
    y = (df["gender"].astype(float).values >= 0.5).astype(int)
    print(f"  Class balance: F={int((y==0).sum()):,} ({(y==0).mean()*100:.1f}%), "
          f"M={int((y==1).sum()):,} ({(y==1).mean()*100:.1f}%)")

    # === Standardization ===
    feat_mean = features.mean(axis=0, keepdims=True)
    feat_std = features.std(axis=0, keepdims=True) + 1e-8
    features_norm = (features - feat_mean) / feat_std

    # === Grid search C via StratifiedKFold ===
    print(f"\nGrid search C ∈ {list(c_grid)} with {n_cv_folds}-fold CV (out-of-fold)...")
    skf_indices = list(StratifiedKFold(n_splits=n_cv_folds, shuffle=True,
                                          random_state=seed).split(features_norm, y))
    cv_results: Dict[float, Dict[str, Any]] = {}
    oof_for_c: Dict[float, np.ndarray] = {}
    for C in c_grid:
        accs: List[float] = []
        oof = np.full(n, -1, dtype=int)
        for tr_idx, va_idx in skf_indices:
            clf_fold = LogisticRegression(max_iter=2000, C=C, n_jobs=-1, random_state=seed)
            clf_fold.fit(features_norm[tr_idx], y[tr_idx])
            preds_va = clf_fold.predict(features_norm[va_idx])
            oof[va_idx] = preds_va
            accs.append(float((preds_va == y[va_idx]).mean()))
        acc_mean = float(np.mean(accs))
        acc_std = float(np.std(accs))
        cv_results[C] = {"acc_mean": acc_mean, "acc_std": acc_std, "fold_accs": accs}
        oof_for_c[C] = oof
        print(f"  C={C:8.2f} : CV acc = {acc_mean:.4f} ± {acc_std:.4f}  "
              f"(folds: {[f'{a:.4f}' for a in accs]})")

    best_C = max(c_grid, key=lambda c: cv_results[c]["acc_mean"])
    best = cv_results[best_C]
    oof = oof_for_c[best_C]
    print(f"\n=> Best C = {best_C} (CV acc = {best['acc_mean']:.4f} ± {best['acc_std']:.4f})")

    # === OOF confusion + per-class for best C ===
    n_f, n_m = int((y == 0).sum()), int((y == 1).sum())
    correct_f = int(((oof == 0) & (y == 0)).sum())
    correct_m = int(((oof == 1) & (y == 1)).sum())
    print(f"OOF per-class accuracy: F = {correct_f/n_f:.4f} ({correct_f:,}/{n_f:,})  "
          f"M = {correct_m/n_m:.4f} ({correct_m:,}/{n_m:,})")
    fn_f = int(((oof == 1) & (y == 0)).sum())
    fp_m = int(((oof == 0) & (y == 1)).sum())
    print(f"Confusion: F→M = {fn_f:,}, M→F = {fp_m:,}")

    # === Refit on full data with best C ===
    print(f"\nRefitting on FULL {n:,} samples with C={best_C} (production model)...")
    clf_prod = LogisticRegression(max_iter=2000, C=best_C, n_jobs=-1, random_state=seed)
    clf_prod.fit(features_norm, y)
    train_acc = float(clf_prod.score(features_norm, y))
    print(f"  Production train accuracy: {train_acc:.4f}  "
          f"(gap with CV acc {best['acc_mean']:.4f} = overfit measure; should be small)")

    metrics = {
        "cv_accuracy_mean": best["acc_mean"],
        "cv_accuracy_std": best["acc_std"],
        "best_C": best_C,
        "train_acc": train_acc,
        "n_train_samples": n,
        "oof_acc_F": correct_f / n_f,
        "oof_acc_M": correct_m / n_m,
        "all_c_results": cv_results,
    }
    return clf_prod, np.concatenate([feat_mean, feat_std], axis=0), metrics


# ============================================================================
# Main classifier (MID lookup + Sapiens probe fallback)
# ============================================================================

class GenderClassifier:
    """Hybrid gender classifier: MID lookup primary + Sapiens probe fallback.

    Attributes:
        mid_mapping: {MID: 0|1}
        probe: trained sklearn classifier (or None if not built)
        probe_norm: (2, D) array [mean, std] for standardization
        sapiens_model_name: model used for feature extraction
    """

    def __init__(self, mid_mapping: Dict[str, int], probe: Optional[object] = None,
                  probe_norm: Optional[np.ndarray] = None,
                  sapiens_model_name: str = "sapiens2_0.1b") -> None:
        self.mid_mapping = mid_mapping
        self.probe = probe
        self.probe_norm = probe_norm
        self.sapiens_model_name = sapiens_model_name

    def predict(self, test_csv: str, image_base_dir: Optional[str] = None) -> np.ndarray:
        """Predict gender for each row of test_csv. Returns int array {0, 1}.
        Uses MID lookup when possible, Sapiens probe for unknown MIDs."""
        df = pd.read_csv(test_csv)
        df["mid"] = df["filename"].apply(extract_mid)
        df["gender_lookup"] = df["mid"].map(self.mid_mapping)
        n_known = int(df["gender_lookup"].notna().sum())
        n_unknown = len(df) - n_known
        print(f"MID lookup: {n_known:,}/{len(df):,} ({100*n_known/len(df):.2f}%) known")

        out = df["gender_lookup"].astype("Int64").to_numpy(dtype=object)
        if n_unknown > 0:
            if self.probe is None or self.probe_norm is None:
                print(f"WARNING: {n_unknown:,} MIDs unknown but no Sapiens probe available "
                      f"→ defaulting unknown to majority (M=1).")
                out = np.where(pd.isna(out), 1, out).astype(int)
            else:
                unknown_mask = df["gender_lookup"].isna().values
                print(f"Running Sapiens probe on {n_unknown:,} unknown samples...")
                unknown_paths = df.loc[unknown_mask, "filename"].tolist()
                features = extract_sapiens_features(unknown_paths, image_base_dir,
                                                       model_name=self.sapiens_model_name)
                features_norm = (features - self.probe_norm[0:1]) / (self.probe_norm[1:2] + 1e-8)
                preds_unknown = self.probe.predict(features_norm)
                out_int = np.zeros(len(df), dtype=int)
                out_int[~unknown_mask] = df.loc[~unknown_mask, "gender_lookup"].astype(int).values
                out_int[unknown_mask] = preds_unknown
                out = out_int
        return out.astype(int)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "mid_mapping": self.mid_mapping,
                "probe": self.probe,
                "probe_norm": self.probe_norm,
                "sapiens_model_name": self.sapiens_model_name,
            }, f)
        print(f"GenderClassifier saved to {path}")

    @classmethod
    def load(cls, path: str) -> "GenderClassifier":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return cls(
            mid_mapping=obj["mid_mapping"],
            probe=obj.get("probe"),
            probe_norm=obj.get("probe_norm"),
            sapiens_model_name=obj.get("sapiens_model_name", "sapiens2_0.1b"),
        )

    @classmethod
    def fit_or_load(cls, train_csv: str, image_base_dir: str,
                     cache_path: str = "cache/gender_classifier.pkl",
                     sample_size: Optional[int] = None,
                     sapiens_model_name: str = "sapiens2_0.1b",
                     force_refit: bool = False) -> "GenderClassifier":
        """Load from cache if present, else fit (MID mapping + Sapiens probe) and cache.

        Default `sample_size=None` = use FULL labeled train set. With CV + grid search
        on C, the production probe is now refit on all data after honest validation.
        """
        if Path(cache_path).exists() and not force_refit:
            print(f"Loading cached GenderClassifier from {cache_path}")
            return cls.load(cache_path)
        print(f"Fitting GenderClassifier (no cache or force_refit=True)...")
        mid_mapping = build_mid_gender_mapping(train_csv)
        print(f"  MID mapping: {len(mid_mapping):,} unique MIDs")
        probe, probe_norm, metrics = train_sapiens_probe(
            train_csv, image_base_dir,
            sample_size=sample_size, model_name=sapiens_model_name,
        )
        cls_obj = cls(mid_mapping=mid_mapping, probe=probe, probe_norm=probe_norm,
                       sapiens_model_name=sapiens_model_name)
        cls_obj._probe_metrics = metrics  # type: ignore[attr-defined]
        cls_obj.save(cache_path)
        return cls_obj

    def validate_against_mid_lookup(
        self, test_csv: str, image_base_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """Real-world accuracy test : run Sapiens probe on the test samples whose MID
        is already known (= ground truth) and compare. This estimates the probe's
        accuracy on the test distribution (vs CV which only tells us train-distribution).
        """
        if self.probe is None or self.probe_norm is None:
            print("WARNING: no probe loaded, skipping validation")
            return {}
        df = pd.read_csv(test_csv)
        df["mid"] = df["filename"].apply(extract_mid)
        df["gender_mid"] = df["mid"].map(self.mid_mapping)
        known = df[df["gender_mid"].notna()].copy()
        n_known = len(known)
        if n_known == 0:
            print("No MID-known test samples — skipping validation")
            return {}
        print(f"\nRunning Sapiens probe on {n_known:,} MID-known test samples for validation...")
        feats = extract_sapiens_features(known["filename"].tolist(), image_base_dir,
                                          model_name=self.sapiens_model_name)
        feats_norm = (feats - self.probe_norm[0:1]) / (self.probe_norm[1:2] + 1e-8)
        probe_preds = self.probe.predict(feats_norm)
        gt = known["gender_mid"].astype(int).values
        acc = float((probe_preds == gt).mean())
        n_f, n_m = int((gt == 0).sum()), int((gt == 1).sum())
        acc_f = float(((probe_preds == 0) & (gt == 0)).sum()) / n_f if n_f else 0.0
        acc_m = float(((probe_preds == 1) & (gt == 1)).sum()) / n_m if n_m else 0.0
        fn_f = int(((probe_preds == 1) & (gt == 0)).sum())
        fp_m = int(((probe_preds == 0) & (gt == 1)).sum())
        print(f"\n=== Sapiens probe accuracy on test (MID-known subset, n={n_known:,}) ===")
        print(f"  Overall  : {acc:.4f}")
        print(f"  F  acc   : {acc_f:.4f}  ({n_f-fn_f:,}/{n_f:,}; missed {fn_f:,})")
        print(f"  M  acc   : {acc_m:.4f}  ({n_m-fp_m:,}/{n_m:,}; missed {fp_m:,})")
        return {"test_mid_acc": acc, "test_mid_acc_F": acc_f, "test_mid_acc_M": acc_m,
                "n_test_mid_known": n_known}
