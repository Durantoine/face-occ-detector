from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import FaceOccDataset
from src.models.face_occ_regressor import FaceOccModel
from src.optimize import _ALIASES
from src.train import _device


def _cfg_from_trial(base, trial: optuna.trial.FrozenTrial):
    """Rebuild a trial's Config: apply its searched params (with the HPO aliases) onto the base."""
    overrides: dict = {}
    for k, v in trial.params.items():
        alias = _ALIASES.get(k, k)
        for a in (alias if isinstance(alias, list) else [alias]):
            overrides[a] = v
    return dataclasses.replace(base, name=f"{base.name}_trial{trial.number}", **overrides)


@torch.no_grad()
def _predict_one(cfg, ckpt: Path, loader: DataLoader, device) -> np.ndarray:
    model = FaceOccModel(cfg.backbone, pretrained=False, pooling_type=cfg.pooling_type,
                         grid_size=cfg.grid_size, attn_queries=cfg.attn_queries,
                         head_mlp_ratio=cfg.head_mlp_ratio).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    model.eval()
    use_amp = device.type == "cuda"
    preds = []
    for b in loader:
        x = b["x"].to(device, non_blocking=(device.type == "cuda"))
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds.append(model(x).float().cpu().numpy())
        else:
            preds.append(model(x).float().cpu().numpy())
    return np.concatenate(preds)


def ensemble_predict(config: str, topk: int, storage: str, out_csv: str) -> None:
    base, _ = load_config(config)
    study = optuna.load_study(study_name=base.name, storage=storage)
    trials = sorted((t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE
                     and t.value is not None and t.value < float("inf")),
                    key=lambda t: t.value)
    if not trials:
        raise SystemExit(f"No finite completed trials in study '{base.name}'")

    device = _device(0)
    df = pd.read_csv(base.test_csv).dropna(subset=[base.image_col])
    df[base.label_col] = 0.0
    loader = DataLoader(FaceOccDataset(df, base, train=False), batch_size=base.batch_size * 2,
                        num_workers=base.num_workers, pin_memory=(device.type == "cuda"))

    preds, used = [], []
    for t in trials:
        cfg = _cfg_from_trial(base, t)
        ckpt = Path(cfg.out_dir) / cfg.name / "best.pt"
        if not ckpt.exists():
            print(f"  skip trial {t.number} (score {t.value:.5f}): {ckpt} missing", flush=True)
            continue
        p = _predict_one(cfg, ckpt, loader, device)
        preds.append(p); used.append((t.number, t.value))
        print(f"  + trial {t.number} (score {t.value:.5f}) -> {ckpt}", flush=True)
        if len(used) >= topk:
            break

    if not preds:
        raise SystemExit("No usable checkpoints found (run the sweep / sync results/ from the bucket).")

    ens = np.clip(np.mean(preds, axis=0), 0.0, 1.0)
    out = pd.DataFrame({base.image_col: df[base.image_col].values, base.label_col: ens, base.gender_col: "x"})
    out.to_csv(out_csv, index=False)
    print(f"\nensembled {len(used)} trials {[(n, round(v, 5)) for n, v in used]}", flush=True)
    print(f"wrote {len(out):,} predictions -> {out_csv}  (mean={ens.mean():.3f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ensemble the top-K Optuna trials of a config on the test set.")
    ap.add_argument("config", help="architecture config name (= optuna study_name)")
    ap.add_argument("--topk", type=int, default=5, help="number of best trials to ensemble")
    ap.add_argument("--storage", default="sqlite:///optuna.db")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()
    ensemble_predict(args.config, args.topk, args.storage, args.out)
