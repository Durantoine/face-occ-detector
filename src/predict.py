from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import Config, load_config
from src.data.dataset import FaceOccDataset
from src.models.face_occ_regressor import FaceOccModel
from src.train import _device


@torch.no_grad()
def predict(cfg: Config, ckpt: str, out_csv: str = "test_predictions.csv") -> None:
    device = _device(0) # Pass 0 as local_rank default
    df = pd.read_csv(cfg.test_csv).dropna(subset=[cfg.image_col])
    df[cfg.label_col] = 0.0
    loader = DataLoader(FaceOccDataset(df, cfg, train=False),
                        batch_size=cfg.batch_size * 2, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))

    model = FaceOccModel(cfg.backbone, pretrained=False, pooling_type=cfg.pooling_type,
                         grid_size=cfg.grid_size, attn_queries=cfg.attn_queries,
                         head_mlp_ratio=cfg.head_mlp_ratio).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    model.eval()

    preds = []
    use_amp = device.type == "cuda"
    for b in loader:
        x = b["x"].to(device, non_blocking=(device.type == "cuda"))
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds.append(model(x).float().cpu().numpy())
        else:
            preds.append(model(x).float().cpu().numpy())
    preds = np.clip(np.concatenate(preds), 0.0, 1.0)

    out = pd.DataFrame({cfg.image_col: df[cfg.image_col].values, cfg.label_col: preds, cfg.gender_col: "x"})
    out.to_csv(out_csv, index=False)
    print(f"wrote {len(out):,} predictions → {out_csv}  (mean={preds.mean():.3f})", flush=True)


def _cfg_from_run(run_id: str, tracking_uri: str) -> Config:
    """Rebuild the Config from an mlflow run's logged params (train.py logs cfg.__dict__)."""
    from dataclasses import fields
    from mlflow.tracking import MlflowClient
    params = MlflowClient(tracking_uri).get_run(run_id).data.params
    flat = {}
    for f in fields(Config):
        if f.name not in params:
            continue
        v, ann = params[f.name], Config.__annotations__.get(f.name)
        try:
            if ann == "bool": flat[f.name] = str(v).lower() == "true"
            elif ann == "int": flat[f.name] = int(float(v))
            elif ann == "float": flat[f.name] = float(v)
            else: flat[f.name] = v
        except (ValueError, TypeError):
            flat[f.name] = v
    return Config(**flat)


if __name__ == "__main__":
    import re
    ap = argparse.ArgumentParser(description="Predict on the test set from a config name OR an mlflow run-id.")
    ap.add_argument("config", help="architecture config name, OR a 32-hex mlflow run-id")
    ap.add_argument("--ckpt", default=None, help="checkpoint path (default: results/<name>/best.pt)")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--tracking-uri", default=os.environ.get("FACE_OCC_TRACKING_URI", "sqlite:///mlflow.db"))
    args = ap.parse_args()
    if re.fullmatch(r"[0-9a-fA-F]{32}", args.config):
        cfg = _cfg_from_run(args.config, args.tracking_uri)
        print(f"[predict] cfg from run {args.config}: name={cfg.name} backbone={cfg.backbone}", flush=True)
    else:
        cfg, _ = load_config(args.config)
    ckpt = args.ckpt or str(Path(cfg.out_dir) / cfg.name / "best.pt")
    predict(cfg, ckpt, args.out)
