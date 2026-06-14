from __future__ import annotations

import argparse
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
def predict(cfg: Config, ckpt: str, out_csv: str = "submission.csv") -> None:
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()
    cfg, _ = load_config(args.config)
    ckpt = args.ckpt or str(Path(cfg.out_dir) / cfg.name / "best.pt")
    predict(cfg, ckpt, args.out)
