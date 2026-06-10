"""Re-estimate the v4 best run (trial 18) VAL performance with a *corrected*
importance reweighting whose tail ratio does not collapse.

Why: the v4 objective used w_imp = P_test/P_train per occlusion bin, but the
assumed test PMF (H_C) decays at the tail, so the ratio collapses to ~0.067 for
y>~0.34 — the hardest samples get almost no weight, making the score optimistic.
This script reloads the exact trained model, reconstructs trial 18's val split
(seed 276, stratified_yg), runs inference on the FULL val, and reports the
challenge score under several weightings + a per-bin breakdown so the tail
support (and its noise) is visible.

This reuses the v4 framework code (branch cora/convnext-baseline lineage), so the
model is rebuilt/loaded exactly as trained.

RUN (on a GPU box; full 20k val) — uses the EXACT P_test/P_train ratio to the
end of the tail (no clamp), unlike v4 which clamped the tail ratio to ~0.067:
    python scripts/eval_v4_reweighted.py \
        --weights v4_best_trial18_model.pth \
        --data-csv data/raw/train.csv --image-dir data/raw

Needs on the machine: this repo (branch cora/v4-best-eval), the mlruns model dir
above, data/raw/train.csv, and the val images under data/raw/. CPU works but is
slow for 20k imgs — prefer a GPU.

NOTE: trial 18 used EMA (decay 0.9998); the mlflow-saved weights are whatever the
trainer saved. If they differ from the EMA-evaluated weights, the reproduced raw
number may differ slightly from the logged 0.000904 — that's expected and does not
affect the *relative* comparison between weightings, which is the point here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.predict import load_model
from src.data.dataset import load_csv_data, _open_rgb
from src.models.dinov3_loader import get_image_processor

# Assumed test PMF over 25 occlusion bins of width 0.02 (same H_C used in training).
N_BINS, BIN_WIDTH = 25, 0.02
TEST_PMF = np.array([
    0.081394, 0.050224, 0.059882, 0.060745, 0.053540, 0.067261, 0.058360, 0.064360,
    0.062282, 0.054443, 0.065130, 0.059760, 0.056048, 0.051151, 0.045269, 0.035512,
    0.025891, 0.020349, 0.012773, 0.007651, 0.003732, 0.001861, 0.001295, 0.000603,
    0.000487,
], dtype=np.float64)
TEST_PMF = TEST_PMF / TEST_PMF.sum()


def occ_bin(y: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(y, float) / BIN_WIDTH).astype(int), 0, N_BINS - 1)


def _werr(pred: np.ndarray, gt: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * (pred - gt) ** 2) / np.sum(w))


def score(pred, gt, gender, w_imp=None):
    base = 1.0 / 30.0 + gt
    w = base if w_imp is None else base * w_imp
    f = gender < 0.5
    m = ~f
    eF, eM = _werr(pred[f], gt[f], w[f]), _werr(pred[m], gt[m], w[m])
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    return {"score": (eF + eM) / 2 + abs(eF - eM), "err_F": eF, "err_M": eM,
            "gap": abs(eF - eM), "ess": ess, "ess_pct": 100 * ess / len(gt)}


@torch.no_grad()
def infer(model, processor, df, image_dir, device, batch_size):
    base = Path(image_dir)
    preds = []
    paths = df["filename"].tolist()
    for i in range(0, len(paths), batch_size):
        imgs = [_open_rgb(p, base) for p in paths[i:i + batch_size]]
        pv = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        out = model(pixel_values=pv)
        logits = out["logits"] if isinstance(out, dict) else out
        preds.append(logits.squeeze(-1).float().cpu().numpy())
        if (i // batch_size) % 20 == 0:
            print(f"  inferred {min(i + batch_size, len(paths))}/{len(paths)}", flush=True)
    return np.clip(np.concatenate(preds), 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-uri", default=None,
                    help="mlflow logged-model dir (e.g. mlruns/1/models/m-8914.../artifacts) or runs:/<id>/model")
    ap.add_argument("--weights", default=None,
                    help="ALT to --model-uri: a bare .pth (pickled FaceOccRegressor) loaded via torch.load. "
                         "Location is free; just point at the file. Needs this branch's code on PYTHONPATH.")
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    ap.add_argument("--data-csv", default="data/raw/train.csv")
    ap.add_argument("--image-dir", default="data/raw")
    ap.add_argument("--val-seed", type=int, default=276,
                    help="trial 18 rotated val seed = 42 + 18*13 = 276 (reproduces the exact val).")
    ap.add_argument("--split-ratio", type=float, default=0.2)
    ap.add_argument("--n-buckets", type=int, default=10)
    ap.add_argument("--tail-floor", type=float, default=0.0,
                    help="DEFAULT 0 = EXACT P_test/P_train ratio kept to the end of the tail (no clamp). "
                         ">0 imposes an arbitrary floor — for sensitivity checks only, NOT recommended "
                         "(over-weights the tail beyond the assumed test distribution).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default="results/v4_reweighted_eval.csv")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[init] device={device} model_uri={args.model_uri}")

    # 1) reconstruct trial 18's exact val split (seed 276, stratified_yg)
    train_df, val_df = load_csv_data(args.data_csv, split_ratio=args.split_ratio,
                                     seed=args.val_seed, n_buckets=args.n_buckets,
                                     val_split_strategy="stratified_yg")
    gt = val_df["FaceOcclusion"].astype(float).values
    gender = val_df["gender"].astype(float).values
    print(f"[val] n={len(val_df)}  (F={int((gender<0.5).sum())}, M={int((gender>=0.5).sum())})  "
          f"y>0.3={int((gt>0.3).sum())}  y>0.5={int((gt>0.5).sum())}")

    # 2) per-bin importance ratio from the TRAIN pool (reproduces v4's collapsing ratio)
    p_train = np.bincount(occ_bin(train_df["FaceOcclusion"].astype(float).values), minlength=N_BINS).astype(float)
    p_train = p_train / p_train.sum()
    ratio_bin = TEST_PMF / (p_train + 1e-12)
    # exact ratio by default; an optional (NOT recommended) floor for sensitivity only
    ratio_used = np.maximum(ratio_bin, args.tail_floor) if args.tail_floor > 0 else ratio_bin
    ratio_label = (f"corrected (arbitrary floor >= {args.tail_floor})" if args.tail_floor > 0
                   else "exact P_test/P_train (no clamp, full tail)")
    bins = occ_bin(gt)

    # 3) inference on the full val
    if args.weights:
        obj = torch.load(args.weights, map_location=device)
        if isinstance(obj, dict):
            raise SystemExit("--weights looks like a state_dict, not a full pickled model; "
                             "use --model-uri with the mlflow model dir instead.")
        model = obj
        processor = get_image_processor(getattr(model, "model_name", "convnext_v2_base"))
    elif args.model_uri:
        model, processor = load_model(args.model_uri, args.tracking_uri)
    else:
        raise SystemExit("provide --weights <model.pth> OR --model-uri <mlflow dir>")
    model.eval().to(device)
    pred = infer(model, processor, val_df, args.image_dir, device, args.batch_size)

    # 4) score under each weighting
    res = {
        "raw (no importance, w=1/30+y)":      score(pred, gt, gender, None),
        ratio_label:                          score(pred, gt, gender, ratio_used[bins]),
    }
    print("\n================ SCORES (val seed {}) ================".format(args.val_seed))
    for k, v in res.items():
        print(f"{k:42s} score={v['score']:.6f}  errF={v['err_F']:.6f} errM={v['err_M']:.6f} "
              f"gap={v['gap']:.6f}  ESS={v['ess']:.0f} ({v['ess_pct']:.0f}%)")

    # 5) per-bin breakdown (where the score comes from / tail support)
    print("\n================ PER-BIN (y range | n | mean sq-err | weight share) ================")
    sq = (pred - gt) ** 2
    rows = []
    for b in range(N_BINS):
        mb = bins == b
        n = int(mb.sum())
        if n == 0:
            continue
        base_w = (1.0 / 30.0 + gt[mb])
        share_raw = base_w.sum()
        share_iw = (base_w * ratio_used[b]).sum()
        rows.append({"bin": b, "y_lo": round(b * BIN_WIDTH, 3), "n": n,
                     "mse": round(float(sq[mb].mean()), 6),
                     "ratio_exact": round(float(ratio_bin[b]), 3),
                     "ratio_used": round(float(ratio_used[b]), 3),
                     "wshare_raw_%": round(100 * share_raw, 2),
                     "wshare_iw_%": round(100 * share_iw, 2)})
    bin_df = pd.DataFrame(rows)
    # normalise weight shares to %
    bin_df["wshare_raw_%"] = (100 * bin_df["wshare_raw_%"] / bin_df["wshare_raw_%"].sum()).round(2)
    bin_df["wshare_iw_%"] = (100 * bin_df["wshare_iw_%"] / bin_df["wshare_iw_%"].sum()).round(2)
    print(bin_df.to_string(index=False))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    bin_df.to_csv(args.out, index=False)
    summ = pd.DataFrame([{"weighting": k, **v} for k, v in res.items()])
    summ.to_csv(args.out.replace(".csv", "_summary.csv"), index=False)
    print(f"\n[saved] {args.out} (+ _summary.csv)")


if __name__ == "__main__":
    main()
