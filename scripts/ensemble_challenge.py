"""Challenge ensemble pipeline.

For each CONFIG, take its top-K Optuna trials and, per trial:
  1. predict the test set from the trial's existing best.pt  (BEST epoch checkpoint)
  2. RE-TRAIN the exact same hyperparameters on 100% of the data (no held-out val,
     epochs fixed = the trial's best epoch)
  3. predict the test set from that full-data model

Predictions are organised under:
  challenge_results/<config>/trial<N>/best/test_predictions.csv   (from best.pt)
  challenge_results/<config>/trial<N>/full/test_predictions.csv   (from the 100%-data retrain)

Finally it ensembles the FULL-DATA predictions two ways:
  challenge_results/ensemble_unweighted.csv   (equal weight per model)
  challenge_results/ensemble_weighted.csv     (weight ~ 1/trial_score: better trial -> more weight)

Usage:
  python scripts/ensemble_challenge.py 3 sapiens-final-v2 dino-vitb16-H100 convnextv2-large-H100
                                        ^K  ^------------- Y configs --------------^
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from src.config import load_config
from src.optimize import _ALIASES
from src.predict import predict
from src.train import train

RESULTS = Path("challenge_results")


def _cfg_from_trial(base, trial: optuna.trial.FrozenTrial):
    """Rebuild a trial's Config: apply its searched params (with the HPO aliases) onto the base."""
    overrides: dict = {}
    for k, v in trial.params.items():
        alias = _ALIASES.get(k, k)
        for a in (alias if isinstance(alias, list) else [alias]):
            overrides[a] = v
    return dataclasses.replace(base, name=f"{base.name}_trial{trial.number}", **overrides)


def _best_epoch(cfg) -> int:
    """The trial's best epoch (from results/<name>/metrics.json), to fix the full-data budget."""
    mj = Path(cfg.out_dir) / cfg.name / "metrics.json"
    try:
        return int(json.loads(mj.read_text()).get("epoch", cfg.epochs - 1))
    except Exception:
        return cfg.epochs - 1


def run(topk: int, configs: list[str], storage: str) -> None:
    full_models: list[tuple[str, float, Path]] = []  # (label, trial_score, full_pred_path)

    for config in configs:
        base, _ = load_config(config)
        study = optuna.load_study(study_name=base.name, storage=storage)
        trials = sorted((t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE
                         and t.value is not None and t.value < float("inf")),
                        key=lambda t: t.value)[:topk]
        if not trials:
            print(f"[{config}] no finite completed trials -> skip", flush=True)
            continue

        for t in trials:
            cfg = _cfg_from_trial(base, t)
            tag = f"trial{t.number}"
            print(f"\n================ {config} {tag} (score {t.value:.5f}) ================", flush=True)

            # 1. predict from the trial's existing best.pt (BEST epoch)
            ckpt = Path(cfg.out_dir) / cfg.name / "best.pt"
            out_best = RESULTS / config / tag / "best" / "test_predictions.csv"
            out_best.parent.mkdir(parents=True, exist_ok=True)
            if ckpt.exists():
                predict(cfg, str(ckpt), str(out_best))
            else:
                print(f"  ! {ckpt} missing -> skipping best-epoch prediction", flush=True)

            # 2. retrain on 100% data, same hyperparameters, epochs = best epoch (+1 to include it)
            be = _best_epoch(cfg)
            cfg_full = dataclasses.replace(cfg, full_data=True, epochs=be + 1,
                                           name=f"{cfg.name}_full", save_qualitative_k=0, log_test_pred=False)
            print(f"  -> retrain on FULL DATA for {be + 1} epochs (trial best epoch = {be})", flush=True)
            train(cfg_full)

            # 3. predict from the full-data model
            ckpt_full = Path(cfg_full.out_dir) / cfg_full.name / "best.pt"
            out_full = RESULTS / config / tag / "full" / "test_predictions.csv"
            out_full.parent.mkdir(parents=True, exist_ok=True)
            predict(cfg_full, str(ckpt_full), str(out_full))
            full_models.append((f"{config}/{tag}", float(t.value), out_full))

    if not full_models:
        raise SystemExit("No full-data predictions produced (no trials / no checkpoints).")

    # ---- ensemble the FULL-DATA predictions (aligned by filename) ----
    base0, _ = load_config(configs[0])
    fcol, lcol, gcol = base0.image_col, base0.label_col, base0.gender_col
    ref = pd.read_csv(full_models[0][2])
    order = ref[fcol].values

    def _aligned(path: Path) -> np.ndarray:
        return pd.read_csv(path).set_index(fcol).reindex(order)[lcol].values

    P = np.stack([_aligned(p) for _, _, p in full_models])          # (M models, N test)
    labels = [name for name, _, _ in full_models]
    scores = np.array([s for _, s, _ in full_models], dtype=float)

    ens_unw = np.clip(P.mean(axis=0), 0.0, 1.0)
    w = 1.0 / np.maximum(scores, 1e-12); w = w / w.sum()           # lower score = better = more weight
    ens_w = np.clip((P * w[:, None]).sum(axis=0), 0.0, 1.0)

    for name, ens in [("ensemble_unweighted", ens_unw), ("ensemble_weighted", ens_w)]:
        out = RESULTS / f"{name}.csv"
        pd.DataFrame({fcol: order, lcol: ens, gcol: "x"}).to_csv(out, index=False)
        print(f"wrote {out}  (mean={ens.mean():.3f})", flush=True)

    print(f"\nensembled {len(full_models)} full-data models:", flush=True)
    for (name, sc, _), wi in zip(full_models, w):
        print(f"  {name:40s} score={sc:.5f}  weight={wi:.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topk", type=int, help="number of best trials to take per config")
    ap.add_argument("configs", nargs="+", help="one or more config names (= optuna study names)")
    ap.add_argument("--storage", default="sqlite:///optuna.db")
    args = ap.parse_args()
    run(args.topk, args.configs, args.storage)
