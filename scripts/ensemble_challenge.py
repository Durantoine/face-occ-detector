"""Challenge ensemble pipeline.

For each CONFIG, take its top-K Optuna trials and, per trial:
  1. predict the test set from the trial's existing best.pt  (BEST epoch checkpoint)
  2. RE-TRAIN the exact same hyperparameters on 100% of the data (no held-out val,
     epochs fixed = the trial's best epoch)
  3. predict the test set from that full-data model

Predictions are organised under:
  challenge_results/<config>/trial<N>/best/test_predictions.csv   (from best.pt)
  challenge_results/<config>/trial<N>/full/test_predictions.csv   (from the 100%-data retrain)

Finally it ensembles BOTH prediction sets x {unweighted mean, 1/val-score weighted} -> 4 submissions:
  challenge_results/ensemble_best_{unweighted,weighted}.csv   (val held-out best.pt models)
  challenge_results/ensemble_full_{unweighted,weighted}.csv   (100%-data retrains)
(weighted: lower val score -> better trial -> more weight)

Usage:
  python scripts/ensemble_challenge.py 3 sapiens-final dino-vitb16
                                        ^K  ^------- Y configs -------^
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
    best_models: list[tuple[str, float, Path]] = []  # (label, trial_score, best.pt pred = val held-out)
    full_models: list[tuple[str, float, Path]] = []  # (label, trial_score, 100%-data retrain pred)

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
                best_models.append((f"{config}/{tag}", float(t.value), out_best))
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

    if not best_models and not full_models:
        raise SystemExit("No predictions produced (no trials / no checkpoints).")

    base0, _ = load_config(configs[0])
    fcol, lcol, gcol = base0.image_col, base0.label_col, base0.gender_col

    # Ensemble a prediction set 2 ways (unweighted mean + 1/val-score weighted), aligned by filename.
    # Run for BOTH sets: 'best' = val-held-out best.pt models, 'full' = 100%-data retrains.
    def _ensemble(models: list[tuple[str, float, Path]], prefix: str) -> None:
        if not models:
            print(f"[{prefix}] no predictions -> skip", flush=True)
            return
        order = pd.read_csv(models[0][2])[fcol].values
        P = np.stack([pd.read_csv(p).set_index(fcol).reindex(order)[lcol].values for _, _, p in models])
        scores = np.array([s for _, s, _ in models], dtype=float)
        w = 1.0 / np.maximum(scores, 1e-12); w = w / w.sum()       # lower val score = better = more weight
        for name, ens in [(f"ensemble_{prefix}_unweighted", P.mean(axis=0)),
                          (f"ensemble_{prefix}_weighted", (P * w[:, None]).sum(axis=0))]:
            ens = np.clip(ens, 0.0, 1.0)
            out = RESULTS / f"{name}.csv"
            pd.DataFrame({fcol: order, lcol: ens, gcol: "x"}).to_csv(out, index=False)
            print(f"wrote {out}  ({len(models)} models, mean={ens.mean():.3f})", flush=True)
        print(f"[{prefix}] members + weights:", flush=True)
        for (name, sc, _), wi in zip(models, w):
            print(f"  {name:40s} score={sc:.5f}  weight={wi:.3f}", flush=True)

    _ensemble(best_models, "best")   # val held-out (best.pt) — val-verified
    _ensemble(full_models, "full")   # 100%-data retrains (val folded back in)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topk", type=int, help="number of best trials to take per config")
    ap.add_argument("configs", nargs="+", help="one or more config names (= optuna study names)")
    ap.add_argument("--storage", default="sqlite:///optuna.db")
    args = ap.parse_args()
    run(args.topk, args.configs, args.storage)
