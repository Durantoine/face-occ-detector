from __future__ import annotations

import argparse
import dataclasses
import os
from typing import Any

import optuna
import torch

from src.config import Config, load_config
from src.train import dist_info, train

_ALIASES = {
    "learning_rate": "lr",
    "dropout": ["head_dropout", "pooling_dropout"]
}


def _active(spec: dict, chosen: dict[str, Any]) -> bool:
    for parent, required in (spec.get("conditional_on") or {}).items():
        if str(chosen.get(parent)) != str(required):
            return False
    return True


def _suggest_one(trial: optuna.Trial, key: str, spec: dict):
    t = spec["type"]
    if t == "categorical":
        return trial.suggest_categorical(key, spec["choices"])
    if t == "int":
        return trial.suggest_int(key, spec["low"], spec["high"])
    return trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))


def _suggest(trial: optuna.Trial, base: Config, search_space: dict) -> Config:
    chosen: dict[str, Any] = {}
    overrides: dict[str, Any] = {}
    for key, spec in search_space.items():
        if not _active(spec, chosen):
            continue
        val = _suggest_one(trial, key, spec)
        chosen[key] = val
        # pretrained_source -> init_backbone_from is resolved centrally in train(), so HPO and
        # direct runs behave identically; here we just carry the chosen value through.
        alias = _ALIASES.get(key, key)
        if isinstance(alias, list):
            for a in alias:
                overrides[a] = val
        else:
            overrides[alias] = val
    if getattr(base, "rotate_seed", False) and "seed" not in overrides:
        # different val split + init per trial -> the top-K trials are decorrelated -> better ensemble
        overrides["seed"] = base.seed + trial.number
    return dataclasses.replace(base, name=f"{base.name}_trial{trial.number}", **overrides)


def _bcast_cfg(obj, is_dist: bool):
    if not is_dist:
        return obj
    import torch.distributed as dist
    lst = [obj]
    dist.broadcast_object_list(lst, src=0)
    return lst[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--storage", default="sqlite:///optuna.db")
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = ap.parse_args()
    config = args.config or os.environ.get("FACE_OCC_ARCH")
    if not config:
        ap.error("config required (positional arg or FACE_OCC_ARCH env var)")

    is_dist, rank, world, local_rank = dist_info()
    if is_dist:
        import torch.distributed as dist
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    base, optuna_cfg = load_config(config)
    search_space = optuna_cfg.get("search_space", {})
    n_trials = args.trials if args.trials is not None else int(optuna_cfg.get("n_trials", 50))
    experiment = f"optuna-{base.name}"

    if rank == 0:
        study = optuna.create_study(
            study_name=base.name, storage=args.storage, direction="minimize", load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=base.seed, multivariate=True),
            pruner=None,
        )

        def objective(trial: optuna.Trial) -> float:
            cfg = _suggest(trial, base, search_space)
            _bcast_cfg(cfg, is_dist)
            # whole body guarded: a single trial's failure (train OR post-processing) must never
            # abort the sweep -- it returns inf and Optuna moves to the next trial.
            try:
                res = train(cfg, optuna_trial=trial, experiment=experiment, tracking_uri=args.tracking_uri)
                if not isinstance(res, dict) or "eval_challenge_score" not in res:
                    print(f"trial {trial.number}: train() returned no score -> inf", flush=True)
                    return float("inf")
                # v35: res IS the best-epoch dict; expose it under explicit best_* names.
                for src, dst in (("eval_challenge_score", "best_score"), ("eval_err_F", "best_err_F"),
                                 ("eval_err_M", "best_err_M"), ("eval_err_diff", "best_err_diff"),
                                 ("eval_mae_pct", "best_mae_pct"), ("epoch", "best_epoch")):
                    if src in res:
                        trial.set_user_attr(dst, float(res[src]))
                score = float(res["eval_challenge_score"])
                print(f"trial {trial.number} finished with score {score:.5f} (best epoch {res.get('epoch')})", flush=True)
                return score
            except Exception as e:
                import traceback
                print(f"trial {trial.number} failed: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                return float("inf")

        study.optimize(objective, n_trials=n_trials)
        _bcast_cfg(None, is_dist)
        print("BEST score:", study.best_value, flush=True)
        print("BEST params:", study.best_params, flush=True)
    else:
        while True:
            cfg = _bcast_cfg(None, is_dist)
            if cfg is None:
                break
            try:
                train(cfg, optuna_trial=None, experiment=experiment, tracking_uri=args.tracking_uri)
            except Exception as e:
                print(f"[rank {rank}] trial worker error: {type(e).__name__}: {e}", flush=True)
                continue

    if is_dist:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
