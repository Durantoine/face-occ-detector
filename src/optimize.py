import copy
import gc
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import optuna
import torch
import yaml
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

from src.train import train
from src.utils.distributed import (
    barrier,
    broadcast,
    cleanup_distributed,
    is_main,
    setup_distributed,
)
from src.utils.environment import setup_environment
from src.utils.mlflow_utils import get_or_create_experiment

setup_environment()

CONFIG: Dict[str, Any] = {
    "architecture": os.environ.get("FACE_OCC_ARCH", "dinov3-vits16-face-occ"),
    "n_trials": 30,
    "study_name": None,
    "tracking_uri": "sqlite:///mlflow.db",
    "storage": "sqlite:///optuna.db",
    "use_mlflow": True,
    "parent_run_id": None,
    "objective_mode": "score",
    "rotate_val_seed": True,
    "test_data_csv": None,
}

_TRAINING_KEYS = {
    "learning_rate", "weight_decay", "num_train_epochs", "warmup_ratio",
    "lr_scheduler_type", "gradient_accumulation_steps", "per_device_train_batch_size",
    "augmentation_level", "ema_decay", "layer_decay",
    "loss_focal_gamma", "loss_fairness_lambda", "use_gender_balanced_sampler",
    "sampler_strategy", "loss_importance_reweight", "loss_gender_reweight", "loss_cell_reweight",
    "loss_patch_mil_alpha", "loss_type", "group_dro_alpha",
}
_MODEL_KEYS = {"hidden_dropout_prob", "pooling", "projection_size", "output_activation"}

_BALANCING_STRATEGY_MAP = {
    "A": {"sampler_strategy": "gender",    "loss_importance_reweight": True,  "loss_gender_reweight": False, "loss_cell_reweight": False},
    "D": {"sampler_strategy": "none",      "loss_importance_reweight": True,  "loss_gender_reweight": True,  "loss_cell_reweight": False},
    "E": {"sampler_strategy": "none",      "loss_importance_reweight": False, "loss_gender_reweight": False, "loss_cell_reweight": True},
    "F": {"sampler_strategy": "occlusion", "loss_importance_reweight": False, "loss_gender_reweight": True,  "loss_cell_reweight": False},
}


def _apply_trial_param(cfg: Dict[str, Any], name: str, value: Any) -> None:
    if name == "balancing_strategy":
        for k, v in _BALANCING_STRATEGY_MAP[str(value)].items():
            cfg["training"][k] = v
        return
    if name in _TRAINING_KEYS:
        cfg["training"][name] = value
        if name == "per_device_train_batch_size":
            cfg["training"]["per_device_eval_batch_size"] = value
    elif name in _MODEL_KEYS:
        cfg["model"][name] = value


def create_trial_config(base_config: Dict[str, Any], trial: optuna.Trial, n: int) -> str:
    cfg = copy.deepcopy(base_config)
    search_space = cfg.get("optuna", {}).get("search_space", {})
    if not search_space:
        raise ValueError(f"No search_space in config '{base_config.get('name')}'")

    for name, spec in search_space.items():
        if name == "seed":
            continue
        t = spec["type"]
        if t == "float":
            v: Any = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=spec.get("log", False))
        elif t == "int":
            v = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        elif t == "categorical":
            v = trial.suggest_categorical(name, spec["choices"])
        else:
            continue
        _apply_trial_param(cfg, name, v)

    cfg["name"] = f"{cfg['name']}_trial{n}"
    out = Path("configs/architectures/optuna_trials") / f"trial_{n}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    return f"optuna_trials/trial_{n}"


def _best_pareto(study: optuna.Study) -> optuna.trial.FrozenTrial:
    return min(study.best_trials, key=lambda t: t.values[0] + 0.5 * t.values[1])


def _make_child_run(
    client: MlflowClient, experiment_id: str, parent_run_id: Optional[str], base_arch: str, trial: optuna.Trial,
) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = client.create_run(
        experiment_id=experiment_id,
        run_name=f"{base_arch}_trial{trial.number}_{ts}",
        tags={MLFLOW_PARENT_RUN_ID: parent_run_id} if parent_run_id else {},
    )
    for k, v in trial.params.items():
        client.log_param(run.info.run_id, k, v)
    return run.info.run_id


def objective(
    trial: optuna.Trial,
    base_arch: str,
    base_config: Dict[str, Any],
    client: Optional[MlflowClient],
    experiment_id: Optional[str],
    parent_run_id: Optional[str],
    tracking_uri: str,
    mode: str,
    rotate_val_seed: bool,
    test_data_csv: Optional[str],
) -> Any:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    seed = base_config.get("training", {}).get("seed", 42)
    trial_data: Optional[Dict[str, Any]] = None
    run_id: Optional[str] = None

    if is_main():
        arch_name = create_trial_config(base_config, trial, trial.number)
        val_seed = (seed + trial.number * 13) if rotate_val_seed else None
        if client and experiment_id:
            run_id = _make_child_run(client, experiment_id, parent_run_id, base_arch, trial)
        trial_data = {"arch": arch_name, "run_id": run_id, "seed": seed, "val_seed": val_seed, "n": trial.number}

    trial_data = broadcast(trial_data)
    assert trial_data is not None
    arch_name = trial_data["arch"]
    run_id = trial_data["run_id"]
    seed = trial_data["seed"]
    val_seed = trial_data["val_seed"]

    if run_id:
        os.environ["MLFLOW_RUN_ID"] = run_id
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

    eval_loss, score, err_diff, err_F, err_M = float("inf"), float("inf"), float("inf"), float("inf"), float("inf")
    try:
        eval_loss, score, err_diff, _, err_F, err_M = train(
            architecture_name=arch_name,
            output_dir=f"./results/optuna_{base_arch}_trial_{trial_data['n']}",
            mlflow_tracking_uri=tracking_uri,
            mlflow_run_id=run_id,
            seed=seed,
            val_seed=val_seed,
            test_data_csv=test_data_csv,
        )
        if is_main() and client and run_id:
            for k, v in [
                ("final_eval_loss", eval_loss), ("best_score", score),
                ("err_diff", err_diff), ("err_F", err_F), ("err_M", err_M),
            ]:
                client.log_metric(run_id, k, v)
            client.set_terminated(run_id, "FINISHED")
            print(f"Trial {trial_data['n']}: score={score:.5f} err_F={err_F:.5f} err_M={err_M:.5f} err_diff={err_diff:.5f}")
        trial.set_user_attr("err_F", float(err_F))
        trial.set_user_attr("err_M", float(err_M))
        trial.set_user_attr("err_diff", float(err_diff))
        trial.set_user_attr("eval_loss", float(eval_loss))
    except Exception:
        eval_loss, score, err_diff, err_F, err_M = float("inf"), float("inf"), float("inf"), float("inf"), float("inf")
        if is_main():
            import traceback
            traceback.print_exc()
            if client and run_id:
                client.set_terminated(run_id, "FAILED")
    finally:
        os.environ.pop("MLFLOW_RUN_ID", None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        barrier()

    if mode == "pareto":
        return (score, err_diff)
    if mode == "loss":
        return eval_loss
    return score


def _save_best_config(arch: str, base: Dict[str, Any], trial: optuna.trial.FrozenTrial, mode: str) -> None:
    cfg = copy.deepcopy(base)
    for k, v in trial.params.items():
        _apply_trial_param(cfg, k, v)
    name = f"{arch}_optuna_best"
    cfg["name"] = name
    out = f"configs/architectures/{name}.yaml"
    Path(out).write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    print(f"Best config saved: {out}")


class _MaxTrials:
    def __init__(self, n: int) -> None:
        self.n = n

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        done = [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]
        if sum(1 for t in study.trials if t.state in done) >= self.n:
            study.stop()


def _create_study_with_retry(study_name: str, storage: str, mode: str) -> optuna.Study:
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    for _ in range(10):
        try:
            if mode == "pareto":
                return optuna.create_study(
                    study_name=study_name, directions=["minimize", "minimize"],
                    storage=storage, load_if_exists=True,
                    sampler=optuna.samplers.NSGAIISampler(),
                )
            return optuna.create_study(
                study_name=study_name, direction="minimize",
                storage=storage, load_if_exists=True, pruner=pruner,
            )
        except Exception as e:
            if "locked" in str(e) or "already exists" in str(e):
                time.sleep(random.uniform(1, 3))
            else:
                raise
    raise RuntimeError(f"Could not create study {study_name} after retries")


def optimize_hyperparameters(
    architecture: str,
    n_trials: int = 30,
    study_name: Optional[str] = None,
    tracking_uri: str = "sqlite:///mlflow.db",
    storage: str = "sqlite:///optuna.db",
    use_mlflow: bool = True,
    parent_run_id: Optional[str] = None,
    objective_mode: str = "score",
    rotate_val_seed: bool = False,
    test_data_csv: Optional[str] = None,
) -> Optional[optuna.Study]:
    local_rank = setup_distributed()

    with open(f"configs/architectures/{architecture}.yaml") as f:
        base_config = yaml.safe_load(f)

    optuna_cfg = base_config.get("optuna", {})
    objective_mode = optuna_cfg.get("objective_mode", objective_mode)
    rotate_val_seed = optuna_cfg.get("rotate_val_seed", rotate_val_seed)
    n_trials = optuna_cfg.get("n_trials", n_trials)
    if study_name is None:
        study_name = f"optuna-{architecture}-{datetime.now().strftime('%Y%m%d')}"

    study: Optional[optuna.Study] = None
    client: Optional[MlflowClient] = None
    experiment_id: Optional[str] = None

    if is_main():
        study = _create_study_with_retry(study_name, storage, objective_mode)
        print(f"Study={study_name} | Trials={n_trials} | Mode={objective_mode} | Dist={local_rank != -1}")
        if use_mlflow:
            client = MlflowClient(tracking_uri=tracking_uri)
            experiment_id = get_or_create_experiment(client, f"optuna-{architecture}")
            if parent_run_id is None:
                parent_run_id = client.create_run(experiment_id=experiment_id, run_name=study_name).info.run_id
                print(f"MLflow parent: {parent_run_id}")

    barrier()
    study = broadcast(study)
    parent_run_id = broadcast(parent_run_id)
    experiment_id = broadcast(experiment_id)

    if is_main():
        assert study is not None
        study.optimize(
            lambda t: objective(t, architecture, base_config, client, experiment_id,
                                parent_run_id, tracking_uri, objective_mode, rotate_val_seed, test_data_csv),
            n_trials=n_trials,
            callbacks=[_MaxTrials(n_trials)],
        )
        broadcast(None)
    else:
        while True:
            data = broadcast(None)
            if data is None:
                break
            try:
                train(architecture_name=data["arch"], output_dir=f"./results/optuna_{architecture}_trial_{data['n']}",
                      mlflow_tracking_uri=tracking_uri, mlflow_run_id=data["run_id"],
                      seed=data["seed"], val_seed=data["val_seed"])
            except Exception:
                pass
            barrier()

    if is_main():
        if use_mlflow and client and parent_run_id:
            client.set_terminated(parent_run_id)
        assert study is not None
        best = _best_pareto(study) if objective_mode == "pareto" else study.best_trial
        print(f"\nBest trial {best.number}: {best.params}")
        _save_best_config(architecture, base_config, best, objective_mode)
        Path("status").mkdir(exist_ok=True)
        (Path("status") / f"{study_name}.done").touch()

    cleanup_distributed()
    return study


if __name__ == "__main__":
    optimize_hyperparameters(
        architecture=CONFIG["architecture"],
        n_trials=CONFIG["n_trials"],
        study_name=CONFIG["study_name"],
        tracking_uri=CONFIG["tracking_uri"],
        storage=CONFIG["storage"],
        use_mlflow=CONFIG["use_mlflow"],
        parent_run_id=CONFIG["parent_run_id"],
        objective_mode=CONFIG["objective_mode"],
        rotate_val_seed=CONFIG["rotate_val_seed"],
        test_data_csv=CONFIG["test_data_csv"],
    )
