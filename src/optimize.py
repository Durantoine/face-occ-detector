import copy
import gc
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    "architecture": os.environ.get("FACE_OCC_ARCH", "dinov3-vitb16-3090-v11"),
    "n_trials": 100,
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
    "augmentation_level",
    "correction_strength", "axis1_power", "axis2_power", "sampler_participation",
    "feature_fairness", "mmd_lambda", "adv_lambda", "ot_lambda", "ot_method", "sinkhorn_eps",
    "loss_focal_gamma",
    "loss_lambda_init", "loss_lambda_lr", "loss_lambda_max", "loss_lambda_ema",
    "loss_query_diversity_lambda",
    "layer_decay",
    "ema_decay",
    "min_lr_rate",   # v12: HPO override of lr_scheduler_kwargs.min_lr_rate
    "early_stopping_patience",
}

_MODEL_KEYS = {
    "head_dropout", "projection_size", "output_activation",
    "backbone_drop_path_rate",
    "pretrained", "pretrained_source", "pooling_type",
    "n_focal", "n_diffuse", "n_free",
    "tau_focal_init", "tau_diffuse_init", "tau_free_init", "learnable_tau",
    "mil_agg", "mil_hidden", "mil_k_top",   # v13: MIL pooling
    "grid_size",                              # v16: grid pooling
    "pool_attn_dropout", "pool_proj_dropout",
}

_FEATURE_FAIRNESS_CHOICES = ("none", "mmd", "dann", "ot")


def _apply_trial_param(cfg: Dict[str, Any], name: str, value: Any) -> None:
    if name == "pretrained_source":
        s = str(value)
        cfg["model"]["pretrained"] = True
        if s in ("lvd", "sapiens_default"):
            cfg["model"]["init_backbone_from"] = None
        elif s.startswith("ibot:"):
            cfg["model"]["init_backbone_from"] = s[len("ibot:"):]
        else:
            raise ValueError(f"Unknown pretrained_source: {s}")
        return
    if name == "feature_fairness":
        if str(value) not in _FEATURE_FAIRNESS_CHOICES:
            raise ValueError(f"Unknown feature_fairness: {value!r}")
        cfg["training"]["feature_fairness"] = str(value)
        return
    if name in _TRAINING_KEYS:
        cfg["training"][name] = value
        if name == "per_device_train_batch_size":
            cfg["training"]["per_device_eval_batch_size"] = value
    elif name in _MODEL_KEYS:
        cfg["model"][name] = value


def _suggest(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
    t = spec["type"]
    if t == "float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=spec.get("log", False))
    if t == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    return None


def _spec_active(spec: Dict[str, Any], sampled: Dict[str, Any]) -> bool:
    cond = spec.get("conditional_on")
    if not cond:
        return True
    for parent, allowed in cond.items():
        if parent not in sampled:
            return False
        if sampled[parent] not in (allowed if isinstance(allowed, list) else [allowed]):
            return False
    return True


def create_trial_config(base_config: Dict[str, Any], trial: optuna.Trial, n: int) -> str:
    cfg = copy.deepcopy(base_config)
    search_space = cfg.get("optuna", {}).get("search_space", {})
    if not search_space:
        raise ValueError(f"No search_space in config '{base_config.get('name')}'")

    sampled: Dict[str, Any] = {}
    for name, spec in search_space.items():
        if name == "seed" or spec.get("conditional_on"):
            continue
        v = _suggest(trial, name, spec)
        if v is None:
            continue
        sampled[name] = v
        _apply_trial_param(cfg, name, v)

    for name, spec in search_space.items():
        if name == "seed" or not spec.get("conditional_on"):
            continue
        if not _spec_active(spec, sampled):
            continue
        v = _suggest(trial, name, spec)
        if v is None:
            continue
        sampled[name] = v
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
    keep_top_n: int = 3,
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
        threshold = _top_n_threshold(trial.study, mode, keep_top_n) if mode != "pareto" else float("inf")
        min_score_to_save = None if mode == "pareto" else threshold
        trial_data = {
            "arch": arch_name, "run_id": run_id, "seed": seed, "val_seed": val_seed,
            "n": trial.number, "min_score_to_save": min_score_to_save,
        }

    # SYNC 1 (start of trial) — broadcast is itself a collective, no separate barrier needed.
    # Non-main rank's broadcast(None) (optimize.py while loop) receives this trial_data.
    trial_data = broadcast(trial_data)
    assert trial_data is not None
    arch_name = trial_data["arch"]
    run_id = trial_data["run_id"]
    seed = trial_data["seed"]
    val_seed = trial_data["val_seed"]
    min_score_to_save = trial_data["min_score_to_save"]

    if run_id:
        os.environ["MLFLOW_RUN_ID"] = run_id
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

    eval_loss, score, err_diff, err_F, err_M = float("inf"), float("inf"), float("inf"), float("inf"), float("inf")
    try:
        eval_loss, score, err_diff, _, err_F, err_M = train(
            architecture_name=arch_name,
            output_dir=f"{os.environ.get('TMPDIR', '/tmp')}/face_occ_results/optuna_{base_arch}_trial_{trial_data['n']}",
            mlflow_tracking_uri=tracking_uri,
            mlflow_run_id=run_id,
            seed=seed,
            val_seed=val_seed,
            test_data_csv=test_data_csv,
            min_score_to_save=min_score_to_save,
            optuna_trial=trial,
        )
    except optuna.exceptions.TrialPruned:
        print(f"Trial {trial.number}: pruned by MedianPruner (intermediate score too high)")
        raise
    except Exception as exc:
        eval_loss, score, err_diff, err_F, err_M = float("inf"), float("inf"), float("inf"), float("inf"), float("inf")
        if is_main():
            import sys
            import traceback
            print(f"!!! TRIAL {trial_data['n']} FAILED: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(file=sys.stdout)
            traceback.print_exc()
            if client and run_id:
                try:
                    client.set_terminated(run_id, "FAILED")
                except Exception:
                    pass

    if is_main() and client and run_id:
        try:
            for k, v in [
                ("final_eval_loss", eval_loss), ("best_score", score),
                ("err_diff", err_diff), ("err_F", err_F), ("err_M", err_M),
            ]:
                client.log_metric(run_id, k, v)
            client.set_terminated(run_id, "FINISHED")
        except Exception as e:
            print(f"WARNING: post-train MLflow logging failed: {e}")
        print(f"Trial {trial_data['n']}: score={score:.5f} err_F={err_F:.5f} err_M={err_M:.5f} err_diff={err_diff:.5f}")
    try:
        trial.set_user_attr("score", float(score))
        trial.set_user_attr("err_F", float(err_F))
        trial.set_user_attr("err_M", float(err_M))
        trial.set_user_attr("err_diff", float(err_diff))
        trial.set_user_attr("eval_loss", float(eval_loss))
        if client and run_id:
            try:
                run_data = client.get_run(run_id).data.metrics
                for k in ["eval_mae_pct", "eval_r2"]:
                    if k in run_data:
                        trial.set_user_attr(k.replace("eval_", ""), float(run_data[k]))
            except Exception as e_pull:
                print(f"WARNING: could not pull metrics from MLflow: {e_pull}")
    except Exception as e:
        print(f"WARNING: trial.set_user_attr failed: {e}")
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


def _completed_scores(study: optuna.Study, mode: str) -> List[Tuple[int, float]]:
    if mode == "pareto":
        return []
    items: List[Tuple[int, float]] = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE or t.value is None:
            continue
        v = float(t.value)
        if v == float("inf") or v != v:
            continue
        items.append((t.number, v))
    items.sort(key=lambda x: x[1])
    return items


def _top_n_threshold(study: optuna.Study, mode: str, n: int) -> float:
    scores = _completed_scores(study, mode)
    if len(scores) < n:
        return float("inf")
    return scores[n - 1][1]


class _PruneRegistryToTopN:
    def __init__(self, client: Optional[MlflowClient], base_arch: str, mode: str, n: int) -> None:
        self.client = client
        self.base_arch = base_arch
        self.mode = mode
        self.n = n

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if self.client is None or self.mode == "pareto":
            return
        scores = _completed_scores(study, self.mode)
        if len(scores) <= self.n:
            return
        top_trial_numbers = {tn for tn, _ in scores[: self.n]}
        for trial_number, _ in scores[self.n:]:
            if trial_number in top_trial_numbers:
                continue
            model_name = f"{self.base_arch}_trial{trial_number}"
            try:
                versions = self.client.search_model_versions(f"name='{model_name}'")
            except Exception:
                continue
            if not versions:
                continue
            for v in versions:
                try:
                    self.client.delete_model_version(model_name, v.version)
                except Exception as e:
                    print(f"WARNING: could not delete {model_name} v{v.version}: {e}")
            try:
                self.client.delete_registered_model(model_name)
                print(f"Pruned {model_name} (out of top-{self.n})")
            except Exception as e:
                print(f"WARNING: could not delete registered model {model_name}: {e}")


def _validate_search_space(base_config: Dict[str, Any]) -> None:
    """Reject legacy keys (v3-v9) explicitly to fail fast on bad yamls."""
    ss = base_config.get("optuna", {}).get("search_space", {})
    LEGACY = {
        "balancing_strategy": "v3 monolithic axis",
        "sampler_strategy": "v4-v7 (sampler retiré en v10)",
        "loss_rw_strategy": "v4-v7",
        "correction_strategy": "v6.5",
        "correction_alpha": "v6.5",
        "axis1_sampler_share": "v8-v9 stick-breaking (retiré v10)",
        "axis1_loss_fraction": "v8 stick-breaking",
        "axis1_share_loss": "v8",
        "axis1_share_aug": "v8",
        "aug_power": "v8",
        "sampler_power": "v9",
        "loss_power": "v9",
        "val_split_alpha": "v9 interpolation (retiré v10)",
        "loss_mmd_alignment": "v8 (remplacé par feature_fairness)",
        "mixup_inter_gender": "v8 (retiré v10)",
        "loss_adv_debiasing": "v8 (remplacé par feature_fairness=dann)",
        "loss_importance_reweight": "v3-v7",
        "loss_cell_reweight": "v3-v7",
        "group_dro_alpha": "v6 (GroupDRO retiré v10)",
        "val_split_strategy": "v6-v9 (pinned test_pmf v10)",
        "aug_share": "v10 (replication retiré v11)",
        "aug_repli_max": "v10 (replication retiré v11)",
        "loss_fairness_lambda": "v3-v10 (remplacé par Lagrangien adaptatif v11)",
    }
    bad = [k for k in LEGACY if k in ss]
    if bad:
        msg = "Legacy search_space params detected:\n"
        for k in bad:
            msg += f"  - {k} : {LEGACY[k]}\n"
        raise ValueError(msg)


def _validate_pretrained_source_choices(base_config: Dict[str, Any], tracking_uri: str) -> None:
    import re
    ss = base_config.get("optuna", {}).get("search_space", {})
    spec = ss.get("pretrained_source")
    if not spec or spec.get("type") != "categorical":
        return
    choices = spec.get("choices") or []
    placeholder_pattern = re.compile(r"__FILL[_A-Z0-9]*__")
    run_id_pattern = re.compile(r"^ibot:runs:/([^/]+)/")
    bad: List[str] = []
    for c in choices:
        s = str(c)
        if placeholder_pattern.search(s):
            bad.append(f"  ✗ '{s}' contains a placeholder")
            continue
        m = run_id_pattern.match(s)
        if m:
            run_id = m.group(1)
            try:
                MlflowClient(tracking_uri=tracking_uri).get_run(run_id)
            except Exception as e:
                bad.append(f"  ✗ '{s}' → MLflow lookup failed: {e}")
    if bad:
        raise ValueError("Invalid pretrained_source:\n" + "\n".join(bad))


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
    raise RuntimeError(f"Could not create study {study_name}")


def optimize_hyperparameters(
    architecture: str,
    n_trials: int = 100,
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
    keep_top_n = int(optuna_cfg.get("keep_top_n", 3))

    if is_main():
        _validate_search_space(base_config)
        if use_mlflow:
            _validate_pretrained_source_choices(base_config, tracking_uri)
    if study_name is None:
        study_name = f"optuna-{architecture}"

    study: Optional[optuna.Study] = None
    client: Optional[MlflowClient] = None
    experiment_id: Optional[str] = None

    if is_main():
        study = _create_study_with_retry(study_name, storage, objective_mode)
        print(f"Study={study_name} | Trials={n_trials} | Mode={objective_mode} | KeepTopN={keep_top_n} | Dist={local_rank != -1}")
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
                                parent_run_id, tracking_uri, objective_mode, rotate_val_seed,
                                test_data_csv, keep_top_n),
            n_trials=n_trials,
            callbacks=[
                _MaxTrials(n_trials),
                _PruneRegistryToTopN(client, architecture, objective_mode, keep_top_n),
            ],
        )
        broadcast(None)
    else:
        # Non-main ranks: same trial loop, mirror rank-0's objective() lifecycle.
        # SYNC 1: broadcast(None) matches rank 0's broadcast(trial_data) in objective.
        # SYNC 2: barrier() at end of iter matches rank 0's barrier() in objective's finally.
        while True:
            data = broadcast(None)
            if data is None:
                break
            try:
                train(architecture_name=data["arch"], output_dir=f"{os.environ.get('TMPDIR', '/tmp')}/face_occ_results/optuna_{architecture}_trial_{data['n']}",
                      mlflow_tracking_uri=tracking_uri, mlflow_run_id=data["run_id"],
                      seed=data["seed"], val_seed=data["val_seed"])
            except Exception:
                pass
            finally:
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
