"""Ensemble the top-N trials of an Optuna sweep.

Requires `rotate_val_seed: false` in the yaml so all trials share the same val split.

Pipeline :
1. Identify top-N trials from Optuna study (sorted by `final_value`)
2. Load each model from MLflow registry
3. For each: predict on val (= same val for all trials thanks to rotate_val_seed=false)
4. Average predictions → ensemble_preds_val
5. Refit per-gender isotonic on ensemble_preds_val + gt (sample-weighted IS)
6. Score ensemble vs best single trial → log improvement
7. Apply per-gender calibration on TEST predictions (averaged) + write submission CSV

Usage :
    python scripts/ensemble_topN.py --arch dinov3-vitb16-3090-v19 --n 3 \
        --output submissions/ensemble_v19_top3.csv

Optional :
    --test-csv data/raw/test_students.csv   (default)
    --gender-csv data/raw/test_students_with_gender.csv
    --tracking-uri sqlite:///mlflow.db
    --no-tta                              (default = TTA on)
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.calibrators import fit_all_calibrators
from src.predict import load_model, predict_images
from src.utils.distribution import _TEST_PMF, N_BINS, BIN_WIDTH, estimate_test_pmf_joint
from src.utils.metrics import compute_score, compute_score_stratified_is


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, help="Optuna study/arch name, e.g. dinov3-vitb16-3090-v19")
    p.add_argument("--n", type=int, default=3, help="Number of top trials to ensemble (default 3)")
    p.add_argument("--test-csv", default="data/raw/test_students.csv")
    p.add_argument("--train-csv", default="data/raw/train.csv")
    p.add_argument("--image-base-dir", default="data/raw")
    p.add_argument("--gender-csv", default="data/raw/test_students_with_gender.csv")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--storage", default="sqlite:///optuna.db")
    p.add_argument("--output", required=True, help="Path to ensemble submission CSV")
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    # === 1. Get top-N trial run_ids from Optuna ===
    study_name = f"optuna-{args.arch}"
    study = optuna.load_study(study_name=study_name, storage=args.storage)
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
              and t.value is not None and t.value != float("inf")]
    trials.sort(key=lambda t: t.value)
    top = trials[:args.n]
    if len(top) < args.n:
        print(f"WARNING: only {len(top)} completed trials, requested {args.n}")
    print(f"Top-{len(top)} trials by val_score:")
    for t in top:
        print(f"  trial {t.number}: val_score = {t.value:.5f}")

    # === 2. Resolve MLflow model URIs ===
    # Convention from _PruneRegistryToTopN: registered_model_name = f"{arch}_trial{N}"
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=args.tracking_uri)
    model_uris: List[str] = []
    for t in top:
        model_name = f"{args.arch}_trial{t.number}"
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                print(f"WARNING: no MLflow registry entry for {model_name} (likely pruned)")
                continue
            v_latest = max(versions, key=lambda v: int(v.version))
            model_uris.append(f"models:/{model_name}/{v_latest.version}")
        except Exception as e:
            print(f"WARNING: cannot resolve {model_name}: {e}")
    if not model_uris:
        print("ERR: no models available for ensembling. Has _PruneRegistryToTopN deleted them ?")
        return 1
    print(f"\nResolved {len(model_uris)} models:\n  " + "\n  ".join(model_uris))

    # === 3. Predict on VAL for each model → average ===
    print("\n=== Step 3: predict on VAL for each model ===")
    val_df = _load_val(args.arch, args.train_csv)
    val_filenames = val_df["filename"].tolist()
    val_gt = val_df["FaceOcclusion"].astype(float).values
    val_gender = val_df["gender"].astype(float).values

    val_preds_per_model: List[np.ndarray] = []
    for i, uri in enumerate(model_uris):
        print(f"\n[{i+1}/{len(model_uris)}] {uri}")
        model, processor = load_model(uri, args.tracking_uri)
        if args.no_tta:
            preds = np.array(predict_images(model, processor, val_filenames, args.image_base_dir, args.batch_size))
        else:
            from src.inference.tta import predict_tta_batch
            preds_t = predict_tta_batch(model, processor, val_filenames,
                                          batch_size=args.batch_size, image_base_dir=args.image_base_dir)
            preds = preds_t.numpy().astype(float)
        val_preds_per_model.append(preds)
        per_trial_score = compute_score(preds, val_gt, val_gender)["challenge_score"]
        print(f"  per-trial val score (raw, no cal) = {per_trial_score:.5f}")

    val_preds_ensemble = np.mean(np.stack(val_preds_per_model), axis=0)
    raw_score = compute_score(val_preds_ensemble, val_gt, val_gender)
    print(f"\nEnsemble val score (RAW, no cal) = {raw_score['challenge_score']:.5f}  "
          f"(err_F={raw_score['err_F']:.5f}, err_M={raw_score['err_M']:.5f})")

    # === 4. Refit per-gender calibrator on ensemble preds ===
    print("\n=== Step 4: fit per-gender calibrator on ensemble val preds ===")
    cals = fit_all_calibrators(val_preds_ensemble, val_gt, val_gender, use_is_weight=True)
    test_pmf_joint = estimate_test_pmf_joint(val_gt, (val_gender >= 0.5).astype(int))

    alphas = np.linspace(0.6, 1.3, 8)
    combo_scores: Dict[Tuple[str, float], float] = {}
    for cal_name, cal in cals.items():
        preds_cal = cal.transform(val_preds_ensemble, val_gender)
        for a in alphas:
            blend = np.clip(a * preds_cal + (1 - a) * val_preds_ensemble, 0, 1)
            s = compute_score_stratified_is(blend, val_gt, val_gender, test_pmf_joint, BIN_WIDTH, N_BINS)
            combo_scores[(cal_name, round(float(a), 2))] = s["challenge_score"]

    best_combo = min(combo_scores, key=combo_scores.get)
    best_cal_name, best_alpha = best_combo
    print(f"Best ensemble combo: ({best_cal_name}, α={best_alpha}) → val_is_strat={combo_scores[best_combo]:.5f}")
    best_single = top[0].value
    delta = combo_scores[best_combo] - best_single
    print(f"  vs best single trial = {best_single:.5f}   Δ = {delta:+.5f}  "
          f"({'IMPROVED' if delta < 0 else 'WORSE'})")

    # === 5. Predict on TEST for each model → average ===
    print("\n=== Step 5: predict on TEST for each model ===")
    test_df = pd.read_csv(args.test_csv).dropna(subset=["filename"])
    test_preds_per_model: List[np.ndarray] = []
    for i, uri in enumerate(model_uris):
        print(f"\n[{i+1}/{len(model_uris)}] {uri}")
        model, processor = load_model(uri, args.tracking_uri)
        if args.no_tta:
            preds = np.array(predict_images(model, processor, test_df["filename"].tolist(),
                                              args.image_base_dir, args.batch_size))
        else:
            from src.inference.tta import predict_tta_batch
            preds_t = predict_tta_batch(model, processor, test_df["filename"].tolist(),
                                          batch_size=args.batch_size, image_base_dir=args.image_base_dir)
            preds = preds_t.numpy().astype(float)
        test_preds_per_model.append(preds)

    test_preds_ensemble = np.mean(np.stack(test_preds_per_model), axis=0)

    # === 6. Load test gender + apply per-gender calibration ===
    print("\n=== Step 6: apply per-gender calibration ===")
    gender_df = pd.read_csv(args.gender_csv)
    gender_map = dict(zip(gender_df["filename"].astype(str), gender_df["gender_predicted"].astype(int)))
    test_gender = np.array([gender_map.get(str(fn), 1) for fn in test_df["filename"]], dtype=float)
    n_unknown = sum(1 for fn in test_df["filename"] if str(fn) not in gender_map)
    if n_unknown > 0:
        print(f"  WARNING: {n_unknown:,} test filenames missing in {args.gender_csv}, defaulted to M")

    cal_best = cals[best_cal_name]
    test_preds_cal = cal_best.transform(test_preds_ensemble, test_gender)
    test_preds_final = np.clip(best_alpha * test_preds_cal + (1 - best_alpha) * test_preds_ensemble, 0, 1)

    # === 7. Save submission ===
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df = test_df[["filename"]].copy()
    out_df["FaceOcclusion"] = test_preds_final
    out_df["gender"] = "x"
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(out_df):,} predictions to {args.output}")
    print(f"Ensemble setup: top-{len(model_uris)}, cal={best_cal_name}, α={best_alpha}, "
          f"val_score={combo_scores[best_combo]:.5f}, vs best single {best_single:.5f} (Δ={delta:+.5f})")
    return 0


def _load_val(arch: str, train_csv: str) -> pd.DataFrame:
    """Reconstruct EXACTLY the val split used by HPO via train.py's _load_train_val.
    Relies on rotate_val_seed=false (same seed=42 across all trials)."""
    from src.train import _load_train_val
    from src.utils.config import load_architecture_config
    cfg = load_architecture_config(arch).to_dict()
    data_cfg = cfg.get("data", {})
    seed = int(cfg.get("training", {}).get("seed", 42))
    _, val_df, _ = _load_train_val(data_cfg, train_csv, None, seed=seed)
    print(f"Reconstructed val split via train.py logic: n={len(val_df)} (seed={seed})")
    return val_df


if __name__ == "__main__":
    sys.exit(main())
