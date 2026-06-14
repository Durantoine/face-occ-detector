# face-occ-detector

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194e2.svg)
![Optuna](https://img.shields.io/badge/Optuna-HPO-7ab800.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b.svg)
![DataChallenge](https://img.shields.io/badge/Idemia%20%C3%97%20T%C3%A9l%C3%A9com%20Paris-DataChallenge%202026-orange.svg)

Idemia / Télécom Paris **DataChallenge 2026** — regression of `FaceOcclusion ∈ [0, 1]` (ratio of occluded area to face area, on 224×224 face crops). The metric **penalises gender disparity** between female and male errors.

## The task

```
Image (224×224 face crop)  →  Model  →  FaceOcclusion ∈ [0, 1]
```

**Per-gender weighted MSE** (the metric weights high occlusions more, `w = 1/30 + GT`):
$$\text{Err}_g = \frac{\sum_{i\in g} w_i (p_i - GT_i)^2}{\sum_{i\in g} w_i}, \qquad w_i = \tfrac{1}{30} + GT_i$$

**Score** (lower = better) — the `|Err_F − Err_M|` term forces gender fairness:
$$\text{Score} = \frac{\text{Err}_F + \text{Err}_M}{2} + \left|\text{Err}_F - \text{Err}_M\right|$$

The interim leaderboard sits **between** `P_train` and `P_test`; the **final** is scored on `P_test`. Test shares ~97% of identities with train → the setting is **transductive** (the val split is a valid test proxy). Details in [docs/PROJECT.md](docs/PROJECT.md).

---

## Pipeline (current)

- **Model** ([`src/models/face_occ_regressor.py`](src/models/face_occ_regressor.py)) — backbone (timm / Sapiens-2 / DINOv3) → pooling `attention` (k learnable queries) **or** `grid` (adaptive `grid_size×grid_size`) → regularized MLP head (`head_mlp_ratio`) → **sigmoid**. Gender is **not** a model input. `pretrained_source` can load an iBOT encoder from MLflow (`ibot:runs:/…`) or the backbone default (`lvd` / `sapiens_default`).

- **Loss** ([`src/utils/losses.py`](src/utils/losses.py)) — per-gender weighted MSE + a **fixed** fairness penalty, optionally **tilted** by the validation gap:
  ```
  loss = (err_F + err_M)/2 + λ · [(1+tilt)·relu(err_M−err_F) + (1−tilt)·relu(err_F−err_M)]
  ```
  `tilt=0` ⇒ symmetric `λ·|err_F−err_M|`. `tilt>0` ⇒ penalise the gender that is worse **on validation** harder. The tilt is set once per epoch from the val gap (`gap_asymmetric: true`), so per-batch noise can't flip the penalty direction.

- **Reweighting** ([`src/utils/distribution.py`](src/utils/distribution.py)) — importance weight `w = P_test_target / (P_pool + is_lambda)` with a **linear intermediate target** `P_target = (1−eval_alpha)·P_train + eval_alpha·P_test` (`eval_alpha=0.6` for sapiens-final-v2). A **y-conditional floor** keeps high-occlusion weight ≥ 1. Training reweighting is fit on the **augmented pool**; the eval/selection score is fit on the **real P_train** (the val is real data). The eval score `[Honest]` is the IS estimate of the metric at `eval_alpha`; `[Full-Pt]` is the same at `alpha=1` (P_test proxy, logged for monitoring, **not** selection — too high-variance).

- **Sampler** ([`src/data/dataset.py`](src/data/dataset.py)) — `sampler_mode: group_batch` = **gender-balanced, no-replacement batch sampler** (both genders in every batch → low-variance per-batch gap → stable fairness gradient). `stratified` = the per-sample importance sampler. `sampler_participation` (rho) splits sampler↔loss.

- **Augmentation** — two complementary kinds:
  - **on-the-fly, label-preserving** (`aug_level: off|light|medium`): flip / color / blur / noise / **affine**, intensity scaled per-sample by the importance weight (interesting ranges augmented more). Does **not** add images.
  - **pre-rendered, occlusion-adding** (`aug_csv`, `aug_proportion`): real new images with adjusted labels, in `data/aug/` (`synthetic_all.csv` + `blend_v3_refined.csv`, high-occ blends). Two-source **leak protection** drops any aug whose train source(s) fell in val.

- **HPO** ([`src/optimize.py`](src/optimize.py)) — Optuna TPE, YAML `search_space`, resumes via `optuna.db` (`load_if_exists`). `rotate_seed: true` → each trial gets a different val split + init → diverse top-K for ensembling.

---

## Quick start (local)

```bash
uv sync                                    # env (Linux: CUDA wheels; Mac: MPS/CPU)

python src/train.py    efficientnet-mini-local        # one supervised run (M3 Max config)
python src/optimize.py sapiens-final-v2               # Optuna HPO (reads configs/architectures/<name>.yaml)
python src/predict.py  sapiens-final-v2               # submission.csv  (cols: filename, FaceOcclusion, gender="x")
python src/predict.py  <mlflow_run_id>               # predict from a specific trial's best.pt (best epoch)
```

Configs live in [`configs/architectures/*.yaml`](configs/architectures/) (one `name:`, `model:`, `data:`, `training:`, `optuna:` per file). Active configs: `sapiens-final-v2`, `dino-vitb16-H100`, `convnextv2-large-H100`, `efficientnet-mini-local`.

---

## Cluster workflow

Data + the iBOT encoders live in the bucket `gs://mon-face-occ-bucket`. The launch scripts sync `data/raw`, `data/aug` (incl. merged blend), `mlruns/1` (iBOT encoder models — **required** by any `pretrained_source: ibot:…`) and `mlflow.db`.

```bash
# ENSTA (SLURM, 1× H100) — sapiens-final-v2
sbatch scripts/optimize_sapiens_final_h100.slurm

# Lyceum (non-SLURM VM) — default config = sapiens-final-v2
bash scripts/cluster_bootstrap.sh --optimize           # or: cluster_bootstrap.sh <config> --optimize

# push local progress / data back to the bucket
bash scripts/sync_to_bucket.sh
```

After a sweep, build the submission ensemble:

```bash
# free ensemble of the top-K existing checkpoints (val-verified)
python -m src.ensemble_predict sapiens-final-v2 --topk 5 --out submission.csv

# full challenge pipeline: per config, predict top-K trials + RETRAIN them on 100% data
# (full_data mode, epochs fixed = best epoch) + predict, then 2 ensembles (unweighted + perf-weighted)
python scripts/ensemble_challenge.py 3 sapiens-final-v2 dino-vitb16-H100 convnextv2-large-H100
#   -> challenge_results/<config>/trial<N>/{best,full}/test_predictions.csv
#   -> challenge_results/ensemble_{unweighted,weighted}.csv
```

---

## Monitoring UIs (MLflow + Optuna + qualitative viewer)

Three UIs read `mlflow.db` / `optuna.db` from the project dir. **Two launchers:**

```bash
# SLURM cluster (ENSTA) — submits a CPU job; auto-pins mlflow to the training version (uv.lock)
sbatch scripts/ui.sh                       # check #SBATCH --partition matches the cluster's CPU partition

# non-SLURM VM (lyceum) or interactive node — background processes
bash scripts/lyceum_run_uis.sh

# ... with custom ports (when the defaults 5050/8090/8511 are taken on the node):
bash scripts/lyceum_run_uis.sh --mlflow-port 6000 --optuna-port 9000 --qual-port 8600
#   (env vars MLFLOW_PORT / OPTUNA_PORT / QUAL_PORT also work)
```

Each launcher prints the **`ssh -L …`** tunnel command (with the chosen ports) to paste on your laptop, then open `http://localhost:<port>`.

> If the viewer errors `Detected out-of-date database schema`, the UI venv's mlflow is newer than the db's writer. `ui.sh` auto-aligns it to `uv.lock` (3.12.0); manual fix: `~/.face_occ_ui_venv/bin/pip install "mlflow==3.12.0"`. **Never** `mlflow db upgrade` a db a live sweep is writing to.

The qualitative viewer shows, per Optuna trial: trials-comparison convergence, diagnostic charts (per-bin × gender error contribution), best/worst-K image galleries, and the **test-prediction distribution** vs `P_test` and the intermediate target, with a **W₁ (Wasserstein-1)** distance.

---

## Project layout

```
src/            train.py · optimize.py · predict.py · ensemble_predict.py  + models/ data/ utils/
scripts/        ensemble_challenge.py · *.slurm · cluster_bootstrap.sh · sync_to_bucket.sh
                ui.sh (SLURM) · lyceum_run_uis.sh (background) · qualitative_viewer.py
configs/architectures/   per-arch YAML (model + data + training + optuna search_space)
data/raw/       train.csv, test_students.csv, database{1,2,3}/, manual_removals.csv
data/aug/       synthetic_all.csv + blend_v3_refined.csv + images/  (augmentation pool)
docs/PROJECT.md design & rationale
```

| Need | Where |
|---|---|
| Task, metric, fairness, reweighting rationale | [docs/PROJECT.md](docs/PROJECT.md) |
| Loss / reweighting / sampler code | [src/utils/losses.py](src/utils/losses.py) · [src/utils/distribution.py](src/utils/distribution.py) · [src/data/dataset.py](src/data/dataset.py) |
| Active configs | [configs/architectures/](configs/architectures/) |
