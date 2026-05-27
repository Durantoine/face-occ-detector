# Configuration — toggles, MLflow logging, project layout

## All toggles, at a glance

Everything below is YAML-controllable and **defaults to a safe value**.

### Hardware / numerics

| Key | Default | Notes |
|---|---|---|
| `training.fp16` | true | P100-compatible |
| `training.bf16` | false | Ampere+ (3090, A100) |
| `training.gradient_checkpointing` | varies | mandatory for ViT-L+ on P100 |
| `training.per_device_train_batch_size` | 64 (ViT-S), 8 (ViT-L+) | per GPU |
| `training.gradient_accumulation_steps` | 1-4 | scales effective batch |

### Model

| Key | Default | Notes |
|---|---|---|
| `model.model_name` | `dinov3_vits16` | DINOv3 local OR any HF model id |
| `model.output_dim` | 1 | regression scalar |
| `model.pooling_type` | attention_k_query | cls / gem / attention_k_query / multihead_attention |
| `model.projection_size` | null | optional projection head |
| `model.output_activation` | sigmoid | sigmoid / clamp / none |
| `model.head_dropout` | 0.1 | |
| `model.n_focal` / `n_diffuse` / `n_free` | 2 / 2 / 2 | only for attention_k_query |
| `model.tau_focal_init` / `tau_diffuse_init` | 0.1 / 1.5 | only for attention_k_query |
| `model.learnable_tau` | true | only for attention_k_query |
| `model.num_heads` | 4 | only for multihead_attention |

### Training

| Key | Default | Notes |
|---|---|---|
| `training.augmentation_level` | medium | none / light / medium / strong |
| `training.sampler_strategy` | gender_x_occ | none / gender / occlusion / gender_x_occ / test_pmf |
| `training.loss_type` | weighted_mse | weighted_mse / group_dro |
| `training.loss_focal_gamma` | 0.0 | 0 = no focal ; >0 = upweight hard examples (weighted_mse only) |
| `training.loss_fairness_lambda` | 1.0 | 0 = off ; >0 = direct fairness penalty in loss (weighted_mse only) |
| `training.group_dro_alpha` | 0.5 | only used if `loss_type: group_dro` |
| `training.ema_decay` | 0.9998 | 0 = no EMA |
| `training.ema_warmup_steps` | 200 | |
| `training.layer_decay` | 0.85 (large), 1.0 (small) | 1.0 = no LLRD |
| `training.early_stopping_patience` | 3 | 0 = disabled |
| `training.metric_for_best_model` | eval_challenge_score_test_estimated | |
| `training.greater_is_better` | false | for regression |
| `training.val_split_strategy` | stratified_yg | stratified_yg / test_pmf |
| `training.eval_importance_reweight` | true | reweight val score to estimate test perf |

### HPO

| Key | Default | Notes |
|---|---|---|
| `optuna.n_trials` | 200 (v4) | budget |
| `optuna.objective_mode` | score | score / loss / pareto(score, err_diff) |
| `optuna.rotate_val_seed` | true | val_seed per trial → robustness |

### Inference

| Flag / config | Default | Where |
|---|---|---|
| `use_tta` (in `predict.py:CONFIG`, or `--no-tta`) | true | hflip-only TTA |
| `--delta-f` / `--delta-m` | none | post-hoc bias correction |
| `match_test_pmf` | true | quantile matching post-inference |
| `submission_format` | true | emits `filename, FaceOcclusion, gender='x'` |

### Top-level

| Flag | Default | Notes |
|---|---|---|
| `use_mlflow` (in `train.py:CONFIG`) | true | set false to disable all MLflow ops |
| `--no-mlflow` (predict) | n/a | predict.py never calls MLflow |

---

## MLflow — what is logged

**Per epoch** (via `compute_metrics`) :
- `eval_challenge_score_test_estimated`, `eval_challenge_score_val`
- `eval_err_F_test_estimated`, `eval_err_M_test_estimated`, `eval_err_diff_test_estimated`
- `eval_err_F_val`, `eval_err_M_val`, `eval_err_diff_val`
- `eval_mae_pct_val`, `eval_mae_pct_test_estimated`
- `eval_r2_val`, `eval_r2_test_estimated`
- `eval_mse_val`, `eval_mae_val`, `eval_loss`

**Run start** (params) :
- `architecture`, `model_*`, `train_*`, `data_*` (tout est loggé en bloc)
- Si `model.init_backbone_from` est set : `init_backbone_from`, `init_backbone_pretrain_run_id`, `init_backbone_missing_keys`, `init_backbone_unexpected_keys`, plus tous les `pretrain_*` params copiés depuis la run pretrain source (tag `pretrain_run_id` pour le lien)
- `test_pmf_ratio_per_bin` : les 20 ratios de shift Y

**Run start** (data-distribution metrics) :
- `data_train_occ_mean/std`, `data_val_occ_mean/std`
- `data_train_occ_female_mean`, `data_train_occ_male_mean`
- `data_val_occ_female_mean`, `data_val_occ_male_mean`
- `num_train_female`, `num_train_male`, `num_val_female`, `num_val_male`, `train_gender_ratio_M_over_F`

**Quantile matching (post-eval)** :
- `eval_challenge_score_matched_test_estimated`, `eval_err_F/M/diff_matched_test_estimated`
- `eval_challenge_score_best_test_estimated` = `min(raw, matched)` — ce qu'on submettrait
- `calibration_helps`, `calibration_delta`
- Param tag : `best_variant_uses_matching` (bool)

**Run end** :
- `val_score`, `val_err_F`, `val_err_M`, `val_err_diff`, `final_eval_loss`

You can plot `err_F` vs `err_M` over epochs to watch the disparity close or widen.

---

## Project layout

```
src/
├── train.py / optimize.py / ensemble_train.py / pretrain_ibot.py / predict.py / evaluate_model.py
├── data/
│   ├── dataset.py        FaceOccDataset, InferenceDataset, balanced_sampler, load_csv_data
│   └── transforms.py     PIL augmentation pipeline (hflip / jitter / rotation / RandAugment)
├── models/
│   ├── face_occ_regressor.py    FaceOccRegressor (HF + DINOv3 dispatch)
│   ├── dinov3_loader.py          weights + variant picker
│   ├── sapiens2_loader.py        Sapiens2 dispatch
│   ├── dinov3_repo/              Meta's vendored DINOv3 code
│   └── weights/                  local .pth files (vits16 shipped)
├── training/callbacks.py         MlflowClient + EMA
├── inference/
│   ├── tta.py                    hflip-only TTA (multi-crop unsafe for ratio regression)
│   ├── ensemble.py               averages scalar predictions across N models
│   └── calibration.py            quantile_match_to_test_pmf + per-gender bias correction
└── utils/
    ├── losses.py                 WeightedMSELoss, GroupDROLoss, make_sampler_keys, _TEST_PMF_0025
    ├── metrics.py                compute_score (gender-aware, official formula)
    ├── config.py / environment.py / mlflow_utils.py / distributed.py

configs/architectures/
├── dinov3-vitb16-3090-v4.yaml             ViT-B/16  — 2× 3090 (v4 — 3 axes orthogonaux)
├── dinov3-vith16plus-3090-v4.yaml         ViT-H+/16 — 2× 3090 (le plus gros DINO en DDP)
├── sapiens2-01b-3090-v4.yaml              Sapiens2-0.1B — 2× 3090
└── sapiens2-08b-3090-v4.yaml              Sapiens2-0.8B — 2× 3090 (le plus gros Sapiens2 en DDP)

scripts/
├── pretrain_ibot_*_2x3090.sh                                   # SLURM iBOT pretrain
├── optimize_*_v4_2x3090.sh                                     # SLURM Optuna HPO v4
├── chain_pretrain.sh / chain_pretrain_two.sh                   # chain N successive sbatch runs
├── chain_optimize.sh / chain_optimize_two.sh                   # chain N Optuna sweeps
├── download_dinov3_weights.sh                                  # URL reference for larger DINOv3 weights
├── launch_ui.sh                                                # start MLflow + Optuna UIs via SLURM
├── mlflow_ui.sh / optuna_dashboard.sh                          # standalone UI launchers
└── qualitative_viewer.py                                       # Streamlit app (Trials comparison + best/worst gallery)
```
