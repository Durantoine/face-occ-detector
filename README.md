# face-occ-detector

Idemia / Télécom Paris **DataChallenge** — regression of `FaceOcclusion ∈ [0, 1]` (ratio of occluded area to total face area on cropped 224×224 face images). The scoring metric explicitly **penalises gender disparity** between female and male errors.

---

## The task

```
Image (224×224 face crop)  →  Model  →  FaceOcclusion ∈ [0, 1]
```

**Loss** (metric-aligned weighted MSE):
$$L = \frac{\sum_i w_i (p_i - GT_i)^2}{\sum_i w_i}, \qquad w_i = \frac{1}{30} + GT_i$$

**Score** (lower is better):
$$\text{Score} = \frac{\text{Err}_F + \text{Err}_M}{2} + \left|\text{Err}_F - \text{Err}_M\right|$$

The weight `w_i = 1/30 + GT_i` makes high-occlusion samples more important.
The `|Err_F − Err_M|` term forces gender fairness.

---

## Quick start

```bash
# 0. Data (already extracted from DataChallenge2026.zip into data/occlusion_datasets/)
ls data/occlusion_datasets/  # train.csv, test_students.csv

# 1. (optional) Download images to data/crops/Crop_224_5fp_100K/
#    Link: https://partage.imt.fr/index.php/s/ntYk27ZFCbeKGqW

# 2. Pipeline (all declarative — edit CONFIG dicts at top of each src/*.py to switch settings)
inv pretrain           # Stage 0 (optional) — iBOT-light domain adaptation on unlabeled faces
inv train              # Stage 1 — single supervised run, baseline
inv optimize           # Stage 2 — Optuna HPO
inv ensemble           # Stage 3 — k-fold ensemble training on best YAML
inv predict            # Stage 4 — emit test_predictions.csv for submission
inv evaluate           # gender-aware score on a labeled CSV

# UIs
inv mlflow-ui          # http://localhost:5000
inv optuna-dashboard   # http://localhost:8080
```

---

## Pipeline overview

```
┌──────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
│ Stage 1 — HPO (Optuna)   │ → │ Stage 2 — k-fold         │ → │ Stage 3 — Inference      │
│ Score-aware objective    │   │ Stratified gender×occ    │   │ TTA (hflip) + ensemble   │
│ rotate_val_seed per trial│   │ 5 fold models per arch   │   │ + post-hoc bias fix      │
└──────────────────────────┘   └──────────────────────────┘   └──────────────────────────┘
```

---

## Fairness — the central problem

The metric penalises gender disparity. **Is metric-only enough? No.** The metric is the objective, but without explicit mechanisms inside training the model has no per-batch signal about gender — it just minimises the average loss. Five mechanisms in the project address fairness, each independently toggleable:

| # | Mechanism | Type | YAML toggle | Default | Effect |
|---|---|---|---|---|---|
| 1 | **Stratified split / k-fold by `gender × occ_bucket`** | implicit | (always on if cols present) | on | balanced val splits, low CV variance |
| 2 | **Sampler strategy** — batches balanced by gender / occ_bucket / gender×occ | implicit | `training.sampler_strategy` ∈ {none, gender, occlusion, gender_x_occ} | `gender_x_occ` | equal F/M (and occ-level) gradient contribution per batch |
| 3 | **Fairness penalty in the loss** — `λ × |Err_F − Err_M|` added to the loss | explicit | `training.loss_fairness_lambda: float` (0 = off) | 0.0 | direct gradient pressure on disparity |
| 4 | **Group-DRO loss** — minimise `(1-α)·mean + α·max` over per-group losses | explicit | `training.loss_type: weighted_mse / group_dro`, `group_dro_alpha: float` | weighted_mse | worst-case fairness, more aggressive than λ-penalty |
| 5 | **Post-hoc per-gender bias correction** — additive `δ_F`, `δ_M` calibrated on val | inference | `predict.py --delta-f --delta-m` | off | always reduces `err_diff`, ~zero cost |

**Recommendation**:
- **Default config** = `sampler_strategy: gender_x_occ` + `loss_type: weighted_mse` + post-hoc bias correction at submission time.
- If `err_diff` remains large after that, **escalate** to `loss_type: group_dro` with `group_dro_alpha=0.5` (or 1.0 for pure worst-case).
- `loss_fairness_lambda` is an alternative, more sensitive to tuning.

The metric (and `objective_mode: pareto`) drives Optuna's selection across these knobs.

---

## External data (optional)

Two slots for extra data, both supported by the existing infrastructure — just drop files in the right place.

### A) Unlabeled faces → iBOT-light pretraining (`data/pretrain/`)

Any directory of face images (recursive scan) OR a WebDataset of `.tar` shards (auto-detected). Set `pretrain_ibot.CONFIG["data_source"]` to point at it. The default expects the local MS1MV3 `.tar` shards.

| Dataset | Size | Cost | Note |
|---|---|---|---|
| **LFW** | 13k | open | aligned faces, immediate use |
| **CelebA** | 200k | research form | strong baseline corpus |
| **VGGFace2** | 3.3M | research form | large, ~36 GB |
| **CASIA-WebFace** | 500k | research form | mid-size alternative |
| **WIDER Face** (crops) | 400k | open | in-the-wild diversity |

Helper: `python -m src.data.external.prepare_pretrain_corpus` — symlinks images from multiple sources (LFW + CelebA + VGGFace2 …) into `data/pretrain/` for a single unified corpus.

### B) Labeled occlusion data → augmentation finetuning (`data/extra/*.csv`)

Set `data.extra_train_csv` in your YAML to a CSV with the **same schema as `train.csv`** (`filename, FaceOcclusion, gender`). It gets concatenated with the Idemia data **before** the stratified split, so it participates in training.

Mapping rules per dataset (heuristics — refine after a first run):

| Dataset | `FaceOcclusion` heuristic | `gender` | Helper script |
|---|---|---|---|
| **CelebA** | weighted sum of attributes (Eyeglasses=+0.10, Wearing_Hat=+0.10, Wearing_Necktie=+0.05, …), clipped to [0, 0.5] | `Male` attribute → 0/1 | `prepare_celeba.py` |
| **MAFA** | `area(mask_bbox ∩ face_bbox) / area(face_bbox)` (continuous) | imputed to 0.5 (neutral) | `prepare_mafa.py` |
| **AR Face** | 0.30 (glasses) / 0.50 (scarf) | annotated | TODO |
| **FaceOcc (if available)** | already continuous | varies | TODO |

Run `python -m src.data.external.prepare_celeba` (or `prepare_mafa`) after downloading the raw dataset; it writes `data/extra/<name>.csv` in our schema. Then set:

```yaml
data:
  data_csv: data/raw/train.csv
  extra_train_csv: data/extra/celeba_occ.csv   # ← stacked with Idemia data
```

**Important caveat**: extra data shifts the train distribution further from the Idemia/test distribution. Combined with `sampler_strategy: gender_x_occ` this is usually a net positive, but monitor `data_train_occ_mean` in MLflow — if the extra data drags it too far from `data_val_occ_mean`, dial back via fewer extra samples or remove that source.

### Where to download

| Dataset | URL | License |
|---|---|---|
| LFW | http://vis-www.cs.umass.edu/lfw/ | open |
| CelebA | http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html | research form |
| VGGFace2 | https://github.com/ox-vgg/vgg_face2 | research form |
| MAFA | http://www.escience.cn/people/geshiming/mafa.html | research |
| WIDER Face | http://shuoyang1213.me/WIDERFACE/ | open |
| AR Face | https://www2.ece.ohio-state.edu/~aleix/ARdatabase.html | research |

Drop the raw downloads into `data/extra/<dataset_name>_raw/`, then run the matching prep script.

---

## Understanding `FaceOcclusion` — two regimes in one label

Visual inspection of the Idemia dataset (samples copied to `inspection/`) reveals that `FaceOcclusion` is **not** just "object covering the face". It mixes two regimes under the same continuous label:

### Regime 1 — Physical occlusion (the obvious one)
- Sunglasses, hats, scarves, masks
- Hands in front of face
- Hair covering forehead/eyes
- Real-world objects in front of the face

### Regime 2 — Information degradation (the hidden one)
Pixels of the face are physically present but the face information is unrecoverable:
- **Noise / pixelation** (old scans, TV captures with rainbow stripes, JPEG artifacts)
- **Stylisation** (pop-art, sketch, engraving, cartoonification)
- **Blur** (motion, defocus, low-res)
- **Synthetic overlays** added by Idemia (e.g. colored paint lines on top of celebrity portraits — see `m.027y_/85-FaceId-0_align.webp`)
- **Heavy degradation** (faded photos, low-light, compression)

Examples at high `FaceOcclusion` values (≥ 0.6) include heavily stylized Obama-style pop-art portraits, engravings, and severely degraded TV captures — none of these have a physical occluder, but Idemia labels them as ~60-100% occluded because **the face is unrecoverable for recognition**.

### Why Idemia mixes both regimes

From an industrial **face recognition** perspective (Idemia's core business), the two regimes are equivalent: whether your photo has sunglasses OR is too blurry/stylized, the face recognition system has the same problem — *the canonical face features are unavailable*. So `FaceOcclusion` is best understood as:

> **The fraction of the face that is unusable for recognition** — caused by physical occlusion, quality degradation, stylisation, or synthetic overlays — measured by an automated face-quality / face-parsing pipeline.

The high-precision decimal values (e.g. `0.024005`, `0.255016`) suggest the labels are computed automatically by such a pipeline, not manually annotated.

### Implications for augmentation strategy

This insight changes which augmentations are valuable. We need **two families** to cover both regimes:

| Augmentation | Covers regime 1 (physical) | Covers regime 2 (quality) |
|---|---|---|
| **MAFA in `extra_train_csv`** | ✅ real masks | ❌ |
| **RandomErasing** (patch on face, with label recompute) | ✅ synthetic patches | ❌ |
| **GaussianBlur, motion blur** | ❌ | ✅ |
| **JPEG compression artifacts** | ❌ | ✅ |
| **Additive Gaussian / shot noise** | ❌ | ✅ |
| **Pixelation / downsample-upsample** | ❌ | ✅ |
| **Style transfer** (cartoon, sketch) | ❌ | ✅ partially |

**Practical recipe**: `transforms.py` should expose a `degradation_level` knob that applies one or more quality-degradation transforms with **proportional label increment** (a Gaussian blur of σ=N corresponds to ~M% extra occlusion, calibrated on the dataset).

### Implications for pretraining sources

This insight **strongly justifies pretraining on MS-Celeb-1M (or VGGFace2 / a similar large face corpus)** — and **changes the previous "MS-Celeb has little value" stance**. Here's why:

1. `database3` (~96% of the Idemia training data) is **literally a subset of MS-Celeb-1M** — same Freebase MIDs, same `_align.webp` naming convention.
2. Pretraining on a larger version of MS-Celeb-1M gives the model **more views of similar identities** at the same canonical alignment, leading to better face manifold representation.
3. **iBOT-style domain adaptation is naturally suited** to specializing an already-trained encoder on a new face corpus — see [Pretraining method](#pretraining-method-ibot-not-mae) below.

Updated dataset priority for `data/pretrain/`:

| # | Dataset | Why |
|---|---|---|
| 1 ⭐ | **MS1MV3 WebDataset** (`gaunernst/ms1mv3-wds` on HF Hub, ~5M images, 100 `.tar` shards, 46 GB) | **Same distribution as `database3`** — optimal domain alignment, ready in WDS format |
| 2 | VGGFace2 (3.3M images, 9k identities) | Available without licence drama, close to MS-Celeb in spirit |
| 3 | CelebA (200k) | Good supplement, easy to get |
| 4 | WIDER Face (400k) | In-the-wild diversity (covers `database1` style images) |
| 5 | LFW (13k) | Too small for serious pretraining alone |

The raw MS-Celeb-1M was retired by Microsoft. InsightFace's cleaned MS1MV3 (5.1M images, 93k identities) is the practical equivalent; `gaunernst/ms1mv3-wds` packages it as WebDataset on HuggingFace Hub.

---

## Pretraining method: iBOT (not MAE)

Several iterations of this project considered MAE as the pretraining objective. **We chose iBOT-light (frozen teacher feature matching) instead.** This section explains why.

### The goal: domain adaptation, not new capability

We want to **specialize an already-trained DINOv3** on the face corpus (MS-Celeb / MS1MV3) that overlaps with the Idemia training set. The encoder already knows what a face looks like (it saw 1.7B images during LVD-1689M pretraining). What we want is to **adjust** its features to the specific face manifold of our downstream task, **without** wiping out what it learned.

This is fundamentally different from "add a new capability" (where MAE would shine — it adds pixel reconstruction skill).

### Why MAE is wrong for our case

| Aspect | MAE | iBOT |
|---|---|---|
| Objective | Reconstruct pixel values of masked patches | Match teacher's **feature embeddings** at masked positions |
| Effect on existing features | **Overwrites** — forces low-level pixel reconstruction | **Preserves** — uses teacher as anchor, just shifts distribution |
| Continuity with DINOv3 pretraining | Different objective (DINOv3 was trained with DINO+iBOT, not MAE) | **Same family** — extends the original pretraining |
| Risk of catastrophic forgetting | High (overrides DINOv3's feature space) | Low (teacher keeps features aligned) |
| HF native implementation | ✅ ready | ❌ but we use Meta's vendored `dinov3_repo/dinov3/loss/ibot_patch_loss.py` as reference |

The cleanest argument: **DINOv3 was originally trained with DINO+iBOT+KoLeo+Gram.** Continuing with the same iBOT objective is "more pretraining of the same kind". Switching to MAE is "different methodology" with no obvious upside.

### Our simplified iBOT (iBOT-light)

Full Meta iBOT uses Sinkhorn-Knopp centering, EMA teacher, multi-crop augmentation, projection heads, prototype distributions, and 4 simultaneous losses (`dino_clstoken_loss + ibot_patch_loss + koleo_loss + gram_loss`). We reproduce the essential mechanism more simply:

- **Student** = DINOv3 ViT-H+ (trainable)
- **Teacher** = same DINOv3 weights, **frozen** (no EMA, no Sinkhorn-Knopp) — anchor for the student
- **Mask ratio** = 0.5 (iBOT range, lower than MAE's 0.75)
- For each batch:
  1. Random binary mask over patches
  2. Student forward **with mask** (uses DINOv3 native `prepare_tokens_with_masks` → masked patches replaced by `mask_token`)
  3. Teacher forward **without mask**, `torch.no_grad()`
  4. Loss = `1 - cosine(student_patches, teacher_patches)` **at masked positions only**
  5. Backprop on student only
- **CLS drift** logged each step (auxiliary metric: how far the student's CLS token has moved from the teacher's)

Optional: switch to EMA teacher (`teacher_frozen: false, teacher_ema_decay: 0.999`) to get self-improving target. Frozen is simpler and avoids EMA bookkeeping.

Implementation: `src/pretrain_ibot.py:DinoV3IBoT` (~100 LOC core).

### Compute budget

| Setup | VRAM per GPU | Time on 2× RTX 3090 (BF16, 5M images, 30 epochs) |
|---|---|---|
| ViT-S/16 | ~5 GB | ~6 h |
| ViT-B/16 | ~9 GB | ~12 h |
| ViT-L/16 (300M) | ~14 GB | ~24 h |
| **ViT-H+/16 (600M)** ⭐ | **~22 GB** (with grad-ckpt) | **~36 h** |

Run with `sbatch scripts/pretrain_ibot_vith16plus_2x3090.sh`. Output: an MLflow run with the student encoder logged as `runs:/<run_id>/encoder`, usable in `train.py` via `resume_from_checkpoint`.

### Why we noticed this late

Honest postmortem: the previous iterations of this project anchored on "MAE" early because HuggingFace provides a ready-to-use `ViTMAEForPreTraining`. We thought implementation-first instead of objective-first. The cue was always there — Meta's iBOT loss is vendored in `dinov3_repo/dinov3/loss/ibot_patch_loss.py` and DINOv3's own ViT has a native `mask_token` and `prepare_tokens_with_masks` method designed exactly for masked-token forward passes. We should have looked at the vendored code earlier.

---

## Distribution shift train → test

Per the task brief (page 3 of `example/task_brief.pdf`):
- **Train**: histogram heavily peaked near 0, decays rapidly (most samples have low occlusion)
- **Test**: more uniform, spread up to 0.5, with substantial mid-occlusion mass

This means a model that minimises naive average MSE on train will be **systematically biased low** on mid-occlusion test cases. Mitigations already wired:

1. **`sampler_strategy: occlusion` or `gender_x_occ`** — rebalances training batches across occlusion levels (the "occlusion" dimension is `occ_bucket(10)` quantiles)
2. **Loss weighting `w = 1/30 + GT`** — already up-weights high-occlusion samples; partial mitigation
3. **Future**: importance reweighting `w_test(GT) / w_train(GT)` if needed (not yet implemented)

---

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
| `model.pooling` | cls | cls / mean / max / attention |
| `model.projection_size` | null | optional projection head |
| `model.output_activation` | sigmoid | sigmoid / clamp / none |
| `model.hidden_dropout_prob` | 0.1 | |

### Training

| Key | Default | Notes |
|---|---|---|
| `training.augmentation_level` | medium | none / light / medium / strong |
| `training.sampler_strategy` | gender_x_occ | none / gender / occlusion / gender_x_occ |
| `training.loss_type` | weighted_mse | weighted_mse / group_dro |
| `training.loss_focal_gamma` | 0.0 | 0 = no focal; >0 = upweight hard examples |
| `training.loss_fairness_lambda` | 0.0 | 0 = off; >0 = direct fairness penalty in loss |
| `training.group_dro_alpha` | 0.5 | only used if `loss_type: group_dro` |
| `training.ema_decay` | 0.9998 | 0 = no EMA |
| `training.ema_warmup_steps` | 200 | |
| `training.layer_decay` | 0.85 (large), 1.0 (small) | 1.0 = no LLRD |
| `training.early_stopping_patience` | 3 | 0 = disabled |
| `training.metric_for_best_model` | eval_score | |
| `training.greater_is_better` | false | for regression |

### HPO

| Key | Default | Notes |
|---|---|---|
| `optuna.n_trials` | 30 | budget |
| `optuna.objective_mode` | score | score / loss / pareto(score,err_diff) |
| `optuna.rotate_val_seed` | true | val_seed per trial → robustness |

### Inference

| Flag / config | Default | Where |
|---|---|---|
| `use_tta` (in `predict.py:CONFIG`, or `--no-tta`) | true | hflip-only TTA |
| `--delta-f` / `--delta-m` | none | post-hoc bias correction |
| `submission_format` | true | emits `filename, FaceOcclusion, gender='x'` |

### Top-level

| Flag | Default | Notes |
|---|---|---|
| `use_mlflow` (in `train.py:CONFIG`) | true | set false to disable all MLflow ops |
| `--no-mlflow` (predict) | n/a | predict.py never calls MLflow |

---

## Backbones

DINOv3 ViT-S/16 (`dinov3_vits16`, 22M) is **vendored with weights** — works out of the box on any hardware.

For larger DINOv3 variants (ViT-L+/16, ViT-H+/16), the weights live behind Meta's DINOv3 License Agreement (`https://dl.fbaipublicfiles.com/dinov3/...` returns 403 without a Meta-signed token). Use `scripts/download_dinov3_weights.sh` as a URL reference, accept the licence on the [DINOv3 GitHub](https://github.com/facebookresearch/dinov3), download manually, and drop the `.pth` into `src/models/weights/` — the loader auto-detects them.

**Alternative open backbones** (HuggingFace, no licence) — recommended for ensemble diversity if you don't have DINOv3-L/H access:

| HF id | Params | Type |
|---|---|---|
| `facebook/dinov2-base` | 86M | SSL transformer |
| `facebook/dinov2-large` | 300M | SSL transformer |
| `facebook/convnextv2-large-22k-224` | 198M | CNN, MAE-pretrained |
| `Yuxin-CV/EVA-02-L-14` | 305M | MIM transformer |
| `google/vit-base-patch16-224` | 86M | ImageNet supervised |

To use any of these, just edit `model.model_name` in a YAML — `FaceOccRegressor._build_backbone` already dispatches DINOv3 vs HuggingFace.

---

## MLflow — what is logged

Per epoch (via `compute_metrics`):
- `eval_score`, `eval_err_F`, `eval_err_M`, `eval_err_diff`, `eval_mse`, `eval_mae`, `eval_loss`

Run start (params):
- `architecture`, `model_name`, `output_dim`, `pooling`, `augmentation_level`, `sampler_strategy`, `loss_type`, `seed`, `num_train`, `num_val`, `num_train_female`, `num_train_male`, `num_val_female`, `num_val_male`, `train_gender_ratio_M_over_F`

Run start (data-distribution metrics):
- `data_train_occ_mean/std`, `data_val_occ_mean/std`
- `data_train_occ_female_mean`, `data_train_occ_male_mean`
- `data_val_occ_female_mean`, `data_val_occ_male_mean`

Run end:
- `val_score`, `val_err_F`, `val_err_M`, `val_err_diff`, `final_eval_loss`
- `test_score`, `test_err_F`, `test_err_M` (if test CSV passed)

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
│   ├── face_occ_classifier.py    FaceOccRegressor (HF + DINOv3 dispatch)
│   ├── dinov3_loader.py          weights + variant picker
│   ├── dinov3_repo/              Meta's vendored DINOv3 code
│   └── weights/                  local .pth files (vits16 shipped)
├── training/callbacks.py         MlflowClient + EMA
├── inference/
│   ├── tta.py                    hflip-only TTA (multi-crop unsafe for ratio regression)
│   ├── ensemble.py               averages scalar predictions across N models
│   └── calibration.py            post-hoc per-gender bias correction
└── utils/
    ├── losses.py                 WeightedMSELoss, GroupDROLoss, make_sampler_keys
    ├── metrics.py                compute_score (gender-aware, official formula)
    ├── config.py / environment.py / mlflow_utils.py / distributed.py

configs/architectures/
├── dinov3-vits16-face-occ.yaml   ViT-S/16 — runs anywhere (weights shipped)
├── dinov3-vitl16plus-p100.yaml   ViT-L+/16 — 2× P100 (requires manual weight download)
└── dinov3-vith16plus-3090.yaml   ViT-H+/16 — 2× 3090 (requires manual weight download)

scripts/
├── train_dinov3_vits16_2gpu.sh / optimize_*.sh / ensemble_*.sh   SLURM, DDP, 30h
├── download_dinov3_weights.sh    URL reference for larger DINOv3 weights
├── connect.sh / clean.sh
```

---

## Audit — fixes applied in this version

| # | Bug / weakness | Status |
|---|---|---|
| 1 | LLRD broken under DDP (`self.model.backbone` raises on wrapped model) | ✅ fixed (`_unwrap` helper) |
| 2 | Unused `nn.functional.mse_loss` in `FaceOccRegressor.forward` | ✅ removed (loss only computed in `WeightedMSETrainer`) |
| 3 | `NaN or 0.0` truthy-check anti-pattern in data-distribution logging | ✅ replaced by `_safe_mean` |
| 4 | No way to disable MLflow for local debug | ✅ `use_mlflow: bool` toggle in `train()` |
| 5 | TTA only available via code, not at submission time | ✅ `use_tta` in `predict.py:CONFIG` + `--no-tta` flag |
| 6 | No mechanism for distribution-shift mitigation (train peaks low, test more uniform) | ✅ `sampler_strategy: occlusion / gender_x_occ` |
| 7 | Only one fairness mechanism (gender-balanced sampler) | ✅ added Group-DRO loss + post-hoc bias correction |
| 8 | DINOv3 weights other than ViT-S/16 hardcoded missing | ✅ all 6 paths registered, loader detects presence |
| 9 | `tqdm` not in pyproject (auto-pulled but implicit) | ✅ added explicit |

---

## Roadmap — next steps

### P0 — Run the baseline
1. Download face crops to `data/crops/Crop_224_5fp_100K/`
2. `inv train` (uses `dinov3-vits16-face-occ.yaml`) — expect `eval_score < 0.005` after a few epochs
3. `sbatch scripts/optimize_dinov3_vits16_2gpu.sh` — HPO
4. `inv ensemble` on the best YAML — produces 5 fold models
5. `inv predict` — emits `test_predictions.csv` for submission

### P1 — Push performance
6. **Post-hoc bias correction**: after a trained model, run `find_optimal_bias(preds_val, gt_val, gender_val)` to get `δ_F`, `δ_M`, then `inv predict --delta-f <df> --delta-m <dm>`. Free fairness fix.
7. **Multi-architecture ensemble**: add DINOv2-base (HF, no license needed) and ConvNeXt v2-large YAMLs → ensemble across 3 backbones × 5 folds = 15 prediction sources.
8. **Resolution upgrade to 384×384**: write a `get_image_processor(name, size=384)` variant. Drop batch by ~3×, expect −0.5 to −1.5 % on `score`.
9. **iBOT-light pretraining** on MS1MV3 (or VGGFace2 / LFW) via `sbatch scripts/pretrain_ibot_vith16plus_2x3090.sh` — see [Pretraining method](#pretraining-method-ibot-not-mae). Marginal expected gain on top of DINOv3 H+ but worth trying if compute is available.

### P2 — Refinements
10. **Importance reweighting** to match test distribution explicitly (compute histogram ratios).
11. **Group-DRO**: try `loss_type: group_dro` with `group_dro_alpha ∈ {0.3, 0.5, 1.0}` if disparity persists.
12. **Pseudo-labeling**: use ensemble predictions on test, keep extreme-confidence samples, retrain.

### P3 — Polish
13. (deprecated — was about DINOv3→MAE mapping, now obsolete since we use iBOT).
14. More tests (DDP smoke test, EMA on a tiny model).

---

## SOTA references

- **Sagawa et al.** *Distributionally Robust Neural Networks for Group Shifts* — Group-DRO. [arXiv:1911.08731](https://arxiv.org/abs/1911.08731)
- **DINOv3 / DINOv2** — Meta (DINOv3 vendored at `src/models/dinov3_repo/`; DINOv2: [arXiv:2304.07193](https://arxiv.org/abs/2304.07193))
- **MAE** — He et al. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- **ConvNeXt v2** — Woo et al. [arXiv:2301.00808](https://arxiv.org/abs/2301.00808)
- **EVA-02** — Fang et al. [arXiv:2303.11331](https://arxiv.org/abs/2303.11331)
- **EMA / SWA** — Izmailov et al. [arXiv:1803.05407](https://arxiv.org/abs/1803.05407)
- **LLRD** — Howard & Ruder (ULMFiT). [arXiv:1801.06146](https://arxiv.org/abs/1801.06146)

---

## Hardware notes

| Platform | VRAM | Precision | Max DINOv3 (this repo) |
|---|---|---|---|
| 1× P100 | 16 GB | FP16 | ViT-S/16 (shipped, comfortable) |
| 2× P100 DDP | 32 GB | FP16 | ViT-L+/16 (requires manual weight DL) |
| 1× RTX 3090 | 24 GB | BF16 | ViT-L/16 |
| 2× RTX 3090 DDP | 48 GB | BF16 | ViT-H+/16 (requires manual weight DL) |

All SLURM scripts use `torchrun --nproc_per_node=2`, 30h time, with the right hardware tags.
