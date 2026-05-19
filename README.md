# face-occ-detector

Binary image classification — occluded vs non-occluded faces. MSIA DataChallenge.

**Stack**: DINOv3 / ViT-B · HuggingFace Transformers · MLflow · Optuna · FocalLoss · DynamicAugDataset · MAE pretraining · k-fold + multi-arch ensemble + TTA.

**Hardware target**: Pascal P100 (16 GB, FP16), with optional RTX 3090 (24 GB, BF16).

---

## Table of contents
- [Quick start](#quick-start)
- [Data format](#data-format)
- [Project layout](#project-layout)
- [Pipeline](#pipeline)
- [DINOv3 — variants & per-hardware configs](#dinov3--variants--per-hardware-configs)
- [Audit (May 2026)](#audit-may-2026)
- [Roadmap — what to do next, in order](#roadmap--what-to-do-next-in-order)
- [SOTA references](#sota-references)

---

## Quick start

```bash
# Full pipeline (in order)
inv pretrain         # Stage 1 — MAE continued pretraining on unlabeled faces (optional)
inv optimize         # Stage 2a — Optuna HPO
inv ensemble         # Stage 2b — k-fold ensemble training on best config

# Single training (skips HPO + ensemble)
inv train

# Evaluate a saved model
inv evaluate --model-uri runs:/RUN_ID/model --data-csv data/raw/val.csv

# UIs
inv mlflow-ui          # http://localhost:5000
inv optuna-dashboard   # http://localhost:8080
```

## Data format

**CSV** — columns `image_path`, `label` (0 = not occluded, 1 = occluded):
```
image_path,label
data/raw/img_001.jpg,0
data/raw/img_002.jpg,1
```

**ImageFolder** — class subdirectories, pre-split or single folder:
```
data/raw/
  train/ not_occluded/ ; occluded/
  val/   not_occluded/ ; occluded/
```

## Project layout

```
src/
├── train.py              # supervised training entry (single run)
├── optimize.py           # Optuna HPO orchestration (Stage 2a)
├── ensemble_train.py     # k-fold ensemble training (Stage 2b)
├── pretrain_mae.py       # MAE continued pretraining (Stage 1)
├── predict.py            # inference on CSV / dir / single / interactive
├── evaluate_model.py     # classification_report on a held-out set
├── data/
│   ├── dataset.py        # FaceOccDataset, DynamicAugDataset, loaders, samplers
│   └── transforms.py     # PIL→PIL augmentation pipeline (light/medium/strong)
├── models/
│   ├── face_occ_classifier.py  # backbone-agnostic classifier (HF + DINOv3 dispatch)
│   ├── dinov3_loader.py        # DINOv3 weights + hardware-aware variant picker
│   ├── dinov3_repo/             # Meta's vendored DINOv3 codebase (torch.hub)
│   └── weights/                 # local .pth checkpoints (vits16 shipped)
├── training/
│   └── callbacks.py      # EMA + MlflowClient + DynamicAug resampler callbacks
├── inference/
│   ├── tta.py            # test-time augmentation (hflip, multi-crop)
│   └── ensemble.py       # multi-model softmax averaging + threshold sweep
└── utils/
    ├── config.py         # type-coerced YAML loader
    ├── environment.py    # device detection + transformers logging
    ├── losses.py         # FocalLoss + class weights
    ├── metrics.py        # F1, precision, recall, accuracy
    ├── mlflow_utils.py   # log_params / log_metrics / get_or_create_experiment
    └── distributed.py    # DDP helpers (setup, broadcast, barrier, is_main)

configs/architectures/
├── vit-base-face-occ.yaml          # ViT-B/16 baseline (works anywhere)
├── dinov3-vits16-face-occ.yaml     # DINOv3 ViT-S/16 baseline (weights shipped)
├── dinov3-vitl16plus-p100.yaml     # DINOv3 ViT-L+/16 for 2× P100 (FP16, grad-ckpt)
└── dinov3-vith16plus-3090.yaml     # DINOv3 ViT-H+/16 for 2× 3090 (BF16, DDP)
```

## Pipeline

```
┌────────────────────────────┐    ┌──────────────────────────┐    ┌────────────────────────┐
│ Stage 1 — Pretraining      │ →  │ Stage 2a — HPO (Optuna)  │ →  │ Stage 2b — k-fold      │
│ MAE on unlabeled faces     │    │ Single split + rotate    │    │ ensemble training      │
│ (optional, costly)         │    │ val_seed per trial       │    │ on best YAML           │
└────────────────────────────┘    └──────────────────────────┘    └────────────────────────┘
                                                                          ↓
                                                          ┌────────────────────────────┐
                                                          │ Stage 3 — Inference         │
                                                          │ TTA + multi-arch ensemble   │
                                                          └────────────────────────────┘
```

### Stage 1 — Domain-adaptive pretraining (`src/pretrain_mae.py`)

Continues MAE-style pretraining on a face-only corpus (CelebA / VGGFace2 / LFW unlabeled) so the encoder learns face-specific spatial features before the classification finetuning. Starts from `facebook/vit-mae-large` (HF native, functional) or DINOv3 (mapping hook is a stub — fill in the encoder weight transfer if you take this path).

Output: an encoder saved to MLflow that `train.py` consumes via `resume_from_checkpoint`.

**Skip conditions**: no unlabeled face corpus, or compute is tight (DINOv3 raw features are already SOTA).

### Stage 2a — Hyperparameter optimization (`src/optimize.py`)

Optuna study with TPE (or NSGA-II for `pareto` mode). For each trial:
1. Samples hyperparameters from the YAML `optuna.search_space`
2. Writes a temp YAML in `configs/architectures/optuna_trials/`
3. Trains once via `train()`
4. Reports F1 / loss / (F1, gap) back to Optuna
5. MedianPruner kills bad trials at epoch 2+

`rotate_val_seed=true` rotates the train/val split seed per trial. Robustness without 5× compute. Best config saved as `{arch}_optuna_best.yaml`.

### Stage 2b — k-fold ensemble training (`src/ensemble_train.py`)

Takes the best YAML and re-trains **N times** with stratified k-fold (k=5 default). Each fold:
1. `data/folds/{arch}/fold_<i>_train.csv` + `fold_<i>_val.csv` written
2. `train(architecture=arch, data_csv=fold_i_train.csv, val_data_csv=fold_i_val.csv, seed=42+i)`
3. Logged to MLflow as a child run with tags `fold=<i>` `ensemble_member=<i>`

Aggregates: parent run logs `cv_f1_macro ± std`. Output: N model URIs ready for ensemble averaging.

### Stage 2b+ — Multi-architecture ensemble

Run `ensemble_train` once per architecture. Mixing inductive biases (transformer-DINO + transformer-MIM + CNN-MAE) gives the biggest leaderboard gains.

```bash
inv ensemble --architecture dinov3-vith16plus-3090_optuna_best     # primary
inv ensemble --architecture eva02-large-face-occ_optuna_best       # add later
inv ensemble --architecture convnextv2-large-face-occ_optuna_best  # add later
```

### Stage 3 — Inference (`src/inference/`)

```python
from src.inference.ensemble import ensemble_predict, threshold_sweep

probs = ensemble_predict(
    model_uris=[f"runs:/{rid}/model" for rid in fold_run_ids + other_arch_run_ids],
    image_paths=test_df["image_path"].tolist(),
    use_tta=True, tta_mode="default",          # default = identity + hflip
)
# Optional threshold tuning on a labeled val set:
# best_tau, score = threshold_sweep(val_probs, val_labels, lambda p, l: f1_score(l, p, average="macro"))
predictions = probs.argmax(dim=-1).numpy()
```

---

## DINOv3 — variants & per-hardware configs

DINOv3 is vendored at `src/models/dinov3_repo/` (Meta's official code, MIT-style license). Shipped weights: **ViT-S/16 only** (83 MB). Larger variants need a download from `https://dl.fbaipublicfiles.com/dinov3/dinov3_<arch>/dinov3_<arch>_pretrain_lvd1689m-<hash>.pth` (DINOv3 License required), dropped into `src/models/weights/`, and registered in `_AVAILABLE_WEIGHTS` in [src/models/dinov3_loader.py](src/models/dinov3_loader.py).

| Variant | Params | Hidden | Train VRAM (fp16, batch 16, grad-ckpt) | Use case |
|---|---|---|---|---|
| `dinov3_vits16` | 22M | 384 | ~2 GB | any device — **shipped** |
| `dinov3_vitb16` | 86M | 768 | ~5 GB | 1× P100 baseline |
| `dinov3_vitl16` | 300M | 1024 | ~10 GB | 1× P100 tight / 1× 3090 comfortable |
| `dinov3_vitl16plus` | 400M | 1024 | ~13 GB | **2× P100 (FP16, DDP)** ← `dinov3-vitl16plus-p100.yaml` |
| `dinov3_vith16plus` | 600M | 1280 | ~17 GB | **2× 3090 (BF16, DDP)** ← `dinov3-vith16plus-3090.yaml` |
| `dinov3_vit7b16` | 7B | 4096 | — | out of reach |

Auto-pick the biggest viable variant:
```python
from src.models.dinov3_loader import recommend_dinov3_variant
arch = recommend_dinov3_variant(headroom_gb=2.0)
```

---

## Audit (May 2026)

### What's strong

- **Modular layout**: data / models / training / inference / utils. Each module is single-purpose.
- **YAML-driven configs** with typed key coercion (`INT_KEYS`, `FLOAT_KEYS`).
- **MLflow + Optuna**: nested runs, NSGA-II Pareto support, MedianPruner, parent-child run topology.
- **Loss toolkit**: FocalLoss (gamma + class_weights + label smoothing), `WeightedRandomSampler` per class, DynamicAug per-epoch resampling.
- **Backbone dispatch**: `FaceOccClassifier` transparently handles HF (`AutoModel`) and DINOv3 (Meta hub) via `_build_backbone` / `_forward_backbone`.
- **DDP / distributed**: extracted to `src/utils/distributed.py`, reused by Optuna.
- **EMA + augmentation + TTA + ensemble**: now first-class — `src/training/callbacks.py`, `src/data/transforms.py`, `src/inference/{tta,ensemble}.py`.
- **Trunc_normal head init** (BERT/ViT convention) applied to classifier + projection + attention_pool.
- **Pipeline scripts**: `pretrain_mae.py`, `optimize.py`, `ensemble_train.py` — one stage per script, all log to MLflow with consistent parent-child topology.
- **Tests pass** (`uv run pytest` green after the refactor).

### What's still missing (ordered by F1 impact)

| # | Gap | Impact | Status |
|---|---|---|---|
| 1 | `RandomErasing` (tensor-level, post-processor) not wired into the collator | **+0.5-1.5% F1** on occlusion tasks specifically | helper exists in `transforms.build_tensor_random_erasing` — needs a collator |
| 2 | Mixup / CutMix not wired | +0.3-0.8% F1 | recipe documented in `transforms.py`, needs `timm.data.Mixup` integration |
| 3 | DINOv3 weights for L+ / H+ not present locally | unblocks the biggest backbones | manual download required (DINOv3 license) |
| 4 | DINOv3 → ViTMAE weight mapping in `pretrain_mae.py` | unlocks MAE pretraining starting from DINOv3 | stub raises NotImplementedError — write key remapping |
| 5 | Layer-wise learning rate decay (LLRD) | +0.2-0.5% F1 on large transformers | not implemented; ~30 LOC |
| 6 | Pseudo-labeling on unlabeled test data | +0.3-1.0% F1 | not implemented; depends on competition rules |
| 7 | Resolution scaling (384/448 instead of 224) | +0.3-0.7% F1 | requires custom processor (HF ViT at higher size) |
| 8 | Face detection / alignment preprocessing | depends on dataset (already-cropped vs in-the-wild) | not implemented |
| 9 | Threshold tuning automation in `evaluate_model.py` | depends on metric | `threshold_sweep` exists in `inference/ensemble.py`, not auto-wired |
| 10 | Tests for new modules (EMA, TTA, ensemble) | code quality | only `test_model.py` exists |

### Refactoring applied this pass

- Extracted `src/utils/mlflow_utils.py` (`log_params`, `log_metrics`, `get_or_create_experiment`) — used by `train.py`, `optimize.py`, `ensemble_train.py`.
- Extracted `src/utils/distributed.py` (DDP helpers) — used by `optimize.py`.
- Created `src/training/callbacks.py` with `EMACallback`, `MlflowClientCallback`, `AugResamplerCallback` (moved out of `train.py`).
- Created `src/data/transforms.py` (augmentation pipeline).
- Created `src/inference/tta.py` + `src/inference/ensemble.py`.
- Split `train.py` into small functions: `_resolve_data_paths`, `_load_train_val`, `_build_datasets`, `_compute_overfit_stats`, `_save_model_to_mlflow`, `_start_or_attach_run`. Main `train()` is now linear and readable.
- `train()` now accepts `val_data_csv` to skip its internal split — this is what unblocks `ensemble_train.py`.
- `train.py` now reads `fp16` / `bf16` / `gradient_checkpointing` / `augmentation_level` / `ema_decay` / `ema_warmup_steps` from the YAML `training` block.
- `FaceOccClassifier` dispatches HF/DINOv3 via `_build_backbone` and supports both `last_hidden_state` and `get_intermediate_layers` via `_forward_backbone`.
- `evaluate_model.py` now respects `image_base_dir`.
- `predict.py` `CONFIG["text"]` (leftover from text-classification origin) → `CONFIG["image"]`.
- `utils/config.py` cleaned of dead keys (`num_epochs`, `train_batch_size`, `synthetic_seed`, `cross_attention_heads`, `extra_transformer_*`, etc.) and aug + EMA keys added.
- `dataset.py` `_open` / `_encode` extracted as free functions, shared between `FaceOccDataset` and `DynamicAugDataset` — also adds a `transform` arg for augmentation.

---

## Roadmap — what to do next, in order

Priority order for **maximising leaderboard rank**. P0 = unblock the pipeline / biggest F1 gain. P3 = polish.

### P0 — Critical (do first)

1. **Download DINOv3 ViT-H+/16 weights** (or ViT-L+/16 for P100).
   `https://dl.fbaipublicfiles.com/dinov3/dinov3_vith16plus/dinov3_vith16plus_pretrain_lvd1689m-<hash>.pth`. Drop in `src/models/weights/` and register the path in `_AVAILABLE_WEIGHTS` of `dinov3_loader.py`.
2. **Wire RandomErasing at the collator level**. The PIL-level pipeline can't do it. Write a `RandomErasingCollator` that wraps `default_data_collator`, then `RandomErasing(p=0.5, scale=(0.02, 0.4))` post-processor. Plug into `train.py` via the `augmentation_level` YAML key.
3. **Wire Mixup / CutMix** via `timm.data.Mixup`. Requires:
   - Add `timm>=0.9.0` to `pyproject.toml`
   - A `MixupCollator` calling `Mixup(...)` on (pixel_values, labels), returning soft labels
   - Adjust `WeightedLossTrainer.compute_loss` to accept soft labels (small change to `FocalLoss`).
4. **Run Stage 1 (MAE pretraining)** on whatever face corpus you can collect — even 50k images of unlabeled faces gives a measurable boost. Use `facebook/vit-mae-large` as the source if DINOv3 mapping isn't ready.

### P1 — High-impact

5. **Per-hardware decision**:
   - 2× P100 only → train `dinov3-vitl16plus-p100`
   - Got 2× 3090 → train `dinov3-vith16plus-3090`
   - One node only → fall back to `dinov3-vits16-face-occ` (ships with weights, no download)
6. **HPO → Ensemble loop**: `inv optimize --architecture dinov3-vith16plus-3090` → `inv ensemble --architecture dinov3-vith16plus-3090_optuna_best`. Expect ~CV F1 within ±0.5% of leaderboard.
7. **Layer-wise learning rate decay (LLRD)**. Pattern: parameters in deeper layers get a smaller LR (factor 0.9^layer). ~30 LOC added to `WeightedLossTrainer.create_optimizer()`. Standard for large-transformer finetuning.
8. **Add a second backbone YAML** for ensemble diversity (recommended: EVA-02-L or ConvNeXt v2-L). Same workflow: HPO → ensemble. Different inductive bias = different errors = better ensemble.

### P2 — Refinements

9. **TTA at evaluate / inference time**: wire `src/inference/tta.predict_tta_batch` into `evaluate_model.py` and `predict.py` (currently single-pass).
10. **Resolution scaling**: write a custom processor wrapper that overrides `size={"height": 384, "width": 384}`. ViT positional embeddings need to be interpolated — DINOv3 handles this natively via `get_intermediate_layers`.
11. **Pseudo-labeling**: if the competition test set is unlabeled but accessible, predict on it with the current ensemble, keep predictions with `max(softmax) > 0.95`, add them as extra training data, retrain. One iteration typically gives +0.3-0.7%.
12. **Face detection / alignment** (if the data is in-the-wild): MTCNN or RetinaFace as a preprocessing step, then crop + align. Adds a `face_align` flag to the dataset config.

### P3 — Polish

13. **Tests** for `EMACallback`, `tta.predict_tta_batch`, `ensemble.ensemble_predict`, `ensemble_train._build_folds`.
14. **DINOv3 → ViTMAE weight mapping** in `pretrain_mae._build_mae_from_dinov3` — currently raises. Walk DINOv3's `qkv` layers and split into HF's separate `query`/`key`/`value`, map `mlp.fc1/fc2` → HF's `intermediate.dense`/`output.dense`. Tedious but bounded.
15. **`__init__.py` exports** for clean external API.

### Sanity-check commands

```bash
uv run pytest                             # all tests green
uv run python -c "from src.train import train"            # imports OK
uv run python -c "from src.inference.ensemble import ensemble_predict"
inv lint && inv typecheck                # style + types
```

---

## SOTA references

### Backbones
- **DINOv3** (Meta, 2024) — see vendored code at `src/models/dinov3_repo/`
- **DINOv2** — Oquab et al. *Learning Robust Visual Features without Supervision*. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- **MAE** — He et al. *Masked Autoencoders Are Scalable Vision Learners*. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- **EVA-02** — Fang et al. [arXiv:2303.11331](https://arxiv.org/abs/2303.11331)
- **ConvNeXt v2** — Woo et al. [arXiv:2301.00808](https://arxiv.org/abs/2301.00808)
- **SigLIP** — Zhai et al. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
- **BEiT v2** — Peng et al. [arXiv:2208.06366](https://arxiv.org/abs/2208.06366)
- **ViT** — Dosovitskiy et al. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

### Face-domain models
- **FaRL** — Zheng et al. *General Facial Representation Learning*. [arXiv:2112.03109](https://arxiv.org/abs/2112.03109)
- **ArcFace** — Deng et al. [arXiv:1801.07698](https://arxiv.org/abs/1801.07698)
- **AdaFace** — Kim et al. [arXiv:2204.00964](https://arxiv.org/abs/2204.00964)
- **MagFace** — Meng et al. [arXiv:2103.06627](https://arxiv.org/abs/2103.06627)
- **InsightFace zoo** — [github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)

### Face occlusion datasets
- **MAFA** — Ge et al. *Detecting Masked Faces in the Wild* (CVPR 2017) — [escience.cn](http://www.escience.cn/people/geshiming/mafa.html)
- **COFW** — Burgos-Artizzu et al. (ICCV 2013) — [vision.caltech.edu](http://www.vision.caltech.edu/xpburgos/ICCV13/)
- **CelebA** — Liu et al. (ICCV 2015) — [mmlab.ie.cuhk.edu.hk](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **VGGFace2** — Cao et al. — [github.com/ox-vgg/vgg_face2](https://github.com/ox-vgg/vgg_face2)
- **WiderFace** — Yang et al. — [shuoyang1213.me](http://shuoyang1213.me/WIDERFACE/)
- **LFW** — [vis-www.cs.umass.edu/lfw](http://vis-www.cs.umass.edu/lfw/)

### Training tricks
- **Mixup** — Zhang et al. [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
- **CutMix** — Yun et al. [arXiv:1905.04899](https://arxiv.org/abs/1905.04899)
- **RandomErasing** — Zhong et al. [arXiv:1708.04896](https://arxiv.org/abs/1708.04896) — the direct analogue of occlusion!
- **RandAugment** — Cubuk et al. [arXiv:1909.13719](https://arxiv.org/abs/1909.13719)
- **Focal Loss** — Lin et al. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- **Label Smoothing** — Müller et al. [arXiv:1906.02629](https://arxiv.org/abs/1906.02629)
- **EMA of weights** — see [`timm.utils.ModelEmaV2`](https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py); our implementation in `src/training/callbacks.py:EMACallback`
- **LLRD** (Layer-wise LR Decay) — Howard & Ruder 2018, ULMFiT-derived. [arXiv:1801.06146](https://arxiv.org/abs/1801.06146)
- **SAM** (Sharpness-Aware Minimisation) — Foret et al. [arXiv:2010.01412](https://arxiv.org/abs/2010.01412)
- **Bag of Tricks** — He et al. *Bag of Tricks for Image Classification with CNNs*. [arXiv:1812.01187](https://arxiv.org/abs/1812.01187)

### Pretraining strategy
- **Don't Stop Pretraining** — Gururangan et al. [arXiv:2004.10964](https://arxiv.org/abs/2004.10964) (origin of the term *domain-adaptive pretraining*)
- **Continued MAE pretraining** — see DINOv2 §3.3 for the multi-stage SSL recipe

### Tooling
- **HuggingFace Transformers** — [docs](https://huggingface.co/docs/transformers)
- **timm** (broader backbone zoo) — [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- **MLflow** — [docs](https://mlflow.org/docs/latest/index.html)
- **Optuna** — [docs](https://optuna.readthedocs.io/)

---

## Hardware notes

| Platform | VRAM | Precision | Max DINOv3 (this repo) |
|---|---|---|---|
| 1× P100 | 16 GB | FP16 (no BF16) | ViT-L/16 tight, ViT-B/16 comfortable |
| 2× P100 (DDP) | 32 GB | FP16 | **ViT-L+/16** (config: `dinov3-vitl16plus-p100`) |
| 1× RTX 3090 | 24 GB | BF16 + TF32 | ViT-L/16 comfortable, ViT-H+/16 feasible |
| 2× RTX 3090 (DDP) | 48 GB | BF16 | **ViT-H+/16** (config: `dinov3-vith16plus-3090`) |

DDP launch:
```bash
torchrun --nproc_per_node=2 src/train.py
torchrun --nproc_per_node=2 src/optimize.py
```
