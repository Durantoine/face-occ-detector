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

# 2. Environment — single pyproject.toml, CUDA index auto-selected on Linux
uv sync                # local (Mac/CPU/MPS): plain PyPI wheels
                       # cluster (Linux):     CUDA 12.6 wheels via [tool.uv.sources] marker

# 3. Pipeline (declarative — edit CONFIG dicts at top of each src/*.py to switch settings)
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

### Cluster workflow (SLURM, 2× RTX 3090)

The three canonical sbatch scripts cover the full A/B comparison:

```bash
# A) Optimize without pretrain (baseline)
sbatch scripts/optimize_dinov3_vith16plus_2x3090.sh
#  → uses configs/architectures/dinov3-vith16plus-3090.yaml

# B) iBOT-light pretrain, then optimize from that encoder
sbatch scripts/pretrain_ibot_vith16plus_2x3090.sh           # single 30h chunk
# OR
./scripts/chain_pretrain.sh 3                                # chain 3× (90h cumulative)
# After pretrain completes:
#   1. read results/pretrain/mlflow_run_id.txt
#   2. open configs/architectures/dinov3-vith16plus-3090-ibot.yaml
#      and replace __FILL_PRETRAIN_RUN_ID__ with that run_id
sbatch scripts/optimize_dinov3_vith16plus_from_pretrain_2x3090.sh
#  → uses configs/architectures/dinov3-vith16plus-3090-ibot.yaml
#  → train.py auto-loads runs:/<run_id>/encoder into the backbone and logs
#    `model_init_backbone_from`, `init_backbone_pretrain_run_id`, all `pretrain_*`
#    params from the pretrain run, and a `pretrain_run_id` tag for traceability.
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

Run with `sbatch scripts/pretrain_ibot_vith16plus_2x3090.sh` (or `./scripts/chain_pretrain.sh 3` to chain three 30 h SLURM submissions). Output: an MLflow run with the student encoder logged as `runs:/<run_id>/encoder`. To use it for finetuning, fill that URI into `model.init_backbone_from` of `configs/architectures/dinov3-vith16plus-3090-ibot.yaml` and launch `sbatch scripts/optimize_dinov3_vith16plus_from_pretrain_2x3090.sh`. `train.py` then loads the weights into the backbone and copies all pretrain params into the finetune run for full provenance.

### Why we noticed this late

Honest postmortem: the previous iterations of this project anchored on "MAE" early because HuggingFace provides a ready-to-use `ViTMAEForPreTraining`. We thought implementation-first instead of objective-first. The cue was always there — Meta's iBOT loss is vendored in `dinov3_repo/dinov3/loss/ibot_patch_loss.py` and DINOv3's own ViT has a native `mask_token` and `prepare_tokens_with_masks` method designed exactly for masked-token forward passes. We should have looked at the vendored code earlier.

---

## Monitoring UIs (MLflow + Optuna) — SLURM workflow

Le gateway `gpu-gw` n'a pas assez de RAM pour faire tourner MLflow UI directement (`Killed` au démarrage). Solution : lancer les UIs sur la partition `CPU` (10 nœuds, 4-day timelimit, pas de QOS GPU) et port-forward depuis ton laptop.

**One-liner pour tout lancer :**

```bash
# Sur le cluster:
./scripts/launch_ui.sh
```

Le script :
1. `sbatch scripts/mlflow_ui.sh` et `scripts/optuna_dashboard.sh` sur la partition `CPU`
2. Poll `squeue` jusqu'à voir les 2 jobs `RUNNING`
3. Extrait les hostnames des nœuds alloués
4. Imprime la commande **`ssh -L ... gpu-gw`** à copy-paste sur ton laptop

Output type :
```
ssh -N -L 5000:nodecpu03:5000 -L 8080:nodecpu05:8080 adurand-25@gpu-gw
```

Sur ton laptop, après le `ssh -N` (qui reste ouvert) :
- `http://localhost:5000` → MLflow UI
- `http://localhost:8080` → Optuna dashboard

Stop avec `scancel <job-ids>` (le script les imprime). Les jobs CPU durent jusqu'à 24h par défaut (modifiable via `#SBATCH --time` dans le sbatch).

---

## Sampler × loss-weight design — the F-occ confound

### The two-axis imbalance

Our 100k train subset (a stratified slice of Idemia's full 752k brief train set — **already partly balanced toward mid-occ**, peak at bin 0 is 32.5 % not 44 %) has two independent imbalances:

1. **Gender** : M/F = 2.086 (67.6 % M, 32.4 % F)
2. **Occlusion** : 32.5 % in `[0, 0.025)`, decaying to <0.1 % above `[0.45, 0.50)`

And a critical **confound** :

| gender | n | occ_mean | occ_std |
|---|---|---|---|
| F (0) | 32,400 | **0.129** | 0.094 |
| M (1) | 67,600 | **0.061** | 0.073 |

→ **Women's faces have 2× more occlusion on average** in the training data. Any model can learn `gender → +0.07 occlusion` as a shortcut instead of looking at actual occlusion cues. This shortcut works on train but degrades test performance and inflates `|Err_F − Err_M|`.

### Four options to decorrelate

The math: with `f_sampler(b)` the per-bin batch frequency under the sampler and `w_imp(b)` the per-bin loss multiplier, the optimizer effectively minimizes `Σ_b f_sampler(b) · w_imp(b) · E[w_metric·err | bin = b]`. To match the test-time evaluation `Σ_b p_test(b) · E[w_metric·err | bin = b]`, we need `f_sampler(b) · w_imp(b) = p_test(b)` per bin.

Following the same pattern as `online-polarization-detector` (which tested `none / class_weights / balanced_sampler / both` in HPO), we ship four strategies:

| Option | Sampler | Loss weights | Pro | Con (especially on our 100k subset) |
|---|---|---|---|---|
| **A** (default) | `gender` (balance F/M only, occ natural) | `w_imp = p_test / p_train` | each sample seen ~1× per epoch, no over-fit risk on rare bins, uses 100 % of train data | does NOT decorrelate gender × occ at the batch level — model can still learn the shortcut, mitigated only by `loss_fairness_lambda` |
| **B** | `gender_x_occ` (uniform over 20 bins × 2 genders) | `w_imp = n_buckets · p_test` | hard decorrelation of (gender × occ) at the batch level, kills the shortcut | bins 17-19 have only 19/62/51 samples each → drawn 100+ times per epoch → memorization; bin 0's 32k samples drawn 0.15×/epoch — **excluded from HPO** |
| **C** | `test_like_x_gender` (sample occ ∝ `_TEST_PMF_0025`, gender 50/50 within each bin) | `w_imp = 1` (sampler does the shift) | conceptually cleanest, batch ≈ test distribution, no double correction | mid-occ samples (bins 5-13) drawn 2-3× per epoch → memorization; bins 17-19 jamais drawn → 132 samples discarded — **excluded from HPO** |
| **D** (loss-only) | `none` (natural train distribution, each sample 1× per epoch) | `w_imp = p_test / p_train`  ×  `w_gender = [1/(2·p_F), 1/(2·p_M)]` per sample | strict 100 % data coverage (no replacement), single-pass per epoch ⇒ no memorization, sampler-loss decoupling is clean, easy to debug | gender balance enforced only on average (across epochs), not per-batch — `λ_fairness` may converge slower since `Err_F` / `Err_M` estimates per step have higher variance |
| **E** (cell-reweight) | `none` | per-cell `w_cell(g, b) = 1/sqrt(count(g, b))` normalized over the **2 × 20 = 40 cells** | hard decorrelation of (gender × occ) **dans le gradient** (chaque case `(g,b)` contribue identiquement au loss), **sans** sur-tirer des cellules rares physiquement | bins ultra-rares (count<20) ont des poids ~5× la moyenne → bruit sur la loss au début ; pas de `w_imp` sur l'axe occ → on n'aligne pas explicitement sur la distribution test |
| **F** (mix inverse de A) | `occlusion` (quantile, 10 buckets équi-effectifs) | `w_gender` only | sampler couvre l'axe occ via quantiles équilibrés (chaque sample vu ~1× par epoch, pas de cellules micro), gender ré-équilibré par loss seul | bucketing par **quantiles** ≠ bucketing par **GT-bins** → ne matche pas la distribution test sur l'axe occ aussi finement que `w_imp` ; gender balance moins fort que sampler `gender` |

### What we ship

**Default in active yamls : Option A** (`sampler_strategy: gender` + `loss_importance_reweight: true`). Optuna teste les **4 stratégies viables `{A, D, E, F}`** sur chaque petit modèle (vitb16, sapiens2-01b) via le categorical `balancing_strategy` du `optuna.search_space`.

Mapping géré dans [optimize.py:_BALANCING_STRATEGY_MAP](src/optimize.py) :

| Strat | `sampler_strategy` | `loss_importance_reweight` | `loss_gender_reweight` | `loss_cell_reweight` |
|---|---|---|---|---|
| A | `gender` | true | false | false |
| D | `none` | true | true | false |
| E | `none` | false | false | true |
| F | `occlusion` | false | true | false |

C deviendra le bon choix sur un **dataset plus grand** (full 752k Idemia) — là, chaque bin a 5k+ samples et le problème de multiplicité disparaît. Tracé comme levier #14b. B a un worst-of-both-worlds profil sur notre subset et est laissé out.

> Known limitation : if you manually set `sampler_strategy: gender_x_occ` outside of HPO, `build_importance_weights` still computes `p_test/p_train` which is incorrect for that sampler. The B-variant formula (`n_buckets · p_test`) is not auto-selected.

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
- When `model.init_backbone_from` is set: `init_backbone_from`, `init_backbone_pretrain_run_id`, `init_backbone_missing_keys`, `init_backbone_unexpected_keys`, plus every `pretrain_*` param copied from the source pretrain run (tag `pretrain_run_id` for the link)

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
├── dinov3-vits16-face-occ.yaml         ViT-S/16  — runs anywhere (weights shipped)
├── dinov3-vitl16-p100.yaml             ViT-L/16  — 2× P100  (requires manual weight download)
├── dinov3-vitl16-3090.yaml             ViT-L/16  — 2× 3090  (requires manual weight download)
├── dinov3-vith16plus-3090.yaml         ViT-H+/16 — 2× 3090  (no pretrain — baseline)
└── dinov3-vith16plus-3090-ibot.yaml    ViT-H+/16 — 2× 3090  (init_backbone_from iBOT pretrain)

scripts/
├── pretrain_ibot_vith16plus_2x3090.sh                       # SLURM iBOT pretrain (30h)
├── optimize_dinov3_vith16plus_2x3090.sh                     # SLURM Optuna HPO, no pretrain
├── optimize_dinov3_vith16plus_from_pretrain_2x3090.sh       # SLURM Optuna HPO from iBOT encoder
├── chain_pretrain.sh                                        # chain N successive sbatch runs
├── download_dinov3_weights.sh                               # URL reference for larger DINOv3 weights
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
| 10 | Two pyproject files (`pyproject.toml` + `pyproject.cluster.toml`) with `cp` swap in every sbatch | ✅ single `pyproject.toml` with `marker = "sys_platform == 'linux'"` CUDA index |
| 11 | MPS autograd `.view()` failure on DINOv3 forced torch `<2.6` and a stack of MPS monkey-patches | ✅ bumped to `torch>=2.7`, removed all MPS shims |
| 12 | Pretrain → finetune wiring opaque (no MLflow trace of which encoder seeded the run) | ✅ `train.py` logs `model_init_backbone_from`, `init_backbone_pretrain_run_id`, all `pretrain_*` params from the source run, and a `pretrain_run_id` tag |

---

## Roadmap — next steps

### Implemented and runnable now

| # | Lever | Where | Status |
|---|---|---|---|
| 1 | **HPO on ViT-H+/16 baseline** | `scripts/optimize_dinov3_vith16plus_2x3090.sh` + `dinov3-vith16plus-3090.yaml` | ready |
| 2 | **iBOT-light pretrain** on MS1MV3 | `scripts/pretrain_ibot_vith16plus_2x3090.sh` + `chain_pretrain.sh` | ready |
| 3 | **HPO from iBOT pretrain** | `scripts/optimize_dinov3_vith16plus_from_pretrain_2x3090.sh` + `dinov3-vith16plus-3090-ibot.yaml` | ready (fill `init_backbone_from` after pretrain) |
| 4 | **Sapiens2-0.8B baseline** (1B human pretrain) | `scripts/optimize_sapiens2_08b_2x3090.sh` + `sapiens2-08b-3090.yaml` | ready |
| 5 | **Importance reweighting** `w(GT) = p_test/p_train` 20 bins | `src/utils/losses.py:build_importance_weights` + yaml `loss_importance_reweight` | ready, on by default for `*-ibot` and sapiens2 yamls |
| 6 | **Gender-balanced sampler** `gender_x_occ` | `src/utils/losses.py:make_sampler_keys` | always on |
| 7 | **Fairness penalty** `λ·|Err_F − Err_M|` | `src/utils/losses.py:WeightedMSELoss` | yaml `loss_fairness_lambda: 1.0` |
| 8 | **Group-DRO** worst-group min | `src/utils/losses.py:GroupDROLoss` | yaml `loss_type: group_dro` |
| 9 | **Post-hoc per-gender bias correction** | `src/inference/calibration.py` | `predict.py --delta-f --delta-m` |
| 10 | **Worst-K validation analysis** | `src/evaluate_model.py:_save_worst_k` | run after training |
| 11 | **EMA + LLRD + grad-ckpt + bf16/fp16** | `src/train.py` | yaml toggles |

### Non-implemented levers — tracked for future work

| # | Lever | Estimated effort | Estimated ROI | Notes |
|---|---|---|---|---|
| 12 | **Val test-like resampling** — sample val from buckets according to `_TEST_PMF_0025` so `eval_score` becomes an honest proxy of test score | ~30 LOC in `data/dataset.py` | metric honesty + ~0.5-1% via better HPO selection | highest priority — currently `eval_score` underestimates the test gap |
| 13 | **PMF calibration via leaderboard** — after 1 submission, fit `_TEST_PMF_*` so that `expected_score(val) ≈ leaderboard_score` | ~20 LOC + 1 submission | refines #5, ~0.3% | requires a baseline submission first |
| 14 | **MAFA + RandomErasing with label recompute** — synthetic occluders pasted on faces, label increment proportional to area covered | ~150 LOC in `transforms.py` + helper script | ~1-2% on score | covers regime 1 (physical occlusion) gap |
| 15 | **Quality-degradation augmentations** — blur, JPEG artifacts, noise, pixelation with proportional label increment | ~100 LOC in `transforms.py` | ~0.5-1% on score | covers regime 2 (information degradation) of the FaceOcclusion label |
| 16 | **DINOv3 ViT-7B + LoRA** as ensemble member | ~100 LOC loader + yaml + sbatch | ensemble diversity, 0.5-1% | full FT impossible on 2×3090, LoRA only |
| 17 | **Sapiens2-5B + LoRA** as ensemble member | similar to #16 | similar | same constraint |
| 18 | **Multi-arch ensemble** add DINOv2-base + ConvNeXt v2-large (HF, no license) | ~2 yamls + sbatch | ensemble diversity, ~1% | low effort if compute available |
| 19 | **Resolution upgrade 384×384** (or Sapiens2 native 1024×768) | ~50 LOC `get_image_processor` + yaml batch downsize | ~0.5-1.5% on score | 3× compute cost |
| 20 | **SAM3-based pseudo-labeling** on CelebA/VGGFace2 for regime-1 occlusion auto-labels → `extra_train_csv` | ~200 LOC offline pipeline | uncertain, only addresses regime 1 | high effort, not yet justified |
| 21 | **Pseudo-labeling** on `test_students.csv` using ensemble high-confidence predictions, retrain | ~80 LOC | ~0.5% if disparity is low | risk of self-confirming bias |
| 22 | **Test-time augmentation beyond hflip** — light crop/scale TTA, ensembled | ~50 LOC in `inference/tta.py` | ~0.2-0.5% | careful with ratio regression invariance |
| 23 | **More tests** — DDP smoke test, EMA correctness, sampler distribution checks | ~200 LOC | maintenance | nice-to-have |
| 24 | **MIL B1 — pure per-patch head**, drop the pooled head entirely. Inference = `mean(sigmoid(patch_logits))` only | ~30 LOC (also touch `predict.py`) | possible if pooled head adds noise vs locality signal | risky : loses CLS + projection context |
| 25 | **MIL B2 — mix at inference** `(pooled_pred + patch_pred) / 2` | ~20 LOC in `predict.py` | cheap ensemble of the two heads | only worthwhile if both heads converge to comparable performance individually |
| 26 | **MIL B4 — learnable mix scalar `β`** at training time, inference uses `β·pooled + (1−β)·patch_pred` | ~40 LOC | clean version of B2, lets the model decide weight | requires extra trainable scalar + careful init |
| 27 | **MIL with face mask** — exclude background patches via SAM3-derived face mask, mean over face patches only | ~80 LOC + offline mask gen | corrects the background-dilution issue in `patch_pred` | depends on SAM3 pseudo-labels (lever #20) |
| 28 | **Per-patch 2-channel decomposition** — per patch: `(occ_p, valid_p) = (sigmoid(head_occ), sigmoid(head_valid))`. Global ratio = `Σ occ_p · valid_p / Σ valid_p`. Adds sparsity regularizer `λ_sparsity · max(0, mean(valid_p) − 0.85)` to force ~15 % background suppression without external face mask. Identifiable by construction (vs the 3-class permutation idea which suffers from inter-image inconsistency). | ~80 LOC in `face_occ_classifier.py` + `losses.py` | force la localisation valid/invalid sans dépendance externe, capture l'intuition "patches utiles seulement" | encore underdetermined sans face mask (modèle peut tricher valid=1 partout) — la régul sparsity est un workaround soft |
| 29 | **Rich pooling head** — remplace le mean/cls/attention single-query par une tête plus expressive consommant **tous** les patch embeddings, pas juste un résumé. Options : (a) **multi-head attention pooling** avec K=8 queries learnables (Perceiver-style), (b) **conv head** qui reshape `(B, N, D) → (B, D, H, W)` puis Conv2D, (c) **mini-transformer head** 1-2 layers au-dessus. Bénéfice : moins de perte info → patch coarseness moins problématique car le réseau exploite la richesse de chaque embedding. | ~20-50 LOC dans `face_occ_classifier.py` | gain attendu si le HPO actuel montre `pooling: attention` > `mean` ; bonne réponse au problème "patch trop grossier" en gardant toute l'info par patch | ajoute des params (10-100k) ; redondant avec le ViT self-attention pour les cas où pooled simple suffit |

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

All SLURM scripts use `torchrun --nproc_per_node=2`, 30 h time, with the right hardware tags. On Linux nodes `uv sync` pulls CUDA 12.6 wheels via the `[tool.uv.sources]` marker; on Mac it falls back to the default PyPI index (CPU/MPS). torch is pinned to `>=2.7,<3.0` to keep MPS autograd working for local sanity tests.
