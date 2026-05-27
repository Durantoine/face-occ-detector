# Données — train, val, sources externes, label `FaceOcclusion`

## Train / val / test (Idemia)

| CSV | Rows | Description |
|---|---|---|
| `data/raw/train.csv` | 100,000 | train set (avec `gender`, `FaceOcclusion`) |
| `data/raw/test_students.csv` | 29,980 | test (n'a que `filename` — pas de GT, pas de gender) |

**Schéma `train.csv`** : `filename`, `FaceOcclusion ∈ [0, 1]`, `gender ∈ {0, 1}` (0 = F, 1 = M).

**Imbalance gender** : M/F = 2.086 (67.6 % M, 32.4 % F).

**Imbalance Y** : 32.5 % in `[0, 0.025)`, decaying to <0.1 % above `[0.45, 0.50)`.

**Confound** : `E[Y | F] ≈ 0.129`, `E[Y | M] ≈ 0.061` → women's faces have 2× more occlusion on average. Voir [fairness.md](fairness.md) §"Sampler × loss-weight design".

---

## Comprendre `FaceOcclusion` — two regimes in one label

Visual inspection of the Idemia dataset (samples copied to `inspection/`) reveals that `FaceOcclusion` is **not** just "object covering the face". It mixes two regimes under the same continuous label :

### Regime 1 — Physical occlusion (the obvious one)
- Sunglasses, hats, scarves, masks
- Hands in front of face
- Hair covering forehead/eyes
- Real-world objects in front of the face

### Regime 2 — Information degradation (the hidden one)

Pixels of the face are physically present but the face information is unrecoverable :
- **Noise / pixelation** (old scans, TV captures with rainbow stripes, JPEG artifacts)
- **Stylisation** (pop-art, sketch, engraving, cartoonification)
- **Blur** (motion, defocus, low-res)
- **Synthetic overlays** added by Idemia (e.g. colored paint lines on top of celebrity portraits)
- **Heavy degradation** (faded photos, low-light, compression)

Examples at high `FaceOcclusion` values (≥ 0.6) include heavily stylized Obama-style pop-art portraits, engravings, and severely degraded TV captures — none of these have a physical occluder, but Idemia labels them as ~60-100% occluded because **the face is unrecoverable for recognition**.

### Why Idemia mixes both regimes

From an industrial **face recognition** perspective (Idemia's core business), the two regimes are equivalent: whether your photo has sunglasses OR is too blurry/stylized, the face recognition system has the same problem — *the canonical face features are unavailable*. So `FaceOcclusion` is best understood as :

> **The fraction of the face that is unusable for recognition** — caused by physical occlusion, quality degradation, stylisation, or synthetic overlays — measured by an automated face-quality / face-parsing pipeline.

The high-precision decimal values (e.g. `0.024005`, `0.255016`) suggest the labels are computed automatically by such a pipeline, not manually annotated.

### Implications for augmentation strategy

This insight changes which augmentations are valuable. We need **two families** to cover both regimes :

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

This insight **strongly justifies pretraining on MS-Celeb-1M (or VGGFace2 / a similar large face corpus)** :

1. `database3` (~96% of the Idemia training data) is **literally a subset of MS-Celeb-1M** — same Freebase MIDs, same `_align.webp` naming convention.
2. Pretraining on a larger version of MS-Celeb-1M gives the model **more views of similar identities** at the same canonical alignment, leading to better face manifold representation.
3. **iBOT-style domain adaptation is naturally suited** to specializing an already-trained encoder on a new face corpus — voir [pretraining.md](pretraining.md).

Updated dataset priority for `data/pretrain/` :

| # | Dataset | Why |
|---|---|---|
| 1 ⭐ | **MS1MV3 @ 224×224 JPG** (`data/pretrain/pretrain_224/`) | Résolution finetune-matched → utilisé par les gros modèles iBOT |
| 2 | **MS1MV3 WebDataset** (`gaunernst/ms1mv3-wds`, 100 `.tar`, 46 GB) | Toujours utilisé pour les petits modèles à 112×112 |
| 3 | VGGFace2 (3.3M images, 9k identities) | Close to MS-Celeb in spirit |
| 4 | CelebA (200k) | Good supplement, easy to get |
| 5 | WIDER Face (400k) | In-the-wild diversity |
| 6 | LFW (13k) | Too small for serious pretraining alone |

---

## External data (optional)

Two slots for extra data, both supported by the existing infrastructure — just drop files in the right place.

### A) Unlabeled faces → iBOT-light pretraining (`data/pretrain/`)

Any directory of face images (recursive scan) OR a WebDataset of `.tar` shards (auto-detected via [`_build_dataset`](../src/pretrain_ibot.py)). Pointer la source via `FACE_OCC_PRETRAIN_SRC` dans le `.sh`. Deux corpus en place :

- `data/pretrain/datasets--gaunernst--ms1mv3-wds/...` — MS1MV3 en WebDataset `.tar`, utilisé pour les **petits modèles à 112×112** (`vitb16`, `sapiens2_0.1b`)
- `data/pretrain/pretrain_224/` — JPG 224×224, utilisé pour les **gros modèles à 224×224** (`vith16plus`, `sapiens2_0.8b`) pour matcher la résolution de finetune

`FACE_OCC_PRETRAIN_IMG_SIZE` contrôle la résolution effective (default 112). Les scripts gros modèles le settent à 224.

| Dataset | Size | Cost | Note |
|---|---|---|---|
| **LFW** | 13k | open | aligned faces, immediate use |
| **CelebA** | 200k | research form | strong baseline corpus |
| **VGGFace2** | 3.3M | research form | large, ~36 GB |
| **CASIA-WebFace** | 500k | research form | mid-size alternative |
| **WIDER Face** (crops) | 400k | open | in-the-wild diversity |

Helper: `python -m src.data.external.prepare_pretrain_corpus` — symlinks images from multiple sources (LFW + CelebA + VGGFace2 …) into `data/pretrain/` for a single unified corpus.

### B) Labeled occlusion data → augmentation finetuning (`data/extra/*.csv`)

Set `data.extra_train_csv` in your YAML to a CSV with the **same schema as `train.csv`** (`filename, FaceOcclusion, gender`). It gets concatenated with the Idemia data **before** the stratified split.

Mapping rules per dataset (heuristics — refine after a first run) :

| Dataset | `FaceOcclusion` heuristic | `gender` | Helper script |
|---|---|---|---|
| **CelebA** | weighted sum of attributes (Eyeglasses=+0.10, Wearing_Hat=+0.10, Wearing_Necktie=+0.05, …), clipped to [0, 0.5] | `Male` attribute → 0/1 | `prepare_celeba.py` |
| **MAFA** | `area(mask_bbox ∩ face_bbox) / area(face_bbox)` (continuous) | imputed to 0.5 (neutral) | `prepare_mafa.py` |
| **AR Face** | 0.30 (glasses) / 0.50 (scarf) | annotated | TODO |
| **FaceOcc (if available)** | already continuous | varies | TODO |

Run `python -m src.data.external.prepare_celeba` (or `prepare_mafa`) after downloading the raw dataset; it writes `data/extra/<name>.csv` in our schema. Then set :

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
