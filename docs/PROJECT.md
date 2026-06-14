# face-occ-detector — design & rationale (v18)

Document unique consolidant le design système, les choix méthodologiques et leur
justification. Pour quickstart et UI, voir [README.md](../README.md). Pour l'historique
des versions, voir `git log`.

---

## Table of contents

1. [Contexte challenge & métrique](#1-contexte-challenge--métrique)
2. [Hypothèse H_C — pivot du framework](#2-hypothèse-h_c--pivot-du-framework)
3. [Données & splits](#3-données--splits)
4. [Architecture — backbones & pooling](#4-architecture--backbones--pooling)
5. [Mécanismes de fairness](#5-mécanismes-de-fairness)
6. [Lagrangien adaptatif (v18 log-ratio)](#6-lagrangien-adaptatif-v18-log-ratio)
7. [Calibration per-gender](#7-calibration-per-gender)
8. [Classifieur de genre test](#8-classifieur-de-genre-test)
9. [Pretrain iBOT](#9-pretrain-ibot)
10. [Infrastructure HPO & DDP](#10-infrastructure-hpo--ddp)
11. [Roadmap v18+](#11-roadmap-v18)
12. [Références](#12-références)

---

## 1. Contexte challenge & métrique

**Tâche** : régression d'`FaceOcclusion ∈ [0, 1]` (proportion de la surface du visage
occluée) sur images 224×224 RGB pré-croppées. Le challenge fournit `gender` sur train
mais PAS sur test (à déduire).

**Métrique officielle** :

```
Err_g  = Σ_{i ∈ g} w_i · (ŷ_i − y_i)²  /  Σ_{i ∈ g} w_i           pour g ∈ {F, M}

avec   w_i = 1/30 + y_i      (pondération qui pénalise plus les fortes occlusions)

Score  = (Err_F + Err_M) / 2  +  |Err_F − Err_M|
```

**Lecture** : la métrique est un Lagrangien explicite à `λ = 1` sur le gap fairness.
La moyenne des erreurs per-gender pèse autant que le gap. Optimiser uniquement la MSE
globale = ignorer ce term. Optimiser uniquement le gap = trivial (zéro tout).
**Le sweet spot est l'équilibre des deux.**

**Pourquoi ce poids `1/30 + y`** : la queue haute occlusion (y > 0.3, ~2% des
samples train) compte plus par sample que les samples y≈0. Sans ce poids, le modèle
ignorerait la queue et tankerait une fois sur P_test (où la queue est ~6%).

---

## 2. Hypothèse H_C — pivot du framework

### 2.1 Énoncé

**H_C : `P_test(G | Y) = P_train(G | Y)`**

Le conditionnel du genre sachant l'occlusion est invariant train→test. La marginale
de Y diffère (c'est le shift qu'on traite), mais le rapport F/M à occlusion fixée
est le même.

### 2.2 Validation empirique

Test indirect via marginale : si H_C tient, alors

```
P_test(F)  =  Σ_y P_train(F | y) · P_test(y)
```

Mesuré sur train + P_test(Y) extraite du PDF :

```
P_test(F) attendu (H_C) ≈ 0.4816
P_test(F) observé (MID lookup, 93.5% coverage) ≈ 0.4879
Delta ≈ 0.5 pp                                          → H_C plausible
```

C'est un test à l'ordre 1 (marginale). La validation à l'ordre 2 (conditionnel exact
par bin Y) nécessiterait Y labels sur test, impossibles à obtenir sans submission.

### 2.3 Conséquences (toute la base théorique)

Sous H_C, par factorisation `P(G, Y) = P(G|Y) · P(Y)` :

```
P_test_joint(G, Y)  =  P_train(G | Y) · P_test(Y)
P_test(Y | G)       =  P_train(G | Y) · P_test(Y) / P_test(G)
```

Donc on peut **dériver toutes les distributions test sans label Y test**, à partir
de :
- `P_train(G | Y)` (mesuré sur train.csv)
- `P_test(Y)` (extrait du PDF, 25 bins en v18)

C'est ce que fait [`src/utils/distribution.py:115`](../src/utils/distribution.py#L115).

### 2.4 IS reweight sous H_C

Le ratio per-sample pour shift train → cible :

```
ratio(g, y)  =  P_target(g, y) / P_train(g, y)
             =  P_train(g|y) · mix_y(α)[y]  /  [P_train(g|y) · P_train(y)]
             =  mix_y(α)[y] / P_train(y)
```

avec `mix_y(α) = (1-α)·P_train(y) + α·P_test(y)`.

**Le facteur `P_train(g|y)` s'annule** : le ratio ne dépend que de Y. C'est ce qui
est implémenté dans [`compute_balancing_weights()`](../src/utils/distribution.py#L131).

À α=1, on shift implicitement `P_train(Y|G) → P_test(Y|G)` POUR CHAQUE GENRE, sans
mécanisme séparé. C'est gratuit grâce à H_C.

---

## 3. Données & splits

### 3.1 Source

- `data/raw/train.csv` : ~96k samples avec colonnes `filename`, `FaceOcclusion`, `gender`
- `data/raw/test_students.csv` : ~30k samples avec uniquement `filename` (gender à déduire)
- Images : `data/raw/database{1,2,3}/.../m.XXXXX/frame_align.webp` (MID = Freebase identifier)

### 3.2 Tri-split v12+

```
Train iid           : 80%  (~80k)   apprentissage
Val iid             : 15%  (~15k)   eval Optuna (stratified IS metric)
Test holdout iid    :  5%  (~5000)  vérification post-hoc, jamais vu par HPO
```

Tous les splits sont iid de `P_train` (pas de stratification par y/g, on garde la
distribution naturelle). La correction vers `P_test` se fait via IS reweight au
moment de l'évaluation.

### 3.3 Distributions empiriques

| Statistique | Train | Test (estimé) |
|---|---|---|
| P(F) | 0.324 | 0.488 |
| P(M) | 0.676 | 0.512 |
| Mass y < 0.05 | ~58% | ~13% |
| Mass y > 0.3 | ~1.6% | ~5.9% |
| Mean y | ~0.078 | ~0.165 |

**Le shift Y est sévère** : train piqué sur faibles occlusions, test beaucoup plus
uniforme jusqu'à y=0.35. Sans IS reweight, le modèle apprend à prédire ~0 par
défaut → catastrophe sur test.

### 3.4 P_test(Y) — 25 bins (v18)

Extraite pixel-par-pixel du PDF page 3 via
[`scripts/extract_test_pmf_from_pdf.py`](../scripts/extract_test_pmf_from_pdf.py) :
- Render PDF page 3 à 300 DPI
- Détection axis frame via lignes noires + tick marks
- Mask RGB bleu matplotlib (142, 186, 217) ± tolérance
- Mean bar height par target bin
- Output : 25 bins × 0.02 couvrant [0, 0.5]

Hardcodée dans [`src/utils/distribution.py:34`](../src/utils/distribution.py#L34).

### 3.5 Pourquoi 25 bins ?

Trade-off resolution vs variance :
- 15 bins (v12-v17) : grossier, lisse les bumps test à y ∈ {0.05, 0.10, 0.15}
- 20 bins : intermédiaire mais saw-tooth d'alignement source/target
- 25 bins : capture les bumps réels, ratio max 4.47 (vs 3.83 en 15-bin, jamais clippé), std sample weight quasi-identique

---

## 4. Architecture — backbones & pooling

### 4.1 Backbones supportés

| Modèle | Params | Hidden D | Source | Image size |
|---|---|---|---|---|
| DINOv3 ViT-B/16 | 86M | 768 | dinov3_loader (LVD pretrain) | 224 |
| Sapiens 0.1B | 100M | 768 | sapiens2_loader (custom pretrain) | 224 |
| CoAtNet-1 | 41M | 768 | timm `coatnet_1_rw_224.sw_in1k` | 224 |
| CoAtNet-3 (v18) | 167M | 768 | timm `coatnet_3_rw_224.sw_in12k` | 224 |
| EfficientNet-B0 | 5M | 1280 | timm `tf_efficientnet_b0` | 224 |

Tous standardisés sur 224×224 (pas de hack de redimensionnement).

### 4.2 Wrapper unifié `FaceOccRegressor`

[`src/models/face_occ_regressor.py`](../src/models/face_occ_regressor.py)

```
input image (B, 3, 224, 224)
  → backbone forward (avec _forward_backbone qui standardise [CLS, patches])
  → backbone output (B, N, D)         où N = 196 patches + 1 CLS pour ViT
  → pooling (au choix HPO)
  → head Linear(pooled_dim, 1) → sigmoid
  → ŷ ∈ [0, 1]
```

### 4.3 Pooling — 3 options HPO

**A. `attention_k_query`** (default)

K queries learnable, chacune calcule une attention pondérée sur les patches :

```
K = n_focal + n_diffuse + n_free       (max 6 en v18)
queries (K, D)                          learnable
tau (K,)                                learnable softmax temperature per query

q = proj_q(queries)                    (K, D)
k = proj_k(patches)                    (B, N, D)
v = proj_v(patches)                    (B, N, D)

scores = einsum("kd,bnd->bkn", q, k) / (sqrt(D) · tau)
weights = softmax(scores, dim=-1)       (B, K, N)
pooled = einsum("bkn,bnd->bkd", weights, v)   (B, K, D)
flat = pooled.flatten(1)                (B, K·D)
out = LayerNorm + Dropout(flat)
```

- `n_focal` ∈ [1, 2] : queries avec tau bas (attention sharp, focus sur 1-2 patches)
- `n_diffuse` ∈ [0, 2] : queries avec tau haut (attention diffuse, global context)
- `n_free` ∈ [0, 2] : queries libres
- `tau_focal_init`, `tau_diffuse_init`, `tau_free_init` HPO

**B. `mil`** (Multi-Instance Learning)

```
mil_agg = "multi" (pinned) → concat de 4 aggregateurs :
  - mean pool patches
  - max pool patches
  - top-k pool (mil_k_top HPO)
  - gated attention pool
→ (B, 4·D)
→ MLP(4D → mil_hidden → mil_hidden) → flat
```

`mil_hidden` ∈ {64, 128, 256}, `mil_k_top` ∈ [10, 50].

**C. `grid`** (spatial pooling)

```
spatial = patches.reshape(B, D, sqrt(N), sqrt(N))    (B, D, 14, 14) pour ViT
pooled = AdaptiveAvgPool2d(grid_size)(spatial)        (B, D, g, g)
flat = pooled.flatten(1)                              (B, g²·D)
```

`grid_size` ∈ {3, 4, 5, 6, 7}.

### 4.4 Critique pooling actuel & plan v19

**Actuel** : output dim = K·D ou g²·D, head Linear(output_dim, 1). Pour Sapiens 1B
(D=1536), K=6 → output 9216 dim → head 9216 params. Pas catastrophique mais **dense**.

**v19 plan** : ajouter `proj_out_dim` optionnel dans `AttentionKQuery` qui projette
chaque query output vers une dim plus petite avant concat :
- Per-query Linear shared (D × proj_out_dim params, pas K×)
- Pour K=6, proj_out_dim=64 → output 384 dim
- Sample-efficient sur 80k samples
- HPO `pool_proj_out_dim` ∈ {None, 32, 64, 128}

### 4.5 GRL pour DANN

`_GradReverse` autograd custom : forward identité, backward `-grad × alpha`. Couplé
à `GenderDiscriminator` (petit MLP) pour adversarial debiasing au feature level.
Activé seulement si `feature_fairness=dann`.

---

## 5. Mécanismes de fairness

3 mécanismes complémentaires, sélectionnables/combinables via HPO.

### 5.1 IS reweight (per-sample loss weight)

[`compute_balancing_weights()`](../src/utils/distribution.py#L131)

```python
ratio_y = (1 - α) · P_train(y) + α · P_test(y)   /   P_train(y)
ratio_y = clip(ratio_y, 0.1, 10.0)
w_i = ratio_y[bin(y_i)] / mean(w)                # normalize to mean=1
```

`α = correction_strength` HPO [0, 1]. À α=1, full IS correction vers P_test.

- Sample weight max observé (full train) : 4.47 (bin y≈0.45, sous le clip)
- Std sample weight : ~1.0
- Effet automatique sur gender marginal : à α=1, F voit weight × 1.51, M × 0.76
  → marginale `P_target(G)` matche `P_test(G)`

**Math propre, fondé sur H_C, c'est le mécanisme principal.**

### 5.2 Lagrangien adaptatif sur `|err_F − err_M|`

Cf section 6 dédiée.

### 5.3 Feature fairness (OT / DANN)

Activé via `feature_fairness ∈ {none, ot, dann}` HPO :

- **`ot`** : `sliced_wasserstein` ou `sinkhorn_distance` entre features F et M
  d'un batch → ajouté à la loss avec poids `ot_lambda`
- **`dann`** : `GenderDiscriminator` prédit gender depuis features avec GRL
  → backbone apprend des features gender-invariantes. Poids `adv_lambda`

**Critique** : redondant avec IS reweight + Lagrangien en théorie. Utile si IS
sous-corrige (e.g., H_C imparfaite). Le HPO peut choisir `none` → la data décide.

### 5.4 Pourquoi 3 mécanismes orthogonaux

| Mécanisme | Niveau d'action | Corrige |
|---|---|---|
| IS reweight | sample-loss | shift de distribution P_train → P_test sous H_C |
| Lagrangien | loss-aggregate | gap mesuré `\|err_F − err_M\|` (pénalité directe) |
| OT/DANN | feature | marginalisation gender dans l'espace latent |

IS = "respect du shift". Lagrangien = "respect du gap mesuré". OT/DANN = "robustesse
représentation". Différentes garanties mathématiques.

---

## 6. Lagrangien adaptatif (v18 log-ratio)

Le cœur du contrôleur fairness pendant le train.

### 6.1 Loss training

```
L_train = (Err_F + Err_M) / 2  +  λ_adapt · |Err_F − Err_M|
```

avec `λ_adapt` ∈ [λ_min, λ_max] mis à jour 1 fois par epoch depuis le signal val.
**En eval, on utilise toujours `λ_metric = 1.0`** (cohérent avec la métrique du
challenge).

### 6.2 Update rule v18

```python
log_ratio = log(max(val_err_diff, ε) / threshold)
Δλ        = lambda_lr · log_ratio
λ_{t+1}   = clip(λ_t + Δλ, lambda_min, lambda_max)
```

avec :
- `val_err_diff` = `|err_F − err_M|` sur val (15k samples) — signal CLEAN
- `lambda_lr = 0.2` (gain en log-space)
- `threshold = 0.0005` (zone d'équilibre)
- `lambda_min = 1.0` (toujours ≥ poids métrique)
- `lambda_max = 3.0` (marge haute)

### 6.3 Justification du log-ratio

**Problème du linéaire** (v17 et antérieur) :

```
Δλ = lr · (val_err_diff − threshold)
```

Asymétrique structurellement : `(val_err − threshold) ∈ [−threshold, +∞)` borné à
gauche, illimité à droite. Donc :
- Ascent rapide (val_err peut être 10× threshold → grand Δλ positif)
- Descent lent (val_err = 0 → Δλ = −threshold · lr, petit)

Conséquence : λ monte facilement, descend pas → reste pinné au max → modèle pousse
trop la fairness → val_err_diff spike → cycle d'oscillation.

**Solution log-ratio** :

```
log(err / threshold)  ∈  (−∞, +∞)  symétrique sur les RATIOS
```

- `err = 10·threshold` → log_ratio = +2.3
- `err = threshold` → log_ratio = 0 (équilibre)
- `err = threshold/10` → log_ratio = −2.3 (symétrique)

`Δλ = lr · log_ratio` donne donc une réponse symétrique multiplicative.

### 6.4 Saturation naturelle de log()

Pas besoin de tanh-cap : log() croît sublinéairement. Pire cas observé dans la DB
(val_err = 0.011 = 22× threshold) :

```
Δλ_max  =  0.2 · log(0.011 / 0.0005)  =  0.2 · 3.09  ≈  +0.62
```

Acceptable comme one-shot. Pour comparaison, l'ancien linéaire avec lr=200 :

```
Δλ_old  =  200 · (0.011 − 0.0005)  =  +2.1     (saturé à cap λ_max)
```

C'est précisément le bug qu'on a éliminé.

### 6.5 Sample weight justifié pour `lambda_min=1.0`

Le challenge metric utilise λ=1. Si en train on permet λ < 1, on optimise un
objectif MOINS fairness-pushing que la métrique elle-même → contraire à l'objectif.
`lambda_min=1.0` garantit qu'on pousse au moins autant que la métrique en
permanence, et plus si fairness en régression.

### 6.6 Plumbing

- [`src/utils/losses.py:WeightedMSELoss.update_lambda()`](../src/utils/losses.py#L73)
- [`src/train.py:LambdaLogCallback.on_evaluate()`](../src/train.py#L132) appelle
  update_lambda 1 fois par epoch après val eval
- `loss_lambda_*` HPO dans yamls : `init`, `lr`, `max`, `min`, `threshold`

---

## 7. Calibration per-gender

### 7.1 Problème

Même après IS reweight (= bonne distribution train) et Lagrangien (= bon gap), le
modèle peut avoir un **biais résiduel `f(x) − E[Y|x]` qui dépend du genre à Y fixé**.

Pourquoi :
- `P(image | Y=y, G=F) ≠ P(image | Y=y, G=M)` (visages F vs M visuellement
  distincts même à occlusion équivalente)
- Le modèle approche `f` localement dans l'espace image
- Sa qualité d'approximation diffère entre les régions F et M à Y fixé
- Résultat : `bias(y, g) := E[f(image) | Y=y, G=g] − y` peut être ≠ entre F et M

C'est **orthogonal à H_C** : pure pathologie modèle.

### 7.2 Solution : 3 calibrateurs per-gender

[`src/inference/calibrators.py`](../src/inference/calibrators.py)

Chaque calibrateur fit un sous-modèle séparé pour F et M sur val (IS-weighted) :

- **isotonic** : Pool Adjacent Violators (PAV), monotone non-paramétrique
- **linear** : 2 params per gender (a, b), monotone si a > 0, closed-form WLS
- **pchip** : monotone cubic spline, plus smooth qu'isotonic

`cal.transform(preds, gender)` applique le bon sous-modèle par sample.

### 7.3 Selection via alpha blend

Pour chaque trial :

```
val_preds_cal = α · cal(preds, gender)  +  (1 − α) · preds
```

`α ∈ {0, 0.1, ..., 1.5}` (16 valeurs). 3 calibrateurs × 16 α = 48 combinaisons
évaluées sur val IS-stratified. Best `(cal_name, α)` loggé en MLflow param.

### 7.4 Application au submit — LE TROU MAJEUR

**Avant v18** : `src/predict.py` n'applique **pas** les calibrateurs. Seul
`bias_correction` (delta scalaires per-gender, simpliste) ou `quantile_match_to_test_pmf`
(global) sont disponibles. Les 6 sous-modèles entraînés sont jetés.

**v18 plan** (en cours) :
1. `train.py` pickle `cals` dict en artifact MLflow (dossier `calibrators/`)
2. `predict.py` :
   - Charge calibrators du best trial via MLflow run_id
   - Charge `data/raw/test_students_with_gender.csv` (cf §8)
   - Applique `cal.transform(preds, gender_predicted)` avec blend `(cal_name, α)`
     loggé

ROI estimé : **+1 à +2 pt** sur le score test. C'est gratuit, pure plumbing.

---

## 8. Classifieur de genre test

### 8.1 Pourquoi nécessaire

Le challenge test_students.csv ne fournit que `filename`. Pour appliquer la
calibration per-gender, il faut deviner le genre.

### 8.2 Pipeline hybride

[`src/inference/gender_classifier.py`](../src/inference/gender_classifier.py)

```
Stage 1 — MID lookup       (93.5% coverage)
  filename contient un Freebase MID (m.XXXXX = identité personne)
  Si MID est dans train.csv → genre 100% déterministe

Stage 2 — Sapiens probe    (6.5% restants)
  Sapiens 0.1b backbone → CLS feature (768d)
  LogisticRegression entraînée sur FULL train (100k samples, v18)
  5-fold CV + grid C ∈ {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000}
  Refit sur 100% avec best C → production model
```

### 8.3 Validation

Le code (v18) ajoute `validate_against_mid_lookup()` :
- Run Sapiens probe sur les ~28k samples test MID-connus
- Compare avec ground truth MID (= 100% par construction)
- Donne **accuracy réelle en production** (vs CV qui mesure sur train)

Cette validation cross-distribution est plus honnête que CV sur train.

### 8.4 CSV séparé `test_students_with_gender.csv`

[`scripts/build_test_gender_csv.py`](../scripts/build_test_gender_csv.py) génère :

```
filename, gender_predicted (0=F, 1=M), gender_source, gender_proba_M
```

- `gender_source ∈ {"mid_lookup", "sapiens_probe"}`
- `gender_proba_M` NaN pour mid_lookup, prob LogReg pour sapiens_probe
- Auditable, stable, découplé de la pipeline d'inférence

### 8.5 Validation marginale H_C

Le script affiche :

```
P_test(F) observé   = depuis prédictions
P_test(F) expected  = Σ_y P_train(F|y) · P_test(y)  [H_C]
Delta               = observé − expected
```

Delta < 1pp = H_C strongly supported. > 7pp = H_C suspect.

---

## 9. Pretrain iBOT

### 9.1 Motivation

Backbones LVD/Sapiens pretrained sur ImageNet/jeu général. Petit drift vers le
domaine "faces croppées 224" peut aider. iBOT = self-supervised distillation avec
masked patches → encoder learn robust patch features.

### 9.2 Setup `src/pretrain_ibot.py`

```
Teacher = backbone EMA (decay HPO 0.999-0.9995) OU frozen
Student = backbone with masked patch input
Loss = student_features · teacher_features (matching dim per masked patch)
Teacher EMA update : θ_t ← d · θ_t + (1-d) · θ_s
```

~50k-100k steps suffisent (cf roadmap : 100-150k pour mieux ressortir vs LVD baseline).

### 9.3 Intégration HPO

`pretrained_source` est un choix catégoriel dans le search_space :

```yaml
pretrained_source:
  type: categorical
  choices:
    - "lvd"                                                  # baseline
    - "ibot:runs:/c1e1ba1a747747688e3691daa4f7a31a/encoder"  # custom iBOT
```

Le modèle init depuis MLflow run_id, partial state_dict load (strict=False).

---

## 10. Infrastructure HPO & DDP

### 10.1 Optuna setup

[`src/optimize.py`](../src/optimize.py)

```
study_name = "optuna-{architecture}"
storage    = sqlite:///optuna.db
sampler    = TPE (default)
pruner     = HyperbandPruner(min_resource=2, max_resource=13, reduction_factor=3)
n_trials   = 100 par yaml
keep_top_n = 3
```

HyperbandPruner (v18, remplace MedianPruner) : alloue exponentiellement plus de
budget (epochs) aux trials qui survivent. Plus efficient sur HPO long.

### 10.2 DDP pattern (v14 restored in v17)

```
Rank 0 :                              Rank 1 :
  setup_distributed                     setup_distributed
  barrier ─────────────────── sync ─────── barrier
  broadcast study, run_id              broadcast (receives)

  study.optimize(objective):           while True:
    while there are trials :              data = broadcast(None)     [SYNC start of trial]
      objective(trial):                   if data is None: break
        trial_data = ...                  try:
        broadcast(trial_data) ─── sync ─── data = trial_data
        train(arch, ...)                   train(arch, ...)
        finally:                          finally:
          barrier ─────────── sync ────── barrier
                                          [defensive try/except v18]

  broadcast(None) ────── sync ────── data = None → break
```

**Pré-requis pour stabilité** :
- `OptunaPruningCallback` broadcast la décision prune au lieu de raise seulement
  rank 0 (sinon rank 1 hang sur backward, NCCL timeout 10 min)
- Try/except autour des `barrier()` finally (sinon trial cassé bloque tout)
- fsync + atomic rename du yaml trial (évite race NFS quand rank 1 read)

### 10.3 MLflow tracking

- Backend : `sqlite:///mlflow.db`
- Artifact store : `./mlruns/` (relatif au CWD)
- Run hiérarchique : parent run = study, child run per trial
- Best trial : top-N preserved via `_PruneRegistryToTopN` callback (post-each-trial)

`mlflow.pytorch.log_model(..., pip_requirements=[])` (v18) : skip `uv export`,
gain ~5-30s par trial.

### 10.4 Scripts launch

```
scripts/optimize_{arch}_v18_2x3090.sh
  SLURM 2× RTX 3090, 8 cpus, 100G mem, 30h walltime
  uv sync env, torchrun --nproc_per_node=2 src/optimize.py
```

### 10.5 Diagnostic v18

- Garde 0-params trainable model AVANT DDP init (fail-fast clair)
- `barrier()` wait time printé en fin de trial (>5s = rank 1 traîne ; ~10min = NCCL
  timeout)
- Rank-aware exception printing (au lieu de `except: pass` silencieux)

---

## 11. Roadmap v18+

### 11.1 Ce qui reste à faire en v18

1. **Pickle calibrateurs en artifact** (train.py) + **load+apply per-gender dans
   predict.py** (utilise `test_students_with_gender.csv`). ROI : +1-2 pt.
2. **Build gender CSV** : lancer `python scripts/build_test_gender_csv.py
   --force-refit` sur cluster pour valider classifier + générer le CSV.
3. **Update `_TEST_P_GENDER`** dans distribution.py avec valeur observée.

### 11.2 v19 candidates

1. **Per-query bottleneck projection** dans AttentionKQuery (cf §4.4) — permet
   d'utiliser gros Sapiens (1B+) sans head explosé.
2. **Ensembling final top-N trials** — avec `rotate_val_seed=false`, moyenner les
   preds des top-3 sur val commune + recalibrer (isotonic) sur ensemble.
3. **SWA (Stochastic Weight Averaging)** — alternative à EMA poids modèle (retiré
   v17). Moyenne des poids sur derniers epochs. Gain typique 0.3-0.8 pt, risque
   minimal (decoupled de DDP).
4. **Mixup/Cutmix** — `augmentation_level=none` actuellement. Pourrait aider sur
   80k samples.
5. **MAFA + augmentation-with-label-recompute** (RandomErasing avec proportional
   label increment, blur/JPEG quality degradation) — estimation +1.5-2.5pt
   (souvent plus impactant que tout le sweep HPO réuni). Pas démarré.
6. **Submission baseline + retro-fit `_TEST_PMF`** pour calibrer sur le score
   leaderboard.
7. **Résolution 384×384** sur gros modèles (Sapiens natif 1024×768) — estimation
   +0.5-1.5pt, ~3× coût compute.

### 11.3 Anti-patterns identifiés (à pas refaire)

- **EMA λ per-batch** : signal noisy + biaisé positivement → λ pinné au cap. Retiré
  v17 pour update per-epoch sur val signal.
- **Linéaire `(err − thr)` asymmetric** : impossible de descendre λ vite. Remplacé
  log-ratio v18.
- **`OptunaPruningCallback` raise rank 0 only** : rank 1 NCCL hang sur prochain
  backward. Corrigé v17 par broadcast décision.
- **Calibrators per-gender entraînés mais pas appliqués au submit** : on a fait le
  hard work pour rien pendant des semaines. Plumbing en cours v18.

---

## 12. Références

### Code clés

- [`src/utils/distribution.py`](../src/utils/distribution.py) : P_test, IS weights,
  test_pmf_joint, empirical PMFs
- [`src/utils/losses.py`](../src/utils/losses.py) : WeightedMSELoss + Lagrangien log-ratio
- [`src/utils/metrics.py`](../src/utils/metrics.py) : compute_score, compute_score_stratified_is
- [`src/models/face_occ_regressor.py`](../src/models/face_occ_regressor.py) : wrapper
  unifié, poolings, GRL
- [`src/inference/calibrators.py`](../src/inference/calibrators.py) : isotonic, linear,
  pchip per-gender
- [`src/inference/gender_classifier.py`](../src/inference/gender_classifier.py) : MID
  lookup + Sapiens probe
- [`src/train.py`](../src/train.py) : training loop, WeightedMSETrainer, callbacks
- [`src/optimize.py`](../src/optimize.py) : Optuna study + DDP coordination

### Scripts

- [`scripts/extract_test_pmf_from_pdf.py`](../scripts/extract_test_pmf_from_pdf.py) :
  re-extraction P_test depuis PDF (25 bins v18)
- [`scripts/build_test_gender_csv.py`](../scripts/build_test_gender_csv.py) :
  génère test_students_with_gender.csv + valide H_C
- [`scripts/estimate_test_gender.py`](../scripts/estimate_test_gender.py) : stats
  MID seules (validation H1 marginal)
- [`scripts/qualitative_viewer.py`](../scripts/qualitative_viewer.py) : UI Streamlit
- [`scripts/optimize_*_v18_2x3090.sh`](../scripts/) : launchers SLURM

### Configs

- [`configs/architectures/dinov3-vitb16-3090-v18.yaml`](../configs/architectures/dinov3-vitb16-3090-v18.yaml)
- [`configs/architectures/sapiens2-01b-3090-v18.yaml`](../configs/architectures/sapiens2-01b-3090-v18.yaml)
- [`configs/architectures/coatnet1-3090-v18.yaml`](../configs/architectures/coatnet1-3090-v18.yaml)
- [`configs/architectures/coatnet3-3090-v18.yaml`](../configs/architectures/coatnet3-3090-v18.yaml)
- [`configs/architectures/efficientnet-b0-3090-v18.yaml`](../configs/architectures/efficientnet-b0-3090-v18.yaml)

### Datasets

- Train + val + test holdout (tri-split iid de train.csv)
- Test challenge (test_students.csv, 30k samples, gender à déduire)
- Pretrain corpora pour iBOT custom (MLflow run logged)

### Hypothèse pivot

H_C : `P_test(G | Y) = P_train(G | Y)`. Validée empiriquement à 0.5pp près sur
marginale P_test(F). Base de tout le framework de correction de distribution.
