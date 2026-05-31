# v15 — Théorie complète et état des implémentations

Doc canonique v15. Reprend la base v12 et formalise les corrections introduites en v13/v14/v15. Lire avant tout sweep ou modification de la loss / des calibrators / du sampler.

---

## 1. Notation

| Symbole | Définition |
|---|---|
| `y ∈ [0, 0.5]` | Occlusion faciale (continue, target régression) |
| `g ∈ {F, M}` | Genre (binaire) |
| `θ` | Paramètres du modèle |
| `ℓ(p, y) = (p - y)²` | Loss per-sample (MSE) |
| `w_metric(y) = 1/30 + y` | **Poids officiel de la métrique** (per-sample) |
| `Err_g` | Erreur weighted per-group : `Σ w·(p-y)² / Σ w` sur g |
| `B` | Taille du batch |
| `α1, α2 ∈ [0, 1]` | Powers axe 1 (Y) et axe 2 (G) du target distribution |

### 1.1 Métrique officielle du concours

```
Score = (Err_F + Err_M) / 2 + |Err_F − Err_M|
      = mean_err + |err_diff|
```

Minimiser cette métrique **récompense conjointement** :
- une mean_err basse (modèle précis dans l'absolu)
- un err_diff bas (modèle équitablement précis sur F et M)

---

## 2. Distributions train / test

### 2.1 Faits connus
- `P_train(y)` empirique : forte masse sur y ≈ 0–0.15, queue rare sur y > 0.3
- `P_test(y)` extraite du PDF du challenge (pixel-reading, 15 bins × 0.0333, cf §11.1)
- `P_train(g)` ≈ (0.10, 0.90) (F minoritaire à ~10%)
- `P_test(g)` ≈ (0.4879, 0.5121) (estimée via MID + DINOv3 probe sur test_students.csv, 100% coverage)

### 2.2 Hypothèse `H_C` — covariate shift Y-only

```
H_C :  P_test(g | y) = P_train(g | y)
   ⇒   P_test(g, y) = P_train(g | y) · P_test(y)
   ⇒   marginale P_test(g) automatiquement dérivée du shift Y
```

**Validation empirique** : `Σ_y P_train(F | y) · P_test(y) ≈ 0.482` vs `P_test(F) ≈ 0.488` observé → écart 0.5 pt → H_C plausible.

---

## 3. Tri-split (train iid 80% / val iid 15% / test holdout 5%)

Construit par `split_train_val_test` ([src/utils/distribution.py](../src/utils/distribution.py)) :

1. **test_holdout** : sampled via `sample_indices_matching_test_pmf` → cells (g, b) tirées selon `P_target(g, b) = P_train(g|b) · P_test(b)` (= P_test(g, b) sous H_C). Sert d'**oracle unbiased** post-train.
2. **val** : 15% iid sur les samples restants. Suit P_train. Sert au best-epoch tracking + fit calibrators + scan α.
3. **train** : le reste (80% des samples, iid P_train).

Cf §4.4 pour la métrique IS-stratifiée appliquée sur val.

---

## 4. Les 6 étages d'intervention

### 4.1 Sample-level — `P_target` factorization + sampler/loss split

#### Chiffres réels (validés sur train.csv)

```
n_train_total = 100 000
P_train(F)    = 0.324      P_train(M)    = 0.676
P_test(F)     ≈ 0.488      P_test(M)     ≈ 0.512   (MID + DINOv3 probe sur test_students.csv)
ΔF            = P_test(F) − P_train(F) = 0.164    (= "pleine distance" à corriger)
```

#### Cible factorisée (chain rule)

```
P_target(g, y) = mix_y(α1)[y] · mix_g(α2)[g | y]            (1)
    mix_y(α1)[y]    = (1-α1)·P_train(y) + α1·P_test(y)
    mix_g(α2)[g|y]  = (1-α2)·P_train(g|y) + α2·P_test(g)
```

**Choix de design** : `P_test(g)` marginal (constant sur y) comme cible axe 2, PAS `P_test(g|y)`. Sous H_C `P_test(g|y) = P_train(g|y)`, donc :
- `(α1=1, α2=0)` ⇒ `P_target = P_test(g, y)` sous H_C ✓ (cible exacte)
- `(α1=1, α2=1)` ⇒ `P_target = P_test(y)·P_test(g)` ≠ P_test(g, y) (suppose G⊥Y dans test)

`α2 > 0` est une **perturbation volontaire** vers la parité G (boost minoritaire au-delà de P_test).

Alternative non implémentée : `axis2_target_g = (0.5, 0.5)` uniforme **per-bin** pour forcer la parité conditionnelle. Plus agressif mais plus risque overfit. La fonction `compute_balancing_weights` accepte le paramètre `axis2_target_g`, switch trivial.

#### Split sampler/loss — framework unifié v15

Deux mécanismes peuvent porter la correction P_train → P_target :
- **Sampler** : modifie la composition des batchs (via `WeightedRandomSampler` per-sample weights)
- **Loss reweight** : pondère les samples dans la loss (via `sample_loss_weight`)

Paramétrisation HPO **orthogonale** :

```
axis1_power, axis2_power ∈ [0, 1]    ← intensités totales par axe (combien on corrige)
sampler_participation ∈ [0, 1]       ← split sampler/loss (qui porte la correction)
```

`axis2_power` décide **combien** on corrige (de 0% à 100% de ΔF). `sampler_participation` décide **comment** ce volume est réparti entre sampler et loss. Indépendants.

#### Mécanique mathématique

```
ratio(g, y) = P_target(g, y) / P_train(g, y)         clip [1/10, 10]

P_sampler(g, y) = (1 - sp) · P_train + sp · P_target = lerp(P_train, P_target, sp)

w_sampler_i = P_sampler / P_train = (1-sp) + sp · ratio_i      → WeightedRandomSampler
w_loss_i    = P_target / P_sampler = ratio_i / ((1-sp) + sp · ratio_i)  → sample_loss_weight
```

#### Cas limites

| (axis2_power, sp) | sampler | loss residual |
|---|---|---|
| (0, *) | neutre (w_sampler=1 uniforme) | neutre (w_loss=1) → ERM pur sur P_train |
| (1, 0) | neutre | porte tout : w_loss = ratio (= correction full) |
| (1, 1) | porte tout : w_sampler = ratio | neutre : w_loss = 1 |
| (1, 0.5) | porte moitié | porte moitié résiduelle |

**Estimateur unbiased** : `E_batch[w_loss · ℓ] = E_P_target[ℓ]` ∀ sp ∈ [0, 1].

#### Exemple numérique (axis2_power=0.5, sp=0.5)

```
correction_totale = 0.5 · ΔF = 0.082            ← décidé par axis2_power
sampler_target_F  = 0.324 + 0.5 · 0.082 = 0.365  ← sampler tire à ce ratio
loss_target_F     = 0.324 + 0.082 = 0.406       ← loss vise toujours le total
```

Le sampler emmène les batchs à 36.5% F (mi-chemin), la loss complète le résidu via w_loss.

#### Trade-offs sampler vs loss

| | Sampler haute | Loss haute |
|---|---|---|
| Stabilité gradients fairness | ✓ Mini-batchs équilibrés réduisent la variance | ✗ Variance haute (peu de samples minoritaires par batch) |
| Diversité effective | ✗ Oversamples les minoritaires (chaque F vue ~2× per epoch à sp=1) | ✓ Chaque sample vu 1× per epoch |
| Risque overfit minoritaire | ✓✗ Plus haut sur F | Plus bas |
| Compatibilité MMD/DANN/OT | Active (≥ 2 samples per gender garanti) | Active (random sampling donne ~41 F per batch=128) |

Optuna explore le plan `(axis_powers × sampler_participation)` → trouve le meilleur compromis pour chaque architecture/setup.

#### Implémentation DDP-safe : `DistributedWeightedSampler`

`torch.utils.data.WeightedRandomSampler` n'est PAS DDP-aware → en multi-GPU, chaque rank tirerait indépendamment, dupliquant des samples et en manquant d'autres. v15 utilise `DistributedWeightedSampler` ([src/data/dataset.py](../src/data/dataset.py)) :

1. Tous les ranks génèrent **la MÊME séquence multinomiale globale** (seeded par `seed + epoch`)
2. Chaque rank prend les indices `i::num_replicas` → **shards disjoints**
3. `set_epoch(e)` est appelé par HF Trainer entre epochs → seed avance, draws différents

Garantit que `Σ_ranks (samples vus) = total_size` sans duplication.

### 4.2 Importance Sampling — bases

Objectif d'optimisation :

```
L(θ) = E_{(g, y) ∼ P_target} [ ℓ(model(x; θ), y) ]              (2)
```

On échantillonne via `P_batch`. Estimateur unbiased :

```
L̂_IS = (1/B) Σ_i  w_i · ℓ_i ,    w_i = P_target(g_i, y_i) / P_batch(g_i, y_i)    (3)

Preuve : E_{i∼P_batch}[w_i · ℓ_i] = ∫ P_batch · (P_target/P_batch) · ℓ = E_P_target[ℓ]
```

**Le dénominateur DOIT être la vraie distribution du batch.** Sinon l'estimateur est biaisé.

### 4.3 Sampler GenderBalanced — P_sampler

`GenderBalancedSampler` ([src/data/dataset.py](../src/data/dataset.py)) force 50% F + 50% M par batch :

```
P_sampler(g, y) = 0.5 · P_train(y | g)                          (4)
```

Activé quand `feature_fairness != "none"` (= MMD / DANN / OT actifs, qui exigent les 2 genres présents par batch pour calculer leurs stats inter-groupe).

### 4.4 Bug double-counting (fix v15) — sampler-aware IS

**Bug v11–v14** : `compute_target_weights` calculait `w = P_target / P_train_emp` quelque soit le sampler. Quand sampler actif, `P_batch ≠ P_train_emp` → estimateur biaisé.

**Démo numérique** (P_train(F)≈0.10, P_train(M)≈0.90, α1=α2=1, P_test(g)≈0.5) :

| | w_i (bug) | batch_dist | effective = batch×w | vs P_target |
|---|---|---|---|---|
| **F**, sans sampler | 5·P_test(y)/P_train(y\|F) | P_train(F,y)=0.1·P_train(y\|F) | 0.5·P_test(y) | = P_target(F,y) ✓ |
| **F**, sampler actif (bug) | idem (calc P_train) | **P_sampler(F,y)=0.5·P_train(y\|F)** | **2.5·P_test(y)** | **5× P_target(F,y) ✗** |
| **F**, sampler actif (fix v15) | P_test(y)/P_train(y\|F) (calc P_sampler) | 0.5·P_train(y\|F) | 0.5·P_test(y) | = P_target(F,y) ✓ |
| **M**, sampler actif (bug) | idem | 0.5·P_train(y\|M) | 0.28·P_test(y) | **0.56× P_target(M,y) ✗** |

Ratio F/M effectif (bug) : **5 / 0.56 ≈ 9× la cible**. Massif.

**Fix v15** ([src/utils/distribution.py:compute_target_weights](../src/utils/distribution.py)) : flag `gender_sampler_active=True` → dénominateur devient :

```
P_batch(g, y) = P_sampler(g, y) = 0.5 · P_train(y|g)            (5)
```

implémenté comme `0.5 · p_train_joint / p_train_marginal_g`. Le flag est câblé depuis [src/train.py:_build_datasets](../src/train.py) sur la base de `use_gender_sampler`.

### 4.5 Batch-level vs Sample-level — quand combiner

Les axes 1+2 (loss reweight) et le sampler (batch composition) sont 2 mécanismes équivalents pour rendre la distribution effective = P_target. **Cumuler les 2 N'EST PAS du double-counting** si et seulement si le weight utilise le bon `P_batch` au dénominateur (= fix v15).

### 4.6 Feature-level — alignement distributionnel

Trois pertes ajoutées à la loss totale (conditionnelles au `feature_fairness` HPO) :

#### MMD — `mmd_rbf` ([src/utils/losses.py](../src/utils/losses.py))

```
MMD²(F, M) = E[k(F,F)] + E[k(M,M)] − 2·E[k(F,M)]
k(x, y) = exp(−||x − y||² / σ²)        σ² = median(||F − M||²)  (median heuristic)
```

Pousse les features F et M vers la même distribution dans l'espace pooled.

#### DANN — Gradient Reversal Layer

```
features ──→ GenderDiscriminator ──→ ŷ_gender
       └─ grad_reverse(α=1) ─┘

L_total = L_task + adv_lambda · CE(ŷ_gender, g)
```

GRL : forward = identité, backward = signe inversé. Le discriminator apprend à prédire le genre, le backbone apprend à **brouiller** le signal de genre.

#### OT — Sliced Wasserstein-2

```
SW²(F, M) = (1/L) Σ_{l=1}^L  W2²(proj_l(F), proj_l(M))
```

Projections aléatoires 1D + OT 1D = sorted L2. L=50 projections.

### 4.7 Loss-level — Lagrangien adaptatif (fix v14)

Loss training :

```
L = (Err_F + Err_M)/2 + λ_adapt · |Err_F − Err_M|               (6)
```

Update de `λ_adapt` (gradient ascent sur la contrainte) :

```
λ_{t+1} = clip( λ_t + η · (err_diff_ema − ε),  0,  λ_max )      (7)
```

**Bug v11–v13** : pas de `ε`, donc `err_diff_ema ≥ 0` toujours ⇒ λ ne fait que monter, sature à `λ_max = 5` dès epoch 2. L'objectif training divergeait de la métrique eval (λ_metric=1).

**Fix v14** : ajout `lambda_threshold ε = 0.0005`. λ peut maintenant descendre si la contrainte est satisfaite (err_diff < ε). Vrai Lagrangien.

### 4.8 Output-level — Post-hoc calibration

Voir §5.

### 4.9 Optim-level — EMA + LLRD

#### EMA double-eval (fix v15)

```
buf_EMA ← decay · buf_EMA + (1-decay) · p_live    ∀ step
```

`evaluate()` lance **2 fois** `super().evaluate()` :
- 1× sur les poids live (prefix `eval_`)
- 1× sur les poids EMA shadow (prefix `ema_`, après `_swap_to_ema` → eval → `_swap_to_live`)

Résultat MLflow : **2 séries propres séparées** (`eval_challenge_score` et `ema_challenge_score`, 1 valeur par epoch chacune).

**Bug v13** : on appelait 2× `super().evaluate()` avec le même `metric_key_prefix="eval"` → HF logge en interne dans les 2 cas sous `eval_*` ⇒ 2 valeurs par step sur la même métrique ⇒ courbe en dents de scie.

#### EMABestTracker (v14)

Callback qui snapshot l'EMA state sur **disque** (`output_dir/ema_best_rank{R}.pt`, local SSD via TMPDIR) au "new best EMA". Snapshot disque (~50ms write) au lieu de RAM pour éviter OOM (sapiens 0.1B en fp32 = ~456MB × 2 ranks = 1GB RAM permanent → OOM 60G).

Post-train : compare `state.best_metric` (HF best live) vs `ema_best_tracker.best_score` → charge le winner dans `trainer.model`. **Tous les ranks DDP** snapshot indépendamment (snapshot identique grâce DDP sync) → pas de broadcast nécessaire, pas de désync.

#### LLRD — Layer-wise LR Decay

LR par groupe = `base_lr · decay^(n_layers − i)`. Bloc 0 (proche input) reçoit le plus petit LR ; head reçoit `base_lr`. Détecte automatiquement `.blocks` (ViT) ou `.encoder.layer` (HF Transformer). **Disabled pour CNN** (MBConv stages ne mappent pas clairement à une profondeur logique).

---

## 5. Post-hoc calibration

### 5.1 Calibrators implémentés

| Nom | Type | Description |
|---|---|---|
| `isotonic` | PAV (Best 1955) | Monotone piecewise-constant, per-gender, IS-weighted |
| `linear` | Platt regression | `y = a·pred + b`, 2 params/gender, closed-form WLS |
| `pchip` | PCHIP spline (Fritsch-Carlson 1980) | Monotone cubic Hermite, ~20 knots, plus smooth qu'isotonic |
| `isotonic_regime` (v14) | Per-(g × y-regime) | Fit 1 isotonic sur y<0.20, 1 autre sur y≥0.20. Transform : blend linéaire dans `[0.15, 0.25]` (`blend_halfwidth=0.05`) pour éviter discontinuité au boundary. Stretch le high-y tail |
| `isotonic_tailboost` (v15) | Single isotonic, weights boostés | `w_i = (1/30+y) · IS_ratio · (1 + boost · max(0, y - y_pivot))`. Pas de boundary, pas d'artefact géométrique. `y_pivot=0.20, boost=5.0` |

### 5.2 IS reweight des calibrators

```
ratio_i = P_test(y_i) / P_train_emp(y_i)         clip(0.1, 10), normalize mean=1
w_i = (1/30 + y_i) · ratio_i
```

Fit per-gender avec ces poids. Sous H_C, le ratio P_test/P_train du gender s'annule au numérateur/dénominateur (cf v12 §4.5).

### 5.3 Alpha-blend

```
pred_blend = α · cal(pred) + (1-α) · pred_raw                  (8)
           = pred_raw + α · (cal(pred) - pred_raw)
```

- `α = 0` → raw seul (pas de correction)
- `α = 1` → correction max (cal pur)
- `α > 1` → over-correction (extrapole au-delà de cal). Utile si isotonic est conservatrice sur la queue par manque de support val.

**Scan** : `α ∈ [0, 1.5]` step 0.1 (16 valeurs).

### 5.4 Sélection (cal, α) — méthodologie

```
(cal*, α*) = argmin_{(cal, α)}  score_val_IS_strat( blend(α, cal(pred_val)) )
```

Sélection sur **val IS-stratifié**, PAS sur test holdout (oracle preserved). Coût méthodologique : sub-optimal possible pour ce trial spécifique, mais test holdout reste mesure indépendante.

`best_alpha_<cal_name>` est aussi loggué per-cal (argmin α pour chaque cal) — utilisé par l'UI pour visualiser la déformation au best α de chaque méthode.

### 5.5 IS-stratified estimator de la métrique (val)

`compute_score_stratified_is` ([src/utils/metrics.py](../src/utils/metrics.py)) — Rao-Blackwellised :

```
Err_g_test  =  Σ_b  P_test(g, b) · mean_in_cell(w·err)         (9)
              ─────────────────────────────────────
               Σ_b  P_test(g, b) · mean_in_cell(w)
```

Moyenne intra-cellule (g, b) sur les samples val, puis pondération par `P_test(g, b)` (= `test_pmf_joint` construit via H_C). Variance bornée par variance intra-cellule. Cellules vides skipped → légère sous-estimation possible sur queue rare, mitigée par val=15k.

### 5.6 Ensemble + calibration : ordre des opérations

| Option | Pipeline |
|---|---|
| A : calibre puis ensemble | `final = mean( cal_i(pred_i) )` — 1 cal fit par modèle |
| **B : ensemble puis calibre** | `final = cal( mean(pred_i) )` — 1 cal fit sur l'ensemble |

**Décision : B**. Raisons :
1. **Cible directement optimale** : B fitte `E[y | mean_pred]`, exactement la fonction à corriger sur la prédiction finale
2. **Variance réduite pour le calibrator** : l'ensemble a variance ~1/√K → cal overfit moins
3. **Biais systémique de sous-prédiction sur hauts y N'EST PAS corrigé par l'ensemble** — seule la calibration post-ensemble peut stretcher la queue
4. **Simplicité** : B = 1 cal × 1 α-scan vs K cals × K α-scans pour A

### 5.7 Pipeline final (post-sweep)

```
1. K modèles (top-K trials Optuna) → preds_val_i, preds_test_i
2. ens_val = mean(preds_val_i)
3. cals = fit_all_calibrators(ens_val, gt_val, gender_val)   # 5 calibrators per-gender IS-weighted
4. (best_cal, best_α) = argmin over (cal, α) val_IS_strat( blend(α, cal(ens_val)) )
5. submit = clip( best_α · best_cal(mean(preds_test_i)) + (1-best_α) · mean(preds_test_i), 0, 1 )
```

---

## 6. Pooling — refonte complète v15

5 options dispo, détection auto `has_cls = not _is_timm_cnn(model_name)` qui pilote `skip_cls` :

| pooling_type | Output | Description | Default pour |
|---|---|---|---|
| `cls` | (B, D) | Token CLS — natif Transformer | DINOv3, Sapiens |
| `gap` (v15) | (B, D) | Mean over patches (skip CLS si ViT) — natif CNN | EfficientNet |
| `mean_var` | (B, 2D) | `concat(mean, std)` sur patches | — |
| `attention_k_query` | (B, K·D) | K queries learnable avec τ par-query (focal/diffuse/free) | — |
| `mil` | (B, 4) ou (B, 1) | Multi-Instance Learning — détails §6.2 | — |

**Retiré v15** : `multihead_attention` (`MultiHeadAttentionPooling`) — redondant avec K-query, jamais utilisé.

### 6.1 K-query attention — détails

```
queries ∈ R^{K×D}  (apprises)
log_τ ∈ R^K        (apprises, init focal/diffuse/free)
score_{k, n} = (queries_k · proj_k(x_n)) / (√D · exp(log_τ_k))
weights_{k, n} = softmax_n(score_{k, n})
pooled_k = Σ_n weights_{k, n} · proj_v(x_n)
pooled = concat_k(pooled_k) → (B, K·D)
```

`√D` = scaling standard "scaled dot product" (Vaswani 2017). `τ_k` = température per-query, learnable → contrôle sharpness (focal init τ=0.1 sharp, diffuse init τ=1.5 uniforme).

### 6.2 MIL — refonte v15

Per-patch scoring head : `scorer(h_i) → score_i ∈ R` (2-layer MLP avec hidden=`mil_hidden`).

5 aggregations possibles (`mil_agg`) :

| Mode | Mécanisme | Sensible à |
|---|---|---|
| `mean` | `mean(scores)` | Flou / dégradation globale uniforme |
| `max` | `max(scores)` | Occlusion sparse extrême |
| `topk_mean` | `mean(topK(scores))`, `K=mil_k_top` | Sparse robuste anti-noise |
| `attention` (fix v15) | **Gated attention Ilse et al. 2018** : `α_i = softmax(w·(tanh(V·h_i) ⊙ σ(U·h_i)))` puis `Σ α_i · score_i`. Attention head SÉPARÉE des scores | Apprend où regarder, découplé du scorer |
| **`multi` (default v15)** | Concat des 4 modes → (B, 4). `Linear(4, output_dim)` apprend la pondération | **Sparse + flou simultanément** |

**Bug v13 fixé v15** : ancienne `attention` agg faisait `Σ softmax(scores) · scores` (Boltzmann smooth-max des scores eux-mêmes). Maintenant attention **découplée** : params dédiés `attn_V, attn_U, attn_w`, conforme paper Ilse.

**Pourquoi `multi` est default** : résout le tradeoff "either/or". Sur face occlusion :
- occlusion physique (lunettes, masque) = sparse → `max` et `topk_mean` capturent
- flou / stylization = uniforme → `mean` capture
- combinant les 4 → le head linear apprend automatiquement quelle agg utiliser selon l'input

### 6.3 GAP (Global Average Pool) — v15

```
pooled = patches.mean(dim=1)      # (B, D)
```

Si ViT : skip CLS (token 0) pour pure mean sur patches. Si CNN : pas de CLS, mean sur toutes les positions spatiales.

**Pourquoi important** : EfficientNet (et CNNs en général) ont été **pretrained avec GAP head** (Avg pool + classifier). Remplacer GAP par K-query/MIL = re-learn la pooling head from scratch, casse l'alignement pretrained. GAP préserve l'alignement.

---

## 7. EfficientNet — pivot B5 → B0 (v15)

Biais v14 identifiés sur `tf_efficientnet_b5` :

| Biais v14 | Impact | Fix v15 |
|---|---|---|
| Input 224 (B5 natif 456) | 4× moins de pixels que designed → capacité sous-utilisée | **Pivot vers B0 (natif 224)** |
| Pas de GAP pooling option | K-query/MIL casse alignement pretrained | `pooling_type: gap` default |
| bf16 sur BatchNorm | Précision insuffisante sur BN running stats | `fp16` au lieu de `bf16` |
| `head_dropout=0.1` | Sous-régularisé (B5 designed 0.4, B0 designed 0.2) | Default 0.2, HPO [0.1, 0.4] |
| `layer_decay` HPO sur MBConv | LLRD ill-defined sur stages MBConv | `layer_decay=1.0` (disabled) |
| `backbone_drop_path_rate` HPO | Ignoré par timm sur EffNet | Retiré du HPO |
| `mil_agg` HPO 4 modes | Redondant avec `multi` (qui les inclut) | `mil_agg=multi` pinned |
| LR cap 2e-4 | Trop haut pour CNN | Cap 1e-4 |

---

## 8. Choix de design critiques

### 8.1 Head bias init via `logit(E[Y])`

Bias du head initialisé à `logit(target_mean_weighted)`. Avec sigmoid output, `sigmoid(bias) ≈ E[Y_target]` au start → évite le warmup gaspillé sur biais 0.5 décentré.

### 8.2 Normalisation `w / w.mean()` (truc v3)

Après calcul des sample weights : `sample_w ← sample_w / mean(sample_w)`. Préserve l'échelle moyenne de la loss à ~1. Permet de garder le même LR à travers différents `(α1, α2)`.

### 8.3 `mil_agg=multi` default

Multi pooling resout le tradeoff "either/or" entre détection sparse (occlusion) et uniforme (flou). Le head linear (B, 4) → (B, 1) apprend la pondération. Pas de HPO sur mil_agg (multi inclut tout).

### 8.4 EMA snapshot disque vs RAM

Disque préféré : `~50ms` write par "new best EMA" (rare, ~5-10×/training), 0 RAM permanent. RAM coûte ~1GB pour sapiens-0.1B × 2 ranks → OOM observé avec --mem 60G. SLURM bumped à 100G + disk snapshot.

### 8.5 Lagrangien adaptatif vs pinned

Adaptatif (v15) avec threshold ε = 0.0005 :
- Si modèle "lazy" sur fairness, λ monte → push sur err_diff
- Si fairness OK, λ descend vers 0
- Tracking via MLflow : metric `lambda_adapt`

Avant fix v14, λ saturait à `λ_max=5` → optimisait un objectif 5× plus pénalisant que la métrique. Le threshold permet `λ` de descendre, restore l'alignement.

---

## 9. HPO Optuna — search space (v15)

### 9.1 DINOv3 / Sapiens (ViT)

| Param | Type | Range | Note |
|---|---|---|---|
| `pretrained_source` | cat | `[lvd, ibot:runs:...]` | (sapiens : sapiens_default + 3 ibot checkpoints) |
| `pooling_type` | cat | `[cls, gap, attention_k_query, mil]` | default cls (natif) |
| `axis1_power` | float | `[0, 1]` | mix Y train↔test |
| `axis2_power` | float | `[0, 1]` | mix G\|Y train↔test marginal |
| `feature_fairness` | cat | `[none, ot, dann]` | active sampler si != none |
| `ot_lambda` | float log | `[0.01, 1.0]` | conditional ot |
| `adv_lambda` | float log | `[0.001, 0.05]` | conditional dann |
| `loss_focal_gamma` | float | `[0, 1.5]` | focal weight pour hard samples |
| `learning_rate` | float log | `[1e-5, 1e-4]` | LR base (cap basé sur v11 sweet spot) |
| `min_lr_rate` | float | `[0.1, 0.5]` | min LR fraction (cosine) |
| `weight_decay` | float | `[0.05, 0.50]` | AdamW WD |
| `head_dropout` | float | `[0, 0.4]` | dropout sur features avant head |
| `backbone_drop_path_rate` | float | `[0, 0.4]` | drop path dans blocks |
| `layer_decay` | float | `[0.65, 0.9]` | LLRD decay |
| K-query (conditional) | various | — | tau_focal/diffuse, n_focal/diffuse/free, pool dropouts, query_diversity_lambda |
| MIL (conditional) | various | — | `mil_hidden` cat [64, 128, 256], `mil_k_top` int [10, 50] |

`mil_agg` est PINNED à `multi` (pas dans HPO).

### 9.2 EfficientNet-B0 (CNN)

| Param | Type | Range | Note |
|---|---|---|---|
| `pooling_type` | cat | `[gap, attention_k_query, mil]` | default gap (natif) |
| `learning_rate` | float log | `[1e-5, 1e-4]` | CNN converge avec LR plus bas |
| `weight_decay` | float | `[0.01, 0.20]` | CNN sous-régularise vite à WD haut |
| `head_dropout` | float | `[0.1, 0.4]` | B0 natif 0.2 |
| `min_lr_rate` | float | `[0.1, 0.5]` | — |
| Pas de `layer_decay`, pas de `backbone_drop_path_rate` | — | — | retirés (mal supportés MBConv) |
| Autres (axis1/2, fairness, focal, K-query, MIL) | idem ViT | — | — |

`n_trials=50` (vs 100 ViT) — search space plus étroit.

---

## 10. Logging MLflow — convention v15

### 10.1 Metrics per-epoch

| Préfixe | Source | Contenu |
|---|---|---|
| `eval_*` | Live model eval | `eval_challenge_score`, `eval_err_F`, `eval_err_M`, `eval_loss`, ... |
| `ema_*` | EMA shadow eval | `ema_challenge_score`, `ema_err_F`, `ema_loss`, ... |
| `lambda_adapt`, `err_diff_ema` | LambdaLogCallback | Lagrangien trajectory |

Logging step = `state.global_step`. 1 valeur par metric par epoch (pas de double-write, pas de zigzag).

### 10.2 Metrics post-train

| Nom | Source |
|---|---|
| `best_live_score` | HF Trainer state.best_metric |
| `best_ema_score` | EMABestTracker |
| `used_ema_weights` | 0 ou 1 selon winner |
| `val_score_<cal>_is_eval` | val IS-strat pour chaque calibrator (α=1) |
| `val_score_<cal>_raw_eval` | val raw (P_train) pour chaque cal |
| `val_score_best_combo_is_eval` | val IS-strat du best (cal, α) |
| `best_alpha_for_submission_value` | α du best combo |
| `best_alpha_<cal>` | best α per-cal (logged for UI) |
| `test_holdout_score_raw` | test holdout sans cal |
| `test_holdout_score_<cal>` | test holdout avec chaque cal (α=1) |
| `test_holdout_score_selected_cal` | test holdout avec (cal, α) sélectionné sur val IS |
| `test_holdout_score_oracle_cal` | test holdout avec best cal post-hoc sur test (ORACLE, biased upper bound) |

### 10.3 Params conditionnels (fix v14)

`adv_lambda / mmd_lambda / ot_lambda` loggués **résolus** (= 0 si feature_fairness=none) au lieu du yaml raw (qui afficherait `adv_lambda=0.01` même sans DANN).

---

## 11. Audit final v15 — checklist

| Composant | Statut |
|---|---|
| `_query_diversity_penalty` (K-query only) | ✅ gate `> 0`, conditional HPO sur K-query |
| MIL `attention` agg | ✅ gated head séparé (Ilse 2018) |
| MIL `multi` default | ✅ |
| `MultiHeadAttentionPooling` | ✅ supprimé (code + yaml HPO + train.py params) |
| `GAPPooling` | ✅ ajouté, `skip_cls=has_cls` auto |
| `has_cls = not _is_timm_cnn(model_name)` | ✅ propagé à GAP/MIL/MeanVar |
| `compute_target_weights` sampler-aware | ✅ `gender_sampler_active` flag |
| Lagrangien threshold | ✅ peut descendre |
| EMA `eval_*` / `ema_*` séparation | ✅ via `metric_key_prefix` |
| EMA snapshot disque | ✅ fix OOM 60G |
| EMABestTracker DDP | ✅ tous ranks snapshot/load (pas de désync) |
| MLflow logging conditionnel | ✅ adv/mmd/ot résolus |
| `ddp_find_unused_parameters=False` | ✅ adv_disc absent quand non-DANN |
| `image_size` configurable | ✅ via model.image_size |
| EfficientNet pivot B0 + GAP natif | ✅ |
| MIL `mil_agg` default `multi` train.py | ✅ |
| Alpha-blend `[0, 1.5]` | ✅ over-correction permise |
| IsotonicTailBoost calibrator | ✅ ajouté, sample weights boostés tail |
| IsotonicRegime `blend_halfwidth=0.05` | ✅ régime smooth |
| UI `eval_challenge_score_*` legacy refs | ✅ fixed (selected/oracle) |
| UI dark theme | ✅ GT amber, P_test target blanc |
| UI best alpha per cal | ✅ logged + applied client-side |
| SLURM `--mem` 100G | ✅ |
| `ema_decay` legacy conflict | ✅ retiré de LEGACY |

---

## 12. Références théoriques

| Concept | Référence |
|---|---|
| MMD + RBF kernel | Gretton et al. 2012, "A Kernel Two-Sample Test" |
| Median heuristic bandwidth | Garreau et al. 2017 |
| DANN + GRL | Ganin & Lempitsky 2015, "Domain-Adversarial Training" |
| Sliced Wasserstein | Bonneel et al. 2015 |
| Lagrangian fairness | Cotter et al. 2019 |
| EMA Polyak averaging | Polyak 1990 |
| LLRD | BERT/ELECTRA/ViT fine-tuning papers |
| Isotonic calibration | Zadrozny & Elkan 2002 |
| PCHIP monotone interpolation | Fritsch & Carlson 1980 |
| Gated MIL attention | Ilse et al. 2018, "Attention-based Deep Multiple Instance Learning" |
| Importance Sampling | Hammersley & Handscomb 1964, ch. 5 |
| Stratified IS / Rao-Blackwellisation | Rubinstein & Kroese 2017 |
| Scaled dot-product attention | Vaswani et al. 2017, "Attention is All You Need" |

Voir [references.md](references.md) pour bibliographie complète.
