# v12 — Théorie complète et état des implémentations

Document de référence consolidé pour la version courante. Couvre :
1. Métrique concours + notation
2. Découverte du shift train/test via MID lookup + tests rigoureux des hypothèses
3. Les 6 étages d'intervention (sample/batch/feature/loss/output/optim)
4. Audit des implémentations (OT, MMD, Lagrangien, EMA, etc.) — avec tests
5. Hyperparams + ranges HPO + analyse empirique v11
6. Différences v11 → v12

Pour l'historique pré-v11 : voir [audits_and_roadmap.md](audits_and_roadmap.md). Pour fairness méthodes détaillées : voir [fairness.md](fairness.md). Architecture pooling : [architecture.md](architecture.md).

---

## 1. Notation et métrique du concours

### 1.1 Variables

- $X$ : image (224×224 RGB)
- $Y \in [0, 1]$ : taux d'occlusion réel (cible)
- $G \in \{F, M\}$ : genre
- $\hat{Y} = f_\theta(X)$ : prédiction du modèle
- $w_i = \frac{1}{30} + y_i$ : poids du sample $i$ dans la métrique

### 1.2 Métrique officielle

$$
\text{Err}_g = \frac{\sum_{i \in g} w_i (\hat{y}_i - y_i)^2}{\sum_{i \in g} w_i}
$$

$$
\text{Score} = \frac{\text{Err}_F + \text{Err}_M}{2} + |\text{Err}_F - \text{Err}_M|
$$

**Propriétés clés** :
- **Métrique par-groupe** : Err_F et Err_M pondérés à égalité, **indépendamment** des effectifs F/M en test
- **Pondération $w_i$** : haute occlusion = plus de poids → la queue rare n'est pas ignorée
- **$\lambda_{\text{metric}} = 1$** sur $|\text{Err}_F - \text{Err}_M|$ → fairness pondérée à égalité avec la moyenne

---

## 2. Distributions train/test et hypothèses

### 2.1 Ce qu'on connaît

| Quantité | Source | Valeur |
|---|---|---|
| $P_{\text{train}}(G, Y)$ | `train.csv` (96k samples) | Mesurée |
| $P_{\text{test}}(Y)$ | PDF page 3, extraction pixel (15 bins) | Mesurée |
| $P_{\text{test}}(G)$ | MID lookup (93.5% coverage) | ≈ (0.477, 0.523) |
| $P_{\text{test}}(G, Y)$ | — | **Inconnue** |

### 2.2 Observations critiques

**Shift marginal gender** : `P_train(F) = 32.4%` → `P_test(F) ≈ 47.7%` (+15 pts)
**Shift marginal Y** : `P_train(Y)` long-tail vers 0 → `P_test(Y)` piqué autour de 0.05-0.25
**Dépendance G↔Y en train** : `P_train(F|Y)` varie de **12%** (Y≈0) à **74%** (Y∈[0.37, 0.40])

### 2.3 Hypothèses candidates

| Nom | Formulation | Interprétation |
|---|---|---|
| $H_A$ | $G \perp Y$ en test | Gender et occlusion indépendants en test |
| $H_B$ | $P_{\text{test}}(Y\|G) = P_{\text{train}}(Y\|G)$ | Patterns Y par gender conservés |
| **$H_C$** | $P_{\text{test}}(G\|Y) = P_{\text{train}}(G\|Y)$ | **Ratio F/M par bin Y conservé** = "H1 covariate shift Y-only" historique |

### 2.4 Tests empiriques rigoureux

**$H_A$ rejetée a priori** : on observe en train une forte dépendance G↔Y (12% → 74%). Improbable que ça disparaisse en test.

**$H_B$ testé empiriquement** :
$$
P_{\text{test}}(Y) \overset{H_B}{=} \sum_g P_{\text{train}}(Y|g) \cdot P_{\text{test}}(g)
$$
L1 distance vs `P_test(Y)` PDF observé : **0.6676** (vs baseline 0.7714 = `P_train(Y)` brut). Amélioration relative seulement **10.4%** → **$H_B$ rejetée**.

**$H_C$ testé empiriquement** :
$$
P_{\text{test}}(F) \overset{H_C}{=} \sum_y P_{\text{train}}(F|y) \cdot P_{\text{test}}(y) = 0.4816
$$
vs observé MID = `0.4766` → **écart 0.5 pts** → **$H_C$ acceptée**.

**IPF (Iterative Proportional Fitting)** confirme : en partant de `P_train(g,y)` comme prior et en forçant les 2 marginales observées, l'IPF "warpe" la conditionnelle $P(G|Y)$ de seulement **0.56%** (négligeable). Solution la plus parcimonieuse = $H_C$.

### 2.5 Interprétation physique

$H_C$ dit : "à occlusion donnée, le ratio F/M est stable train→test". Ce qui change train→test = la fréquence de chaque niveau d'occlusion (test a plus de Y moyens-hauts). Cohérent avec patterns biologiques :
- Y≈0 : hommes dominent (rasés, sans cheveux, sans maquillage)
- Y moyens : femmes dominent (cheveux longs, maquillage léger)
- Y hauts : femmes dominent (cheveux longs/voile, maquillage important)

Voir [`scripts/estimate_test_gender.py`](../scripts/estimate_test_gender.py) pour l'extraction MID + validation $H_C$.

---

## 3.bis Pourquoi `test_pmf_joint` est calculé via H_C dans le code

Chaque fois qu'on a besoin d'une estimation de `P_test(g, y)` (val split, test holdout sampling, IS stratifié), on l'estime par :
```
P_test(g, y)  =  P_train(g | y)  ×  P_test(y)        ← H_C
```
**Justification** :
- `P_test(y)` connu (extrait du PDF, 15 bins).
- `P_train(g | y)` mesurable sur train (déterministe, pas de bruit).
- Le seul ingrédient hypothétique est H_C, qu'on a validé empiriquement à 0.5 pt près sur la marginale `P_test(F)` via MID lookup (cf §2-3).

**Alternatives rejetées** :
- `P_test(g) × P_test(y)` (= H_A indépendance) : faux empiriquement (G ↔ Y corrélés dans train).
- `P_test(y | g)` via P_train(y | g) (= H_B) : implique mauvaise marginale `P_test(y)` (L1 = 0.67).
- IPF avec contraintes (P_test(g) MID + P_test(y) PDF) : revient à H_C à 0.56% près.

Implémentation centralisée : [`distribution.estimate_test_pmf_joint`](../src/utils/distribution.py).

## 3.ter Tri-split v12 (train iid / val iid / test holdout P_test)

**Évolution v11 → v12** : on passe d'un split bipartite (train / val resampled-to-P_test) à un split tripartite :

```
Train  ~80%  iid de P_train
Val    ~15%  iid de P_train  + métrique estimée via IS stratifié
Test    ~5%  resampled match P_test (H_C)  — utilisé UNIQUEMENT pour valider post-hoc isotonic
```

**Motivation : pourquoi changer ?**

Le val v11 (resampled match P_test) avait un **effet pervers** : sur les bins moyens-hauts (Y∈[0.3, 0.5], dominants en test mais rares en train), le resampling stratifié amputait jusqu'à 60% du train. Or ces samples sont précisément ceux dont le modèle a le plus besoin pour bien généraliser au test challenge.

Quantification (sur train.csv) :

| Bin Y | Train brut | Val v11 (resampled P_test) | Train v11 retenu | Train v12 retenu (iid val) |
|---|---|---|---|---|
| [0.00, 0.03] | ~33000 | 1700 (5%) | 31300 (95%) | ~28000 (85%) |
| [0.10, 0.13] | ~5000 | 1500 (30%) | 3500 (70%) | ~4250 (85%) |
| [0.30, 0.33] | ~1300 | 780 (60%) | 520 (40%) | ~1034 (80%) |
| [0.37, 0.40] | ~470 | 280 (60%) | 190 (40%) | ~327 (70%) |

Sur les bins rares (importants pour test), **v12 garde ~70% en train** vs 40% en v11 → **+75% de samples rares en train**.

**Importance Sampling stratifié pour la métrique val** : voir §3.quater.

**Test holdout 5%** : sert UNIQUEMENT à valider, après training, l'effet du post-hoc isotonic sur une distribution test-like (pas vue pendant training/HPO). Logged dans MLflow comme `test_holdout_score_raw`, `test_holdout_score_iso`, `test_holdout_iso_gain`. Préds sauvegardées en artifact `qualitative/test_holdout_predictions.csv` pour visualisation UI.

Implémentation : [`distribution.split_train_val_test`](../src/utils/distribution.py).

## 3.quater Importance Sampling stratifié pour val métrique

**Problème** : val est tirée iid de `P_train` (≠ `P_test`). Comment estimer la métrique sur `P_test` sans biais ?

**IS naïf** : poids per-sample `r_i = P_test(x_i) / P_train(x_i)`. Variance explose sur bins rares en train (ratios extrêmes dominent).

**IS stratifié (Rao-Blackwellisé)** : on partitionne par cellule (g, bin Y), on calcule la moyenne empirique par cellule, puis on combine avec les poids connus `P_test(g, b)` :

```
                sum sur b  [ P_test(g, b)  × MoyenneErr_dans_cellule(g, b) ]
Err_g_test  =  ────────────────────────────────────────────────────────────
                sum sur b  [ P_test(g, b)  × MoyenneW_dans_cellule(g, b) ]

avec :
   MoyenneErr_dans_cellule(g, b) = sum_{i: g_i=g, bin_i=b} w_i · (p_i - y_i)²  /  n_cellule
   MoyenneW_dans_cellule(g, b)   = sum_{i: g_i=g, bin_i=b} w_i                /  n_cellule
   w_i = 1/30 + y_i                                ← poids challenge per-sample
   P_test(g, b)  estimé via H_C                    ← P_train(g|y) × P_test(y)
```

Variance bornée par la variance INTRA-cellule (généralement petite avec ~500 samples/cellule sur val 15k). Une cellule sans sample val est skip-pée (logué via `_is_stratified_skipped_cells`).

Avantage clé : pas de samples extrêmes qui dominent → estimation stable, utilisable pour HPO et early stopping (`metric_for_best_model: eval_challenge_score`).

Implémentation : [`metrics.compute_score_stratified_is`](../src/utils/metrics.py).

## 3. Val split — basée sur $H_C$

Implémenté dans [`dataset._split_val_to_match_test_pmf`](../src/data/dataset.py).

$$
P_{\text{val}}(g, y) = P_{\text{train}}(g|y) \cdot P_{\text{test}}(y)
$$

**Effet automatique** : val a ~48% F (re-balancé automatiquement par le mix avec `P_test(y)`, sans avoir à le demander explicitement). C'est cohérent avec l'observation MID `P_test(F) ≈ 48%`.

---

## 4. Les 6 étages d'intervention

### 4.1 Sample-level — rebalancing target unifié

[`losses.compute_target_weights`](../src/utils/losses.py)

$$
P_{\text{target}}(g, y) = \text{mix}_y(\alpha_1) \times \text{mix}_g(\alpha_2)
$$

avec :
- $\text{mix}_y(\alpha_1) = (1-\alpha_1) P_{\text{train}}(y) + \alpha_1 P_{\text{test}}(y)$ ← **axe 1** (`axis1_power`)
- $\text{mix}_g(\alpha_2) = (1-\alpha_2) P_{\text{train}}(g|y) + \alpha_2 \cdot 0.5$ ← **axe 2** (`axis2_power`)

**Sample weight** :
$$
w_i = \text{clip}\left(\frac{P_{\text{target}}(g_i, y_i)}{P_{\text{train}}(g_i, y_i)}, \frac{1}{10}, 10\right)
$$
puis normalisé : $w \leftarrow w / \bar{w}$ (hygiène d'échelle, **truc v3 critique**).

**Pourquoi `axis2 → 0.5` uniforme** : on a découvert que `P_test(F) ≈ 0.48`, donc la cible uniforme 50/50 est presque juste (écart négligeable). Pas besoin de raffiner.

### 4.2 Batch-level — `GenderBalancedSampler`

[`dataset.GenderBalancedSampler`](../src/data/dataset.py)

**Principe** : à chaque epoch, sampling sans remplacement (autant que possible) pour garantir :
- Tous les samples M vus exactement 1× (n_M = 67600 pour Sapiens train split)
- Les samples F oversamplés (~2× chacun) pour atteindre n_M

Total par epoch : `2 × max(n_F, n_M) = 135 200` samples → 528 steps avec batch=256.

**Activation** : seulement si `feature_fairness != none`. Raison : MMD/DANN/OT ont besoin de F+M **physiquement présents** dans chaque batch pour calculer leurs stats croisées.

### 4.3 Feature-level — alignement distributionnel

Trois choix (mutuellement exclusifs car même objectif : $P(\text{feat}|F) \approx P(\text{feat}|M)$) :

#### MMD (Maximum Mean Discrepancy) — [`losses.mmd_rbf`](../src/utils/losses.py)

$$
\text{MMD}^2 = \mathbb{E}[k(x, x')] + \mathbb{E}[k(y, y')] - 2\mathbb{E}[k(x, y)]
$$

avec kernel RBF $k(a, b) = \exp(-\|a-b\|^2 / \sigma^2)$ et **σ² = median heuristic** (median of pairwise squared distances dans le batch).

**Statut v12** : retiré du search_space, code conservé dormant. Raison : redondant avec OT/DANN, espace HPO réduit.

#### DANN — adversarial via Gradient Reversal Layer

[`face_occ_regressor.GenderDiscriminator`](../src/models/face_occ_regressor.py) + [`grad_reverse`](../src/models/face_occ_regressor.py)

Discriminateur MLP `features → P(gender)`. Loss = cross-entropy. Via GRL, backbone reçoit -∇ donc apprend des features **invariantes** au gender.

**Statut v12** : actif (`feature_fairness=dann` dans HPO). `adv_lambda` en HPO conditionnel [0.001, 0.05] log.

#### OT (Sliced Wasserstein-2) — [`losses.sliced_wasserstein`](../src/utils/losses.py)

$$
\text{SW}_2^2 = \frac{1}{L} \sum_{l=1}^L W_2^2(\pi_l \# X, \pi_l \# Y)
$$

avec $\pi_l$ = projection sur la $l$-ème direction unitaire aléatoire ($L=50$). Pour chaque projection, $W_2^2$ entre 1D distributions = mean squared diff des quantiles sortés.

**Validation empirique** (tests internes) :
| Setup | SW² calculé | Attendu |
|---|---|---|
| 2× N(0, I), N=100 | 0.063 | ~0 (bruit) |
| N(0, I) vs N(1, I), N=100 | **1.093** | ≈ shift² = 1.0 ✓ |
| Sizes inégales (20, 35) | 0.200 | petit ✓ |
| Gradient flow | non-nul | ✓ |
| Convergence n_proj | stable dès 10 | 50 = safe default |

**Avantages OT vs MMD vs DANN** :
- Différentiable directement
- Pas de bandwidth à tuner (vs MMD)
- Pas d'adversarial instable (vs DANN)
- Batch-size agnostic (interpolation pour tailles inégales)

**Statut v12** : actif (`feature_fairness=ot` dans HPO). `ot_lambda` en HPO conditionnel [0.01, 1.0] log.

### 4.4 Loss-level — Lagrangien adaptatif

[`losses.WeightedMSELoss`](../src/utils/losses.py)

**Formulation training** :
$$
\mathcal{L}_{\text{train}}(\theta, \lambda_t) = \frac{\text{Err}_F + \text{Err}_M}{2} + \lambda_t \cdot |\text{Err}_F - \text{Err}_M|
$$

**Update Lagrangien** (gradient ascent sur la violation) :
$$
\text{EMA}_t = \text{EMA}_{t-1} \cdot \alpha + |\text{Err}_F - \text{Err}_M|_t \cdot (1 - \alpha)
$$
$$
\lambda_{t+1} = \text{clip}(\lambda_t + \eta_\lambda \cdot \text{EMA}_t,\ 0,\ \lambda_{\max})
$$

**Hyperparams pinned** :
- $\lambda_0 = 1.0$ (démarre à la valeur métrique officielle)
- $\eta_\lambda = 0.5$ (lambda_lr — assez fort pour adaptation visible sur 8-12 epochs)
- $\lambda_{\max} = 5.0$ (cap anti-divergence)
- $\alpha = 0.9$ (EMA lisse sur ~10 steps, filtre passe-bas sur la violation bruitée per-batch)

**DDP** : `all_reduce(err_diff, op=AVG)` avant l'update → λ identique sur tous les ranks.

**CRITICAL séparation train/eval** :
```python
if self.training:
    lam = self.lambda_adapt  # adaptive
else:
    lam = torch.tensor(1.0, ...)  # FORCED to 1.0 (= challenge metric)
```

**Bug fix v11**: HF Trainer `model.eval()` ne propage PAS à `loss_fct` (module séparé). Sans correction, `loss_fct.training` reste True pendant eval → `all_reduce` appelé pendant `torch.no_grad()` prediction step → **DDP deadlock** (hang à 14/30 batches reproductible).

Fix dans [`train.py compute_loss`](../src/train.py) :
```python
self.loss_fct.train(model.training)  # sync explicitly
```

### 4.5 Output-level — Post-hoc isotonic per-gender

[`inference.isotonic.GenderConditionalIsotonic`](../src/inference/isotonic.py)

Fit sur val : `IsotonicRegression(weighted=w_i = 1/30 + y_i)` séparé par gender. Apply à l'inférence test (nécessite le gender).

**À l'inférence test** : gender obtenu via MID lookup (93.5% coverage) + Sapiens linear probe pour les 6.5% restants (à implémenter).

### 4.6 Optim-level — EMA + LLRD

#### EMA weights — [`train.EMAWeightCallback`](../src/train.py)

$$
\theta_{\text{EMA}}^{(t)} = \alpha \cdot \theta_{\text{EMA}}^{(t-1)} + (1-\alpha) \cdot \theta^{(t)}, \quad \alpha = 0.999
$$

**Double-eval intelligent** : à chaque eval epoch, on évalue raw ET EMA, on garde le meilleur. HF `load_best_model_at_end=True` charge ensuite la version gagnante.

**Optimisation** : skip double-eval pour les early epochs (`ema_eval_min_epoch_frac=0.5` → uniquement seconde moitié). Évite de gaspiller compute quand l'EMA est encore "chaud" (presque l'init).

**Bug fix EMA** : buffers initialisés sur CPU puis HF déplace le modèle vers cuda → mismatch device au premier `on_step_end`. Fix : migration lazy via `buf.to(p.device)` au premier usage.

#### LLRD — Layer-wise Learning Rate Decay

[`WeightedMSETrainer.create_optimizer`](../src/train.py)

$$
\text{lr}_l = \text{base\_lr} \cdot (\text{layer\_decay})^{n_{\text{layers}} - l}
$$

Embeddings/cls_token : `lr × decay^(n+1)`. Head : `base_lr`. Standard pour fine-tuner ViT pré-entraîné.

**Range v12** : `layer_decay ∈ [0.65, 0.9]` (était [0.65, 1.0] en v11 — 1.0 = LLRD off causait divergences avec LR haut).

---

## 4.7 Choix de design critiques (détaillés)

### 4.7.1 Head bias init via $\text{logit}(\mathbb{E}[Y])$

**Bug v3-v10** : `nn.init.zeros_(self.head.bias)` → `sigmoid(0)=0.5`, mais `E[Y_train] ≈ 0.085`. Modèle démarre avec offset +0.4 sur les prédictions → gradient initial dominé par cet offset → 1-2 epochs gaspillés à juste compenser. Pour un fine-tuning de 8-12 epochs c'est ~15% du training perdu.

**Fix** : init `bias = logit(E[Y_train]) ≈ -2.40` → `sigmoid(-2.40) ≈ 0.085`. Modèle démarre proche du target → gradient utile dès step 1.

### 4.7.2 Normalisation `w / w.mean()` (truc v3)

Après le clip des ratios `P_target / P_train`, on normalise pour que `mean(w) = 1`. Sans ça, l'échelle moyenne de la loss varie selon le tirage du batch (bruit additionnel sur le LR effectif). Avec, l'échelle reste stable → LR/WD comparables entre trials. Hygiène d'échelle critique.

### 4.7.3 P_test PMF — extraction pixel-par-pixel

L'histogramme de `P_test(Y)` du PDF page 3 est une image matplotlib rasterisée (pas de tableau extractible). Extraction via :
1. Render PDF page 3 à 300 DPI (`pdfplumber.to_image`)
2. Détecter bars bleus via masque RGB `(B > R + 20) & (B > 180)`
3. Calibrer axes via OCR positions des labels Y et X
4. Mesurer la hauteur (en pixels) du top de chaque colonne bar
5. Convertir hauteur → count via calibration, puis agréger sur 15 bins

Précision : ±1% par bin (vs ±5% extraction visuelle). Code reproductible : `_TEST_PMF` hardcoded dans [`distribution.py`](../src/utils/distribution.py).

### 4.7.4 Focal weighting

```python
w_i_focal = w_i × (err_i + 0.05)^γ,   γ ∈ [0, 1.5]  (HPO)
```

Forme `(err + 0.05)^γ` choisie après comparaison avec v3 `1 + err^γ` (multiplicateur ~1.01 pour γ=1 → quasi-inactif). Le `+0.05` shift évite que les samples avec `err ≈ 0` aient `w → 0` (focal trop agressif).

Empiriquement v3/v4 ont gagné avec `γ=0` (focal off) → ce n'est pas un levier critique. Mais on garde dans le HPO pour exploration.

### 4.7.5 Binning : 15 × 0.033

Compromis entre :
- **10 bins × 0.05** (v10) : trop grossier, perte de forme sur P_test
- **20 bins × 0.025** (v3) : un bin a 6 samples train → ratio explose
- **15 bins × 0.033** (v11+) : queue protégée par clip=10, ratio max 8.3 sur 117 samples (statistiquement OK)

Aucun bin n'atteint le clip=10 sur le train réel → le clip est juste un safety net. Voir §6.1 "Bin vs continue" pour les alternatives (KDE, etc.) et pourquoi 15 bins est suffisant.

## 5. Audit implémentations (récap validations)

| Composant | Statut | Notes |
|---|---|---|
| `WeightedMSELoss` (Lagrangien) | ✅ OK | sync train/eval avec model fixé |
| `sliced_wasserstein` (OT) | ✅ OK | validé empiriquement, gradient flow OK |
| `mmd_rbf` (MMD) | ✅ OK (dormant) | median heuristic, kernel = exp(-d²/σ²) |
| `GenderDiscriminator` + GRL (DANN) | ✅ OK | actif en HPO v12 |
| `EMAWeightCallback` | ✅ OK (avec fix device) | double-eval skip early |
| `GenderBalancedSampler` | ✅ OK | no-replacement F oversample, all M seen |
| `compute_target_weights` (axes 1/2) | ✅ OK | clip=10 + normalize mean=1 |
| `_split_val_to_match_test_pmf` | ✅ OK | implémente $H_C$ |
| `IsotonicCalibrator` (post-hoc) | ✅ OK | val-side validé, test-side nécessite gender pipeline |
| `MlflowClientCallback` | ✅ OK | rank 0 guard + try/except non-blocking |
| `mlflow_utils.log_*` | ✅ OK | rank 0 guard + try/except |
| LLRD `create_optimizer` | ✅ OK | catch-all `lambda n: True, base_lr` pour leftover params |
| Head bias init via `target_mean` | ✅ OK | `bias = logit(E[Y_train])` ≈ -2.40 |

---

## 6. Hyperparams + ranges HPO v12

### 6.1 Pinned (training section)

| Param | Valeur | Raison |
|---|---|---|
| `per_device_train_batch_size` | 128 | effective 256 en DDP 2 GPUs |
| `per_device_eval_batch_size` | 256 | eval rapide |
| `warmup_ratio` | 0.1 | standard |
| `min_lr_rate` | **0.3** | LR final = 30% du peak (v11 était 0.1, decay trop agressif) |
| `max_grad_norm` | 1.0 | clip standard |
| `early_stopping_patience` | 3 | |
| `bf16` | true | mixed precision |
| `augmentation_level` | light | HFlip + ColorJitter |
| `ema_decay` | 0.999 | running avg ~1000 steps |
| `loss_lambda_init` | 1.0 | start at metric λ |
| `loss_lambda_lr` | 0.5 | gradient ascent step |
| `loss_lambda_max` | 5.0 | cap |
| `loss_lambda_ema` | 0.9 | EMA filter on err_diff |
| `logging_steps` | 25 | MLflow refresh granularity |

### 6.2 Search space HPO

| Param | Range | Note |
|---|---|---|
| `pretrained_source` | {lvd, ibot:...} | catégorique |
| `pooling_type` | {cls, attention_k_query, multihead_attention} | catégorique |
| `axis1_power` | [0, 1] | 0 = pas de rebal Y |
| `axis2_power` | [0, 1] | 0 = pas de rebal gender |
| `feature_fairness` | {none, ot, dann} | **MMD retiré, DANN remis** |
| `ot_lambda` | [0.01, 1.0] log | si OT |
| `adv_lambda` | [0.001, 0.05] log | si DANN |
| `loss_focal_gamma` | [0, 1.5] | 0 = focal off |
| `learning_rate` | **[1e-5, 2e-4]** log | **resserré vs [1.5e-5, 4e-4] qui divergeait** |
| `num_train_epochs` | [6, 12] | |
| `weight_decay` | [0.05, 0.50] | |
| `head_dropout` | [0, 0.4] | |
| `backbone_drop_path_rate` | [0, 0.4] | |
| `layer_decay` | **[0.65, 0.9]** | **resserré vs [0.65, 1.0] — 1.0 = LLRD off causait divergences** |
| `pool_attn_dropout` | [0, 0.3] | v4 range restauré |
| `pool_proj_dropout` | [0, 0.3] | v4 range |
| `loss_query_diversity_lambda` | [0, 0.2] | si attention_k_query |
| `tau_focal_init` | [0.05, 0.3] log | v4 range restauré |
| `tau_diffuse_init` | [1.0, 3.0] | v4 range |
| `n_focal` | [1, 4] | v4 range |
| `n_diffuse` | [1, 4] | v4 range |
| `n_free` | [0, 3] | v4 range |
| `num_heads` | {2, 4, 8, 16} | si multihead |

---

## 7. Empirical findings v11 (analysés depuis `mlflow.db`)

### 7.1 Best score v11 = 0.00186 (Sapiens trial4)

| Param | Valeur |
|---|---|
| `learning_rate` | 2.12e-4 |
| `layer_decay` | 0.87 |
| `final_loss` | 0.0043 |

**Progression** : v11 best (0.00186) **meilleur que v8/v9/v10** (~0.00176) mais encore au-dessus de v4 (0.0012).

### 7.2 Pattern de divergence identifié

| LR | layer_decay | score | final_loss | Verdict |
|---|---|---|---|---|
| 2.12e-4 | 0.87 | **0.00186** | 0.0043 | best |
| 1.46e-4 | 0.84 | 0.00261 | 0.0054 | OK |
| **3.24e-4** | **0.99** | 0.00973 | 0.0322 | divergent |
| **2.66e-4** | **0.98** | 0.00855 | 0.0583 | divergent |
| **2.07e-4** | **0.93** | 0.00865 | 0.0604 | divergent |

**Conclusion** : la combo `LR > 2e-4 + layer_decay > 0.9` (LLRD presque off) cause divergence. Sweet spot : `LR ∈ [1.3e-4, 2.1e-4]` avec `layer_decay ∈ [0.8, 0.87]`.

**Fix v12** : `LR ∈ [1e-5, 2e-4]` + `layer_decay ∈ [0.65, 0.9]` → exclut la zone toxique.

---

## 8. Différences v11 → v12

| Param | v11 | v12 | Raison |
|---|---|---|---|
| `learning_rate` high | 4.0e-4 | **2.0e-4** | v11 divergences à LR > 2e-4 |
| `layer_decay` high | 1.0 | **0.9** | LLRD off + LR haut = divergence |
| `min_lr_rate` | 0.1 | **0.3** | decay cosine trop agressif en v11 |
| `feature_fairness` | [none, ot] | **[none, ot, dann]** | rééquilibre exploration |
| `n_focal` range | [1, 3] | **[1, 4]** | restauré v4 (more capacity) |
| `n_diffuse` range | [1, 3] | **[1, 4]** | restauré v4 |
| `n_free` range | [0, 2] | **[0, 3]** | restauré v4 |
| `pool_attn_dropout` max | 0.5 | **0.3** | v11 sur-régularisait |
| `pool_proj_dropout` max | 0.5 | **0.3** | idem |
| `tau_focal_init` low | 0.08 | **0.05** | v4 permettait très sharp |
| `tau_diffuse_init` high | 4.0 | **3.0** | trop diffus inutile |
| `logging_steps` | 50 | **25** | refresh MLflow plus rapide |

---

## 9. Bug fixes pushed (v11 et v12)

Tous ces fixes touchent le code Python (mutualisé), donc bénéficient v11 et v12 :

1. **Head bias init** : `nn.init.zeros_` → `constant_(logit(E[Y]))` (≈ -2.40 pour notre data)
2. **EMA device migration** : lazy `buf.to(p.device)` au premier `on_step_end`
3. **EMA double-eval** : skip pour `epoch < num_epochs/2`
4. **`loss_fct.train(model.training)`** : sync explicite dans `compute_loss` → évite DDP deadlock pendant eval
5. **DDP `find_unused_parameters=False`** quand DANN désactivé → -5-10% step time
6. **DDP `all_reduce(err_diff)`** dans Lagrangien → λ_adapt synchronisé entre ranks
7. **Custom `GenderBalancedSampler`** sans replacement → tous les M vus chaque epoch
8. **MLflow rank 0 guard** : `_is_rank_zero()` dans `mlflow_utils.log_*` + `MlflowClientCallback`
9. **MLflow try/except non-blocking** : sqlite contention → warning, pas hang
10. **`output_dir → /tmp`** : évite NFS save lent (1-5 min) → /tmp local SSD (< 5 sec)
11. **`PYTHONUNBUFFERED=1`** dans SLURM → logs en temps réel
12. **`logging_steps=25`** : plus de logs visibles dans MLflow UI

---

## 10. Pipeline MID + Sapiens (à implémenter)

Pour le post-hoc isotonic à l'inférence test :

1. **MID lookup** : parser le `filename` → extraire `m.XXXXX` → lookup dans mapping `MID → gender` construit depuis `train.csv`. Coverage : **93.5%** du test set.
2. **Sapiens linear probe** : pour les 6.5% MIDs inconnus, entraîner un linear probe sur features Sapiens pour prédire le gender.
3. **Application** : `gender_test = mid_lookup(filename) or sapiens_probe(image)`, puis `y_calibrated = isotonic_per_gender.transform(y_pred, gender_test)`.

Script `scripts/estimate_test_gender.py` fait déjà le MID lookup (validation $H_C$). Reste à intégrer dans `predict.py` + entraîner le Sapiens probe.

---

## 11. Références théoriques

| Concept | Référence |
|---|---|
| MMD + RBF kernel | Gretton et al. 2012, "A Kernel Two-Sample Test" |
| Median heuristic bandwidth | Garreau et al. 2017 |
| DANN + GRL | Ganin & Lempitsky 2015, "Domain-Adversarial Training" |
| Sliced Wasserstein | Bonneel et al. 2015, "Sliced and Radon Wasserstein Barycenters" |
| Iterative Proportional Fitting | Deming & Stephan 1940, Sinkhorn-Knopp 1967 |
| Lagrangian fairness | Cotter et al. 2019 ; Augmented Lagrangian methods |
| EMA Polyak averaging | Polyak 1990 |
| LLRD | BERT/ELECTRA/ViT fine-tuning papers |
| Isotonic calibration | Zadrozny & Elkan 2002 |
| Perceiver IO (cross-attn queries) | Jaegle et al. 2021 |
| Group fairness impossibility | Chouldechova 2017 |

Voir [references.md](references.md) pour bibliographie complète.
