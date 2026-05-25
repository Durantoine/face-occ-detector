# Revue des méthodes d'équilibrage et d'adaptation de domaine

Ce document fait l'inventaire des mécanismes actuellement implémentés dans le repo, clarifie ce qu'ils corrigent vraiment (et ce qu'ils ne corrigent pas), puis liste les familles de méthodes manquantes avec une recommandation pour le search space v3.

---

## 1. Cadre : deux problèmes distincts

Il est crucial de distinguer **deux objectifs orthogonaux** qui sont souvent confondus :

### 1.1. Distribution shift train → test (sur la cible Y)

Le test set a une PMF d'occlusion *différente* du train set (vu dans [src/utils/losses.py:6-9](src/utils/losses.py#L6-L9), `_TEST_PMF_0025`). L'objectif est d'estimer correctement la métrique sur la distribution test, et idéalement d'entraîner un modèle qui performe bien sur cette distribution.

→ Méthodes : importance reweighting basé sur le ratio `p_test(y) / p_train(y)`.

### 1.2. Fairness / invariance démographique (entre groupes G)

Indépendamment de la distribution de Y, on veut que l'erreur soit **comparable entre hommes et femmes** (et non pas une moyenne globale qui cache un déséquilibre). C'est une contrainte de robustesse cross-groupe.

→ Méthodes : balanced sampling, gender reweighting, group-DRO, fairness penalty (|err_F − err_M|).

**Confusion fréquente** : « le sampler par genre corrige le distribution shift » → faux. Le sampler par genre corrige le **déséquilibre de représentation** entre groupes au sein du train ; il ne corrige pas le shift sur Y.

---

## 2. Briques actuellement implémentées

### 2.1. Sampler (sélection des batches)

`make_sampler_keys` ([src/utils/losses.py:191](src/utils/losses.py#L191)) + `create_balanced_sampler` ([src/data/dataset.py:18](src/data/dataset.py#L18)) :

| `sampler_strategy` | Clé d'équilibrage | But |
|---|---|---|
| `none` | (pas de sampler) | Aucun rééquilibrage côté data |
| `gender` | 0/1 | Égaliser ratio F/M par batch |
| `occlusion` | bucket(y, 10 bins) | Égaliser couverture des niveaux d'occlusion (proxy pour distribution shift sur Y) |
| `gender_x_occ` | gender × bucket(y) | Égaliser conjointement (F/M) × (niveau d'occlusion) |

**Coût** : O(1) à l'init. Pas d'overhead à l'entraînement. Inconvénient : le `WeightedRandomSampler` *réduit le nombre d'échantillons par epoch* (`n_samples = min_group_count × n_groups`), donc on perd potentiellement de la donnée.

### 2.2. Reweighting dans la loss (`WeightedMSELoss`, [src/utils/losses.py:56](src/utils/losses.py#L56))

Trois flags orthogonaux qui se combinent multiplicativement avec le poids de base `w = 1/30 + y` (poids cible de la métrique) :

| Flag | Multiplicateur | But |
|---|---|---|
| `loss_importance_reweight` | `w ×= p_test(y_bin) / p_train(y_bin)` (clipé à [0.1, 10]) | Corrige le shift Y, expose le train à la distribution test |
| `loss_gender_reweight` | `w ×= class_weight[gender]` (∝ 1/count) | **No-op** dans notre loss fairness-aware : le facteur constant par genre se factorise dans $\overline{e}_F$ et $\overline{e}_M$ (preuve dans le README, §3). |
| `loss_cell_reweight` | `w ×= 1/√count[gender, bin]` | Combine genre × bin d'occlusion (proche de stratify) |

**Coût** : O(B) par batch, négligeable.

### 2.3. Fairness penalty (`loss_fairness_lambda`)

Implémentée dans `WeightedMSELoss.forward` ([src/utils/losses.py:134](src/utils/losses.py#L134)) :
```
loss = (err_F + err_M) / 2 + λ · |err_F − err_M|
```

C'est une **pénalité L1 sur le gap inter-groupe**. Λ = 0 → simple moyenne (mais déséquilibrée puisque les groupes ont des effectifs différents). λ grand → force `err_F ≈ err_M`, au risque de dégrader la moyenne.

**Coût** : O(B). Effet : pousse vers l'invariance démographique au niveau de l'erreur.

### 2.4. Group-DRO (`GroupDROLoss`, [src/utils/losses.py:137](src/utils/losses.py#L137))

```
loss = (1−α) · mean(err_g) + α · max(err_g)
```

α = 0 → ERM (moyenne); α = 1 → pessimiste pur (Sagawa et al. 2019). Activé via `loss_type: group_dro`. Implémentation actuelle : worst-group computation par batch, sans EMA sur les poids de groupes ni adversaire dual (la version « online » de Group-DRO).

**Coût** : O(G × B), G = 2 ici. Négligeable.

### 2.5. Eval reweighting (`eval_importance_reweight`)

Au moment de l'évaluation, le `eval_score` est reweighté avec `p_test/p_train` (`compute_metrics`). C'est un **estimateur honnête de la métrique sur le test set** à partir du val set. Ne change pas l'entraînement, juste la métrique reportée à Optuna.

---

## 3. Stratégies A, D, E, F (présentes) — et B, C (absentes)

Dans [src/optimize.py:59-65](src/optimize.py#L59-L65) :

| ID | Sampler | importance_rw | gender_rw | cell_rw | Hypothèse / cas d'usage |
|---|---|---|---|---|---|
| **A** | gender | ✓ | ✗ | ✗ | Sampler équilibre genres → importance loss corrige le shift Y. **Cleanest baseline** : data et loss agissent sur deux dimensions différentes. |
| **D** | none | ✓ | ✓ (no-op) | ✗ | Pas de réduction d'effectif via sampler. `gender_rw=T` est un no-op (cf. README), donc D ≡ "no sampler + importance_rw seul" en pratique. |
| **E** | none | ✗ | ✗ | ✓ | Cell reweight = 1/√count(g,y) → équivalent stratifié train×gender mais en soft. Pas de correction explicite du shift test. |
| **F** | occlusion | ✗ | ✓ | ✗ | Sampler stratifie sur Y (proxy pour shift), loss compense le genre. |

**B et C n'existent pas.** Probable trou dans la nomenclature historique. À tracer dans le git log si l'historique est utile.

**Discussion rapide** (voir README §"Train↔test distribution shift" pour le détail mathématique) :
- **A** = théoriquement propre (sampler équilibré + shift Y corrigé). **Mais** : sampler gender coûte ~20-40% des données par epoch.
- **D** = `loss_gender_reweight` est un **no-op** dans notre formule (cf. README) puisque le facteur $c_G$ se factorise dans la moyenne par-groupe. Donc D ≡ "no sampler + importance_rw seul" en pratique. Garde toutes les données → empiriquement meilleure que A.
- **E** = `loss_cell_reweight` est l'unique vrai reweighting joint $Y \times G$ (ne se factorise pas car varie avec le bin). Pas d'estimation explicite de $P_{\text{test}}$ par contre.
- **F** = ressemble à A inversée (sampler Y, loss G). Sampler `occlusion` peut perdre encore plus de données.

---

## 4. Familles de méthodes manquantes

Ces familles ne sont **pas implémentées** mais pertinentes au problème.

### 4.1. Adversarial debiasing / DANN

**Idée** : un discriminateur tente de prédire le genre (ou un attribut nuisible) à partir des features ; un gradient reversal layer pousse le backbone à produire des features *non-prédictives* du genre. Référence : Ganin & Lempitsky 2015 (DANN), Zhang et al. 2018 (adversarial debiasing).

- **Corrige** : invariance démographique au niveau des **features** (plus profond que `fairness_lambda` qui agit sur l'erreur).
- **Coût** : modeste (+1 tête, +1 forward sur features). Convergence parfois instable, nécessite warmup et tuning du coefficient λ_adv.
- **Pertinence pour notre tâche** : forte. Une caractéristique d'occlusion *physique* est invariante au genre, donc forcer des features invariantes au genre est plausible. Risque : si l'occlusion est elle-même corrélée au genre dans le train (ex. plus d'hommes avec lunettes), DANN peut sur-pénaliser.

### 4.2. MMD / Wasserstein alignment

**Idée** : aligner les distributions de features entre groupes via une mesure de distance entre distributions :
- **MMD (Maximum Mean Discrepancy)** : Long et al. 2015. Différence de moyennes dans un RKHS. O(B²) par batch dans le kernel trick.
- **Wasserstein** : Shen et al. 2018. Distance optimale transport entre distributions de features. Plus coûteux mais théoriquement plus solide.

```
loss = task_loss + λ · MMD(features | gender=F, features | gender=M)
```

- **Corrige** : alignement géométrique des features entre groupes.
- **Coût** : MMD ~O(B²·D), Wasserstein ~O(B³·D) ou O(B²·D·iters) avec Sinkhorn.
- **Pertinence** : intermédiaire. Plus stable que DANN, mais moins flexible (force tout l'espace de features à être aligné, alors que seul certains axes devraient l'être).

### 4.3. IRM (Invariant Risk Minimization)

**Idée** : Arjovsky et al. 2019. Apprendre un prédicteur dont le risque est invariant à travers les environnements (groupes). Pénalité : `||∇_w R^g(w)||²` doit être nul pour tout g.

- **Corrige** : invariance causale (les features apprises capturent les causes, pas les corrélations spurieuses).
- **Coût** : modeste (gradient de gradient).
- **Pertinence** : forte conceptuellement, mais en pratique IRM peut être instable et difficile à tuner. Plusieurs critiques (Rosenfeld et al. 2020) montrent qu'il ne dépasse pas ERM dans beaucoup de setups.

### 4.4. Contrastive group-aware (SupCon variants)

**Idée** : Khosla et al. 2020 (SupCon) + variantes group-aware. Apprendre des features où :
- Même occlusion + même gender → proche (positives strictes)
- Même occlusion + gender différent → proche (positives invariantes au genre)
- Occlusion différente → loin (négatives)

Forme un objectif auxiliaire sur les embeddings du pooling.

- **Corrige** : structure de l'espace de features (invariance par groupe, séparation par cible).
- **Coût** : moyen (besoin de batch large, ou banque de prototypes).
- **Pertinence** : élégante mais demande un peu d'ingénierie (paires positives/négatives). Probablement overkill pour ce projet.

### 4.5. Distribution Matching via Sample Mixing (Mixup variants)

- **Mixup** (Zhang et al. 2018) inter-genre : interpole entre une image F et une image M de même niveau d'occlusion → label = même niveau, gender = mix.
- **GroupMix** / **ManifoldMix** : variantes sur les features intermédiaires.

- **Corrige** : à la fois data augmentation et invariance par groupe.
- **Coût** : nul (juste interpolation linéaire).
- **Pertinence** : intéressante pour notre tâche, simple à implémenter, et complémentaire des autres méthodes.

### 4.6. Two-stage Calibration

Entraîner d'abord sur ERM, puis ajuster la dernière couche (logit adjustment ou re-balanced classifier) pour corriger les biais sur la distribution test. Référence : Menon et al. 2021 (logit adjustment), Kang et al. 2020 (decoupled representation/classifier).

- **Corrige** : distribution shift de Y au niveau du classifier seulement.
- **Coût** : ~10% du training (juste la dernière couche réentraînée).
- **Pertinence** : élégante pour le shift Y ; pour la fairness, moins direct.

---

## 5. Recommandations pour le search space v3

### 5.1. Garder pour la baseline solide
- **A, D, E, F** (déjà en place). Les conserver toutes pour A/B/C/D testing.

### 5.2. Nouveaux IDs à ajouter (prioritaires)

| ID | Description | Effort impl. | Risque |
|---|---|---|---|
| **G** | DANN (adversarial debiasing sur gender) | ★★☆ (tête adversaire + GRL + λ_adv warmup) | Convergence instable |
| **H** | MMD alignment inter-genre sur les features pooled | ★☆☆ (RBF kernel sur batch features) | Faible, peu de tuning |
| **I** | Inter-gender Mixup (mix F↔M même bucket Y) | ★☆☆ (DataCollator custom) | Faible |

### 5.3. À écarter pour v3

- **IRM** : trop instable empiriquement, ROI questionnable
- **Contrastive group-aware** : trop d'ingénierie, gain marginal vs DANN/MMD
- **Two-stage calibration** : à explorer hors Optuna (post-hoc sur le best trial)

### 5.4. Search space proposé

```yaml
balancing_strategy:
  type: categorical
  choices: [A, D, E, F, G, H, I]

# Conditional hyperparams
adv_lambda:                    # G uniquement
  conditional_on: { balancing_strategy: G }
  type: float
  low: 0.01
  high: 1.0
  log: true

mmd_lambda:                    # H uniquement
  conditional_on: { balancing_strategy: H }
  type: float
  low: 0.01
  high: 1.0
  log: true

mixup_alpha:                   # I uniquement
  conditional_on: { balancing_strategy: I }
  type: float
  low: 0.1
  high: 0.5
```

---

## 6. Plan d'implémentation (à valider)

1. **G (DANN)** :
   - Ajouter `GenderDiscriminator(features_dim → 2)` dans le regressor (head séparée)
   - Gradient reversal layer (custom autograd Function)
   - Loss = task_loss + λ_adv · CE(disc(features), gender)
   - Wiring dans `WeightedMSETrainer.compute_loss`

2. **H (MMD)** :
   - Fonction `mmd_rbf(feat_F, feat_M, sigmas=[1, 5, 10])` standalone
   - Récupérer les features pooled via `outputs["features"]` (à exposer dans le regressor)
   - Pénalité ajoutée dans compute_loss

3. **I (Mixup inter-genre)** :
   - Custom `data_collator` ou wrapper dans `compute_loss` qui interpole les batches
   - Sample des paires F-M dans le même bucket de Y
   - Compatible avec `WeightedMSELoss` (label-mixed)

Chacune se branche dans `_BALANCING_STRATEGY_MAP` avec ses flags propres. Le `compute_loss` du trainer fait le dispatch selon les flags actifs.

---

## 7. Distinction finale : ce qui adresse quoi

| Méthode | Distribution shift Y | Fairness G | Both |
|---|---|---|---|
| `loss_importance_reweight` | ✓ | ✗ | |
| `sampler_strategy: occlusion` | ~ (partiel) | ✗ | |
| `loss_gender_reweight` | ✗ | ✓ | |
| `loss_cell_reweight` | ✗ | ✗ | ✓ |
| `sampler_strategy: gender_x_occ` | ✗ | ✗ | ✓ |
| `loss_fairness_lambda` | ✗ | ✓ | |
| `GroupDROLoss` | ✗ | ✓ | |
| **G — DANN** | ✗ | ✓ | |
| **H — MMD** | ✗ | ✓ | |
| **I — Mixup inter-G** | ✗ | ✓ | |
| `eval_importance_reweight` (eval seul) | ✓ | ✗ | |

→ **Trou notable** : aucune méthode purement « distribution shift Y » au-delà de `loss_importance_reweight`. Pour explorer ce trou, on pourrait ajouter du **Mixup intra-genre sur Y** (interpoler deux samples du même genre mais buckets Y différents) — à considérer si E + A ne suffisent pas.
