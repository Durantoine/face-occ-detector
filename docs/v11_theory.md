# v11 — Théorie complète

Document théorique de la version 11. Couvre : métrique du concours, découverte du shift de distribution train↔test (via MID lookup), validation rigoureuse des hypothèses de transport, et tous les mécanismes implémentés (5 étages d'intervention).

Pour l'historique antérieur (v3 baseline, v10 régression), voir [audits_and_roadmap.md](audits_and_roadmap.md). Pour les détails d'architecture et de pooling, voir [architecture.md](architecture.md).

---

## 1. Notation et métrique du concours

### 1.1 Variables

- $X$ : image d'un visage (entrée du modèle)
- $Y \in [0, 1]$ : taux d'occlusion réel (cible de régression)
- $G \in \{F, M\}$ : genre de la personne (sensible attribute)
- $\hat{Y} = f_\theta(X)$ : prédiction du modèle, paramètres $\theta$
- $w_i = \frac{1}{30} + y_i$ : poids du sample $i$ dans la formule officielle

### 1.2 Métrique du concours

$$
\text{Err}_g = \frac{\sum_{i \in g} w_i \cdot (\hat{y}_i - y_i)^2}{\sum_{i \in g} w_i}, \quad g \in \{F, M\}
$$

$$
\text{Score} = \frac{\text{Err}_F + \text{Err}_M}{2} + |\text{Err}_F - \text{Err}_M|
$$

**Propriétés clés** :

1. **Métrique par-groupe** : pondère $\text{Err}_F$ et $\text{Err}_M$ à égalité, **indépendamment** du nombre de samples F ou M dans le test set.
2. **Pondération $w_i$** : plus un sample a une occlusion élevée, plus il compte. Évite que la queue rare soit ignorée.
3. **Coefficient $\lambda_{\text{metric}} = 1$** sur $|\text{Err}_F - \text{Err}_M|$ : composante "fairness" pondérée à égalité avec la moyenne d'erreur.

### 1.3 Pourquoi cette métrique est intéressante

- Standard MSE pondérerait par sample → minorité (F = 32% train) sous-représentée
- Métrique par-groupe force le modèle à être bon sur **les deux** groupes
- Terme $|\text{Err}_F - \text{Err}_M|$ pénalise les modèles biaisés (un groupe sacrifié pour l'autre)

---

## 2. Distributions train et test

### 2.1 Données disponibles

| Quantité | Source | Statut |
|---|---|---|
| $P_{\text{train}}(G, Y)$ | `train.csv` (96k samples, $G$ et $Y$ connus) | **Mesuré directement** |
| $P_{\text{test}}(Y)$ | PDF page 3, ré-extrait via reading pixel | **Mesuré** (15 bins × 0.033) |
| $P_{\text{test}}(G)$ | MID lookup (93.5% coverage du test set) | **Estimé** ($\approx 0.477$ F) |
| $P_{\text{test}}(G, Y)$ | — | **Inconnu** (Y non labellisé en test) |

### 2.2 Observations critiques

#### A. Shift sur la marginale gender

| | Train | Test (MID) | Delta |
|---|---|---|---|
| $P(F)$ | 32.4% | **47.7%** | +15.3 pts |
| $P(M)$ | 67.6% | 52.3% | -15.3 pts |

**Conséquence apparente** : le test a presque autant de femmes que d'hommes, alors que le train est très déséquilibré.

**Mais** : la métrique est par-groupe, donc cette différence de proportion n'affecte PAS la formule métrique. Elle affecte seulement la variance d'estimation de $\text{Err}_F$ et $\text{Err}_M$.

#### B. Shift sur la marginale Y

`P_test(Y)` est plus piqué autour de $Y \in [0.05, 0.25]$ que `P_train(Y)` (qui est long-tail vers 0). Les hommes sans occlusion (rasés, sans cheveux, sans maquillage) dominent le bin $Y \approx 0$ en train (52% du bin 0 est M, vs 12% F).

#### C. Dépendance gender ↔ occlusion en train

`P_train(F|Y)` varie fortement avec $Y$ :

| Bin Y | $P_{\text{train}}(F\|Y)$ | Interprétation |
|---|---|---|
| [0.00, 0.03] | **12%** | Hommes dominent "pas d'occlusion" |
| [0.10, 0.17] | 55-56% | Femmes commencent à dominer |
| [0.30, 0.40] | **66-74%** | Femmes dominent (cheveux longs, maquillage, voile, etc.) |

Cette dépendance reflète des **patterns biologiques/sociologiques réels** (les femmes ont plus souvent des cheveux longs qui occluent le visage, plus de maquillage qui modifie la peau, etc.).

---

## 3. Hypothèses de distribution test

Comme $P_{\text{test}}(G, Y)$ est inconnu, on doit faire une hypothèse pour pouvoir :
1. Construire un val split représentatif de test
2. Calibrer les mécanismes de rebalancing

Trois hypothèses candidates :

| Nom | Formulation | Sens |
|---|---|---|
| **$H_A$** | $G \perp Y$ dans test | "Le gender est indépendant de l'occlusion en test" |
| **$H_B$** | $P_{\text{test}}(Y\|G) = P_{\text{train}}(Y\|G)$ | "Les patterns d'occlusion par gender sont stables train→test" |
| **$H_C$** | $P_{\text{test}}(G\|Y) = P_{\text{train}}(G\|Y)$ | "À occlusion donnée, le ratio F/M est stable train→test" — **= "H1 covariate shift Y-only" du projet historique, cf. [fairness.md](fairness.md)** |

### 3.1 Test rigoureux de $H_A$ (g ⊥ y)

Si $H_A$, alors $P_{\text{test}}(G, Y) = P_{\text{test}}(G) \times P_{\text{test}}(Y)$ → notamment $P_{\text{test}}(F | Y) = P_{\text{test}}(F) = 0.48$ uniformément.

**Rejeté a priori** : on observe en train que $P(F|Y)$ varie de 12% à 74% selon le bin. Il serait surprenant que cette dépendance disparaisse au test alors qu'elle reflète des invariants biologiques (cheveux, maquillage).

### 3.2 Test rigoureux de $H_B$ (P(Y|G) conservé)

Si $H_B$, alors par la loi des probabilités totales :
$$
P_{\text{test}}(Y) = \sum_g P_{\text{test}}(Y|g) \cdot P_{\text{test}}(g) = \sum_g P_{\text{train}}(Y|g) \cdot P_{\text{test}}(g)
$$

**Calcul empirique** (sur 15 bins) :

```
P_test_y_implied_HB = 0.48 × P_train(y|F) + 0.52 × P_train(y|M)
```

- L1 distance vs `P_test(y)` PDF observé : **0.6676**
- Baseline (juste `P_train(y)` vs `P_test(y)`) : 0.7714
- Amélioration relative : **seulement 10.4%**

**Verdict** : $H_B$ est **FAUX**. Les distributions Y par gender ne sont pas conservées train↔test.

### 3.3 Test rigoureux de $H_C$ (P(G|Y) conservé)

Si $H_C$, alors :
$$
P_{\text{test}}(G) = \sum_y P_{\text{test}}(G|y) \cdot P_{\text{test}}(y) = \sum_y P_{\text{train}}(G|y) \cdot P_{\text{test}}(y)
$$

**Calcul empirique** :

```
P_test_F_implied_HC = Σ_y P_train(F|y) × P_test(y) = 0.4816
P_test_F_observed (MID lookup)                     = 0.4766
Écart                                              = 0.0050  (0.5 points)
```

**Verdict** : $H_C$ est **COHÉRENTE** (écart < 1 point). C'est l'hypothèse retenue.

### 3.4 IPF — confirmation par maximum-entropy

L'Iterative Proportional Fitting (Sinkhorn-Knopp) trouve $P_{\text{test}}(G, Y)$ le plus proche d'un prior donné (en KL divergence) qui respecte les marginales observées.

Algorithme :

```
M ← P_train(g, y)              # prior
repeat:
  M[g, :] ← M[g, :] × P_test(g) / sum(M[g, :])    # force marginale gender
  M[:, y] ← M[:, y] × P_test(y) / sum(M[:, y])    # force marginale Y
until convergence
```

**Résultat avec prior $P_{\text{train}}$** :

| | Warp max nécessaire |
|---|---|
| Conditionnelle $P(G\|Y)$ | **0.56%** (presque rien) |
| Conditionnelle $P(Y\|G)$ | **31.8%** (énorme) |

→ L'IPF "naturellement" conserve $P(G|Y)$ et modifie $P(Y|G)$. C'est exactement ce que dit $H_C$.

**Subtilité importante** : ce résultat IPF est **partiellement circulaire** (le prior est `P_train`, l'IPF y converge naturellement si les marginales sont compatibles). Le vrai test indépendant de $H_C$ est le calcul de la section 3.3 : il **n'utilise pas l'IPF** mais juste 3 quantités observées indépendamment (`P_train(F|y)`, `P_test(y)`, `P_test(F)`).

### 3.5 Pourquoi $H_C$ et pas $H_B$, intuitivement

$H_C$ dit : "le test est un re-mixage de train avec poids par bin $Y$, à chaque bin on conserve les proportions F/M". Ce qui change train→test, c'est la **fréquence** de chaque niveau d'occlusion, pas la composition gender à un niveau donné.

C'est cohérent avec l'idée que le test set a été samplé pour avoir une distribution Y "plus utile" pour l'évaluation (plus de masse sur les Y moyens où la régression est challenging), sans changer les patterns biologiques sous-jacents.

---

## 4. Val split — implications de $H_C$

### 4.1 Construction du val

Notre val split applique $H_C$ par cell $(g, y)$ :

$$
P_{\text{val}}(g, y) = P_{\text{train}}(g|y) \cdot P_{\text{test}}(y)
$$

Implémenté dans [`src/data/dataset.py::_split_val_to_match_test_pmf`](../src/data/dataset.py).

### 4.2 Conséquences observées

- val a ~48% F (auto-rebalancé vs train 32% F, par le mix avec $P_{\text{test}}(y)$)
- val a la même marginale Y que test (par construction)
- val est **représentative de test** sous $H_C$

### 4.3 Pourquoi val a 48% F sans qu'on l'ait demandé explicitement

Mathématiquement : $P_{\text{val}}(F) = \sum_y P_{\text{train}}(F|y) \cdot P_{\text{test}}(y) = 0.4816$. C'est **automatique** du mix.

Intuition : on tire plus d'échantillons des bins Y moyens-hauts (où $P_{\text{test}}(y)$ a plus de masse), et ces bins sont dominés par F en train (P_train(F|y) ≈ 55-74%). Résultat : plus de F dans val.

---

## 5. Mécanismes implémentés — 6 étages

L'approche v11 attaque le problème à **6 étages distincts**, chacun avec un rôle non-redondant.

### 5.1 Sample-level : rebalancing target unifié

Implémenté dans [`losses.compute_target_weights`](../src/utils/losses.py).

**Cible** : $P_{\text{target}}(g, y) = \text{mix}_y(\alpha_1) \times \text{mix}_g(\alpha_2)$

Avec :
- $\text{mix}_y(\alpha_1) = (1-\alpha_1) \cdot P_{\text{train}}(y) + \alpha_1 \cdot P_{\text{test}}(y)$  ← **axe 1**
- $\text{mix}_g(\alpha_2) = (1-\alpha_2) \cdot P_{\text{train}}(g|y) + \alpha_2 \cdot 0.5$  ← **axe 2**

**Poids per-sample** : $w_i = P_{\text{target}}(g_i, y_i) / P_{\text{train}}(g_i, y_i)$, clippé à $[1/10, 10]$, puis **normalisé pour que mean = 1** (truc v3 critique, voir section 6.2).

#### Axe 1 : axis1_power $\in [0, 1]$

- $\alpha_1 = 0$ : pas de re-balancing (target = $P_{\text{train}}(y)$)
- $\alpha_1 = 1$ : target = $P_{\text{test}}(y)$ exactement (re-balancing maximum)
- Continu entre les deux

#### Axe 2 : axis2_power $\in [0, 1]$

- $\alpha_2 = 0$ : ratio F/M par bin conservé (= conditionnel train)
- $\alpha_2 = 1$ : ratio F/M par bin forcé à 50/50

**Note avec $H_C$** : on a vu que test a $P_{\text{test}}(F) \approx 0.48 \approx 0.5$. Donc axe 2 → uniform est **bien calibré** par rapport à la réalité test (l'écart 50/50 vs 48/52 est négligeable).

### 5.2 Batch-level : gender-balanced sampler

Implémenté dans [`dataset.GenderBalancedSampler`](../src/data/dataset.py).

**Principe** : à chaque epoch, échantillonner $n = 2 \cdot \max(n_F, n_M)$ samples tels que :
- Tous les samples M sont vus exactement une fois
- Les samples F sont oversamplés (sans remplacement par passe) jusqu'à atteindre $\max(n_F, n_M)$

**Différence avec WeightedRandomSampler standard** : pas de replacement aléatoire qui ferait perdre ~50% des samples M. Garantie statistique forte sur la couverture.

**Quand activé** : seulement si `feature_fairness != none`. Raison : OT, MMD, DANN ont besoin de samples F+M **physiquement présents par batch** pour calculer leurs statistiques croisées.

### 5.3 Feature-level : alignement distributionnel

Implémenté dans [`losses.sliced_wasserstein`](../src/utils/losses.py) et `losses.mmd_rbf`.

**Objectif** : forcer $P(\text{features}|F) \approx P(\text{features}|M)$ dans l'espace latent du backbone.

Trois mécanismes possibles (mutuellement exclusifs, car ils visent la même cible) :

| Méthode | Mécanisme | Référence |
|---|---|---|
| **MMD** | Distance RBF kernel entre stats F vs M, bandwidth via median heuristic | Gretton et al. 2012 |
| **DANN** | Adversarial : gender discriminator + Gradient Reversal Layer | Ganin & Lempitsky 2015 |
| **OT (sliced Wasserstein-2)** | Projection sur $n$ directions 1D, Wasserstein 1D = L2 entre quantiles | Bonneel et al. 2015 |

**Choix v11** : `feature_fairness ∈ {none, ot}` — on garde uniquement OT (le plus défensable théoriquement, différentiable, batch-agnostic).

**Raison du choix exclusif** : combiner MMD + DANN ou MMD + OT crée deux signaux gradient sur la même cible avec des échelles différentes → ils se "battent" plutôt que se compléter.

### 5.4 Loss-level : Lagrangien adaptatif

Implémenté dans [`losses.WeightedMSELoss`](../src/utils/losses.py) (`forward()` method).

#### Formulation

**Loss de training** :
$$
\mathcal{L}_{\text{train}}(\theta, \lambda) = \frac{\text{Err}_F + \text{Err}_M}{2} + \lambda \cdot |\text{Err}_F - \text{Err}_M|
$$

C'est un problème min-max : minimiser sur $\theta$, maximiser sur $\lambda$ (pour qu'$\lambda$ enforce la contrainte de fairness).

**Update Lagrangien** (gradient ascent sur $\lambda$) :
$$
\lambda_{t+1} = \text{clip}\left(\lambda_t + \eta_\lambda \cdot \text{EMA}_t(|\text{Err}_F - \text{Err}_M|),\ 0,\ \lambda_{\max}\right)
$$

Avec :
- $\eta_\lambda = 0.5$ (lambda_lr) — step size du GA
- $\lambda_{\max} = 5.0$ — cap pour éviter divergence
- EMA decay = 0.9 (filter passe-bas sur la violation per-batch, lisse sur ~10 steps)
- $\lambda_0 = 1.0$ — démarre à la valeur métrique officielle

**DDP** : `all_reduce(err_diff, op=AVG)` avant l'update pour synchroniser $\lambda$ entre ranks.

#### Métrique officielle vs loss training

**Cruciale séparation** :
- **Training** : utilise $\lambda_{\text{adapt}}$ (peut être supérieur à 1 si la fairness est difficile à atteindre)
- **Évaluation / métrique loggée** : forcée à $\lambda = 1.0$ (formule officielle du concours)

Implémenté via `if self.training: lam = lambda_adapt else: lam = 1.0` dans le forward.

Conséquence : `metric_for_best_model = eval_challenge_score` (calculé par [`metrics.compute_score`](../src/utils/metrics.py) avec λ=1.0) est **toujours la vraie métrique du concours**, indépendamment de la dynamique $\lambda_{\text{adapt}}$.

#### Pourquoi le Lagrangien et pas $\lambda$ fixe ?

- $\lambda = 1.0$ fixe (métrique officielle) : équilibre prédéfini fairness vs accuracy
- $\lambda$ adaptatif : équilibre **appris** selon la trajectoire d'optimisation. Si la fairness est facile, $\lambda$ reste bas et on optimise plus d'accuracy ; si la fairness est difficile, $\lambda$ monte et force la contrainte.

C'est la même intuition que LBFGS / Augmented Lagrangian methods : $\lambda$ s'ajuste automatiquement aux violations observées.

### 5.5 Output-level : post-hoc isotonic calibration

Implémenté dans [`inference.isotonic.GenderConditionalIsotonic`](../src/inference/isotonic.py).

**Principe** : après training, fit une régression isotonique séparée par gender sur (pred, gt) du val set. Apply à l'inférence test.

Régression isotonique = monotone non-paramétrique, garantit que l'ordre relatif des prédictions est conservé (donc R²/AUC inchangé), mais peut corriger des biais de calibration sample-conditionnels.

**Poids** : utilise $w_i = 1/30 + y_i$ (la pondération officielle) pour fit, donc la calibration optimise directement la métrique.

**Limite à l'inférence** : nécessite le gender au test time. Voir section 7 (pipeline MID + Sapiens).

### 5.6 Optim-level : EMA + LLRD

#### EMA des poids

Implémenté dans [`train.EMAWeightCallback`](../src/train.py).

$$
\theta_{\text{EMA}, t} = \alpha \cdot \theta_{\text{EMA}, t-1} + (1-\alpha) \cdot \theta_t, \quad \alpha = 0.999
$$

**Double evaluation** : à chaque epoch, on évalue à la fois les poids raw ET les poids EMA, on garde le meilleur (par `eval_challenge_score`). Le best checkpoint sauvé contient donc les poids gagnants (raw ou EMA selon le cas).

`load_best_model_at_end=True` charge ensuite le meilleur epoch (raw ou EMA, selon le gagnant à cet epoch).

#### LLRD — Layer-wise Learning Rate Decay

Implémenté dans `WeightedMSETrainer.create_optimizer()`.

$$
\text{lr}_l = \text{base\_lr} \cdot (\text{layer\_decay})^{n_{\text{layers}} - l}
$$

Plus une couche est profonde (proche de l'entrée), plus son LR est petit. Embeddings ont LR $\cdot \text{decay}^{n+1}$ ; head a LR $\cdot \text{decay}^0 = \text{base\_lr}$.

Standard pour fine-tuner les ViT pré-entraînés sans détruire les features bas-niveau.

---

## 6. Choix de design critiques

### 6.1 Head bias init via $\text{logit}(\mathbb{E}[Y])$

Implémenté dans [`face_occ_regressor.FaceOccRegressor._init_weights`](../src/models/face_occ_regressor.py).

**Bug v3-v10** : `nn.init.zeros_(self.head.bias)` → $\sigma(0) = 0.5$, mais $\mathbb{E}[Y_{\text{train}}] \approx 0.085$. Modèle démarre avec un offset de +0.4 sur les prédictions → gradient initial dominé par cet offset → ~1-2 epochs gaspillés à juste compenser.

**Fix** : init `bias = logit(target_mean) ≈ -2.40`, donc $\sigma(\text{bias}) \approx 0.085 = \mathbb{E}[Y_{\text{train}}]$.

### 6.2 Normalisation $w / w.\text{mean}()$ (truc v3)

Sans normalisation : si un bin extrême a $w = 8$ et la majorité $w \approx 1$, ces samples dominent la loss. Échelle moyenne de loss varie d'un batch à l'autre → LR effectif change → bruit additionnel sur le gradient.

Avec normalisation : $w / \text{mean}(w)$ garantit $\text{mean}(w) = 1$ toujours. Les ratios relatifs sont préservés, mais l'échelle absolue reste stable.

C'est de l'**hygiène d'échelle** essentielle pour cohérence LR/WD à travers les trials HPO.

### 6.3 P_test PMF — extraction pixel-par-pixel

Implémenté via `pdfplumber` + traitement d'image (cf. `_TEST_PMF` dans [`losses.py`](../src/utils/losses.py)).

**Procédure** :
1. Render PDF page 3 à 300 DPI
2. Détecter les bars bleus (matplotlib alpha=0.5) via masque RGB
3. Calibrer les axes via OCR des labels (8 ticks Y, 6 ticks X)
4. Pour chaque colonne pixel, mesurer la hauteur du top de la bar
5. Convertir hauteur (pixels) → count via calibration Y
6. Agréger sur les 15 bins target

**Résultat** : `_TEST_PMF` 15 valeurs sommant à 1, précision ~±1% sur chaque bin (vs extraction visuelle à ±5%).

### 6.4 Focal weighting

$$
w_i^{\text{focal}} = w_i \cdot (\text{err}_i + 0.05)^\gamma, \quad \gamma \in [0, 1.5]
$$

Forme `(err + 0.05)^γ` choisie après comparaison avec `1 + err^γ` (v3) qui était quasi-inactive (multiplicateur ~1.01 pour γ=1).

`+0.05` shift évite que les samples avec err≈0 aient $w \to 0$.

### 6.5 Binning : 15 × 0.033

Compromis entre :
- **10 bins × 0.05** (v10) : trop grossier, perte d'info sur la forme de $P_{\text{test}}(y)$
- **20 bins × 0.025** (v3) : un bin extrême a 6 samples train → ratio explose (×8.6)
- **15 bins × 0.033** : queue protégée par clip=10, ratio max 8.3 sur 117 samples (statistiquement OK)

Aucun bin n'atteint le clip=10 sur le train réel — le clip est juste un safety net.

---

## 7. Pipeline MID + Sapiens — gender à l'inférence test

### 7.1 Motivation

Le post-hoc isotonic (§5.5) nécessite le gender à l'inférence. Or `test_students.csv` n'a que le `filename`, pas le gender.

### 7.2 Procédure hybride

1. **MID lookup** : extraire `m.xxxxx` du chemin (Freebase Machine ID), lookup dans le mapping `MID → gender` construit depuis `train.csv`. Couverture : **93.5%** du test set.
2. **Sapiens linear probe** : pour les 6.5% restants (MID absent du train), inférer le gender via un linear probe entraîné sur Sapiens features.

### 7.3 Justification de la coverage 93%

Les MIDs (Freebase identifiers) identifient une personne unique. Si la même personne a plusieurs images, elles partagent le MID. Les datasets de visages célèbres (MS1MV3, etc.) sont souvent utilisés pour train + test → forte overlap des MIDs.

### 7.4 Application

```python
gender_test = mid_lookup(filename) or sapiens_probe(image)
y_calibrated = isotonic_per_gender.transform(y_pred, gender_test)
```

À implémenter dans `predict.py` (futur).

---

## 8. Synthèse — recap visuel

### 8.1 Les 6 étages d'intervention

```
            INPUT
              │
              ▼
   ┌──────────────────────┐
   │ 1. Sample weight     │  axis1_power (P_test(y)) + axis2_power (P(g|y) → uniform)
   │    (compute_target_w)│  ← normalize w/w.mean()
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ 2. Batch composition │  GenderBalancedSampler (50/50 F+M par batch, no replacement)
   │    (Sampler)         │  ← activé si feature_fairness != none
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ MODEL  fθ            │  ViT/Sapiens backbone + K-query pool + head
   │  ← LLRD (layer_decay)│  ← head bias init = logit(E[Y])
   │  ← EMA θ_EMA         │
   └──────────────────────┘
              │
       features │ logits
              │
              ▼
   ┌──────────────────────┐
   │ 3. Feature alignment │  Sliced Wasserstein OT entre P(feat|F) et P(feat|M)
   │    (sliced_wasser.)  │  ← option exclusive (pas MMD ni DANN simultané)
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────────┐
   │ 4. Loss : Lagrangian Fairness Loss            │
   │    L = (Err_F + Err_M)/2 + λ_adapt · |diff|   │
   │    λ_adapt updated via GA on EMA(|diff|)      │
   │    ← train: λ_adapt | eval/metric: λ=1.0     │
   │    ← DDP all_reduce on err_diff               │
   └──────────────────────────────────────────────┘
              │
              ▼ (backprop)
       ⊕  θ_EMA tracking via callback
       ⊕  Best model selection : double-eval (raw vs EMA)
              │
              ▼  (post-training)
   ┌──────────────────────┐
   │ 5. Post-hoc          │  IsotonicRegression(weighted) per gender
   │    (isotonic.py)     │  ← uses val (gender known) for fit
   └──────────────────────┘
              │
              ▼  (inference test)
   ┌──────────────────────┐
   │ 6. Gender at test    │  MID lookup (93.5%) + Sapiens probe (6.5%) → gender_test
   └──────────────────────┘
              │
              ▼
        FINAL PREDICTION
```

### 8.2 Complémentarité (pas redondants)

| Étage | Action | Vise quoi | Redondant avec |
|---|---|---|---|
| 1. Sample weight (axe 1) | Re-pondère par bin Y | Distribution Y | rien |
| 1. Sample weight (axe 2) | Re-pondère par cell (g,y) | Distribution F/M intra-Y | partiellement avec 2 |
| 2. Sampler | Composition batch 50/50 | Composition physique batch | partiellement avec 1 (axe 2) |
| 3. Feature OT | Distance Wasserstein F vs M | Distributions des features | (autres options retirées) |
| 4. Lagrangien | Multiplicateur adaptatif sur diff | Métrique agrégée | rien |
| 5. Post-hoc isotonic | Calibration per-gender | Sortie finale | rien |
| 6. EMA / LLRD | Stabilité optim | Trajectoire / fine-tuning | rien |

**Le seul couplage est entre axe 2 et sampler** : tous deux poussent vers balance gender, mais à étages distincts (loss weight vs batch composition). Effets cumulatifs mais distincts. Optuna détermine le mix optimal.

---

## 9. Search space HPO v11

Voir [yaml v11 DINOv3](../configs/architectures/dinov3-vitb16-3090-v11.yaml) et [yaml v11 Sapiens2](../configs/architectures/sapiens2-01b-3090-v11.yaml) pour la liste exhaustive.

Récap des dimensions clés :

| Catégorie | Param | Range | Note |
|---|---|---|---|
| Sample | axis1_power | [0, 1] | 0 = off |
| Sample | axis2_power | [0, 1] | 0 = off |
| Feature | feature_fairness | {none, ot} | exclusif |
| Feature | ot_lambda | [0.01, 1.0] log | si OT actif |
| Loss | loss_focal_gamma | [0, 1.5] | 0 = no focal |
| Optim | learning_rate | [1.5e-5, 4e-4] log | range historique productif |
| Optim | num_train_epochs | [6, 12] | |
| Optim | weight_decay | [0.05, 0.50] | |
| Optim | layer_decay | [0.65, 1.0] | 1.0 = LLRD off |
| Reg | head_dropout | [0, 0.4] | |
| Reg | backbone_drop_path_rate | [0, 0.4] | |
| Arch | pretrained_source | {lvd, ibot:...} | DINOv3 only |
| Arch | pooling_type | {cls, attention_k_query, multihead_attention} | |
| Pool K-query | n_focal | [1, 3] | conditional |
| Pool K-query | n_diffuse | [1, 3] | conditional |
| Pool K-query | n_free | [0, 2] | conditional |

**Pinned (pas dans HPO)** :
- `augmentation_level: light` (HFlip + ColorJitter)
- `loss_lambda_init: 1.0`, `loss_lambda_lr: 0.5`, `loss_lambda_max: 5.0`, `loss_lambda_ema: 0.9`
- `ema_decay: 0.999`
- `per_device_train_batch_size: 128` (effective 256 en DDP 2 GPUs)

**Retiré du HPO vs v10** :
- `loss_fairness_lambda` (remplacé par Lagrangien adaptatif)
- `aug_share` (replication mechanism retiré)
- `mmd_lambda`, `adv_lambda` (MMD/DANN remplacés par OT seul)

---

## 10. Références théoriques

| Concept | Référence |
|---|---|
| Maximum Mean Discrepancy (MMD) | Gretton et al. 2012, "A Kernel Two-Sample Test" |
| Median heuristic bandwidth | Garreau et al. 2017 |
| DANN (Gradient Reversal Layer) | Ganin & Lempitsky 2015 |
| Sliced Wasserstein | Bonneel et al. 2015, "Sliced and Radon Wasserstein Barycenters" |
| Iterative Proportional Fitting | Deming & Stephan 1940, Sinkhorn-Knopp 1967 |
| Lagrangian fairness constraints | Cotter et al. 2019, "Two-player game" + Augmented Lagrangian |
| EMA weights for ML | Polyak averaging (1990) |
| Layer-wise LR Decay | BERT fine-tuning paper, ELECTRA, ViT papers |
| Isotonic calibration | Zadrozny & Elkan 2002 |
| Perceiver IO / cross-attn queries | Jaegle et al. 2021 |
| Group fairness impossibility | Chouldechova 2017, Kleinberg et al. 2016 |

Voir aussi [references.md](references.md) pour le détail bibliographique.
