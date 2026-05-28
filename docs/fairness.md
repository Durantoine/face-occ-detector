# Fairness, équilibrage & shift train→test

> Référence théorique complète des mécanismes d'équité (gender) et de compensation du shift sur Y.
> Pour la cohérence des combos dans le search space Optuna et le câblage code, voir aussi
> [audits_and_roadmap.md](audits_and_roadmap.md) §"Audit v4".

> **Mise à jour v6** — Le search space Optuna v6 modifie les 3 axes (justifications basées
> sur les data v4, 34 trials actifs) :
> - **Axe 1 (sampler)** : retirés `gender` (no-op cf §3), `gender_x_occ` (avg 0.01345),
>   `occlusion` (redondant avec `test_pmf`). Ajoutés `gender_within_occ` et
>   `gender_within_test_pmf` (50/50 F/M intra-bin, sans sur-poids des cellules rares).
> - **Axe 2 (loss_rw)** : retiré `cell_joint` (avg 0.00639). Ajoutés `cell_within_occ` et
>   `cell_within_test_pmf` (équivalent loss-side des nouveaux samplers).
> - **Axe 3 (feature_fairness)** : retiré `dann` (avg 0.01246, adversarial training instable).
>   Reste `none`, `mmd`, `mixup_gender`. La stratégie G (DANN) ci-dessous reste documentée
>   à titre de référence théorique, mais le code DANN n'est plus activé par le sweep v6.
>
> **Mise à jour v6.5 — Paired-α design** : les axes 1 et 2 ont été **collapsés** en un seul
> axe `correction_strategy` (4 choix) + un continu `correction_alpha ∈ [0, 1]`.
> Constat clé : chaque paire (sampler, loss) ciblait la même correction r(y, g), juste
> appliquée à des stages différents. Avec α, on partage la correction :
> sampler ∝ r^α, loss ∝ r^(1−α), effet combiné = r exact (∀α).
> α=1 ⇔ pur sampler (legacy), α=0 ⇔ pur loss, α=0.5 ⇔ hybride √-strength (poids effectifs
> divisés par √r côté chaque mécanisme → moins d'extrêmes, moins de variance).
> Voir §"Paired-α design" en fin de doc.

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

## Train↔test distribution shift on Y — and the Y×G coupling

**Le shift existe** : on a estimé la distribution d'occlusion du test set par inférence sur les images, et elle diffère de celle du train. Codée en dur dans [`_TEST_PMF_0025`](../src/utils/losses.py#L6) — vecteur de $B = 20$ bins de largeur $\delta = 0{,}025$ couvrant $[0, 0{,}5]$.

**Le coupling Y↔G** : dans le train, $\mathbb{E}[Y \mid G = F] > \mathbb{E}[Y \mid G = M]$. Donc si on corrige le shift sur $Y$ (test plus chargé vers le haut), les femmes (sur-représentées dans les bins haut-$Y$) reçoivent implicitement plus de poids. La question : **est-ce qu'on doit re-corriger côté genre par-dessus ?**

### Notations et dimensions

| Symbole | Définition | Domaine / dim |
|---|---|---|
| $N$ | taille d'un batch (ou d'un epoch selon contexte) | scalaire |
| $i$ | index sample | $1 \le i \le N$ |
| $y_i$ | label d'occlusion | $y_i \in [0, 1]$ |
| $g_i$ | genre | $g_i \in \{0, 1\}$ ($0 = F$, $1 = M$) |
| $\hat y_i$ | prédiction du modèle | $\hat y_i \in [0, 1]$ |
| $e_i = (\hat y_i - y_i)^2$ | erreur quadratique | $e_i \ge 0$ |
| $B$ | nombre de bins | $B = 20$ |
| $\delta$ | largeur d'un bin | $\delta = 0{,}025$ |
| $b(y) = \lfloor y / \delta \rfloor$ | indice du bin | $b(y) \in \{0, \dots, B-1\}$ |
| $P_{\text{train}} \in \mathbb{R}^B$ | PMF empirique train sur $Y$ | $P_{\text{train}}[b] = \tfrac{1}{N}\sum_{i} \mathbb{1}\{b(y_i) = b\}$ |
| $P_{\text{test}} \in \mathbb{R}^B$ | PMF test pré-estimée | constante codée |
| $w^{\text{imp}} \in \mathbb{R}^B$ | ratio d'importance par bin | défini ci-dessous |
| $w_i^{\text{base}} = \tfrac{1}{30} + y_i$ | poids de la métrique cible | scalaire par sample |
| $w_i^{\text{gender}} \in \{c_F, c_M\}$ | poids gender_class_weights[$g_i$] | scalaire par sample |
| $n_F = \lvert\{i : g_i = 0\}\rvert$, $n_M$ | effectifs par genre dans un batch | scalaire |

### 1. Importance reweighting (correction du shift Y)

Sous l'hypothèse de **covariate shift sur Y uniquement** :
$$P_{\text{test}}(g \mid y) = P_{\text{train}}(g \mid y) \quad \text{pour tout } (y, g),$$
le ratio de Radon-Nikodym se simplifie :
$$\frac{P_{\text{test}}(y, g)}{P_{\text{train}}(y, g)} \;=\; \frac{P_{\text{test}}(y) \cdot P_{\text{test}}(g \mid y)}{P_{\text{train}}(y) \cdot P_{\text{train}}(g \mid y)} \;=\; \frac{P_{\text{test}}(y)}{P_{\text{train}}(y)}.$$

Le bon ratio est donc **marginal sur $Y$**, et il **traite automatiquement la corrélation $Y \times G$** : pas besoin de corriger en plus côté genre, puisque $P(g \mid y)$ est supposé identique.

Implémenté dans [`build_importance_weights`](../src/utils/losses.py#L12) :
$$w^{\text{imp}}[b] \;=\; \mathrm{clip}\!\left(\frac{P_{\text{test}}[b]}{\max\!\left(P_{\text{train}}[b],\; 10^{-6}\right)},\; \tfrac{1}{10},\; 10\right), \qquad w^{\text{imp}} \leftarrow \frac{w^{\text{imp}}}{\bar{w}^{\text{imp}}}.$$

Le clip $[1/10, 10]$ empêche les ratios divergents sur bins quasi-vides ; la renormalisation à moyenne 1 garde l'échelle de loss stable.

### 2. Loss formula (avec ou sans `gender_rw`)

Posons le poids effectif par sample (`loss_importance_reweight=True`) :
$$w_i \;=\; w_i^{\text{base}} \cdot w^{\text{imp}}[b(y_i)] \qquad \text{(shape: } (N,)\text{)}.$$

Avec `loss_gender_reweight=True` ([src/utils/losses.py:109](../src/utils/losses.py#L109)) on ajoute :
$$w_i^{(\text{D})} \;=\; w_i \cdot w_i^{\text{gender}} \;=\; w_i \cdot c_{g_i}.$$

Comme `gender` est toujours présent chez nous, [src/utils/losses.py:131-134](../src/utils/losses.py#L131-L134) calcule :
$$\overline{e}_G \;=\; \frac{\sum_{i :\, g_i = G} w_i^{(\cdot)} \cdot e_i}{\sum_{i :\, g_i = G} w_i^{(\cdot)}}, \qquad \mathcal{L} \;=\; \tfrac{1}{2}\!\left(\overline{e}_F + \overline{e}_M\right) + \lambda \cdot \left\lvert\overline{e}_F - \overline{e}_M\right\rvert.$$

### 3. Pourquoi `gender_rw` est un **no-op** dans cette formule

Le facteur $c_G$ est **constant à l'intérieur d'un groupe**. Au numérateur et au dénominateur de $\overline{e}_G$ il se factorise et **se simplifie** :
$$\overline{e}_F^{(\text{avec gender\_rw})} \;=\; \frac{\sum_{i :\, g_i = 0}\; c_F \cdot w_i \cdot e_i}{\sum_{i :\, g_i = 0}\; c_F \cdot w_i} \;=\; \frac{c_F \cdot \sum_{i :\, g_i = 0} w_i \cdot e_i}{c_F \cdot \sum_{i :\, g_i = 0} w_i} \;=\; \overline{e}_F^{(\text{sans gender\_rw})}.$$

Idem pour $\overline{e}_M$ avec $c_M$. **Conclusion : activer ou désactiver `loss_gender_reweight` ne change strictement rien au gradient** (tant que le batch contient au moins un sample F et un sample M).

> Note importante : `loss_cell_reweight` n'a PAS cette propriété, parce que $w^{\text{cell}}[g_i, b(y_i)]$ **varie à l'intérieur d'un groupe** (en fonction du bin $Y$). Le facteur ne se simplifie pas → c'est un vrai mécanisme de reweighting joint $Y \times G$.

### 4. Si on suspectait un shift sur G aussi (non couvert ici)

Si $P_{\text{test}}(g) \ne P_{\text{train}}(g)$, l'hypothèse $P_{\text{test}}(g \mid y) = P_{\text{train}}(g \mid y)$ tombe, et il faudrait estimer $P_{\text{test}}(g \mid y)$ via un classifieur d'attributs sur les images test, puis :
$$w(y, g) \;=\; \frac{P_{\text{test}}(y, g)}{P_{\text{train}}(y, g)}.$$

On n'a pas cette estimation aujourd'hui, donc on s'en tient à l'hypothèse Y-only.

---

## Stratégies d'équilibrage — 3 axes orthogonaux (v4)

### Les 4 aspects à corriger

Avant les mécanismes, on sépare les 4 problèmes distincts :

| # | Aspect | Description |
|---|---|---|
| 1 | **Déséquilibre F/M sur train** | Plus d'un genre que l'autre dans le train set |
| 2 | **Déséquilibre Y sur train** | Distribution d'occlusion concentrée près de 0 dans le train |
| 3 | **Corrélation Y × G** | Les visages F ont en moyenne plus d'occlusion que M sur le train (hard correlation à décorréler) |
| 4 | **Shift train→test sur Y** | Distribution Y du test diffère de celle du train (test plus chargé en haut-Y) |

### Les 3 axes du search space

Optuna sample **3 axes indépendamment** par trial, ce qui permet de mesurer l'**importance** de chaque axe via `optuna.importance.get_param_importances(study)` après le sweep.

#### Axe 1 — `sampler_strategy` (équilibrage data-level)

Quel sampler de batch utiliser ? Le tableau liste **toutes les valeurs historiques** (v3/v4),
avec un drapeau pour celles activées dans le search space v6.

| Valeur | v6 | Effet | Aspects adressés |
|---|---|---|---|
| `none` | ✅ | DataLoader standard (séquentiel/random) | aucun |
| `gender` | ❌ | WeightedRandomSampler équilibrant F/M *(retiré v6 : no-op cf §3)* | **(1)** |
| `occlusion` | ❌ | WeightedRandomSampler équilibrant les buckets Y *(retiré v6 : redondant avec `test_pmf`)* | **(2)** |
| `gender_x_occ` | ❌ | WeightedRandomSampler par cellule (G × Y_bucket) *(retiré v6 : sur-poids des cellules rares, avg score v4 = 0.01345)* | **(1), (2), (3)** |
| `test_pmf` | ✅ | WeightedRandomSampler avec poids `P_test(Y)/P_train(Y)` | **(2), (4)** |
| `gender_within_occ` | ✅ *(v6)* | 50/50 F/M intra-bin Y, Y reste P_train | **(1), (3)** — décorrélation Y×G sans changer P(Y) |
| `gender_within_test_pmf` | ✅ *(v6)* | 50/50 F/M intra-bin Y + Y → P_test | **(1), (3), (4)** — décorrélation Y×G + shift Y |

> Les samplers `gender_within_*` égalisent F/M *à l'intérieur de chaque bin Y* sans surreprésenter
> les cellules rares (≠ `gender_x_occ` qui tirait les bins haut-Y 100×/epoch → memorization).

#### Axe 2 — `loss_rw_strategy` (reweighting dans la loss)

| Valeur | v6 | Effet | Aspects adressés |
|---|---|---|---|
| `none` | ✅ | Loss standard (juste `w_i = 1/30 + y_i`) | aucun |
| `imp_rw` | ✅ | `w_i *= P_test(bin_y)/P_train(bin_y)` | **(4)** |
| `cell_joint` | ❌ | imp_rw + `1/√count(G, bin_y)` *(retiré v6 : avg score v4 = 0.00639)* | **(3), (4)** |
| `cell_within_occ` | ✅ *(v6)* | `W[g, b] = 0.5 × P_train[b] / count(g, b)` — 50/50 F/M intra-bin via loss | **(1), (3)** |
| `cell_within_test_pmf` | ✅ *(v6)* | `W[g, b] = 0.5 × P_test[b] / count(g, b)` — 50/50 F/M intra-bin + Y → P_test | **(1), (3), (4)** |

> Les `cell_within_*` sont l'équivalent **loss-side** des nouveaux samplers `gender_within_*` :
> chaque sample est vu 1×/epoch (pas de re-tirage des rares) mais reçoit un poids qui égalise
> les contributions F/M intra-bin. Voir [`build_cell_weights_within`](../src/utils/losses.py).

> Conditional : quand `sampler_strategy ≠ none`, l'axe 2 est forcé à `none` (évite la double
> correction). Quand `loss_type=group_dro`, l'axe 2 est aussi forcé à `none`.

#### Axe 3 — `feature_fairness` (invariance genre au niveau features)

| Valeur | v6 | Effet | Aspects adressés |
|---|---|---|---|
| `none` | ✅ | Aucune contrainte sur les features | aucun |
| `dann` | ❌ | Discriminateur G + GRL *(retiré v6 : avg score v4 = 0.01246, adversarial instable)* | **(1), (3)** |
| `mmd` | ✅ | Pénalité MMD entre features F et M | **(1), (3)** |
| `mixup_gender` | ✅ | Mixup inter-genre dans le même bucket Y | **(1), (3)** |

### Combos couverts

**v6** : 4 (sampler) × 4 (loss_rw) × 3 (feature_fairness) = **48 combinaisons théoriques**.
Filtrage du conditional (loss_rw forcé à `none` quand sampler ≠ `none`, et idem pour
`loss_type=group_dro`) → **~22 combinaisons valides** par loss_type.

Quelques combos notables (v6) :

| Combo (sampler, loss_rw, feature_fairness) | Sens |
|---|---|
| `(none, none, none)` | Baseline pur — aucune correction |
| `(none, imp_rw, none)` | Juste correction shift Y |
| `(none, cell_within_occ, none)` | 50/50 F/M intra-bin loss-side, Y reste P_train |
| `(none, cell_within_test_pmf, none)` | 50/50 F/M intra-bin loss-side + Y → P_test |
| `(none, imp_rw, mmd)` | Correction shift Y + alignement géométrique des features F/M |
| `(none, imp_rw, mixup_gender)` | Correction shift Y + décorrélation par data aug |
| `(test_pmf, none, none)` | Compensation shift Y au niveau sampler |
| `(test_pmf, none, mmd)` | Sampler aligned + MMD G |
| `(gender_within_occ, none, mixup_gender)` | Sampler 50/50 intra-bin + mixup G |
| `(gender_within_test_pmf, none, mmd)` | Triple correction : Y→P_test + 50/50 F/M + MMD |

→ Le 3-axes v6 couvre les corrections **(1) F/M, (2) Y, (3) Y×G, (4) shift train→test**
de manière orthogonale, sans cellules rares sur-représentées.

### Budget Optuna et trial par combo

```
200 trials sur ~22 combos valides = ~9 trials/combo en moyenne (TPE concentre vite sur les bons)
→ ~20-30 trials sur les 3-5 meilleurs combos, ~1-2 sur les mauvais
```

Suffisant pour identifier le gagnant et mesurer l'importance par axe.

### Pénalité fairness `loss_fairness_lambda` (toujours dans le search space)

**Indépendamment** des 3 axes ci-dessus, Optuna sample un `loss_fairness_lambda ∈ [0, 2]` qui ajoute `λ · |err_F − err_M|` à la loss. C'est une 4e dimension de hard decorrelation (au niveau de l'aggregation des erreurs), toujours disponible — conditionnel sur `loss_type=weighted_mse`.

---

## Méthodes — détails de chaque stratégie de fairness

> Pour chaque méthode du search space v4 : formule, dimensions, coût et motivation théorique. LaTeX rendu en MathJax (GitHub direct).

### Notations communes

| Tenseur | Shape | Description |
|---|---|---|
| $\hat y, y, w, e$ | $(B,)$ | prédictions, labels Y, poids par-sample, erreurs $e_i = (\hat y_i - y_i)^2$ |
| $g$ | $(B,)$ | labels genre, valeurs $\in \{0, 1\}$ |
| $w^{\text{imp}}$ | $(B_{\text{bins}},)$ avec $B_{\text{bins}} = 20$ | poids d'importance par bin de Y |
| $b(y) = \lfloor y / \delta \rfloor$ | scalaire $\in \{0, \dots, B_{\text{bins}} - 1\}$ | indice du bin pour le sample |
| $\Phi$ | $(B,\, D')$ | features pooled |

Toutes ces stratégies se combinent (multiplicativement pour les poids, additivement pour les pénalités) avec la loss de base :
$$\underbrace{\mathcal{L}_{\text{base},i}}_{\text{scalaire}} = \underbrace{w_i}_{\text{scalaire}} \cdot (\hat y_i - y_i)^2, \qquad w_i = \underbrace{(\tfrac{1}{30} + y_i)}_{\text{poids métrique}} \cdot \underbrace{w^{\text{imp}}[b(y_i)]}_{\text{shift Y, optionnel}}.$$

L'aggregation finale dans `WeightedMSELoss` (avec gender présent) :
$$\overline{e}_G = \frac{\sum_{i: g_i = G} w_i e_i}{\sum_{i: g_i = G} w_i} \in \mathbb{R}, \qquad \mathcal{L}_{\text{task}} = \tfrac{1}{2}(\overline{e}_F + \overline{e}_M) + \lambda_{\text{fair}} \cdot |\overline{e}_F - \overline{e}_M| \in \mathbb{R}_+.$$

### Stratégie A — Sampler gender + importance reweighting

```yaml
sampler_strategy: gender
loss_importance_reweight: true
```

- **Côté data** : `WeightedRandomSampler` avec poids $\propto 1/n_G$ → batches équilibrés F/M en attendu
- **Effet** : $n_{\text{samples/epoch}} = 2 \cdot \min(n_F, n_M)$ → perte de 20-40 % de données par epoch selon imbalance
- **Côté loss** : $w_i$ inclut $w^{\text{imp}}$ (corrige shift Y)

### Stratégie D — No sampler + importance + gender_reweight (no-op)

```yaml
sampler_strategy: none
loss_importance_reweight: true
loss_gender_reweight: true     # ← no-op
```

- **`gender_rw`** : multiplie $w_i$ par une constante par-genre $c_{g_i}$. Mais cette constante **se factorise** dans $\overline{e}_G$ → effet nul (preuve §3 ci-dessus)
- **Effet net** : équivalent à « no sampler + importance_rw seul »
- **Empiriquement** : bat A car utilise 100 % des données par epoch

### Stratégie E — Cell reweight (joint Y × G)

```yaml
sampler_strategy: none
loss_cell_reweight: true
```

Construit $W^{\text{cell}} \in \mathbb{R}^{2 \times B}$ avec :
$$W^{\text{cell}}[g, b] = \frac{1}{\sqrt{|\{i : g_i = g \wedge b(y_i) = b\}|}}, \quad \text{normalisé à moyenne 1}.$$

Puis $w_i \mathrel{\*}= W^{\text{cell}}[g_i, b(y_i)]$.

- **Spécificité** : le poids **varie par bin Y à l'intérieur d'un genre** → **ne se factorise pas** dans $\overline{e}_G$ → c'est un vrai mécanisme actif (contrairement à `gender_rw`)
- **Effet** : compense la sous-représentation des cellules $(g, b)$ rares, sans utiliser $P_{\text{test}}$ (pas de correction de shift Y explicite)

### Stratégie G — Adversarial debiasing (DANN)

```yaml
sampler_strategy: none
loss_importance_reweight: true
loss_adv_debiasing: true       # → enable_adv_disc=True dans le model
adv_lambda: ε ∈ [0.01, 1.0] (log scale)
```

Architecture additionnelle : un MLP discriminateur de genre.

| Paramètre / tenseur | Shape | Description |
|---|---|---|
| $\Phi$ | $(B,\, D')$ | features pooled (input du disc) |
| $W_1$ | $(256,\, D')$ | première couche linéaire |
| $W_2$ | $(2,\, 256)$ | deuxième couche linéaire (2 classes : F/M) |
| $\text{Disc}(\Phi)$ | $(B,\, 2)$ | logits gender |

$$\underbrace{\text{Disc}(\Phi)}_{(B,2)} = W_2 \cdot \text{ReLU}(W_1 \cdot \Phi + b_1) + b_2.$$

Loss adversariale :
$$\mathcal{L}_{\text{adv}} = \mathrm{CE}\bigl(\text{Disc}(\text{GRL}(\Phi)),\; g\bigr), \qquad \mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{task}} + \lambda_{\text{adv}} \cdot \mathcal{L}_{\text{adv}}.$$

**Gradient Reversal Layer (GRL)** : identité au forward, sign-flip au backward.

- Le **discriminateur** apprend à classifier le genre depuis les features
- Le **backbone** apprend à produire des features dont le genre n'est pas décodable
- $\lambda_{\text{adv}}$ trop grand → backbone explose ; trop petit → effet nul

### Stratégie H — MMD alignment

```yaml
feature_fairness: mmd
mmd_lambda: ε ∈ [0.01, 1.0] (log scale)
```

Aligne les distributions de features pooled entre F et M via Maximum Mean Discrepancy² dans un RKHS gaussien.

$$\text{MMD}^2(\Phi_F, \Phi_M) = \mathbb{E}_{x, x' \sim \Phi_F}[k(x, x')] + \mathbb{E}_{y, y' \sim \Phi_M}[k(y, y')] - 2 \, \mathbb{E}_{x \sim \Phi_F, y \sim \Phi_M}[k(x, y)].$$

Kernel RBF multi-bandwidth :
$$k(x, y) = \frac{1}{|S|} \sum_{\sigma \in S} \exp\!\left( -\frac{\|x - y\|^2}{2 \sigma^2} \right), \quad S = \{1, 5, 10\}.$$

Loss totale : $\mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{task}} + \lambda_{\text{MMD}} \cdot \widehat{\text{MMD}^2}(\Phi_F, \Phi_M)$.

- **Coût** : ~$B^2 \cdot D'$ par batch (négligeable)
- **Avantage vs DANN** : pas d'adversaire à entraîner → stable, peu de tuning
- **Limite** : aligne toute la distribution feature-space, alors qu'on voudrait seulement aligner conditionnellement sur Y

### Stratégie I — Inter-gender Mixup

```yaml
feature_fairness: mixup_gender
mixup_alpha: ε ∈ [0.1, 1.0]   # v6 : étendu de [0.1, 0.5] (log scale)
```

Pour chaque sample F dans le batch, on cherche un partenaire M dans le bucket Y le plus proche, puis on interpole.

**Étape 1 — pairing** :
$$\text{partner}(i) = \arg\min_{j \in m_{\text{idx}}} |b(y_i) - b(y_j)|.$$

**Étape 2 — sampling du poids de mix** : $\lambda_i \sim \text{Beta}(\alpha, \alpha)$. Pour $\alpha < 1$, la distribution Beta est U-shaped → $\lambda$ concentré près de 0 ou 1 (mix faible). Pour $\alpha = 1$, uniforme. Pour $\alpha > 1$ (hors range), centré sur 0.5.

**Étape 3 — interpolation** :
$$\tilde x_i = \lambda_i \cdot x_i^F + (1 - \lambda_i) \cdot x_{\text{partner}(i)}^M, \qquad \tilde y_i = \lambda_i \cdot y_i^F + (1 - \lambda_i) \cdot y_{\text{partner}(i)}^M.$$

- **Coût** : O($n_F \cdot n_M$) + O($n_F \cdot C \cdot H \cdot W$). Négligeable.
- **Avantage** : zéro params nouveaux, juste de la data aug

### Stratégie J — Cell-within reweight (v6, équivalent loss-side des samplers `gender_within_*`)

```yaml
loss_rw_strategy: cell_within_occ        # ou cell_within_test_pmf
```

Construit $W^{\text{cell-within}} \in \mathbb{R}^{2 \times B}$ avec :
$$W^{\text{cell-within}}[g, b] = \frac{0.5 \cdot P_{\text{target}}[b]}{\max\!\big(\lvert\{i : g_i = g \wedge b(y_i) = b\}\rvert,\, 1\big)},$$
puis normalisé pour que $\sum_{g, b} \text{count}(g, b) \cdot W[g, b] / N = 1$ (moyenne pondérée
= 1, échelle de loss préservée) et clippé à `median × 10` pour éviter les cellules ultra-rares.

- $P_{\text{target}}$ = $P_{\text{train}}$ (cas `cell_within_occ`) ou $P_{\text{test}}$ (cas `cell_within_test_pmf`)
- **Différence vs `cell_joint`** : pas de $1/\sqrt{N}$ qui sur-pondère les cellules rares, ici on
  cible directement 50/50 F/M intra-bin sans surreprésenter les bins peu peuplés
- **Code** : [`build_cell_weights_within`](../src/utils/losses.py)

### Résumé : matrice méthode × objectif

| Strat. | Sampler | Imp_rw | Mécanisme actif | v6 | Cible théorique |
|---|---|---|---|---|---|
| A | gender | ✓ | aucun (sampler suffit pour G) | ❌ | shift Y + fairness G via sampler |
| D | none | ✓ | gender_rw=no-op | ❌ | shift Y + fairness G via aggregation par-groupe |
| E | none | ✗ | cell_rw | ❌ | équité Y×G implicite |
| F | occlusion | ✗ | gender_rw=no-op | ❌ | shift Y via sampler |
| **G** | none | ✓ | **DANN** | ❌ | fairness G via features invariantes au gender |
| **H** | none | ✓ | **MMD** | ✅ | fairness G via alignement de distributions de features |
| **I** | none | ✓ | **Mixup inter-G** | ✅ | invariance par augmentation conditionnelle Y |
| **J** | none | ✗ | **cell_within_***  *(v6)* | ✅ | 50/50 F/M intra-bin sans sur-poids des cellules rares |
| **K** | `gender_within_*` *(v6)* | ✗ | aucun (sampler suffit) | ✅ | équivalent data-side de J |

---

## Choix du validation split — `val_split_strategy`

Comment construire le val à partir du train ? Deux options dans v4 :

### `stratified_yg` (default)

Random split stratifié par `(gender × occlusion_bucket)`. Le val a **la même distribution que le train**, donc P_val ≠ P_test (shifted).

- **Métriques** : `challenge_score_val` mesure la perf sur val (biaisée comme estimateur de test) ; `challenge_score_test_estimated` reweighte le val à `P_test` pour estimer la perf test sans biais.
- **Avantages** : val maximal (20 % des données = 20k samples), stable.
- **Inconvénients** : l'estimation test (`*_test_estimated`) souffre de variance amplifiée sur les bins haut-Y rares.

### `test_pmf` (nouveau)

Resample le val à partir du train pour que **P_val(Y) = P_test(Y)** par construction. Chaque bin de Y reçoit `P_test(bin) × val_size` samples.

- **Métriques** : `challenge_score_val` est **directement** l'estimateur de la perf test (pas de reweighting nécessaire). `challenge_score_test_estimated` ≈ `challenge_score_val`.
- **Avantages** : éval propre, pas de magouille d'estimation, BatchNorm/stats internes au val matchent test, lecture plus intuitive.
- **Inconvénients** : val plus petit (~10-15k au lieu de 20k), plus de variance par trial.
- **Implémentation** : [`_split_val_to_match_test_pmf()`](../src/data/dataset.py).

### Comment choisir

| Si tu veux... | Utilise |
|---|---|
| Stabilité, gros val, courbes lisses | `stratified_yg` |
| Estimation directe perf test, plus rigoureux | `test_pmf` |
| Comparer empiriquement les 2 | Lance 2 sweeps en parallèle |

---

## Compenser le shift train→test — 3 mécanismes complémentaires

### Mécanisme 1 — Reweighter la loss (axe 2 = `imp_rw`)

**Quand** : pendant l'entraînement, à chaque batch.

**Comment** : chaque sample reçoit un multiplicateur `w_imp = P_test(bin) / P_train(bin)`. Les samples haut-Y (rares sur train, sur-représentés sur test) voient leur loss multipliée par un grand nombre.

**Flag yaml** : `loss_importance_reweight: true` (via `loss_rw_strategy: imp_rw`).

**Avantages** : toutes les samples sont vues à chaque epoch. Implémentation simple.

**Inconvénients** : gradients bruyants quand les poids sont extrêmes. Stats BatchNorm restent calculées sur la distribution train.

### Mécanisme 2 — Resampler les données (axe 1 = `test_pmf`)

**Quand** : pendant l'entraînement, AU NIVEAU DU SAMPLER.

**Comment** : `WeightedRandomSampler` avec poids `P_test[bin] / P_train[bin]` clippés à [0.1, 10]. Le modèle voit effectivement une distribution qui matche test à chaque batch.

**Avantages vs Mécanisme 1** :
- BatchNorm et stats internes calibrées correctement
- Loss landscape propre (pas de poids extrêmes)
- Théoriquement équivalent en espérance, mais différent en pratique

**Inconvénients** :
- Les samples haut-Y rares sont vus plusieurs fois par epoch → risque d'overfit (compensable par data aug forte)
- Les samples bas-Y vus moins souvent

**Quand préférer 2 sur 1** : si le modèle diverge avec `loss_importance_reweight` (gradients explosent), ou si BatchNorm (peu d'impact ici car DINOv3/Sapiens2 utilisent LayerNorm).

### Mécanisme 3 — Calibration POST-INFERENCE (quantile matching)

**Quand** : APRÈS le training, sur les prédictions finales du test set, juste avant de soumettre.

**Idée** : histogram matching standard. On regarde la distribution empirique des prédictions du modèle sur le test set, et on applique une transformation monotone pour qu'elle matche exactement `P_test`.

**Algorithme** :
1. Le modèle prédit $\hat y_1, \dots, \hat y_N$ sur le test
2. On trie par ordre croissant
3. La k-ième prédiction → quantile $k/N$ de la distribution empirique
4. On la remplace par la valeur de Y qui correspond au même quantile dans P_test
5. Résultat : distribution des prédictions matche exactement P_test, mais l'**ORDRE** est préservé

**Code** : [`quantile_match_to_test_pmf()`](../src/inference/calibration.py) (~30 lignes).

**Pourquoi ça aide** : décomposition standard `erreur = biais² + variance`. Si le modèle sous-prédit les hautes occlusions (joue safe), le quantile mapping rehausse ces prédictions vers la vraie distribution test → **réduit le biais sans changer la variance**.

**Conditions de succès** :
- ✅ Le modèle a un bon RANKING (préserve l'ordre)
- ❌ Si les erreurs sont purement aléatoires (bruit blanc), ça ne change rien

```python
from src.inference.calibration import quantile_match_to_test_pmf
from src.utils.losses import _TEST_PMF_0025
from src.utils.metrics import compute_score

score_raw = compute_score(val_preds, val_gt, val_gender)["score"]
val_preds_calibrated = quantile_match_to_test_pmf(val_preds, _TEST_PMF_0025)
score_calibrated = compute_score(val_preds_calibrated, val_gt, val_gender)["score"]
print(f"raw={score_raw:.5f}  calibrated={score_calibrated:.5f}  Δ={score_raw - score_calibrated:+.5f}")
```

### Comment combiner les 3 mécanismes

| Stage | Action | Toujours actif ? |
|---|---|---|
| Training (loss) | `loss_rw_strategy: imp_rw / cell_joint` | optionnel, Optuna sample |
| Training (sampler) | `sampler_strategy: test_pmf` | optionnel, Optuna sample (incompatible avec axe 2 ≠ none) |
| Inference | `match_test_pmf: True` dans predict.py | **toujours essayer** sur val, activer si ça aide |

Note : Mécanismes 1 et 2 sont **mutuellement exclusifs** dans le sweep (gating Optuna). Le mécanisme 3 est **toujours additif**.

### Subtilité : pourquoi pas juste le mécanisme 3 ?

Le mécanisme 3 corrige UNIQUEMENT le biais marginal sur Y. Il ne corrige pas :
- Les erreurs **dépendantes de l'image** (un visage difficile reste mal prédit)
- Le **biais conditionnel sur le genre**
- Les **interactions Y × G**

→ Mécanisme 3 = **filet de sécurité** post-training. Mécanismes 1/2 = bonne distribution dès le départ.

---

## Annexe — Sampler × loss-weight design (analyse fine sur 100k subset)

### The two-axis imbalance

Notre 100k train subset a deux déséquilibres :

1. **Gender** : M/F = 2.086 (67.6 % M, 32.4 % F)
2. **Occlusion** : 32.5 % in `[0, 0.025)`, decaying to <0.1 % above `[0.45, 0.50)`

Et un **confound** critique :

| gender | n | occ_mean | occ_std |
|---|---|---|---|
| F (0) | 32,400 | **0.129** | 0.094 |
| M (1) | 67,600 | **0.061** | 0.073 |

→ **Women's faces have 2× more occlusion on average** in the training data. Any model can learn `gender → +0.07 occlusion` as a shortcut. This shortcut works on train but degrades test and inflates `|Err_F − Err_M|`.

### Quatre options de décorrélation (analyse historique v3)

The math: avec `f_sampler(b)` la fréquence par-bin du sampler et `w_imp(b)` le multiplicateur de loss, l'optimizer minimise `Σ_b f_sampler(b) · w_imp(b) · E[w_metric·err | bin = b]`. Pour matcher l'éval test `Σ_b p_test(b) · E[...]`, il faut `f_sampler(b) · w_imp(b) = p_test(b)`.

| Option | Sampler | Loss weights | Pro | Con |
|---|---|---|---|---|
| **A** | `gender` | `w_imp = p_test / p_train` | each sample ~1×/epoch, 100% data | does NOT decorrelate gender × occ |
| **B** | `gender_x_occ` | `w_imp = n_buckets · p_test` | hard decorrelation Y×G | bins 17-19 (<62 samples) drawn 100×/epoch → memorization — **excluded from HPO** |
| **C** | `test_like_x_gender` | `w_imp = 1` | conceptually cleanest | bins 17-19 jamais drawn → samples discarded — **excluded** |
| **D** | `none` | `w_imp` + `w_gender` | 100% data, no memorization | gender balance only on average |
| **E** | `none` | per-cell `w_cell(g, b) = 1/sqrt(count(g,b))` | hard decorrelation Y×G dans le gradient, sans memorization | bins ultra-rares ont des poids ~5× la moyenne |
| **F** | `occlusion` (quantile) | `w_gender` only | sampler quantile-balanced sur occ | quantile bins ≠ GT bins → ne matche pas la distribution test aussi finement |

→ Dans v4, ces stratégies sont remplacées par les **3 axes orthogonaux** (sampler, loss_rw, feature_fairness), avec la même couverture conceptuelle mais en samplant les axes indépendamment.

---

## Paired-α design (v6.5)

### Constat motivant le refactor

Le design 3-axes v6 traite `sampler_strategy` et `loss_rw_strategy` comme indépendants — mais en pratique chaque "stratégie sampler" a un jumeau loss qui vise **exactement la même correction** r(y, g) :

| Cible r(y, g) = P_target / P_train | Sampler version | Loss version |
|---|---|---|
| P_test(y) / P_train(y) | `test_pmf` | `imp_rw` |
| 0.5 · P_train(y) / P_train(y, g) | `gender_within_occ` | `cell_within_occ` |
| 0.5 · P_test(y) / P_train(y, g) | `gender_within_test_pmf` | `cell_within_test_pmf` |

Le `conditional_on: sampler_strategy=["none"]` v6 forçait à choisir un seul stage (sampler **ou** loss). Le paired-α relâche ça en faisant **les deux à intensité partagée**.

### Formulation

Pour chaque correction de cible r :

$$\underbrace{p_{\text{sampler}}(y, g)}_{\text{poids de tirage}} \;\propto\; r(y, g)^{\alpha} \qquad
\underbrace{w_{\text{loss}}(y, g)}_{\text{poids dans la loss}} \;\propto\; r(y, g)^{1-\alpha}$$

**Effet combiné sur le gradient** :

$$\mathbb{E}_{(y,g) \sim p_{\text{sampler}}} \!\left[w_{\text{loss}}(y, g) \cdot \mathcal{L}\right]
\;=\; \mathbb{E}_{(y,g) \sim p_{\text{train}}} \!\left[r(y, g)^{\alpha} \cdot r(y, g)^{1-\alpha} \cdot \mathcal{L}\right]
\;=\; \mathbb{E}_{p_{\text{train}}} \!\left[r(y, g) \cdot \mathcal{L}\right]$$

→ La correction effective est **exactement r**, quel que soit α. Seule la **répartition** entre data-side et gradient-side change.

### Pourquoi α=0.5 est intéressant

Si r(y, g) = 10 pour les cellules rares :
- **α=1 (pur sampler)** : ces samples sont tirés 10× par epoch → memorization si très peu de samples uniques
- **α=0 (pur loss)** : gradients ×10 sur 1 sample → grosse variance, batch dominé par 1 sample
- **α=0.5 (hybride √)** : tirage √10 ≈ 3.16× + poids √10 ≈ 3.16. Aucun extrême, partage la charge.

### Cas particulier : α aux extrêmes

- α=1 : `loss_power=0` → train.py court-circuite la loss reweight (irrelevant). Équivalent legacy "pur sampler".
- α=0 : `sampler_power=0` → train.py court-circuite la création du sampler. Équivalent legacy "pur loss".

→ Les extrêmes reproduisent exactement le comportement v6, donc le sweep paired-α généralise strictement v6.

### Search space v6.5

```yaml
correction_strategy:
  type: categorical
  choices: ["none", "test_pmf", "gender_within_occ", "gender_within_test_pmf"]
correction_alpha:
  type: float
  low: 0.0
  high: 1.0
  conditional_on:
    correction_strategy: ["test_pmf", "gender_within_occ", "gender_within_test_pmf"]
    loss_type: weighted_mse
```

4 stratégies × 1 dim continue = ~50 trials/stratégie avec un budget de 200 → TPE peut tuner α finement par cible.

### Pourquoi α n'est pas appris en bout de modèle

Le sampler est **discret** (sélection de samples → batch), pas différentiable. Donc ∂L/∂α n'existe pas côté data-side. Les workarounds (REINFORCE, bilevel, sampler-soft) sont soit trop coûteux (extra training), soit défont l'intérêt du sampler. **TPE = la bonne solution** : 1 trial = 1 sample (α, score), apprentissage bayésien gratuit au niveau outer-loop.

### Implémentation

- [src/utils/losses.py](../src/utils/losses.py) — `build_importance_weights(..., power)`, `build_cell_weights_within(..., power)` : appliquent `r^power` avec normalisation conservée.
- [src/data/dataset.py](../src/data/dataset.py) — `create_test_pmf_sampler(..., power)`, `create_gender_within_bin_sampler(..., power)` : idem côté sampler.
- [src/optimize.py](../src/optimize.py) — `_CORRECTION_STRATEGY_PAIRS` map les 4 corrections vers leur paire (sampler, loss). `_apply_trial_param` propage `correction_alpha` → `sampler_power=α`, `loss_power=1-α`.
- [src/train.py](../src/train.py) — short-circuit propre quand power=0 (skip sampler/loss respectivement).

### Extension v7 : `correction_strength` (β)

v7 ajoute un coefficient β ∈ [0.3, 1.0] qui contrôle l'**intensité totale** de la correction (orthogonal à α qui contrôle la répartition sampler/loss) :

$$p_{\text{sampler}} \propto r^{\alpha \cdot \beta} \qquad w_{\text{loss}} \propto r^{(1-\alpha) \cdot \beta}$$

$$\text{Effet combiné sur le gradient} = r^{\alpha\beta + (1-\alpha)\beta} = r^{\beta}$$

- β=1 : correction complète (= v6, équivalent r^1)
- β=0.5 : √r, correction à mi-puissance (utile si la correction complète overfit sur les bins rares high-Y)
- β=0.3 : r^0.3, correction très douce (limite basse, on s'approche du cas "presque pas de correction")

**Pourquoi pas β ∈ [0, 1]** : β→0 est déjà couvert par `correction_strategy=none`. Le range [0.3, 1.0] évite la zone redondante et concentre l'exploration sur la portion utile.

[src/optimize.py](../src/optimize.py) propage via `_refresh_correction_powers()` : recalcul de `sampler_power` et `loss_power` à chaque set de α ou β, ordre d'application des params indifférent.

---

## v8 design — refonte axes orthogonaux

Inspiré par le constat empirique : `cell_joint + MMD` (top v4, score 0.00120) battait les
designs v6/v7 qui forçaient 50/50 F/M intra-bin (déviation P_test). Le design v8 sépare
proprement Y-shift et G-fairness sur 5 axes indépendants, et réintroduit `cell_rw` soft
(la mécanique gagnante v4).

### Hypothèse théorique

H1 — covariate shift Y-only : P_test(genre | occlusion) = P_train(genre | occlusion).
Sous H1, le sampler optimal pour matcher P_test(G, Y) est `test_pmf` (corrige uniquement
Y, ne touche pas G). Forcer 50/50 F/M intra-bin dévie de P_test → contre-productif sauf si
la métrique impose autre chose.

La métrique impose **l'égalité d'erreurs** (`|err_F - err_M|`), pas l'égalité de distribution.
On peut atteindre cet objectif sans modifier P(G|Y) train, via :
- `loss_focal_gamma` : auto-pondère les hard examples (modalités rares mal apprises)
- `feature_fairness=mmd` : aligne features F/M dans le RKHS
- `loss_fairness_lambda` : pression directe sur le gap d'erreur
- `axis2_power` (cell_rw soft) : compensation douce des cellules rares (top v4)

### Axe 1 — Y shift correction (3 mécaniques)

Les 3 mécaniques partagent la même correction `r(b) = P_test(b) / P_train(b)` mais
l'appliquent à des stages différents :

| Mécanique | Effet | Implementation |
|---|---|---|
| **Sampler test_pmf** (poids `a × γ`) | Tire plus souvent les bins haut-Y | [`create_test_pmf_sampler(power=...)`](../src/data/dataset.py) |
| **Loss imp_rw** (poids `b × γ`) | Pondère gradient par r^power | [`build_importance_weights(power=...)`](../src/utils/losses.py) |
| **Aug Y-conditional** (poids `c × γ`) | Réplique k_i = r^power copies augmentées | [`YConditionalAugDataset`](../src/data/dataset.py) |

Paramètres v8 :
- `axis1_strength` (γ) ∈ [0.5, 1.0] : intensité totale de la correction
- `axis1_sampler_share` (a) ∈ [0, 1]
- `axis1_loss_share` (b) ∈ [0, 1]
- `aug_share` (c) = max(0, 1 − a − b) (déduit, normalisé si a+b>1)

Effet combiné sur le gradient pour un sample dans bin b :
> tirage `r^(γa)` × loss weight `r^(γb)` × aug count `r^(γc)` = `r^(γ × (a+b+c)) = r^γ`

γ=1 → correction complète. γ=0.5 → correction modérée (√r). La répartition (a, b, c)
détermine **comment** la correction est appliquée (data-side, gradient-side, ou diversity).

### Aug Y-conditional — stochastic Bernoulli rounding

Pour chaque sample `i` dans bin `b(i)`, on calcule l'espérance de copies :
> `k_float_i = r(b_i)^c`

On tire le nombre réel de copies via Bernoulli rounding :
- `k_int = floor(k_float)` (copies garanties)
- Plus 1 copie supplémentaire avec proba `k_float - floor(k_float)`

Cette procédure préserve **exactement** E[k] = k_float (vérifié à 30 seeds dans le test
de validation). Chaque copie passe par la pipeline d'augmentation standard, donc chaque
copie est une vue **différente** du même sample base. Combiné, ça produit la diversité
demandée sans memo des samples rares.

`set_epoch()` re-roll le mapping virtuel à chaque epoch via un `TrainerCallback` dédié,
évitant que la même sélection de copies se fige sur tout le training.

### Axe 2 — Soft G compensation (cell_rw)

`axis2_power` ∈ [0, 1] contrôle l'intensité de la compensation des cellules (g, b) rares :

> `W[g, b] = (1 / sqrt(count(g, b)))^power`, normalisé à mean=1

- power=0 → pas d'effet (W=1 partout)
- power=1 → standard 1/sqrt (top v4)
- power=0.5 → encore plus doux (1/count^0.25)

Cellules rares (F-bin18, M-bin18) reçoivent ~5-8× la médiane à power=1, vs ~20-35× pour
`cell_within_*` strict 50/50 (v6). Le soft sqrt est précisément ce qui évite la memo
catastrophique des cellules ultra-rares.

### Reproduction du top v4 via le nouveau design

Top v4 (Sapiens trial 18, score 0.00120) = `(none, cell_joint, mmd)` correspond à :
- `axis1_strength=1.0, axis1_sampler_share=0, axis1_loss_share=1` → pure imp_rw (loss-side)
- `axis2_power=1.0` → cell_rw sqrt full
- `feature_fairness=mmd`
- `loss_focal_gamma, loss_fairness_lambda` libres

Le sweep v8 inclut cette configuration et toutes ses voisines, en plus de couvrir le
sweet spot 3-mécaniques (sampler + loss + aug Y-conditional).

### Implémentation

- [src/data/dataset.py](../src/data/dataset.py) — `YConditionalAugDataset`, `create_sampler_weights_for_virtual`
- [src/utils/losses.py](../src/utils/losses.py) — `build_cell_weights(..., power)` (cell_rw soft)
- [src/optimize.py](../src/optimize.py) — `_refresh_axis1_powers()` : recompute sampler_power/loss_power/aug_share depuis γ × shares
- [src/train.py](../src/train.py) — wiring complet, callback `_YCondAugEpochCallback` pour re-roll virtual mapping
- Pinned : `ema_decay=0`, `loss_type=weighted_mse`, `augmentation_level=medium`, `val_split_strategy=test_pmf`, `eval_importance_reweight=false`
