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

## Train↔test distribution shift on Y — and the Y×G coupling

**Le shift existe** : on a estimé la distribution d'occlusion du test set par inférence sur les images, et elle diffère de celle du train. Codée en dur dans [`_TEST_PMF_0025`](src/utils/losses.py#L6) — vecteur de $B = 20$ bins de largeur $\delta = 0{,}025$ couvrant $[0, 0{,}5]$.

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

Implémenté dans [`build_importance_weights`](src/utils/losses.py#L12) :
$$w^{\text{imp}}[b] \;=\; \mathrm{clip}\!\left(\frac{P_{\text{test}}[b]}{\max\!\left(P_{\text{train}}[b],\; 10^{-6}\right)},\; \tfrac{1}{10},\; 10\right), \qquad w^{\text{imp}} \leftarrow \frac{w^{\text{imp}}}{\bar{w}^{\text{imp}}}.$$

Le clip $[1/10, 10]$ empêche les ratios divergents sur bins quasi-vides ; la renormalisation à moyenne 1 garde l'échelle de loss stable.

### 2. Loss formula (avec ou sans `gender_rw`)

Posons le poids effectif par sample (`loss_importance_reweight=True`) :
$$w_i \;=\; w_i^{\text{base}} \cdot w^{\text{imp}}[b(y_i)] \qquad \text{(shape: } (N,)\text{)}.$$

Avec `loss_gender_reweight=True` ([src/utils/losses.py:109](src/utils/losses.py#L109)) on ajoute :
$$w_i^{(\text{D})} \;=\; w_i \cdot w_i^{\text{gender}} \;=\; w_i \cdot c_{g_i}.$$

Comme `gender` est toujours présent chez nous, [src/utils/losses.py:131-134](src/utils/losses.py#L131-L134) calcule :
$$\overline{e}_G \;=\; \frac{\sum_{i :\, g_i = G} w_i^{(\cdot)} \cdot e_i}{\sum_{i :\, g_i = G} w_i^{(\cdot)}}, \qquad \mathcal{L} \;=\; \tfrac{1}{2}\!\left(\overline{e}_F + \overline{e}_M\right) + \lambda \cdot \left\lvert\overline{e}_F - \overline{e}_M\right\rvert.$$

### 3. Pourquoi `gender_rw` est un **no-op** dans cette formule

Le facteur $c_G$ est **constant à l'intérieur d'un groupe**. Au numérateur et au dénominateur de $\overline{e}_G$ il se factorise et **se simplifie** :
$$\overline{e}_F^{(\text{avec gender\_rw})} \;=\; \frac{\sum_{i :\, g_i = 0}\; c_F \cdot w_i \cdot e_i}{\sum_{i :\, g_i = 0}\; c_F \cdot w_i} \;=\; \frac{c_F \cdot \sum_{i :\, g_i = 0} w_i \cdot e_i}{c_F \cdot \sum_{i :\, g_i = 0} w_i} \;=\; \overline{e}_F^{(\text{sans gender\_rw})}.$$

Idem pour $\overline{e}_M$ avec $c_M$. **Conclusion : activer ou désactiver `loss_gender_reweight` ne change strictement rien au gradient** (tant que le batch contient au moins un sample F et un sample M, ce qui est le cas typique avec un sampler équilibré ou des batchs assez grands).

> Note importante : `loss_cell_reweight` n'a PAS cette propriété, parce que $w^{\text{cell}}[g_i, b(y_i)]$ **varie à l'intérieur d'un groupe** (en fonction du bin $Y$). Le facteur ne se simplifie pas → c'est un vrai mécanisme de reweighting joint $Y \times G$.

### 4. Alors pourquoi la stratégie D bat la stratégie A en pratique ?

Récap des deux stratégies dans [`_BALANCING_STRATEGY_MAP`](src/optimize.py#L59) :

| | A | D |
|---|---|---|
| `sampler_strategy` | `gender` | `none` |
| `loss_importance_reweight` | True | True |
| `loss_gender_reweight` | False | **True** (mais on a montré que c'est un no-op) |

→ **La seule différence active** est le **sampler**. Avec `sampler_strategy: gender` ([`create_balanced_sampler`](src/data/dataset.py#L18)) :
$$n_{\text{samples par epoch}} \;=\; 2 \cdot \min(n_F, n_M).$$

Si le train est par exemple 60/40 F/M sur 100k samples, **A voit ~80k samples par epoch (perte de 20%)** alors que D voit les 100k. À budget d'epochs égal, D fait plus de gradient steps sur la full distribution.

De plus, la formule de loss $\tfrac{1}{2}(\overline{e}_F + \overline{e}_M) + \lambda\lvert\overline{e}_F - \overline{e}_M\rvert$ **fait déjà le boulot de fairness côté loss** (moyenne par groupe = pondération implicite, $\lambda$ pénalise le gap). Forcer un batch équilibré F/M via sampler ajoute peu une fois cette aggregation par-groupe en place.

**Hypothèse empirique principale pour D > A** : D utilise simplement plus de données par epoch.

### 5. Recommandation pratique

| Objectif | Mécanisme | Statut |
|---|---|---|
| Honnêteté de la métrique val→test | `eval_importance_reweight` (val score reweighté à $P_{\text{test}}$) | par défaut `True` |
| Correction du shift Y à l'entraînement | `loss_importance_reweight` | `True` |
| Fairness G **sur la distribution test** | `loss_fairness_lambda > 0` ou `loss_type: group_dro` | tunable |
| `loss_gender_reweight` | **sans effet** sous notre formule | indifférent — laisse Optuna trancher |
| `sampler_strategy: gender` | optionnel ; **coûte de la donnée** (20–40% selon imbalance) | À éviter par défaut. Préférer `none` ou `gender_x_occ`. |

→ **Strategy A** (sampler=gender) reste théoriquement propre mais paie en données.
→ **Strategy D** (no sampler + gender_rw_no-op) ≡ **Strategy "no sampler + importance_rw seul"** en pratique. C'est probablement la baseline à battre.

### 6. Si on suspectait un shift sur G aussi (non couvert ici)

Si $P_{\text{test}}(g) \ne P_{\text{train}}(g)$, l'hypothèse $P_{\text{test}}(g \mid y) = P_{\text{train}}(g \mid y)$ tombe, et il faudrait estimer $P_{\text{test}}(g \mid y)$ via un classifieur d'attributs sur les images test, puis :
$$w(y, g) \;=\; \frac{P_{\text{test}}(y, g)}{P_{\text{train}}(y, g)}.$$

On n'a pas cette estimation aujourd'hui, donc on s'en tient à l'hypothèse Y-only.

---

## Méthodes — référence détaillée

> Cette section donne pour chaque méthode du search space v3 : la formule, les dimensions, le coût et la motivation théorique. Elle utilise du LaTeX rendu en MathJax — viewable sur GitHub directement, ou en local avec une extension Markdown supportant MathJax.

### Notations communes (glossaire complet)

**Variables aléatoires et instances** — convention : *majuscule* pour la variable aléatoire, *minuscule indicée par $i$* pour la réalisation du sample $i$.

| Symbole | Définition | Type / domaine |
|---|---|---|
| $Y$ | variable aléatoire d'**occlusion** : la quantité à prédire (% de visage occlus, normalisé) | continue, $Y \in [0, 1]$ |
| $y_i$ | valeur d'occlusion observée pour le sample $i$ du dataset | $y_i \in [0, 1]$, scalaire |
| $\hat y_i$ | prédiction du modèle pour le sample $i$ | $\hat y_i \in [0, 1]$, scalaire (sortie sigmoid) |
| $G$ | variable aléatoire de **genre** | binaire, $G \in \{0, 1\}$ (0 = féminin, 1 = masculin) |
| $g_i$ | genre observé pour le sample $i$ | $g_i \in \{0, 1\}$, scalaire |
| $P_{\text{train}}(\cdot)$, $P_{\text{test}}(\cdot)$ | distributions de probabilité sur le train et le test set | mesures de probabilité |

**Bin / histogramme sur $Y$** :

| Symbole | Définition | Domaine |
|---|---|---|
| $B_{\text{bins}}$ | nombre de bins pour discrétiser $Y$ | $B_{\text{bins}} = 20$ |
| $\delta$ | largeur d'un bin | $\delta = 0{,}025$, couvre $[0,\, 0{,}5]$ |
| $b(y) = \lfloor y / \delta \rfloor$ | indice du bin contenant $y$ | $b(y) \in \{0, \dots, B_{\text{bins}} - 1\}$ |
| $P_{\text{train}} \in \mathbb{R}^{B_{\text{bins}}}$, $P_{\text{test}} \in \mathbb{R}^{B_{\text{bins}}}$ | PMF empiriques sur les bins | vecteurs, somme à 1 |
| $w^{\text{imp}} \in \mathbb{R}^{B_{\text{bins}}}$ | ratio d'importance $P_{\text{test}}[b] / P_{\text{train}}[b]$, clippé et normalisé | vecteur |

**Architecture du modèle** :

| Symbole | Définition | Dimensions / domaine |
|---|---|---|
| $B$ | taille du batch | scalaire |
| $N$ | nombre de tokens en sortie du backbone ($= 1$ CLS $+ N_p$ patches) | scalaire (= 197 pour ViT-B/16 à 224×224) |
| $D$ | dimension cachée du backbone | $D \in \{768, 1024, 1280\}$ pour ViT-B / L / H+ |
| $X$ | tenseur des features de sortie du backbone (CLS à l'index 0) | $\mathbb{R}^{B \times N \times D}$ |
| $\Phi$ | features après pooling, input du head linéaire final | $\mathbb{R}^{B \times D'}$, $D'$ dépend du pooling |
| $D'$ | dim de sortie du pool : $D$ pour CLS/GeM/MHA, $K \cdot D$ pour K-query | scalaire |
| $h$ | sortie scalaire du head | $h \in \mathbb{R}^B$ |
| $\sigma(h) = (1 + e^{-h})^{-1}$ | sigmoid, $\hat y = \sigma(h)$ | applique element-wise |

**Métriques** :

| Symbole | Définition | Type |
|---|---|---|
| $e_i = (\hat y_i - y_i)^2$ | erreur quadratique du sample $i$ | scalaire ≥ 0 |
| $w_i^{\text{base}} = \tfrac{1}{30} + y_i$ | poids de la métrique du challenge | scalaire > 0 |
| $w_i$ | poids effectif du sample dans la loss : $w_i = w_i^{\text{base}} \cdot w^{\text{imp}}[b(y_i)]$ si reweighting actif, sinon $w_i = w_i^{\text{base}}$ | scalaire > 0 |
| $\overline{e}_G = \dfrac{\sum_{i: g_i = G} w_i \, e_i}{\sum_{i: g_i = G} w_i}$ | moyenne pondérée des erreurs dans le groupe $G$ | scalaire ≥ 0 |
| $\text{score} = \tfrac{1}{2}(\overline{e}_F + \overline{e}_M) + |\overline{e}_F - \overline{e}_M|$ | **métrique officielle du challenge** | scalaire ≥ 0, à minimiser |
| $\text{MSE} = \tfrac{1}{N}\sum_i e_i$ | Mean Squared Error (non-pondérée) | scalaire ≥ 0 |
| $\text{MAE} = \tfrac{1}{N}\sum_i \lvert\hat y_i - y_i\rvert$ | Mean Absolute Error (référence intuitive : même unité que $y$) | scalaire ≥ 0 |
| $\text{eval\_loss}$ | loss minimisée par le modèle ($= \tfrac{1}{2}(\overline{e}_F + \overline{e}_M) + \lambda_{\text{fair}} \cdot \lvert\overline{e}_F - \overline{e}_M\rvert$ avec $\lambda_{\text{fair}}$ tunable) | scalaire ≥ 0 |
| $\text{eval\_score}$ | métrique du challenge évaluée sur val (avec $\lambda = 1$ fixe) | scalaire ≥ 0 |
| $\text{eval\_score\_raw}$ | idem mais sans $w^{\text{imp}}$ (= métrique sur val sous l'hypothèse $P_{\text{val}} = P_{\text{test}}$) | scalaire ≥ 0 |

### Ordres de grandeur — pourquoi le score est si petit ?

Trois facteurs combinés :

**1. La métrique est une MSE pondérée → erreurs au carré**. Avec $|\hat y - y| \approx 0{,}05$ (erreur typique d'un bon modèle), on a $e_i = 0{,}0025$.

**2. Les valeurs de $Y$ sont concentrées près de 0**. La PMF test :
$$P_{\text{test}} = [0{,}105,\, 0{,}090,\, 0{,}090,\, 0{,}090,\, 0{,}092,\, 0{,}088,\, 0{,}083,\, 0{,}072,\, 0{,}067,\, 0{,}063,\, 0{,}055,\, 0{,}045,\, 0{,}028,\, 0{,}017,\, 0{,}010,\, 0{,}003,\, 0{,}002,\, 0,\, 0,\, 0]$$
donne $\mathbb{E}[Y_{\text{test}}] \approx 0{,}13$, avec 95 % des samples sous $0{,}3$. Donc même une prédiction médiocre ne s'éloigne pas tellement de $y$ en absolu.

**3. Décomposition d'un score réel observé**. Pour `score=0.00268 | err_diff=0.00080` :
$$\overline{e}_F \approx \overline{e}_M \approx 0{,}001,\quad |\overline{e}_F - \overline{e}_M| \approx 0{,}0008,$$
$$\text{score} = \tfrac{1}{2}(0{,}001 + 0{,}001) + 0{,}0008 = 0{,}0027 \;\checkmark$$

Conversion vers le MAE (intuition concrète) :
$$\text{MAE} \approx \sqrt{\overline{e}_G} = \sqrt{0{,}001} \approx 0{,}032,$$
soit le modèle prédit l'occlusion à **±3,2 points de pourcentage** près en moyenne. Pour la difficulté de la tâche, c'est bon.

**Implication pour interpréter les trials Optuna** :
- ne pas lire les scores en absolu : un passage de $0{,}00185$ à $0{,}00268$ semble petit mais c'est **+45 %** en relatif → trial nettement pire
- activer l'**affichage log** sur l'axe Y dans MLflow / optuna-dashboard
- les améliorations gagnantes se chiffrent en quelques $10^{-4}$ d'écart, c'est normal

---

### Poolings

#### Pooling 1 — CLS

Le plus simple : on garde le premier token (le [CLS] entraîné par DINO pour agréger globalement).

| Tenseur | Shape | Description |
|---|---|---|
| $X$ | $(B,\, N,\, D)$ | sortie du backbone (input du pool) |
| $\Phi = X[:, 0, :]$ | $(B,\, D)$ | features (input du head) |

- **Params appris dans le pool** : 0 (le [CLS] est déjà entraîné par le backbone)
- **Output dim** : $D' = D$
- **Coût** : nul
- **Quand l'utiliser** : baseline propre quand le backbone DINO/iBOT est de qualité ; suffisant si le signal est global et déjà capturé par [CLS]
- **Limite** : ne mélange pas l'info des patches au-delà de ce que le backbone a déjà fait

#### Pooling 2 — GeM (Generalized Mean)

Généralisation de mean et max (Radenović et al. 2018). $p$ apprenable contrôle la sharpness.

| Tenseur / paramètre | Shape | Description |
|---|---|---|
| $X$ | $(B,\, N,\, D)$ | input |
| $\tilde p$ | $()$ scalaire | paramètre brut |
| $p = \text{softplus}(\tilde p) + \varepsilon$ | $()$ | exposant strictement positif |
| $X[:, 1:, :]$ | $(B,\, N_p,\, D)$ | patches (skip [CLS] à l'index 0), $N_p = N - 1$ |
| $\Phi$ | $(B,\, D)$ | features pooled |

$$\Phi_{b, d} = \left( \frac{1}{N_p} \sum_{n=1}^{N_p} \max(X[b, n, d],\, \varepsilon)^{p} \right)^{1/p}.$$

- **Cas limites** : $p \to 1$ → mean ; $p \to \infty$ → max
- **Params appris dans le pool** : 1 (le scalaire $\tilde p$)
- **Output dim** : $D' = D$
- **Coût** : O($B \cdot N_p \cdot D$), négligeable
- **Limite v2 → v3** : sur features ViT non-activées (peuvent être négatives), le clamp à $\varepsilon$ tue la moitié du signal. Acceptable comme baseline ; reconsidérer si GeM perd contre les autres.

#### Pooling 3 — Attention K-query (le v2 actuel)

$K = n_{\text{focal}} + n_{\text{diffuse}} + n_{\text{free}}$ queries apprenables, chacune avec sa propre température $\tau_k$ apprenable.

| Tenseur / paramètre | Shape | Description |
|---|---|---|
| $X$ | $(B,\, N,\, D)$ | input du pool |
| $Q$ | $(K,\, D)$ | matrice des $K$ queries apprenables |
| $\log \tau$ | $(K,)$ | log-températures apprenables, $\tau_k = \exp(\log \tau_k) > 0$ |
| $W_K \in \mathbb{R}^{D \times D}$, $W_V \in \mathbb{R}^{D \times D}$ | $(D,\, D)$ | projections clé/valeur |
| $K_X = X W_K^\top$, $V_X = X W_V^\top$ | $(B,\, N,\, D)$ | clés/valeurs projetées |
| $A$ (attention weights) | $(B,\, K,\, N)$ | poids softmax par query sur les patches |
| pooled (avant flatten) | $(B,\, K,\, D)$ | $K$ vecteurs poolés par sample |
| $\Phi$ après concat + LayerNorm | $(B,\, K \cdot D)$ | features finales |

Init différenciée des $\tau$ pour forcer la diversité des rôles :
- $\tau_k = \tau_{\text{focal}}$ ≈ 0.1 pour les $n_{\text{focal}}$ premières queries → attention piquée, capture des indices locaux
- $\tau_k = \tau_{\text{diffuse}}$ ≈ 1.5 pour les $n_{\text{diffuse}}$ suivantes → attention plate, capture des signaux globaux
- $\tau_k = \tau_{\text{free}}$ = 1 pour les $n_{\text{free}}$ dernières → neutre

Forward :
$$\underbrace{\text{scores}_{b,k,n}}_{(B,K,N)} = \frac{\overbrace{Q_{k,:}}^{(D,)} \cdot \overbrace{K_X[b,n,:]}^{(D,)}}{\sqrt{D} \cdot \tau_k}, \qquad \underbrace{A_{b,k,:}}_{(N,)} = \text{softmax}_n(\text{scores}_{b,k,:}),$$
$$\underbrace{\Phi_b}_{(K \cdot D,)} = \text{LayerNorm}\!\left( \mathrm{concat}_{k=1}^{K} \underbrace{\sum_{n=1}^{N} A_{b,k,n} \cdot V_X[b,n,:]}_{(D,)} \right).$$

- **Output dim** : $D' = K \cdot D$ (par défaut $K = 6$, soit $6 \cdot 768 = 4608$ pour ViT-B/16)
- **Params appris dans le pool** : $K \cdot D$ (queries) + $K$ ($\log \tau$) + $2 D^2$ (projections K/V) + $2 \cdot K \cdot D$ (LayerNorm γ/β)
- **Coût** : O($B \cdot K \cdot N \cdot D$), modeste
- **Pénalité optionnelle** : `loss_query_diversity_lambda` ajoute une pénalité cosinus entre paires de queries pour empêcher le collapse. Notons $\widetilde A_{b,k,:} = A_{b,k,:} / \|A_{b,k,:}\|_2 \in \mathbb{R}^{N}$, alors :
$$\Omega_{\text{div}} = \frac{1}{B \cdot K(K-1)} \sum_b \sum_{k \neq j} \widetilde A_{b,k,:} \cdot \widetilde A_{b,j,:} \in [-\tfrac{1}{K-1}, 1].$$

#### Pooling 4 — Multi-Head Attention (MHA)

1 query apprenable découpée en $H$ heads de dim $D/H$. Pas de $\tau$ per-head — la diversité émerge des projections aléatoirement initialisées.

| Tenseur / paramètre | Shape | Description |
|---|---|---|
| $X$ | $(B,\, N,\, D)$ | input |
| $Q$ (la query unique) | $(D,)$ | paramètre apprenable |
| $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$ | $(D,\, D)$ | projections par-token |
| $W_{\text{out}} \in \mathbb{R}^{D \times D}$ | $(D,\, D)$ | projection de sortie |
| $q = W_Q Q$ réorganisée | $(H,\, D/H)$ | query splittée en $H$ heads |
| $k = W_K X$ réorganisée | $(B,\, H,\, N,\, D/H)$ | clés par-head |
| $v = W_V X$ réorganisée | $(B,\, H,\, N,\, D/H)$ | valeurs par-head |
| $A$ (attention weights) | $(B,\, H,\, N)$ | softmax par-head sur les patches |
| pooled (avant concat) | $(B,\, H,\, D/H)$ | un vecteur poolé par head |
| $\Phi = W_{\text{out}} \cdot \mathrm{concat}(\ldots)$ | $(B,\, D)$ | features finales |

Forward :
$$\underbrace{\text{scores}_{b,h,n}}_{(B,H,N)} = \frac{\overbrace{q_h}^{(D/H,)} \cdot \overbrace{k_{b,h,n,:}}^{(D/H,)}}{\sqrt{D/H}}, \qquad A_{b,h,:} = \text{softmax}_n(\text{scores}),$$
$$\underbrace{\Phi_b}_{(D,)} = W_{\text{out}} \cdot \mathrm{concat}_{h=1}^{H} \underbrace{\sum_n A_{b,h,n} \cdot v_{b,h,n,:}}_{(D/H,)}.$$

- **Output dim** : $D' = D$
- **Params appris dans le pool** : $D$ (query) + $4 D^2$ (4 projections linéaires) + $2D$ (LayerNorm)
- **Coût** : O($B \cdot H \cdot N \cdot D/H) = O(B \cdot N \cdot D)$, identique à un transformer block standard
- **Search space** : `num_heads ∈ {2, 4, 8, 16}` (diviseurs communs ViT-B/L/H+)
- **Contrainte** : $D$ doit être divisible par $H$. Pour ViT-B (D=768), ViT-L (D=1024), ViT-H+ (D=1280), les valeurs $\{2, 4, 8, 16\}$ sont toutes des diviseurs.

---

### Stratégies d'équilibrage & fairness

**Notations pour cette section** :

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

#### Stratégie A — Sampler gender + importance reweighting

```yaml
sampler_strategy: gender
loss_importance_reweight: true
```

- **Côté data** : `WeightedRandomSampler` avec poids $\propto 1/n_G$ → batches équilibrés F/M en attendu
- **Effet** : $n_{\text{samples/epoch}} = 2 \cdot \min(n_F, n_M)$ → perte de 20-40 % de données par epoch selon imbalance
- **Côté loss** : $w_i$ inclut $w^{\text{imp}}$ (corrige shift Y)
- **Théoriquement** : la plus propre, sépare les préoccupations (data corrige gender, loss corrige Y)
- **Pratiquement** : la perte de données peut dégrader la convergence vs D

#### Stratégie D — No sampler + importance + gender_reweight (no-op)

```yaml
sampler_strategy: none
loss_importance_reweight: true
loss_gender_reweight: true     # ← no-op, voir §"Train↔test shift"
```

- **`gender_rw`** : multiplie $w_i$ par une constante par-genre $c_{g_i}$. Mais cette constante **se factorise** dans $\overline{e}_G$ → effet nul (preuve ci-dessus)
- **Effet net** : équivalent à « no sampler + importance_rw seul »
- **Empiriquement** : bat A car utilise 100 % des données par epoch

#### Stratégie E — Cell reweight (joint Y × G)

```yaml
sampler_strategy: none
loss_cell_reweight: true
```

Construit $W^{\text{cell}} \in \mathbb{R}^{2 \times B}$ avec :
$$W^{\text{cell}}[g, b] = \frac{1}{\sqrt{|\{i : g_i = g \wedge b(y_i) = b\}|}}, \quad \text{normalisé à moyenne 1}.$$

Puis $w_i \mathrel{\*}= W^{\text{cell}}[g_i, b(y_i)]$.

- **Spécificité** : le poids **varie par bin Y à l'intérieur d'un genre** → **ne se factorise pas** dans $\overline{e}_G$ → c'est un vrai mécanisme actif (contrairement à `gender_rw`)
- **Effet** : compense la sous-représentation des cellules $(g, b)$ rares, sans utiliser $P_{\text{test}}$ (pas de correction de shift Y explicite)
- **Coût** : O($B$), nul à la création (poids pré-calculés)

#### Stratégie F — Sampler occlusion + gender_reweight (partiel)

```yaml
sampler_strategy: occlusion
loss_gender_reweight: true     # ← no-op
```

- **Sampler** : équilibre les buckets d'occlusion (10 buckets quantiles) → couverture uniforme sur Y → proxy implicite pour la correction de shift
- **`gender_rw`** : toujours no-op (cf. D)
- **Perte de données** : encore plus forte qu'A si les buckets sont déséquilibrés

#### Stratégie G — Adversarial debiasing (DANN)

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
| $b_1$ | $(256,)$ | bias |
| $W_2$ | $(2,\, 256)$ | deuxième couche linéaire (2 classes : F/M) |
| $b_2$ | $(2,)$ | bias |
| $\text{Disc}(\Phi)$ | $(B,\, 2)$ | logits gender |
| $\text{CE}(\text{Disc}, g)$ | $()$ scalaire | cross-entropy moyenne sur batch |

$$\underbrace{\text{Disc}(\Phi)}_{(B,2)} = \underbrace{W_2}_{(2,256)} \cdot \underbrace{\text{ReLU}\bigl(\underbrace{W_1}_{(256,D')} \cdot \underbrace{\Phi}_{(D',)} + b_1\bigr)}_{(256,)} + b_2 \quad \text{(broadcast sur le batch)}.$$

Loss adversariale :
$$\mathcal{L}_{\text{adv}} = \mathrm{CE}\bigl(\text{Disc}(\text{GRL}(\Phi)),\; g\bigr) \in \mathbb{R}_+, \qquad \mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{task}} + \lambda_{\text{adv}} \cdot \mathcal{L}_{\text{adv}}.$$

**Gradient Reversal Layer (GRL)** : identité au forward, sign-flip au backward.

| Direction | Opération | Shape input → output |
|---|---|---|
| forward | $\Phi \to \Phi$ (identité) | $(B, D') \to (B, D')$ |
| backward | $\nabla_\Phi \mathcal{L}_{\text{adv}} \to -\nabla_\Phi \mathcal{L}_{\text{adv}}$ | $(B, D') \to (B, D')$ |

**Conséquence du sign-flip** :
- Le **discriminateur** reçoit le gradient normal ($+\nabla$) → il **apprend à classifier** le genre depuis les features
- Le **backbone** reçoit le gradient inversé ($-\nabla$) → il **apprend à produire des features dont le genre n'est pas décodable**
- Équilibre dynamique entre les deux ; $\lambda_{\text{adv}}$ trop grand → backbone explose, $\lambda_{\text{adv}}$ trop petit → effet nul

**Coût total disc** :
- Pour pooling CLS/GeM/MHA ($D' = D = 768$) : $W_1$ a $256 \cdot 768 = 197$k params + $W_2$ a $2 \cdot 256 = 512$ params → ~198k params
- Pour pooling attention K-query ($D' = K \cdot D = 4608$ avec $K=6$) : $W_1$ a $256 \cdot 4608 = 1.18$M params

Négligeable vs backbone ViT-B (86M).

#### Stratégie H — MMD alignment

```yaml
sampler_strategy: none
loss_importance_reweight: true
loss_mmd_alignment: true
mmd_lambda: ε ∈ [0.01, 1.0] (log scale)
```

Aligne les distributions de features pooled entre F et M via Maximum Mean Discrepancy² dans un RKHS gaussien.

| Tenseur | Shape | Description |
|---|---|---|
| $\Phi$ | $(B,\, D')$ | features pooled, $D'$ dépend du pooling |
| $\Phi_F$ | $(n_F,\, D')$ | sous-matrice des samples F dans le batch |
| $\Phi_M$ | $(n_M,\, D')$ | sous-matrice des samples M dans le batch ($n_F + n_M = B$) |
| Matrice $K_{FF} = k(\Phi_F, \Phi_F^\top)$ | $(n_F,\, n_F)$ | kernel intra-F |
| Matrice $K_{MM}$ | $(n_M,\, n_M)$ | kernel intra-M |
| Matrice $K_{FM}$ | $(n_F,\, n_M)$ | kernel inter-genre |
| $\widehat{\text{MMD}^2}$ | $()$ scalaire | estimateur empirique |

Définition mathématique :
$$\text{MMD}^2(\Phi_F, \Phi_M) = \mathbb{E}_{x, x' \sim \Phi_F}[k(x, x')] + \mathbb{E}_{y, y' \sim \Phi_M}[k(y, y')] - 2 \, \mathbb{E}_{x \sim \Phi_F, y \sim \Phi_M}[k(x, y)].$$

Kernel RBF multi-bandwidth (somme de gaussiennes à différents $\sigma$ pour robustesse) :
$$k(x, y) = \frac{1}{|S|} \sum_{\sigma \in S} \exp\!\left( -\frac{\|x - y\|^2}{2 \sigma^2} \right) \in \mathbb{R}, \quad S = \{1, 5, 10\},\; x, y \in \mathbb{R}^{D'}.$$

Implémenté en batch via produits scalaires (formule de polarisation) :
$$\underbrace{\|x - y\|^2}_{(n_F, n_M)} = \underbrace{\|x\|^2}_{(n_F, 1)} + \underbrace{\|y\|^2}_{(1, n_M)} - 2 \underbrace{\langle x, y \rangle}_{(n_F, n_M)}.$$

**Estimateur empirique** sur batch :
$$\widehat{\text{MMD}^2} = \underbrace{\frac{1}{n_F^2} \sum_{i,j \in F} k(\phi_i, \phi_j)}_{\text{mean}(K_{FF})} + \underbrace{\frac{1}{n_M^2} \sum_{i,j \in M} k(\phi_i, \phi_j)}_{\text{mean}(K_{MM})} - \underbrace{\frac{2}{n_F n_M} \sum_{i \in F, j \in M} k(\phi_i, \phi_j)}_{\text{mean}(K_{FM})}.$$

Loss totale :
$$\mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{task}} + \lambda_{\text{MMD}} \cdot \widehat{\text{MMD}^2}(\Phi_F, \Phi_M).$$

- **Coût** : O($(n_F^2 + n_M^2 + n_F n_M) \cdot D'$), soit ~$B^2 \cdot D'$ par batch. Pour $B=32, D'=4608$ : ~4.7M flops par bandwidth × 3 bandwidths = ~14M flops. Négligeable.
- **Avantage vs DANN** : pas d'adversaire à entraîner → stable, peu de tuning
- **Limite** : aligne **toute** la distribution feature-space, alors qu'on voudrait seulement aligner conditionnellement sur Y
- **Edge case** : si $n_F = 0$ ou $n_M = 0$ dans le batch (rare avec sampler équilibré), MMD = 0 par convention

#### Stratégie I — Inter-gender Mixup

```yaml
sampler_strategy: none
loss_importance_reweight: true
mixup_inter_gender: true
mixup_alpha: ε ∈ [0.1, 0.5]
```

Pour chaque sample F dans le batch, on cherche un partenaire M dans le bucket Y le plus proche, puis on interpole.

| Tenseur | Shape | Description |
|---|---|---|
| $x$ (pixel_values) | $(B,\, C,\, H, W)$ | images du batch, $C = 3$, $H = W = 224$ |
| labels (concaténation y + g) | $(B,\, 2)$ | colonne 0 = y, colonne 1 = g |
| $f_{\text{idx}}$ | $(n_F,)$ | indices des samples F |
| $m_{\text{idx}}$ | $(n_M,)$ | indices des samples M |
| matrice de distance $|b(y_i^F) - b(y_j^M)|$ | $(n_F,\, n_M)$ | distance en bins entre chaque F et chaque M |
| $\lambda$ | $(n_F,)$ | poids du sample F, échantillonné par Beta |
| $\lambda_x = \lambda$ reshape | $(n_F,\, 1,\, 1,\, 1)$ | broadcast vers $(C, H, W)$ |
| $\lambda_y = \lambda$ | $(n_F,)$ | broadcast vers la dimension Y |
| $\tilde x$ (mixed images) | $(B,\, C,\, H,\, W)$ | tenseur retourné, F-slots remixés |
| $\tilde{\text{labels}}$ | $(B,\, 2)$ | labels retournés, F-slots avec Y interpolé |

**Étape 1 — pairing** : pour chaque sample $i \in f_{\text{idx}}$, on cherche son partenaire :
$$\text{partner}(i) = \arg\min_{j \in m_{\text{idx}}} |b(y_i) - b(y_j)|, \quad \text{tie-break par perm aléatoire}.$$

**Étape 2 — sampling du poids de mix** :
$$\lambda_i \sim \text{Beta}(\alpha, \alpha) \in (0, 1), \quad i \in f_{\text{idx}}.$$

Pour $\alpha \in [0.1, 0.5]$, la distribution Beta est U-shaped → $\lambda$ concentré près de 0 ou 1 (mix doux : mostly-F ou mostly-M).

**Étape 3 — interpolation** :
$$\underbrace{\tilde x_i}_{(C,H,W)} = \lambda_i \cdot \underbrace{x_i^F}_{(C,H,W)} + (1 - \lambda_i) \cdot \underbrace{x_{\text{partner}(i)}^M}_{(C,H,W)},$$
$$\tilde y_i = \lambda_i \cdot y_i^F + (1 - \lambda_i) \cdot y_{\text{partner}(i)}^M, \quad \tilde g_i = g_i^F = 0 \text{ (inchangé)}.$$

**Étape 4 — sortie** : on retourne `(mixed_pixel_values, mixed_labels)`, les samples M restent intouchés ; ce nouveau batch est passé au forward+compute_loss normalement.

- **Coût** : O($n_F \cdot n_M$) pour la matrice de distance argmin + O($n_F \cdot C \cdot H \cdot W$) pour les interpolations. Tout en GPU. Négligeable.
- **Avantage** : zéro params nouveaux, juste de la data aug
- **Si un batch n'a que F ou que M** : pas de mix possible, batch retourné tel quel (early-return dans `inter_gender_mixup`)

---

### Résumé : matrice méthode × objectif

| | Strat. | Sampler | Imp_rw | Mécanisme actif | Cible théorique |
|---|---|---|---|---|---|
| A | gender | ✓ | ✓ | aucun (sampler suffit pour G) | shift Y + fairness G via sampler |
| D | none | ✓ | gender_rw=no-op | aucun nouveau | shift Y + fairness G via aggregation par-groupe |
| E | none | ✗ | cell_rw | cell joint $Y \times G$ | équité Y×G implicite |
| F | occlusion | ✗ | gender_rw=no-op | aucun nouveau | shift Y via sampler |
| **G** | none | ✓ | **DANN** | fairness G via features invariantes au gender |
| **H** | none | ✓ | **MMD** | fairness G via alignement de distributions de features |
| **I** | none | ✓ | **Mixup inter-G** | invariance par augmentation conditionnelle Y |

→ **A et D** s'attaquent au shift Y avec la fairness G en sous-produit (sampler / aggregation).
→ **E** est joint Y×G sans estimer $P_{\text{test}}$.
→ **F** est l'inverse d'A (sampler sur Y, pas explicitement sur G).
→ **G/H/I** ajoutent des mécanismes de fairness *au niveau des features*, qui complètent (et non remplacent) le shift Y handling via `loss_importance_reweight=True`.

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

Run with `sbatch scripts/pretrain_ibot_vith16plus_2x3090.sh` (or `./scripts/chain_pretrain.sh 3` to chain three 30 h SLURM submissions). Output: an MLflow run with the student encoder logged as `runs:/<run_id>/encoder`.

**v3 workflow** — pour utiliser ce pretrain custom dans Optuna, fill l'URI dans la choice `ibot:runs:/<run_id>/encoder` du search space `pretrained_source` du yaml d'architecture (par exemple `configs/architectures/dinov3-vitb16-3090-v3.yaml`). Optuna comparera alors automatiquement :
- `"lvd"` (ou `"sapiens_default"` pour Sapiens2) : poids de base Meta
- `"ibot:runs:/<run_id>/encoder"` : notre pretrain custom par-dessus

`train.py` charge les poids dans le backbone et copie tous les `pretrain_*` params dans la run finetune pour la traçabilité complète. Une validation runtime ([`_validate_pretrained_source_choices`](src/optimize.py)) vérifie l'existence des runs MLflow avant le démarrage du sweep — pas de trial gâché sur un `runs:/__FILL__/encoder` oublié.

### Why we noticed this late

Honest postmortem: the previous iterations of this project anchored on "MAE" early because HuggingFace provides a ready-to-use `ViTMAEForPreTraining`. We thought implementation-first instead of objective-first. The cue was always there — Meta's iBOT loss is vendored in `dinov3_repo/dinov3/loss/ibot_patch_loss.py` and DINOv3's own ViT has a native `mask_token` and `prepare_tokens_with_masks` method designed exactly for masked-token forward passes. We should have looked at the vendored code earlier.

---

## Branches

- **`main`** — version v1 : `FaceOccRegressor` avec pooling simple (cls / mean / max / attention single-query) + tête MIL auxiliaire (`patch_head` + `loss_patch_mil_alpha`)
- **`v2-attention-pooling`** — version v2 : `FaceOccRegressor` avec **K=6 multi-head attention pooling** à températures mixtes (focal/diffuse/free), apprises ; tête MIL retirée ; 3 niveaux de régularization séparés

Voir la section [v2 attention pooling](#v2-attention-pooling) pour les détails de l'architecture v2.

---

## v2 attention pooling

Implémenté sur la branche `v2-attention-pooling`. Inspiré de [Set Transformer (PMA)](https://arxiv.org/abs/1810.00825), [Perceiver IO](https://arxiv.org/abs/2107.14795), et [Dual-Attention MIL (2024)](https://www.mdpi.com/2079-9292/13/22/4445).

### Motivation

Le pooling v1 (mean / cls / attention single-query) compresse `(B, N, D)` → `(B, D)` en jetant 98 % de l'info. Avec un DINOv3 pretrained qui produit des embeddings très riches par patch (768 dims pour ViT-B/16), c'est gaspiller.

L'architecture v2 garde la richesse : `(B, N, D)` → K queries attendent indépendamment sur tous les patches → `(B, K·D)` → tête finale → scalaire.

### Architecture détaillée

```
                       backbone (DINOv3 / Sapiens2)
                                  │
                          (B, N+1, D)   ← +1 = CLS
                                  │
                       ┌──────────┴──────────┐
                       │ AttentionPooling    │
                       │                     │
                       │  K=6 queries:       │
                       │  • 2 focal (τ=0.1)  │ ← capture occluders localisés (régime 1)
                       │  • 2 diffuse (τ=1.5)│ ← capture qualité globale (régime 2)
                       │  • 2 free (τ=1.0)   │ ← libre de spécialiser
                       │                     │
                       │  τ apprises via log_tau = nn.Parameter
                       │  (positivité garantie par exp)
                       │                     │
                       │  scores = Q·K / (scale × τ)
                       │  weights = softmax(scores)        ← dropout pool_attn_dropout
                       │  pooled = weights @ V             (par query)
                       └──────────┬──────────┘
                                  │
                          (B, K=6, D)
                                  │
                          flatten → (B, K·D)
                                  │
                              LayerNorm                    ← dropout pool_proj_dropout
                                  │
                            [projection]                   ← optionnel, Linear→Norm→GELU→Dropout
                                  │
                              dropout                      ← head_dropout
                                  │
                              Linear                       ← (K·D, 1)
                                  │
                              sigmoid
                                  │
                              scalar prediction
```

### Les 3 niveaux de régularization (yaml params)

| Niveau | Param | Effet |
|---|---|---|
| **Backbone** | `backbone_drop_path_rate` ∈ [0, 0.2] | Stochastic depth dans le ViT (DINOv3) / `drop_rate` (Sapiens2). Passé au constructor du backbone. |
| **Pool** | `pool_attn_dropout` ∈ [0, 0.3] | Dropout appliqué sur les **poids d'attention** dans la pool (entre softmax et la multiplication par V). |
| **Pool** | `pool_proj_dropout` ∈ [0, 0.3] | Dropout sur la **sortie agrégée** de la pool (après LayerNorm, avant la tête). |
| **Head** | `head_dropout` ∈ [0, 0.3] | Dropout avant le Linear final (et dans la projection optionnelle). |

### Query diversity penalty

Au-delà des dropouts, on peut **pénaliser la redondance entre queries** :

```python
loss = main_loss + λ_div × diversity_penalty(attn_weights)
```

où `diversity_penalty` = moyenne des cosinus pairwise entre les K=6 distributions d'attention :
- 0 si queries totalement orthogonales (attention sur des patches disjoints)
- 1 si queries identiques (collapse)

Param yaml : `loss_query_diversity_lambda ∈ [0, 0.2]` (Optuna search). À λ=0, aucune contrainte ; à λ=0.2, forte pression vers diversité.

### Structure des K=6 queries (yaml params)

| Param | Default | Notes |
|---|---|---|
| `n_focal` | 2 | Queries init avec τ basse → softmax sharp → attention concentrée (occluders) |
| `n_diffuse` | 2 | Queries init avec τ haute → softmax flat → attention uniforme (qualité globale) |
| `n_free` | 2 | Queries init avec τ=1 → neutres |
| `tau_focal_init` | 0.1 | (Optuna explore [0.05, 0.3]) |
| `tau_diffuse_init` | 1.5 | (Optuna explore [1.0, 3.0]) |
| `tau_free_init` | 1.0 | Fixé en pratique |
| `learnable_tau` | true | Si false, les τ restent fixes à leurs valeurs init |

Au cours du training, les τ apprises peuvent **diverger arbitrairement** — la spécialisation focale/diffuse n'est qu'un prior d'init, pas une contrainte stricte. Si le modèle décide qu'il a besoin de 6 queries diffuses, il peut converger là.

### Workflow v2

```bash
# Bascule sur la branche v2:
git checkout v2-attention-pooling

# Lance le HPO (search_space inclut tous les nouveaux params):
sbatch scripts/optimize_dinov3_vitb16_2x3090.sh

# Compare avec main (v1, MIL+simple pooling) via MLflow UI
```

L'A/B v1 vs v2 sur ViT-B/16 dira si l'enrichissement architectural vaut le coup avant d'investir sur le H+/16.

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
│   ├── face_occ_regressor.py    FaceOccRegressor (HF + DINOv3 dispatch)
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
| 28 | **Per-patch 2-channel decomposition** — per patch: `(occ_p, valid_p) = (sigmoid(head_occ), sigmoid(head_valid))`. Global ratio = `Σ occ_p · valid_p / Σ valid_p`. Adds sparsity regularizer `λ_sparsity · max(0, mean(valid_p) − 0.85)` to force ~15 % background suppression without external face mask. Identifiable by construction (vs the 3-class permutation idea which suffers from inter-image inconsistency). | ~80 LOC in `face_occ_regressor.py` + `losses.py` | force la localisation valid/invalid sans dépendance externe, capture l'intuition "patches utiles seulement" | encore underdetermined sans face mask (modèle peut tricher valid=1 partout) — la régul sparsity est un workaround soft |
| 29 | ~~**Rich pooling head**~~ — **IMPLEMENTÉ** sur la branche [`v2-attention-pooling`](#v2-attention-pooling) (K=6 multi-head attention pooling, mixed temperature init, learnable τ, dual-attention régime 1/2 prior). Remplace MIL + simple pooling sur cette branche. | done | gain attendu si HPO révèle des τ apprises divergentes (focal vs diffuse) | nécessite Optuna fresh study car search_space change |

---

## SOTA references

### Backbones & SSL
- **DINOv3** — Meta, vendored at `src/models/dinov3_repo/`. ViT + DINO + iBOT + KoLeo + Gram losses.
- **DINOv2** — Oquab et al. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- **MAE** — He et al. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- **iBOT** (image BERT-like SSL) — Zhou et al. [arXiv:2111.07832](https://arxiv.org/abs/2111.07832)
- **Sapiens** — Khirodkar et al., Meta foundation for human vision (1B human images)
- **Sapiens 2** — Meta, custom architecture (RoPE + GQA + SwiGLU + RMSNorm), [github.com/facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2)
- **ConvNeXt v2** — Woo et al. [arXiv:2301.00808](https://arxiv.org/abs/2301.00808)
- **EVA-02** — Fang et al. [arXiv:2303.11331](https://arxiv.org/abs/2303.11331)

### Face image quality / occlusion analysis (most relevant to our task)
- **A Comprehensive Review of Face Detection Techniques for Occluded Faces** (CMES 2025) — [techscience.com/CMES/v143n3/62822](https://www.techscience.com/CMES/v143n3/62822/html). Survey 4 categories: feature-based, ML, DL, hybrid. Note la trend ViT/Swin.
- **A Survey of Face Recognition Techniques under Occlusion** (IET Biometrics 2021) — [Zeng et al.](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/bme2.12029). 3 stratégies: visible-parts → reconstruction, fusion sub-regions, adversarial.
- **CLIB-FIQA** (CVPR 2024) — Ou et al. Confidence calibration for face quality. [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Ou_CLIB-FIQA_Face_Image_Quality_Assessment_with_Confidence_Calibration_CVPR_2024_paper.pdf)
- **CR-FIQA** (Boutros 2023) — Sample relative classifiability for FIQA
- **FaceQNet** — Hernández-Ortega et al. [arXiv:1904.01740](https://arxiv.org/abs/1904.01740)
- **AOFD: Adversarial Occlusion-aware Face Detection** — segmentation + detection jointly. [arXiv:1709.05188](https://arxiv.org/abs/1709.05188)

### Multiple Instance Learning (MIL) + attention pooling
- **Attention-based Deep MIL** — Ilse et al. 2018. [arXiv:1802.04712](https://arxiv.org/abs/1802.04712). Notre baseline conceptuel.
- **Set Transformer / PMA (Pooling by Multihead Attention)** — Lee et al. 2019. [arXiv:1810.00825](https://arxiv.org/abs/1810.00825). Base théorique du multi-head attention pooling (levier #29).
- **Perceiver IO** — Jaegle et al. 2021. [arXiv:2107.14795](https://arxiv.org/abs/2107.14795). K-query attention pour set→scalar.
- **Rethinking Attention-Based MIL** (2024) — [arXiv:2404.00351](https://arxiv.org/abs/2404.00351). État de l'art MIL attention.
- **Dual-Attention MIL** (Electronics 2024) — 2 attentions parallèles pour WSI classification. [MDPI](https://www.mdpi.com/2079-9292/13/22/4445). **Base de notre levier #30 dual-attention régime 1/2.**
- **CAMIL: Channel Attention MIL** (Bioinformatics 2025) — [Oxford Academic](https://academic.oup.com/bioinformatics/article/41/2/btaf024/7958575)
- **Neighborhood Attention MIL** (2024) — locality + attention pour WSI. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11390382/)

### Fairness in face analysis
- **Group-DRO** — Sagawa et al. *Distributionally Robust Neural Networks for Group Shifts*. [arXiv:1911.08731](https://arxiv.org/abs/1911.08731). Implémenté dans `src/utils/losses.py:GroupDROLoss`.
- **Component-Based Fairness in Face Attribute Classification** (FAccT 2025) — Bayesian network + meta-learning. [arXiv:2505.01699](https://arxiv.org/abs/2505.01699)
- **Toward Fairer Face Recognition Datasets** (2024) — [arXiv:2406.16592](https://arxiv.org/abs/2406.16592)

### Optimization / regularization
- **EMA / SWA** — Izmailov et al. [arXiv:1803.05407](https://arxiv.org/abs/1803.05407). Implémenté.
- **LLRD** — Howard & Ruder (ULMFiT). [arXiv:1801.06146](https://arxiv.org/abs/1801.06146). Implémenté.
- **Focal Loss** — Lin et al. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002). Adapté en `loss_focal_gamma`.
- **Importance weighting / covariate shift** — Shimodaira 2000 (classical reference for `w = p_test/p_train`)

### Notre niche : continuous occlusion ratio regression
Aucune publication ne fait **exactement** notre tâche (Idemia metric `(Err_F + Err_M)/2 + |Err_F − Err_M|` avec `w = 1/30 + GT`). Le champ FIQA produit des scores de qualité multi-facteur (incluant occlusion comme une dimension parmi d'autres), mais pas de regression isolée sur le ratio d'occlusion. → **Setup spécifique au challenge**, on combine des briques validées individuellement.

---

## Hardware notes

| Platform | VRAM | Precision | Max DINOv3 (this repo) |
|---|---|---|---|
| 1× P100 | 16 GB | FP16 | ViT-S/16 (shipped, comfortable) |
| 2× P100 DDP | 32 GB | FP16 | ViT-L+/16 (requires manual weight DL) |
| 1× RTX 3090 | 24 GB | BF16 | ViT-L/16 |
| 2× RTX 3090 DDP | 48 GB | BF16 | ViT-H+/16 (requires manual weight DL) |

All SLURM scripts use `torchrun --nproc_per_node=2`, 30 h time, with the right hardware tags. On Linux nodes `uv sync` pulls CUDA 12.6 wheels via the `[tool.uv.sources]` marker; on Mac it falls back to the default PyPI index (CPU/MPS). torch is pinned to `>=2.7,<3.0` to keep MPS autograd working for local sanity tests.
