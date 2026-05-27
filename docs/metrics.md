# Métriques — guide complet

Cette section documente **les 14 métriques** que [`compute_score()`](../src/utils/metrics.py) calcule à chaque évaluation. Comprendre ces métriques est fondamental pour interpréter les sweeps Optuna et le `qualitative_viewer`.

---

## La métrique du challenge

Formule :

$$\text{score} = \tfrac{1}{2}(\overline{e}_F + \overline{e}_M) + \lvert\overline{e}_F - \overline{e}_M\rvert$$

avec

$$\overline{e}_G = \frac{\sum_{i:\, g_i = G} w_i \cdot (\hat y_i - y_i)^2}{\sum_{i:\, g_i = G} w_i}, \qquad w_i = \tfrac{1}{30} + y_i$$

→ La pondération `w_i = 1/30 + GT` rend les fortes occlusions plus importantes.
→ Le terme `|err_F − err_M|` force la fairness genre.
→ Les moyennes intra-groupe ($\overline{e}_F$, $\overline{e}_M$) font que **le déséquilibre F/M dans le train n'affecte pas la métrique** (chaque groupe est traité indépendamment).

---

## Convention de nommage

Toutes les métriques v4 utilisent un **suffixe explicite** :

| Suffixe | Calcul | À quoi ça sert |
|---|---|---|
| `_val` | mesure **directe** sur le validation set, sans pondération de shift | Voir la perf brute du modèle sur val |
| `_test_estimated` | val reweighté par $w^{\text{imp}}[b(y_i)] = P_{\text{test}}[b]/P_{\text{train}}[b]$ | **Estimation non-biaisée** de la perf sur le test set |

→ Quand `loss_importance_reweight: false` et `eval_importance_reweight: false`, les deux versions sont identiques.
→ Quand activé, l'écart `_test_estimated` − `_val` mesure directement **l'effet de la correction de shift** sur la métrique.

---

## Les 8 métriques officielles du challenge

| Métrique | Sens | Type |
|---|---|---|
| `challenge_score_test_estimated` | **LA cible Optuna** — score du challenge, reweighté à $P_{\text{test}}$ | scalar, à minimiser |
| `challenge_score_val` | Score du challenge sur val direct (sans correction shift) | scalar, diagnostic |
| `err_F_test_estimated` | $\overline{e}_F$ reweighté à $P_{\text{test}}$ | scalar |
| `err_M_test_estimated` | $\overline{e}_M$ reweighté à $P_{\text{test}}$ | scalar |
| `err_diff_test_estimated` | $\lvert\overline{e}_F - \overline{e}_M\rvert$ reweighté | scalar, **mesure de fairness genre** |
| `err_F_val`, `err_M_val`, `err_diff_val` | versions val direct | scalar |

Le `metric_for_best_model: eval_challenge_score_test_estimated` dans les yaml v4 fait que **HF Trainer charge le meilleur checkpoint sur cette métrique**, et Optuna l'optimise.

### Variantes "matched" (quantile mapping) et "best"

| Métrique | Sens |
|---|---|
| `eval_challenge_score_matched_test_estimated` | Score après quantile-matching des preds vers `P_test` (cf [fairness.md](fairness.md) §"Mécanisme 3") |
| `eval_challenge_score_best_test_estimated` | `min(raw, matched)` — score qu'on submetterait réellement |

→ **Optuna optimise `best`** : si le matching aide, `eval_score` est réécrit à `s_matched` (cf [src/train.py:832-833](../src/train.py#L832-L833)) puis retourné comme objectif Optuna.

---

## Les 6 métriques de référence (interprétation humaine)

Pas optimisées directement, mais essentielles pour comprendre **qualitativement** où en est le modèle.

| Métrique | Formule | Lecture | Plage |
|---|---|---|---|
| `mse_val` | $\frac{1}{N}\sum (\hat y - y)^2$ | MSE non-pondérée — homogène à $Y^2$ (illisible) | $[0,\, 0{,}25]$ typique |
| `mae_val` | $\frac{1}{N}\sum \lvert\hat y - y\rvert$ | MAE — même unité que $Y$ | $[0,\, 0{,}5]$ typique |
| **`mae_pct_val`** | $\text{MAE} \times 100$ | **"le modèle se trompe en moyenne de X points d'occlusion"** | $[0,\, 50]$ % typique |
| **`mae_pct_test_estimated`** | idem, reweighté à $P_{\text{test}}$ | Même intuition mais estimation test perf | idem |
| **`r2_val`** | $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$ | **0 = modèle trivial, 1 = parfait** | $(-\infty, 1]$ |
| **`r2_test_estimated`** | idem, reweighté à $P_{\text{test}}$ | Idem pour le test | $(-\infty, 1]$ |

### Pourquoi pas `1 - MAE` comme "accuracy"

Tentant mais **trompeur**. La distribution de $Y$ est concentrée près de 0 ($\mathbb{E}[Y] \approx 0{,}13$). Donc un modèle trivial qui prédit toujours 0 obtient :

$$\text{MAE}_{\text{trivial}} = \mathbb{E}[\lvert 0 - Y\rvert] = \mathbb{E}[Y] \approx 0{,}13$$
$$1 - \text{MAE}_{\text{trivial}} \approx 0{,}87 = \text{"87 % accuracy"}$$

→ Le modèle inutile ressemble à du 87 % bon. **R² évite ce piège** car il normalise par la variance de $Y$ : un modèle trivial obtient $R^2 = 0$ par construction.

---

## Ordres de grandeur typiques

```
Modèle bien entraîné (~5 epochs ViT-B/16 + iBOT) :
  challenge_score_test_estimated ≈ 0.0025 - 0.0040
  mae_pct_test_estimated         ≈ 3 - 5 points de %
  r2_test_estimated              ≈ 0.85 - 0.95
  err_diff_test_estimated        ≈ 0.0005 - 0.0010  (gap F-M résiduel)

Modèle trivial (predict mean) :
  challenge_score ~ 0.03
  mae_pct ~ 13 %
  r2 = 0
```

### Pourquoi le score est si petit

**1. La métrique est une MSE pondérée → erreurs au carré**. Avec $|\hat y - y| \approx 0{,}05$, on a $e_i = 0{,}0025$.

**2. Les valeurs de $Y$ sont concentrées près de 0**. PMF test :
$$P_{\text{test}} = [0{,}105,\, 0{,}090,\, 0{,}090,\, 0{,}090,\, 0{,}092,\, 0{,}088,\, 0{,}083,\, 0{,}072,\, 0{,}067,\, 0{,}063,\, 0{,}055,\, 0{,}045,\, 0{,}028,\, 0{,}017,\, 0{,}010,\, 0{,}003,\, 0{,}002,\, 0,\, 0,\, 0]$$
donne $\mathbb{E}[Y_{\text{test}}] \approx 0{,}13$, avec 95 % des samples sous $0{,}3$.

**3. Décomposition d'un score réel**. Pour `score=0.00268 | err_diff=0.00080` :
$$\overline{e}_F \approx \overline{e}_M \approx 0{,}001,\quad |\overline{e}_F - \overline{e}_M| \approx 0{,}0008,$$
$$\text{score} = \tfrac{1}{2}(0{,}001 + 0{,}001) + 0{,}0008 = 0{,}0027 \;\checkmark$$

Conversion vers le MAE (intuition concrète) :
$$\text{MAE} \approx \sqrt{\overline{e}_G} = \sqrt{0{,}001} \approx 0{,}032,$$
soit le modèle prédit l'occlusion à **±3,2 points de pourcentage** près en moyenne.

**Implication pour interpréter les trials Optuna** :
- ne pas lire les scores en absolu : un passage de $0{,}00185$ à $0{,}00268$ semble petit mais c'est **+45 %** en relatif → trial nettement pire
- activer l'**affichage log** sur l'axe Y dans MLflow / optuna-dashboard
- les améliorations gagnantes se chiffrent en quelques $10^{-4}$ d'écart, c'est normal

---

## Comment lire les métriques en pratique

**Dans MLflow** :
- Tous les noms apparaissent avec le préfixe `eval_` (par exemple `eval_challenge_score_test_estimated`)
- Active l'**axe Y log** sur les courbes — les scores sont en $10^{-3}$, l'échelle linéaire écrase tout
- Pour comparer 2 trials : un passage de 0,00185 à 0,00268 = **+45 % en relatif** (significatif), pas "petit" !

**Dans Optuna dashboard** :
- L'objectif est `challenge_score_test_estimated` (ou `_best_test_estimated` après quantile matching)
- Les `user_attrs` exposent `err_F`, `err_M`, `err_diff`, `eval_loss`

**Dans le qualitative_viewer** :
- Panneau "Métrique du challenge" : `score`, `err_F`, `err_M`, `err_diff` côte-à-côte avec leurs versions test-estimated
- Panneau "Métriques humaines" : `mae_pct` (en %) et `r2`
- Onglet 📊 **Diagnostic charts** : les graphes ci-dessous

---

## Diagnostic charts (générés à chaque trial)

Quand `save_qualitative_k > 0` dans le yaml, [`_save_diagnostic_charts()`](../src/train.py) génère un PNG dans `qualitative/diagnostics/error_vs_occlusion_and_density.png`, automatiquement uploadé comme artifact MLflow.

C'est un **4-panel** (16×10 inches) :

### Panel A — MAE par bin × genre, **brute** (avec CI 95%)

Vue intrinsèque : où le modèle pèche absolument, sans pondération. Courbes F (rouge) et M (bleu) avec error-bars ±1.96·SE par bin.

**Lecture** :
- Si la courbe overall plafonne haut en bin-Y élevé → le modèle se casse sur les fortes occlusions
- Si F est **systématiquement au-dessus** de M → biais structurel
- Si l'écart F-M **explose** dans les bins haut-Y → fairness vient des cas extrêmes

### Panel B — Contribution au score par bin × genre

$$\text{contribution}_{G,b} = \frac{\sum_{i \in G \wedge b(y_i)=b} w_i \cdot (p_i - y_i)^2}{\sum_{i \in G} w_i}$$

Vue **réelle dans la loss** : un gros gap à haut-Y avec peu de samples → barre microscopique ici → l'écart n'a pas d'impact sur le score officiel. C'est ÇA qui pilote `err_F`, `err_M`, `err_diff`.

→ Quand `eval_pmf_ratio` est appliqué, les contributions matchent `err_*_test_estimated` (= cible Optuna). Sinon elles matchent `err_*_val`.

### Panel C — Densité de samples par bin × genre

Contexte : où vivent les données, où le ratio F/M se déforme. Permet de voir d'un coup d'œil pourquoi un bin haut-Y a une grosse MAE mais une faible contribution (peu de samples).

### Panel D — Distribution d'erreur par genre

Histogramme de $\lvert\hat y - y\rvert$ split F vs M, avec lignes verticales aux moyennes (MAE_F, MAE_M).

**Lecture** :
- Deux distributions superposées → modèle bien calibré sur les genres
- Distribution F shiftée vers la droite → biais systématique
- Queue lourde sur F (mais pas sur M) → quelques très mauvaises preds polluent la moyenne, pas un biais global

→ Ces 4 vues sont **complémentaires** : A donne l'intuition brute, B donne l'impact réel, C donne le contexte data, D donne la forme de la distribution d'erreur.
