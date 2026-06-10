# Ré-estimation de la perf VAL du meilleur run v4 (trial 18) — repondération corrigée

Cette branche (`cora/v4-best-eval`, lignée `cora/convnext-baseline`) reproduit **exactement**
la config du meilleur run v4 et fournit un script pour **recharger le modèle entraîné, prédire
sur la val complète, et recalculer le score challenge** avec une repondération d'importance
**dont la queue ne s'effondre pas**.

- Config exacte du run : [`configs/architectures/convnextv2-base-v4-best.yaml`](../configs/architectures/convnextv2-base-v4-best.yaml)
  (= `trial_18.yaml`, pooling `cls`, sampler `gender`, feature_fairness `none`, aug `light`, lr 3.11e-5…).
- Script : [`scripts/eval_v4_reweighted.py`](eval_v4_reweighted.py)

## Pourquoi
Le score « officiel » de v4 (0.001005) utilise un poids `w = (1/30+y) × ratio`, où
`ratio = P_test/P_train` par bin. Or la PMF test supposée décroît à la queue → le ratio
**s'effondre à ~0.067** pour `y>0.34` : les pires échantillons ne pèsent presque rien → score
optimiste. Ce script floore le ratio (`--tail-floor`) pour rendre la queue son poids, et sort
le **détail par bin** pour rendre visible le (manque de) support en queue.

## Ce qu'il faut sur la machine (GPU de préférence)
1. **Ce repo, branche `cora/v4-best-eval`** :
   ```bash
   git clone https://github.com/Durantoine/face-occ-detector && cd face-occ-detector
   git checkout cora/v4-best-eval
   uv sync --no-dev            # ou: pip install -e . / les deps du projet
   ```
2. **Le modèle entraîné** (logged-model mlflow du run 54e63c3a) — à copier depuis le cluster :
   ```bash
   # depuis le cluster (≈350 Mo) :
   scp -r cluster:'~/face-occ-detector/mlruns/1/models/m-8914eed6b51740e4ad43667735756f13' \
          ./mlruns/1/models/
   ```
3. **`data/raw/train.csv`** + **les images** sous `data/raw/` (mêmes que l'entraînement :
   `data/raw/database{1,2,3}/...`). Extraire `crops.zip` ici si besoin.

## Lancer (val complète, 20k images)
```bash
python scripts/eval_v4_reweighted.py \
    --model-uri mlruns/1/models/m-8914eed6b51740e4ad43667735756f13/artifacts \
    --data-csv data/raw/train.csv --image-dir data/raw \
    --val-seed 276 --tail-floor 0.5
```
- `--val-seed 276` reproduit **exactement** la val du trial 18 (= `42 + 18×13`). Ne pas changer.
- `--tail-floor` : plancher du ratio par bin (0.5 par défaut ; mets `0` pour désactiver, ou
  une autre valeur pour tester la sensibilité).
- CPU possible (`--device cpu`) mais lent sur 20k images → préférer un GPU.

## Sortie
- Tableau **3 pondérations** : `raw` (1/30+y), `test_estimated` (ratio original qui s'effondre),
  `corrected` (ratio flooré) — avec err_F / err_M / gap / ESS.
- Détail **par bin** (`results/v4_reweighted_eval.csv`) : n, MSE, ratio original vs flooré,
  part de poids — pour voir où le score se concentre et où le support manque.

## Limites à garder en tête
- **C'est la val de sélection** (seed 276, celle sur laquelle le run a early-stoppé) → score
  intrinsèquement **optimiste** ; ce n'est pas un holdout indépendant (v4 n'en a pas).
- **EMA** (decay 0.9998) : les poids mlflow sauvés peuvent légèrement différer des poids
  EMA-évalués → le `raw` reproduit peut s'écarter un peu du 0.000904 loggé. La **comparaison
  relative entre pondérations** (l'objet du script) reste valide.
- **Rareté de la queue** : la val n'a que ~5 échantillons `y>0.5`. Même avec un ratio non
  effondré, l'extrême queue reste **bruitée** — c'est une limite de **données**, pas de pondération.
