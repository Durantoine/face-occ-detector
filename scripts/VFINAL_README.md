# convnextv2-base-vfinal — recette v4 gagnante + pondération continue v36 + aug

Branche **`cora/convnext-vfinal`** (lignée `cora/convnext-baseline` = framework v4).
Reprend le **trial-18 v4 gagnant** (pooling `cls`, **gender batch sampler**, aug `light`,
lr 3.1e-5) mais remplace la pondération fairness **binnée** (`cell_joint`) par la
**pondération CONTINUE de v36** :

    w(y, g) = P_test(y) / (P_train(y, g) + is_lambda)      (KDE gaussien + spline P_test)

- **per-gender ("joint")** = `cell_joint` en continu → up-pondère la queue occludée, surtout
  les **hommes fortement occludés** (cellule rare), down-pondère le bulk peu occludé.
- **un seul poids partagé loss + eval** → plus d'artefact de binning (la queue ne s'effondre
  plus à 0.067 comme dans v4 ; cf. `scripts/plot_distributions.py`).
- `is_lambda` (=0.10) = plancher de régularisation sur `P_train` (dompte la variance de queue).

## Mécanique fairness (= v4, mais continue)
1. **Gender batch sampler** (`sampler_strategy: gender`) — équilibrage F/M au niveau data.
2. **Continuous cell_joint** (`loss_rw_strategy: continuous_joint`) — poids `w(y,g)` dans la loss.

## Fichiers
- `src/utils/continuous_weight.py` — la pondération continue (porté de v36).
- `scripts/plot_distributions.py` — trace densités + ratios (3 panneaux). 
- `configs/architectures/convnextv2-base-vfinal.yaml` — la config.
- Loss/eval câblés : `src/utils/losses.py`, `src/utils/metrics.py`, `src/train.py`.

## Données d'augmentation (data/aug)
Le pool d'occluders synthétiques est appendé au **TRAIN uniquement** (drop anti-fuite val via
`source_filename`). `data/` est gitignoré → il faut **mettre les images sur la machine de run** :
```
data/aug/                       # images À PLAT (matchées par basename)
    p1_syn_000000.webp ...
    synthetic.csv               # colonnes: filename, FaceOcclusion, gender, source_filename
```
(295 images du pilote pour l'instant ; le run 13k se branchera au même endroit.)

## Lancer
```bash
# tracer les distributions/ratios (sanity check de la pondération) :
python scripts/plot_distributions.py --csv data/raw/train.csv --target joint --out docs/assets/ratio_regularized.png

# entraînement (framework v4) :
FACE_OCC_ARCH=convnextv2-base-vfinal python src/train.py     # ou via le bootstrap/sbatch du repo
```

## Limites (honnête)
- Implémenté et **validé en isolation** (config, pondération continue sur données réelles, compile),
  mais **PAS** exécuté end-to-end sur GPU ici (pas de carte locale). À vérifier au 1er run :
  le `weight_fn` fait un aller-retour numpy par batch (OK pour batch 32) ; le chargement des
  images `data/aug` (chemins absolus) ; la clé `metric_for_best_model`.
- `scipy` ajouté aux deps (KDE via sklearn, spline via scipy) → `uv sync` requis.
