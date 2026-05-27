# Audits, stratégie compétitive & roadmap

## Stratégie compétitive — où sont les vrais gains

> Auto-critique honnête. Ce qu'on a fait est solide d'un point de vue ingénierie, mais
> probablement **insuffisant pour gagner le challenge** sans investir aussi sur la donnée.

### Ce qui est solide

- **Fairness** : décomposition en 3 axes orthogonaux + preuve formelle que `gender_rw` est un no-op + multiple mécanismes (sampler, λ-penalty, Group-DRO, DANN, MMD, mixup, post-hoc). Mathématiquement propre.
- **Shift Y** : `_TEST_PMF_0025` + `importance_reweight` + `test_pmf` sampler + val split test-like → estimation honnête de la perf test.
- **iBOT-light vs MAE** : choix correct (continuité avec DINOv3, frozen teacher safe pour ne pas abîmer le backbone).
- **Infra** : chain SLURM, MLflow continu, snapshot intermédiaires en search_space, sidecar `.meta.json`, validation runtime des choix `pretrained_source`. Niveau industriel.

### Faiblesses stratégiques identifiées

**1. Pretrain budget anémique pour les gros modèles** — 50k steps × eff_batch=64 = 3.2M faces vues = **0.6 epoch MS1MV3**. Court pour une domain adaptation iBOT.

  *Nuance importante* : le pretrain part **bien de LVD / sapiens_default** ([src/models/dinov3_loader.py:60-66](../src/models/dinov3_loader.py#L60-L66), `pretrained=True` par défaut + teacher frozen = LVD). Donc même après 0.6 epoch, l'encoder reste proche du baseline avec un petit drift — pas de risque de "100h gaspillées" au sens strict. Le risque réel : **le delta iBOT vs LVD peut être trop petit pour ressortir dans le sweep Optuna**. Augmenter à 100-150k steps maximise les chances que `ibot:encoder_*` batte `lvd` de façon mesurable.

**2. Le plus gros levier (data labellisée) n'est pas pris** — Le label `FaceOcclusion` mixte deux régimes (cf [data.md](data.md) §"Understanding FaceOcclusion"). Aucune donnée externe avec recompute du label n'est intégrée :
  - **MAFA** (occlusion physique annotée) → mapping `area(mask ∩ face) / area(face)` direct
  - **RandomErasing avec label recompute** → patch synthétique sur visage, label incrémenté
  - **Quality degradation** (blur, JPEG, noise, pixelation) avec proportional label increment → couvre le régime 2

  Roadmap #14-15 estime ~1.5-2.5% combiné. **Probablement plus impactant que tout le sweep HPO réuni.**

**3. Résolution sous-utilisée** — Sapiens2 natif est 1024×768, on l'utilise à 224. Lever #19 (résolution 384×384) estimé +0.5-1.5%, non implémenté.

**4. Ensemble multi-arch pas encore généralisé** — `inv ensemble` fait du 5-fold sur **un seul yaml**. Le plan : k-fold séparé sur chaque arch (vith16plus + sapiens 0.8b + best petit), puis moyenne au submit. Estimation +0.5-1%.

**5. `_TEST_PMF` ni vérifié ni calibré** — La PMF cible [`_TEST_PMF_0025`](../src/utils/losses.py#L8) est codée en dur. Deux risques :
  - **Méthodologie d'estimation incertaine** : si la PMF d'origine vient d'un modèle faible, elle est biaisée → tout `importance_reweight` / `test_pmf` sampler tire le sweep dans la mauvaise direction.
  - **Pas de calibration leaderboard** : une submission permettrait de fitter `_TEST_PMF` pour que `expected_score(val) ≈ leaderboard_score`. Lever #13, pas fait.

### Ordre de priorité recommandé pour gagner

| # | Action | Effort | Impact estimé | Statut |
|---|---|---|---|---|
| 1 | **MAFA + augm-with-label-recompute** (blur/JPEG/erase) | 1-2 jours | **+1.5-2.5%** | à faire |
| 2 | **Vérifier `_TEST_PMF_0025`** — méthodo d'origine, ré-estimation via inférence baseline, cross-check post-1ère submission | 2-4h | **+0.5-1.5%** si la PMF actuelle est fausse | à faire |
| 3 | **Ensemble multi-arch au submit time** — k-fold séparé sur 3 archis, moyenne des prédictions | 2-3h infra + N k-folds | **+0.5-1%** | infra k-fold OK pour 1 arch, à généraliser |
| 4 | **1 submission baseline + fit `_TEST_PMF`** sur le score leaderboard | 1h | **+0.3-1%** | dépend de #2 |
| 5 | **Résolution 384 pour les gros modèles** | 3-4h | **+0.5-1.5%** | non démarré ; ~3× coût compute |
| 6 | **Plus de pretrain** (100-150k steps @224, ou EMA teacher) | budget cluster | +0.3-0.8% | en cours |
| 7 | HPO sweep actuel (3 axes) | en cours | +0.2-0.5% | tourne, mais c'est de la finition |

**Synthèse** : si on a une semaine, **4 jours sur #1-#4** + **3 jours de sweep en parallèle** > l'inverse. Le challenge se gagne typiquement sur la donnée, pas sur le modèle.

---

## Roadmap — next steps

### Implemented and runnable now

| # | Lever | Where | Status |
|---|---|---|---|
| 1 | **HPO on ViT-H+/16 baseline** | `scripts/optimize_dinov3_vith16plus_2x3090.sh` + `dinov3-vith16plus-3090-v4.yaml` | ready |
| 2 | **iBOT-light pretrain** on MS1MV3 | `scripts/pretrain_ibot_vith16plus_2x3090.sh` + `chain_pretrain.sh` | ready |
| 3 | **HPO from iBOT pretrain** (snapshots en search_space) | yaml `pretrained_source` choices | ready (fill run_id after pretrain) |
| 4 | **Sapiens2-0.8B baseline** (1B human pretrain) | `scripts/optimize_sapiens2_08b_2x3090.sh` + yaml | ready |
| 5 | **Importance reweighting** `w(GT) = p_test/p_train` 20 bins | `src/utils/losses.py:build_importance_weights` + yaml `loss_rw_strategy` | ready |
| 6 | **Gender-balanced sampler** `gender_x_occ` | `src/utils/losses.py:make_sampler_keys` | always available |
| 7 | **Fairness penalty** `λ·|Err_F − Err_M|` | `src/utils/losses.py:WeightedMSELoss` | yaml `loss_fairness_lambda` |
| 8 | **Group-DRO** worst-group min | `src/utils/losses.py:GroupDROLoss` | yaml `loss_type: group_dro` |
| 9 | **Quantile matching** post-inférence | `src/inference/calibration.py` | `predict.py` |
| 10 | **Post-hoc per-gender bias correction** | `src/inference/calibration.py` | `predict.py --delta-f --delta-m` |
| 11 | **Worst-K + best-K diagnostic** | `src/train.py:_save_qualitative_examples` | auto on each trial |
| 12 | **Diagnostic charts (4-panel)** | `src/train.py:_save_diagnostic_charts` | PNG auto-uploaded to MLflow |
| 13 | **EMA + LLRD + grad-ckpt + bf16/fp16** | `src/train.py` | yaml toggles |

### Non-implemented levers — tracked for future work

| # | Lever | Estimated effort | Estimated ROI | Notes |
|---|---|---|---|---|
| 14 | **MAFA + RandomErasing with label recompute** — synthetic occluders pasted on faces, label increment proportional to area covered | ~150 LOC in `transforms.py` + helper script | ~1-2% on score | covers regime 1 (physical occlusion) gap |
| 15 | **Quality-degradation augmentations** — blur, JPEG artifacts, noise, pixelation with proportional label increment | ~100 LOC in `transforms.py` | ~0.5-1% on score | covers regime 2 (information degradation) |
| 16 | **DINOv3 ViT-7B + LoRA** as ensemble member | ~100 LOC loader + yaml + sbatch | ensemble diversity, 0.5-1% | full FT impossible on 2×3090, LoRA only |
| 17 | **Sapiens2-5B + LoRA** as ensemble member | similar to #16 | similar | same constraint |
| 18 | **Multi-arch ensemble** add DINOv2-base + ConvNeXt v2-large (HF, no license) | ~2 yamls + sbatch | ensemble diversity, ~1% | low effort if compute available |
| 19 | **Resolution upgrade 384×384** (or Sapiens2 native 1024×768) | ~50 LOC `get_image_processor` + yaml batch downsize | ~0.5-1.5% on score | 3× compute cost |
| 20 | **SAM3-based pseudo-labeling** on CelebA/VGGFace2 for regime-1 occlusion auto-labels | ~200 LOC offline pipeline | uncertain | high effort |
| 21 | **Pseudo-labeling** on `test_students.csv` using ensemble high-confidence predictions, retrain | ~80 LOC | ~0.5% if disparity is low | risk of self-confirming bias |
| 22 | **Test-time augmentation beyond hflip** — light crop/scale TTA, ensembled | ~50 LOC in `inference/tta.py` | ~0.2-0.5% | careful with ratio regression invariance |

---

## Audit v4 — leçons et limitations

### Ce qu'on a fait mieux qu'en v3

- **Décomposition en 3 axes orthogonaux** au lieu de 9 stratégies fixes : on peut mesurer l'importance Optuna axe par axe (`optuna.importance.get_param_importances(study)`)
- **`no_balancing` baseline accessible** comme combo `(none, none, none)` → mesure l'apport des autres mécanismes
- **Validation split flexible** : `stratified_yg` (default) ou `test_pmf` (val matche directement P_test)
- **`pretrained_source` en search_space** : compare LVD/sapiens_default vs iBOT snapshots @ 25k/50k/75k/encoder final (100k @224) dans le même study
- **Quantile mapping post-inférence** : 3e mécanisme de compensation shift, indépendant du training
- **Métriques nommées explicitement** : `challenge_score_test_estimated` vs `_val`, `mae_pct_*`, `r2_*` — pas d'aliases ambigus
- **Diagnostic charts par trial** : 4 panels (MAE brute par bin, contribution au score, densité, distribution d'erreur)
- **Sidecar `.meta.json`** par prédiction : trace complète

### Limitations qu'on assume

| Limitation | Conséquence | Mitigation possible |
|---|---|---|
| **52 combos valides, 200 trials → ~4 trials/combo en moyenne** | Importance Optuna par axe approximative | Bumper `n_trials` si on veut conclure rigoureusement |
| `cell_joint` couple toujours `cell_rw + imp_rw` | On ne peut pas tester `cell_rw` seul | Ajouter une 4e option `cell_only` à l'axe 2 |
| Naming inconsistant : `loss_rw_strategy` vs `feature_fairness` | Cognitive load mineur | Renommer |
| `loss_gender_reweight` câblé en code mais hors search space | Option dormante | Documenter clairement ; ou retirer du code |

### Ce qu'on aurait pu faire encore mieux

**1. Source unique pour les flags d'équilibrage** — Actuellement définis à plusieurs endroits (`_LOSS_RW_STRATEGY_MAP`, params du Trainer, `train_cfg.get`). Un `Pydantic.BaseModel` centraliserait.

**2. Tests + CI** — Aucun test automatisé sur les 3 axes. Au minimum :
```python
def test_strategy_map_complete():
    for k, v in _LOSS_RW_STRATEGY_MAP.items():
        assert set(v.keys()) == {'loss_importance_reweight', 'loss_cell_reweight'}
```

**3. Search space planifié de bout en bout AVANT de coder** — Plusieurs itérations ont laissé des incohérences (no-ops, double-counts) qu'on a corrigées au fil de l'eau.

---

## Audit — fixes applied in v3 → v4

| # | Bug / weakness | Status |
|---|---|---|
| 1 | LLRD broken under DDP | ✅ fixed (`_unwrap` helper) |
| 2 | Unused `nn.functional.mse_loss` in `FaceOccRegressor.forward` | ✅ removed |
| 3 | `NaN or 0.0` truthy-check anti-pattern in data-distribution logging | ✅ replaced by `_safe_mean` |
| 4 | No way to disable MLflow for local debug | ✅ `use_mlflow: bool` toggle |
| 5 | TTA only available via code, not at submission time | ✅ `use_tta` in `predict.py` + `--no-tta` |
| 6 | No mechanism for distribution-shift mitigation | ✅ 3 mécanismes (loss imp_rw, test_pmf sampler, quantile matching) |
| 7 | Only one fairness mechanism | ✅ 5 mécanismes (axes 1/2/3 + λ-penalty + post-hoc) |
| 8 | DINOv3 weights other than ViT-S/16 hardcoded missing | ✅ all 6 paths registered |
| 9 | `tqdm` not in pyproject | ✅ added explicit |
| 10 | Two pyproject files with `cp` swap | ✅ single `pyproject.toml` with marker |
| 11 | MPS autograd `.view()` failure forced torch `<2.6` | ✅ bumped to `torch>=2.7` |
| 12 | Pretrain → finetune wiring opaque | ✅ MLflow trace complet |
