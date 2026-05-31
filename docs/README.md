# Documentation — index

Documentation théorique du projet face-occ-detector. Pour l'utilisation pratique (commandes, workflows), voir le [README principal](../README.md).

## Théorie

| Fichier | Contenu |
|---|---|
| **[v12_theory.md](v12_theory.md)** | **v12 : doc de référence courante.** Tri-split + IS stratifié, Lagrangien adaptatif, OT, EMA double-eval, validation H_C via MID, 6 étages d'intervention + choix de design, bug fixes, audit complet. |
| [fairness.md](fairness.md) | Fairness historique : 3 axes équilibrage, shift train↔test, mécanismes détaillés (DANN, MMD, mixup, cell reweight). H1 = H_C cohérence vérifiée v11+. |
| [architecture.md](architecture.md) | Poolings (CLS, K-query attention, MHA), backbones |
| [metrics.md](metrics.md) | Métriques officielles + référence, ordres de grandeur, diagnostic charts |
| [data.md](data.md) | Dataset Idemia, `FaceOcclusion`, external data (CelebA/MAFA/etc.) |
| [pretraining.md](pretraining.md) | iBOT-light, compute budget, v4 workflow |

## Référence

| Fichier | Contenu |
|---|---|
| [configuration.md](configuration.md) | YAML toggles, MLflow logging, project layout |
| [scaling.md](scaling.md) | Hardware (P100, 3090), DDP, refactor FSDP/ZeRO-3 pour 1B+ |
| [audits_and_roadmap.md](audits_and_roadmap.md) | Stratégie compétitive, audit v3→v4, roadmap |
| [references.md](references.md) | Bibliographie SOTA |
