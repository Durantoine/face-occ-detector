# Documentation — index

Documentation théorique du projet face-occ-detector. Pour l'utilisation pratique (commandes, workflows), voir le [README principal](../README.md).

## Théorie

| Fichier | Contenu |
|---|---|
| **[v11_theory.md](v11_theory.md)** | **v11 : référence principale.** Lagrangien adaptatif, OT, EMA double-eval, validation rigoureuse H_C via MID lookup, 6 étages d'intervention. |
| [fairness.md](fairness.md) | Fairness, équilibrage 3 axes, shift train↔test, 3 mécanismes de compensation, méthodes détaillées (DANN, MMD, mixup, cell reweight). H1 = H_C cohérence vérifiée v11. |
| [architecture.md](architecture.md) | Poolings (CLS, GeM, K-query attention, MHA), v2 attention pooling, backbones |
| [metrics.md](metrics.md) | 14 métriques officielles + référence, ordres de grandeur, diagnostic charts 4-panel |
| [data.md](data.md) | Dataset Idemia, comprendre `FaceOcclusion` (2 régimes), external data (CelebA/MAFA/etc.) |
| [pretraining.md](pretraining.md) | iBOT-light (pourquoi pas MAE), compute budget, v4 workflow |

## Référence

| Fichier | Contenu |
|---|---|
| [configuration.md](configuration.md) | Tous les YAML toggles, MLflow logging, project layout |
| [scaling.md](scaling.md) | Hardware (P100, 3090), ce qui passe en DDP, refactor FSDP/ZeRO-3 pour 1B+ |
| [audits_and_roadmap.md](audits_and_roadmap.md) | Stratégie compétitive, audit v3→v4, roadmap |
| [references.md](references.md) | Bibliographie SOTA |
| [balancing_review.md](balancing_review.md) | Notes brutes sur les stratégies d'équilibrage (historique) |
