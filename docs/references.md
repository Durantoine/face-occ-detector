# SOTA references

## Backbones & SSL

- **DINOv3** — Meta, vendored at `src/models/dinov3_repo/`. ViT + DINO + iBOT + KoLeo + Gram losses.
- **DINOv2** — Oquab et al. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- **MAE** — He et al. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- **iBOT** (image BERT-like SSL) — Zhou et al. [arXiv:2111.07832](https://arxiv.org/abs/2111.07832)
- **Sapiens** — Khirodkar et al., Meta foundation for human vision (1B human images)
- **Sapiens 2** — Meta, custom architecture (RoPE + GQA + SwiGLU + RMSNorm), [github.com/facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2)
- **ConvNeXt v2** — Woo et al. [arXiv:2301.00808](https://arxiv.org/abs/2301.00808)
- **EVA-02** — Fang et al. [arXiv:2303.11331](https://arxiv.org/abs/2303.11331)

## Face image quality / occlusion analysis (most relevant to our task)

- **A Comprehensive Review of Face Detection Techniques for Occluded Faces** (CMES 2025) — [techscience.com/CMES/v143n3/62822](https://www.techscience.com/CMES/v143n3/62822/html). Survey 4 categories: feature-based, ML, DL, hybrid. Note la trend ViT/Swin.
- **A Survey of Face Recognition Techniques under Occlusion** (IET Biometrics 2021) — [Zeng et al.](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/bme2.12029). 3 stratégies: visible-parts → reconstruction, fusion sub-regions, adversarial.
- **CLIB-FIQA** (CVPR 2024) — Ou et al. Confidence calibration for face quality. [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Ou_CLIB-FIQA_Face_Image_Quality_Assessment_with_Confidence_Calibration_CVPR_2024_paper.pdf)
- **CR-FIQA** (Boutros 2023) — Sample relative classifiability for FIQA
- **FaceQNet** — Hernández-Ortega et al. [arXiv:1904.01740](https://arxiv.org/abs/1904.01740)
- **AOFD: Adversarial Occlusion-aware Face Detection** — segmentation + detection jointly. [arXiv:1709.05188](https://arxiv.org/abs/1709.05188)

## Multiple Instance Learning (MIL) + attention pooling

- **Attention-based Deep MIL** — Ilse et al. 2018. [arXiv:1802.04712](https://arxiv.org/abs/1802.04712). Notre baseline conceptuel.
- **Set Transformer / PMA** — Lee et al. 2019. [arXiv:1810.00825](https://arxiv.org/abs/1810.00825). Base théorique du multi-head attention pooling.
- **Perceiver IO** — Jaegle et al. 2021. [arXiv:2107.14795](https://arxiv.org/abs/2107.14795). K-query attention pour set→scalar.
- **Rethinking Attention-Based MIL** (2024) — [arXiv:2404.00351](https://arxiv.org/abs/2404.00351).
- **Dual-Attention MIL** (Electronics 2024) — 2 attentions parallèles pour WSI classification. [MDPI](https://www.mdpi.com/2079-9292/13/22/4445).
- **CAMIL: Channel Attention MIL** (Bioinformatics 2025) — [Oxford Academic](https://academic.oup.com/bioinformatics/article/41/2/btaf024/7958575)
- **Neighborhood Attention MIL** (2024) — locality + attention pour WSI. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11390382/)

## Fairness in face analysis

- **Group-DRO** — Sagawa et al. *Distributionally Robust Neural Networks for Group Shifts*. [arXiv:1911.08731](https://arxiv.org/abs/1911.08731). Implémenté dans `src/utils/losses.py:GroupDROLoss`.
- **DANN** (Domain-Adversarial Neural Networks) — Ganin et al. [arXiv:1505.07818](https://arxiv.org/abs/1505.07818). Implémenté pour fairness gender.
- **MMD** (Maximum Mean Discrepancy) — Gretton et al. Two-sample test in RKHS, used here for feature alignment.
- **Component-Based Fairness in Face Attribute Classification** (FAccT 2025) — Bayesian network + meta-learning. [arXiv:2505.01699](https://arxiv.org/abs/2505.01699)
- **Toward Fairer Face Recognition Datasets** (2024) — [arXiv:2406.16592](https://arxiv.org/abs/2406.16592)

## Optimization / regularization

- **EMA / SWA** — Izmailov et al. [arXiv:1803.05407](https://arxiv.org/abs/1803.05407). Implémenté.
- **LLRD** — Howard & Ruder (ULMFiT). [arXiv:1801.06146](https://arxiv.org/abs/1801.06146). Implémenté.
- **Focal Loss** — Lin et al. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002). Adapté en `loss_focal_gamma`.
- **Importance weighting / covariate shift** — Shimodaira 2000 (classical reference for `w = p_test/p_train`)
- **Mixup** — Zhang et al. [arXiv:1710.09412](https://arxiv.org/abs/1710.09412). Adapté en `mixup_inter_gender`.

## Notre niche : continuous occlusion ratio regression

Aucune publication ne fait **exactement** notre tâche (Idemia metric `(Err_F + Err_M)/2 + |Err_F − Err_M|` avec `w = 1/30 + GT`). Le champ FIQA produit des scores de qualité multi-facteur (incluant occlusion comme une dimension parmi d'autres), mais pas de regression isolée sur le ratio d'occlusion. → **Setup spécifique au challenge**, on combine des briques validées individuellement.
