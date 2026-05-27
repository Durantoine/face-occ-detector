# Architecture du modèle — pooling, head, backbones

> Référence détaillée des composants du modèle : poolings (CLS / GeM / K-query attention / MHA),
> head, et choix de backbones.

---

## Notations communes

| Symbole | Définition | Dimensions / domaine |
|---|---|---|
| $B$ | taille du batch | scalaire |
| $N$ | nombre de tokens en sortie du backbone ($= 1$ CLS $+ N_p$ patches) | scalaire (= 197 pour ViT-B/16 à 224×224) |
| $D$ | dimension cachée du backbone | $D \in \{768, 1024, 1280\}$ pour ViT-B / L / H+ |
| $X$ | tenseur des features de sortie du backbone (CLS à l'index 0) | $\mathbb{R}^{B \times N \times D}$ |
| $\Phi$ | features après pooling, input du head linéaire final | $\mathbb{R}^{B \times D'}$ |
| $D'$ | dim de sortie du pool : $D$ pour CLS/GeM/MHA, $K \cdot D$ pour K-query | scalaire |
| $h$ | sortie scalaire du head | $h \in \mathbb{R}^B$ |
| $\sigma(h) = (1 + e^{-h})^{-1}$ | sigmoid, $\hat y = \sigma(h)$ | applique element-wise |

---

## Poolings

### Pooling 1 — CLS

Le plus simple : on garde le premier token (le [CLS] entraîné par DINO pour agréger globalement).

| Tenseur | Shape | Description |
|---|---|---|
| $X$ | $(B,\, N,\, D)$ | sortie du backbone (input du pool) |
| $\Phi = X[:, 0, :]$ | $(B,\, D)$ | features (input du head) |

- **Params appris dans le pool** : 0
- **Output dim** : $D' = D$
- **Coût** : nul
- **Quand l'utiliser** : baseline propre quand le backbone DINO/iBOT est de qualité ; suffisant si le signal est global
- **Limite** : ne mélange pas l'info des patches au-delà de ce que le backbone a déjà fait

### Pooling 2 — GeM (Generalized Mean)

Généralisation de mean et max (Radenović et al. 2018). $p$ apprenable contrôle la sharpness.

| Tenseur / paramètre | Shape | Description |
|---|---|---|
| $X$ | $(B,\, N,\, D)$ | input |
| $\tilde p$ | $()$ scalaire | paramètre brut |
| $p = \text{softplus}(\tilde p) + \varepsilon$ | $()$ | exposant strictement positif |
| $X[:, 1:, :]$ | $(B,\, N_p,\, D)$ | patches (skip [CLS]), $N_p = N - 1$ |
| $\Phi$ | $(B,\, D)$ | features pooled |

$$\Phi_{b, d} = \left( \frac{1}{N_p} \sum_{n=1}^{N_p} \max(X[b, n, d],\, \varepsilon)^{p} \right)^{1/p}.$$

- **Cas limites** : $p \to 1$ → mean ; $p \to \infty$ → max
- **Params appris dans le pool** : 1 (le scalaire $\tilde p$)
- **Output dim** : $D' = D$
- **Limite** : sur features ViT non-activées (peuvent être négatives), le clamp à $\varepsilon$ tue la moitié du signal

### Pooling 3 — Attention K-query (v2, recommandé)

$K = n_{\text{focal}} + n_{\text{diffuse}} + n_{\text{free}}$ queries apprenables, chacune avec sa propre température $\tau_k$ apprenable.

| Tenseur / paramètre | Shape | Description |
|---|---|---|
| $X$ | $(B,\, N,\, D)$ | input du pool |
| $Q$ | $(K,\, D)$ | matrice des $K$ queries apprenables |
| $\log \tau$ | $(K,)$ | log-températures apprenables |
| $W_K, W_V$ | $(D,\, D)$ | projections clé/valeur |
| $A$ (attention weights) | $(B,\, K,\, N)$ | poids softmax par query sur les patches |
| pooled (avant flatten) | $(B,\, K,\, D)$ | $K$ vecteurs poolés par sample |
| $\Phi$ après concat + LayerNorm | $(B,\, K \cdot D)$ | features finales |

Init différenciée des $\tau$ pour forcer la diversité des rôles :
- $\tau_k = \tau_{\text{focal}}$ ≈ 0.1 → attention piquée, capture des indices locaux
- $\tau_k = \tau_{\text{diffuse}}$ ≈ 1.5 → attention plate, capture des signaux globaux
- $\tau_k = \tau_{\text{free}}$ = 1 → neutre

Forward :
$$\text{scores}_{b,k,n} = \frac{Q_{k,:} \cdot K_X[b,n,:]}{\sqrt{D} \cdot \tau_k}, \qquad A_{b,k,:} = \text{softmax}_n(\text{scores}_{b,k,:}),$$
$$\Phi_b = \text{LayerNorm}\!\left( \mathrm{concat}_{k=1}^{K} \sum_{n=1}^{N} A_{b,k,n} \cdot V_X[b,n,:] \right).$$

- **Output dim** : $D' = K \cdot D$ (par défaut $K = 6$, soit $6 \cdot 768 = 4608$ pour ViT-B/16)
- **Params appris dans le pool** : $K \cdot D$ (queries) + $K$ ($\log \tau$) + $2 D^2$ (projections K/V) + $2 \cdot K \cdot D$ (LayerNorm γ/β)
- **Pénalité optionnelle** : `loss_query_diversity_lambda` ajoute une pénalité cosinus entre paires de queries pour empêcher le collapse :
$$\Omega_{\text{div}} = \frac{1}{B \cdot K(K-1)} \sum_b \sum_{k \neq j} \widetilde A_{b,k,:} \cdot \widetilde A_{b,j,:}.$$

### Pooling 4 — Multi-Head Attention (MHA)

1 query apprenable découpée en $H$ heads de dim $D/H$. Pas de $\tau$ per-head — la diversité émerge des projections aléatoirement initialisées.

| Tenseur / paramètre | Shape | Description |
|---|---|---|
| $X$ | $(B,\, N,\, D)$ | input |
| $Q$ (la query unique) | $(D,)$ | paramètre apprenable |
| $W_Q, W_K, W_V, W_{\text{out}}$ | $(D,\, D)$ | projections |
| $A$ (attention weights) | $(B,\, H,\, N)$ | softmax par-head sur les patches |
| $\Phi$ | $(B,\, D)$ | features finales |

Forward :
$$\text{scores}_{b,h,n} = \frac{q_h \cdot k_{b,h,n,:}}{\sqrt{D/H}}, \qquad A_{b,h,:} = \text{softmax}_n(\text{scores}),$$
$$\Phi_b = W_{\text{out}} \cdot \mathrm{concat}_{h=1}^{H} \sum_n A_{b,h,n} \cdot v_{b,h,n,:}.$$

- **Output dim** : $D' = D$
- **Search space** : `num_heads ∈ {2, 4, 8, 16}` (diviseurs communs ViT-B/L/H+)

---

## v2 attention pooling — architecture complète

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
| **Pool** | `pool_attn_dropout` ∈ [0, 0.3] | Dropout appliqué sur les **poids d'attention** dans la pool. |
| **Pool** | `pool_proj_dropout` ∈ [0, 0.3] | Dropout sur la **sortie agrégée** de la pool (après LayerNorm). |
| **Head** | `head_dropout` ∈ [0, 0.3] | Dropout avant le Linear final. |

### Query diversity penalty

Au-delà des dropouts, on peut **pénaliser la redondance entre queries** :

```python
loss = main_loss + λ_div × diversity_penalty(attn_weights)
```

où `diversity_penalty` = moyenne des cosinus pairwise entre les K=6 distributions d'attention :
- 0 si queries totalement orthogonales (attention sur des patches disjoints)
- 1 si queries identiques (collapse)

Param yaml : `loss_query_diversity_lambda ∈ [0, 0.2]` (Optuna search).

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

Au cours du training, les τ apprises peuvent **diverger arbitrairement** — la spécialisation focale/diffuse n'est qu'un prior d'init, pas une contrainte stricte.

---

## Backbones

DINOv3 ViT-S/16 (`dinov3_vits16`, 22M) est **vendored with weights** — works out of the box on any hardware.

Pour les variants plus gros (ViT-L+/16, ViT-H+/16), les poids vivent derrière la DINOv3 License Agreement de Meta. Use `scripts/download_dinov3_weights.sh` as a URL reference, accept the licence on the [DINOv3 GitHub](https://github.com/facebookresearch/dinov3), download manually, and drop the `.pth` into `src/models/weights/` — le loader auto-détecte.

**Alternative open backbones** (HuggingFace, no licence) — recommandé pour la diversité d'ensemble :

| HF id | Params | Type |
|---|---|---|
| `facebook/dinov2-base` | 86M | SSL transformer |
| `facebook/dinov2-large` | 300M | SSL transformer |
| `facebook/convnextv2-large-22k-224` | 198M | CNN, MAE-pretrained |
| `Yuxin-CV/EVA-02-L-14` | 305M | MIM transformer |
| `google/vit-base-patch16-224` | 86M | ImageNet supervised |

To use any of these, just edit `model.model_name` in a YAML — `FaceOccRegressor._build_backbone` already dispatches DINOv3 vs HuggingFace.

### Backbones supportés et tailles

| Backbone | Params | Tient en 2× RTX 3090 ? | Statut |
|---|---|---|---|
| DINOv3 ViT-B/16 | 86M | ✅ BS=128 | pretrain + finetune OK |
| DINOv3 ViT-L/16 | 300M | ✅ BS=64 (avec grad_ckpt) | OK |
| DINOv3 ViT-H+/16 | 600M | ✅ BS=8 + grad_ckpt | **le plus gros DINO sans refactor** |
| Sapiens2 0.1B | 100M | ✅ BS=128 | pretrain + finetune OK |
| Sapiens2 0.4B | 400M | ✅ BS=8 + grad_ckpt | OK |
| Sapiens2 0.8B | 800M | ✅ BS=4 + grad_ckpt | **le plus gros Sapiens2 sans FSDP** |
| Sapiens2 1B+ | 1.5B+ | ❌ | nécessite FSDP / DeepSpeed (cf [scaling.md](scaling.md)) |
