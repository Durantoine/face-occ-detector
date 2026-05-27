# Hardware notes & scaling

## Hardware support matrix

| Platform | VRAM | Precision | Max DINOv3 (this repo) |
|---|---|---|---|
| 1× P100 | 16 GB | FP16 | ViT-S/16 (shipped, comfortable) |
| 2× P100 DDP | 32 GB | FP16 | ViT-L+/16 (requires manual weight DL) |
| 1× RTX 3090 | 24 GB | BF16 | ViT-L/16 |
| 2× RTX 3090 DDP | 48 GB | BF16 | ViT-H+/16 (requires manual weight DL) |

All SLURM scripts use `torchrun --nproc_per_node=2`, 30 h time, with the right hardware tags. On Linux nodes `uv sync` pulls CUDA 12.6 wheels via the `[tool.uv.sources]` marker ; on Mac it falls back to the default PyPI index (CPU/MPS). torch is pinned to `>=2.7,<3.0` to keep MPS autograd working for local sanity tests.

## Scaling — what fits on 2× RTX 3090 (24GB) and what doesn't

### Backbones qui passent en DDP standard

| Backbone | Params | Tient en DDP ? | BS effectif | Statut |
|---|---|---|---|---|
| DINOv3 ViT-B/16 | 86M | ✅ | 128 | pretrain + finetune OK |
| DINOv3 ViT-L/16 | 300M | ✅ (avec grad_ckpt) | 64 | OK |
| **DINOv3 ViT-H+/16** | **600M** | ✅ (BS=8 + grad_ckpt) | 32 | **le plus gros DINO sans refactor** |
| Sapiens2 0.1B | 100M | ✅ | 128 | pretrain + finetune OK |
| Sapiens2 0.4B | 400M | ✅ (BS=8 + grad_ckpt) | 32 | OK |
| **Sapiens2 0.8B** | **800M** | ✅ (BS=4 + grad_ckpt) | 64 | **le plus gros Sapiens2 sans FSDP** |

### Backbones bloqués (besoin de refactor FSDP/ZeRO-3)

| Backbone | Params | Pourquoi ça ne passe pas | Bloqueur |
|---|---|---|---|
| Sapiens2 1B | 1.5B | Adam fp32 (~12 GB) + student+teacher BF16 (~6 GB) + grads + activations > 24 GB / GPU | FSDP / DeepSpeed ZeRO-3 |
| Sapiens2 5B | 5B | Modèle BF16 seul ~10 GB ; optim ~40 GB. Impossible même avec FSDP sur 2× 24GB | GPU ≥ A100 80GB, ou ZeRO-3 + offload CPU/NVMe |

## Ce qu'il faudrait modifier pour débloquer Sapiens2 1B+ (backlog v5)

### Option A — FSDP (PyTorch natif)

Modifier `src/pretrain_ibot.py` :
1. Wrapper student/teacher avec `FullyShardedDataParallel` au lieu de DDP (`sharding_strategy=FULL_SHARD`, `MixedPrecision(param_dtype=bfloat16)`)
2. `EncoderSnapshotCallback` : utiliser `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)` pour obtenir le state_dict avant `mlflow.pytorch.log_model`
3. Si `teacher_frozen=False` : `ema_update_teacher` avec `summon_full_params`
4. Resumability : checkpoints FSDP (ne pas réutiliser le format HF Trainer par défaut)
5. HF Trainer wrapper iBOT custom contient 2 nn.Modules (`student` + `teacher`) → wrapping manuel nécessaire

**Effort estimé** : 2-3 jours dev + debug. Risque de NaN à debug en BF16 sharded.

### Option B — DeepSpeed ZeRO-3

Plus invasif mais mieux documenté.
1. `pip install deepspeed`, ajouter à `pyproject.toml`
2. Config `configs/deepspeed_zero3.json` (Adam offload, params offload, grad accumulation)
3. `TrainingArguments(deepspeed="configs/deepspeed_zero3.json", ...)`
4. Snapshots : `deepspeed.checkpoint_engine` + conversion vers state_dict standard

**Effort estimé** : 1-2 jours. Plus stable que FSDP en pratique.

### Pour Sapiens2 5B et plus

Impossible sur 2× RTX 3090 même avec ZeRO-3. Pistes :
- Migrer vers A100/H100 ≥ 80GB
- ZeRO-3 + offload CPU + NVMe (très lent, ~10× facteur)
- **Ne pas pretrain** : utiliser directement les poids Meta `sapiens2_5b` avec `pretrained_source: "sapiens_default"`. Sapiens2 est déjà human-centric par construction, donc l'iBOT custom apporte peut-être peu à cette taille.

### Quand prendre la décision

```
Sweep v4 Optuna terminé → quelle source pretrained gagne ?
│
├── "lvd" / "sapiens_default" gagne → pas besoin d'iBOT custom aux grosses tailles
│   → utiliser sapiens2_1b/5b directement avec sapiens_default
│   → mais FSDP toujours nécessaire pour finetune 1B+
│
└── "ibot:encoder_*" gagne → l'iBOT custom apporte
    → Option A (FSDP) ou B (DeepSpeed) pour pretrain 1B+
    → ou rester à 0.8B comme plafond pratique
```

Liens : [PyTorch FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) · [HF Trainer FSDP](https://huggingface.co/docs/transformers/main/en/fsdp) · [DeepSpeed ZeRO-3](https://www.deepspeed.ai/tutorials/zero/)
