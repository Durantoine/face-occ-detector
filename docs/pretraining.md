# Pretraining method: iBOT (not MAE)

Several iterations of this project considered MAE as the pretraining objective. **We chose iBOT-light (frozen teacher feature matching) instead.** This section explains why.

---

## The goal: domain adaptation, not new capability

We want to **specialize an already-trained DINOv3** on the face corpus (MS-Celeb / MS1MV3) that overlaps with the Idemia training set. The encoder already knows what a face looks like (it saw 1.7B images during LVD-1689M pretraining). What we want is to **adjust** its features to the specific face manifold of our downstream task, **without** wiping out what it learned.

This is fundamentally different from "add a new capability" (where MAE would shine — it adds pixel reconstruction skill).

## Why MAE is wrong for our case

| Aspect | MAE | iBOT |
|---|---|---|
| Objective | Reconstruct pixel values of masked patches | Match teacher's **feature embeddings** at masked positions |
| Effect on existing features | **Overwrites** — forces low-level pixel reconstruction | **Preserves** — uses teacher as anchor, just shifts distribution |
| Continuity with DINOv3 pretraining | Different objective (DINOv3 was trained with DINO+iBOT, not MAE) | **Same family** — extends the original pretraining |
| Risk of catastrophic forgetting | High (overrides DINOv3's feature space) | Low (teacher keeps features aligned) |
| HF native implementation | ✅ ready | ❌ but we use Meta's vendored `dinov3_repo/dinov3/loss/ibot_patch_loss.py` as reference |

The cleanest argument: **DINOv3 was originally trained with DINO+iBOT+KoLeo+Gram.** Continuing with the same iBOT objective is "more pretraining of the same kind". Switching to MAE is "different methodology" with no obvious upside.

## Our simplified iBOT (iBOT-light)

Full Meta iBOT uses Sinkhorn-Knopp centering, EMA teacher, multi-crop augmentation, projection heads, prototype distributions, and 4 simultaneous losses (`dino_clstoken_loss + ibot_patch_loss + koleo_loss + gram_loss`). We reproduce the essential mechanism more simply:

- **Student** = DINOv3 ViT-H+ (trainable)
- **Teacher** = same DINOv3 weights, **frozen** (no EMA, no Sinkhorn-Knopp) — anchor for the student
- **Mask ratio** = 0.4 (iBOT range, lower than MAE's 0.75)
- For each batch:
  1. Random binary mask over patches
  2. Student forward **with mask** (uses DINOv3 native `prepare_tokens_with_masks` → masked patches replaced by `mask_token`)
  3. Teacher forward **without mask**, `torch.no_grad()`
  4. Loss = `1 - cosine(student_patches, teacher_patches)` **at masked positions only**
  5. Backprop on student only
- **CLS drift** logged each step (auxiliary metric: how far the student's CLS token has moved from the teacher's)

Optional: switch to EMA teacher (`teacher_frozen: false, teacher_ema_decay: 0.999`) to get self-improving target. Frozen is simpler and avoids EMA bookkeeping.

Implementation: `src/pretrain_ibot.py:DinoV3IBoT` (~100 LOC core).

## Compute budget

| Setup | VRAM per GPU | Time on 2× RTX 3090 (BF16, 5M images, 30 epochs) |
|---|---|---|
| ViT-S/16 | ~5 GB | ~6 h |
| ViT-B/16 | ~9 GB | ~12 h |
| ViT-L/16 (300M) | ~14 GB | ~24 h |
| **ViT-H+/16 (600M)** ⭐ | **~22 GB** (with grad-ckpt) | **~36 h** |

Run with `sbatch scripts/pretrain_ibot_vith16plus_2x3090.sh` (ou `./scripts/chain_pretrain.sh 8 scripts/pretrain_ibot_vith16plus_2x3090.sh` pour chainer 8 links de 30h, target 100k steps @224 = ~1.2 epoch MS1MV3). Output : run MLflow avec snapshots `encoder_25000/50000/75000/encoder` (final à 100k) — stop-early possible via scancel.

**v6 workflow** — pour utiliser ce pretrain custom dans Optuna, fill l'URI dans la choice `ibot:runs:/<run_id>/encoder` du search space `pretrained_source` du yaml d'architecture (par exemple `configs/architectures/dinov3-vitb16-3090-v6.yaml`). Optuna comparera alors automatiquement :
- `"lvd"` (ou `"sapiens_default"` pour Sapiens2) : poids de base Meta
- `"ibot:runs:/<run_id>/encoder"` : notre pretrain custom par-dessus

`train.py` charge les poids dans le backbone et copie tous les `pretrain_*` params dans la run finetune pour la traçabilité complète. Une validation runtime ([`_validate_pretrained_source_choices`](../src/optimize.py)) vérifie l'existence des runs MLflow avant le démarrage du sweep — pas de trial gâché sur un `runs:/__FILL__/encoder` oublié.
