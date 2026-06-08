# Dataset synthétique d'occluders (Tier 2)

Ce dossier accueille un dataset d'augmentation pour la régression `FaceOcclusion`.
On part de **vrais visages propres** du train (occlusion < 0.05) et on **peint un
occluder réaliste par inpainting** (diffusion) dans une zone qu'on contrôle — donc
le label est **exact** : `FaceOcclusion = label_source + aire_du_masque` (plafonné à 1).
Le placement est sémantique (cheveux/coiffe en haut, écharpe en bas, main sur le visage)
et tout ce qui est **hors** du masque reste **identique** à l'original (recomposition).

Types d'occluders : **cheveux, coiffes, écharpes, mains**. Les aires sont biaisées
vers **0.30–0.70** pour fabriquer la queue haute-occlusion qui manque au train.
Un **garde-fou qualité** rejette automatiquement les ratés (barres noires / non-occlusions)
et régénère à la place.

> Le générateur géométrique historique (`scripts/generate_synthetic_occluders.py`,
> aplats noirs) est **abandonné** — ne pas l'utiliser.

---

## 0. Pré-requis

- **GPU NVIDIA** fortement recommandé (SDXL en CPU = plusieurs minutes/image).
- **Images sources** extraites : `crops.zip` → `data/raw/Crop_224_5fp_100K/database{1,2,3}/...`
  (les chemins de `train.csv` se résolvent contre ce dossier).
- **`train.csv`** (colonnes `filename,FaceOcclusion,gender`) — adapter le chemin ci-dessous.
- 1ʳᵉ exécution : téléchargement du modèle depuis Hugging Face (SDXL-inpaint ≈ 6.9 Go,
  **public/non-gated**, aucune authentification requise).

Dépendances : on utilise un environnement éphémère `uv` (évite d'installer tout le
projet, ex. Sapiens). Remplace `<TRAIN_CSV>` et `<IMAGE_DIR>` par tes chemins.

```bash
TRAIN_CSV=../occlusion_datasets/train.csv          # adapte si besoin
IMAGE_DIR=data/raw/Crop_224_5fp_100K
DEPS="--with torch --with diffusers --with transformers --with accelerate \
      --with huggingface_hub --with pillow --with numpy --with pandas --with safetensors"
```

---

## 1. Smoke-test d'abord (≈48 images) — VALIDER la qualité

À regarder en priorité : le rendu des **mains** (le type le plus risqué).

```bash
cd face-occ-detector
uv run $DEPS python scripts/generate_inpaint_occluders.py \
    --train-csv "$TRAIN_CSV" --image-dir "$IMAGE_DIR" \
    --output-dir data/synthetic/_smoke \
    --n-samples 48 --types hair headwear scarf hand \
    --model diffusers/stable-diffusion-xl-1.0-inpainting-0.1 --work-size 1024 \
    --batch-size 4 --steps 30
```

Inspecter `data/synthetic/_smoke/images/` + `synthetic.csv`. Si les mains déçoivent,
soit les retirer (`--types hair headwear scarf`), soit augmenter `--steps`/`--guidance`.
Quand c'est bon, supprimer `_smoke` et lancer le run complet.

---

## 2. Génération complète (≈13 000 images)

```bash
cd face-occ-detector
uv run $DEPS python scripts/generate_inpaint_occluders.py \
    --train-csv "$TRAIN_CSV" --image-dir "$IMAGE_DIR" \
    --output-dir data/synthetic/occluder_inpaint \
    --n-samples 13000 --types hair headwear scarf hand \
    --model diffusers/stable-diffusion-xl-1.0-inpainting-0.1 --work-size 1024 \
    --batch-size 4 --steps 30
```

Le script **checkpointe** le CSV régulièrement (reprise possible / pas de perte si crash).
Sortie :

```
data/synthetic/occluder_inpaint/
    images/syn_000000.webp ...
    synthetic.csv   # filename, FaceOcclusion, gender, source_filename,
                    # occluder_type, actual_area, original_label
```

### Variante plus rapide (si pas de gros GPU)
Modèle 512px, ~3× plus rapide, qualité un peu en dessous :
```
    --model Lykon/dreamshaper-8-inpainting --work-size 512 --batch-size 8
```

### Sur un cluster SLURM
Un script sbatch prêt à l'emploi : `scripts/generate_inpaint_occluders_1xA100.sh`
(réglages via variables d'env `N_SAMPLES`, `TYPES`, `MODEL`, `WORK`, `BATCH`, `OUTDIR`).
Voir l'en-tête du fichier pour la commande de smoke-test.

---

## 3. Brancher à l'entraînement

Dans la config YAML, pointer `extra_train_csv` sur le CSV produit (v19 concatène
automatiquement avec le train) et activer l'augmentation de robustesse Tier 1 :

```yaml
data:
  data_csv: data/raw/train.csv
  extra_train_csv: data/synthetic/occluder_inpaint/synthetic.csv
  train_aug: tier1_safe      # flou/pixel/JPEG/couleur, label INCHANGÉ (cf. src/data/transforms.py)
```

---

## Principaux réglages (`scripts/generate_inpaint_occluders.py --help`)

| Option | Défaut | Rôle |
|---|---|---|
| `--types` | hair headwear scarf hand | occluders à générer |
| `--area-min/--area-max` | 0.30 / 0.70 | plage d'aire occultée (biais queue haute) |
| `--model` | dreamshaper-8 (script) / SDXL (sbatch) | modèle d'inpainting (non-gated) |
| `--work-size` | 512 | 512 pour SD1.5/dreamshaper, **1024 pour SDXL** |
| `--steps` / `--guidance` | 28 / 8.0 | qualité vs vitesse |
| `--no-quality-guard` | off | désactive le rejet auto des ratés |
| `--max-retries` | 2 | re-essais d'un échantillon rejeté |
| `--clean-label-max` | 0.05 | seuil « visage propre » des sources |

> Note : les `stabilityai/*-inpainting` sont **gated** (401 sans token HF) — utiliser
> les modèles non-gated ci-dessus.
