"""
QLoRA fine-tuning of Qwen2.5-VL-7B-Instruct for face occlusion regression.

Run 1 (baseline) : SFT direct sur {"percentage": 0.xxxx}
Run 2 (actuel)   : CoT structuré + signal dropout 30%
  - Chain-of-thought : le modèle analyse chaque signal avant de répondre
  - Signal dropout : 30% de chance de masquer chaque signal → force l'usage de la vision
  - LoRA r=32, LR=1e-4, early stopping sur challenge_score
  - 20K samples recommandé (100K ≈ 65h/époque sur RTX 5080)

Usage:
    # 1. Précalculer les signaux
    uv run python src/precompute_signals.py --n 22000 --seed 42 --append

    # 2. Run 2 — CoT + dropout
    uv run python src/qwen_finetune.py --n-train 20000 --n-val 1000 --epochs 3 --cot

    # Run 1 (baseline sans CoT)
    uv run python src/qwen_finetune.py --n-train 5000 --n-val 500 --epochs 3
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    default_data_collator,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.metrics import compute_score
from src.zero_shot_eval import OCC_BINS, stratified_sample

QWEN_REPO   = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_BASE  = Path(
    "/home/matt/Programmation/704_IADATA_ML_avance/datachallenge"
    "/DataChallenge2026/occlusion_datasets/raw/Crop_224_5fp_100K"
)
DATA_CSV        = Path("data/train.csv")
SIGNALS_CSV     = Path("data/signals_precomputed.csv")
OUTPUT_DIR      = Path("results/qwen_finetune")

PROMPT_BASE = (
    "Estimate the face occlusion fraction in this image.\n"
    "Occlusion = fraction of the face surface that is hidden, from 0.0000 (fully visible) "
    "to 1.0000 (fully hidden).\n"
    "What counts: hand, mask, glasses, hat, blur, shadow, hair covering face area.\n"
    "Reply ONLY as JSON: {{\"percentage\": <float 4 decimal places>}}"
)

PROMPT_WITH_SIGNALS = (
    "Estimate the face occlusion fraction in this image.\n"
    "Occlusion = fraction of the face surface that is hidden, from 0.0000 (fully visible) "
    "to 1.0000 (fully hidden).\n"
    "What counts: hand, mask, glasses, hat, blur, shadow, hair covering face area.\n\n"
    "Pre-computed signals (use as quantitative context — they may be noisy):\n"
    "  blur_score  : {blur_score:.1f}   (Laplacian variance; <80 = blurry/occluded)\n"
    "  hair_pct    : {hair_pct:.4f}  (Segformer: fraction of frame with hair pixels)\n"
    "  yaw_deg     : {yaw_deg:.1f}°   (MediaPipe head rotation; >30° = partial profile)\n"
    "  pose_occ    : {pose_occ:.4f}  (MediaPipe geometric occlusion from pose)\n"
    "  orientation : {orientation}\n\n"
    "Reply ONLY as JSON: {{\"percentage\": <float 4 decimal places>}}"
)


def build_prompt(signals: Optional[Dict[str, Any]] = None, cot: bool = False) -> str:
    """Construit le prompt avec ou sans signaux selon disponibilité."""
    reply_fmt = (
        'Reply as JSON with signal analysis then percentage:\n'
        '{"signal_analysis": {"blur": "...", "hair": "...", "pose": "..."}, '
        '"dominant": "hair|pose|blur|physical_object|none", '
        '"percentage": <float 4 decimal places>}'
        if cot else
        'Reply ONLY as JSON: {"percentage": <float 4 decimal places>}'
    )
    if signals is None:
        return (
            "Estimate the face occlusion fraction in this image.\n"
            "Occlusion = fraction of the face surface that is hidden, from 0.0000 "
            "(fully visible) to 1.0000 (fully hidden).\n"
            "What counts: hand, mask, glasses, hat, blur, shadow, hair covering face area.\n\n"
            + reply_fmt
        )
    hair_str = f"{signals['hair_pct']:.4f}" if signals["hair_pct"] >= 0 else "n/a"
    return (
        "Estimate the face occlusion fraction in this image.\n"
        "Occlusion = fraction of the face surface that is hidden, from 0.0000 "
        "(fully visible) to 1.0000 (fully hidden).\n"
        "What counts: hand, mask, glasses, hat, blur, shadow, hair covering face area.\n\n"
        "Pre-computed signals (use as quantitative context — they may be noisy):\n"
        f"  blur_score  : {signals['blur_score']:.1f}   (Laplacian variance; <80 = blurry/occluded)\n"
        f"  hair_pct    : {hair_str}  (Segformer: fraction of frame with hair pixels)\n"
        f"  yaw_deg     : {signals['yaw_deg']:.1f}°   (MediaPipe head rotation; >30° = partial profile)\n"
        f"  pose_occ    : {signals['pose_occ']:.4f}  (MediaPipe geometric occlusion from pose)\n"
        f"  orientation : {signals['orientation']}\n\n"
        + reply_fmt
    )


def _signal_blur_desc(blur: float) -> str:
    if blur < 30:
        return f"blur_score={blur:.1f} → très flou, probable occlusion blur"
    if blur < 80:
        return f"blur_score={blur:.1f} → légèrement flou, contribution possible"
    return f"blur_score={blur:.1f} → net, pas d'occlusion blur"


def _signal_hair_desc(hair: float) -> str:
    if hair < 0:
        return "n/a (Segformer non calculé)"
    if hair > 0.40:
        return f"hair_pct={hair:.2f} → très fort, cheveux couvrent large surface"
    if hair > 0.20:
        return f"hair_pct={hair:.2f} → modéré, contribution hair probable"
    return f"hair_pct={hair:.2f} → faible, cheveux peu occultants"


def _signal_pose_desc(yaw: float, pose_occ: float, orient: str) -> str:
    if abs(yaw) > 30 or pose_occ > 0.05:
        return f"yaw={yaw:.1f}°, pose_occ={pose_occ:.2f} → {orient}, occlusion par pose"
    return f"yaw={yaw:.1f}°, pose_occ={pose_occ:.2f} → frontal, aucune occlusion pose"


def _dominant_signal(signals: Dict[str, Any], gt: float) -> str:
    """Identifie rétrospectivement le signal dominant cohérent avec le GT."""
    if gt <= 0.04:
        return "none"
    blur  = signals.get("blur_score", 999)
    hair  = signals.get("hair_pct", -1)
    yaw   = abs(signals.get("yaw_deg", 0))
    pose  = signals.get("pose_occ", 0)
    scores: Dict[str, float] = {}
    if blur < 80:
        scores["blur"] = (80 - blur) / 80 * gt
    if hair > 0.15:
        scores["hair"] = hair * gt
    if yaw > 20 or pose > 0.03:
        scores["pose"] = max(yaw / 90, pose) * gt
    if not scores:
        scores["physical_object"] = gt
    return max(scores, key=lambda k: scores[k])


def build_cot_answer(
    signals: Optional[Dict[str, Any]],
    gt: float,
    dropout_rate: float = 0.30,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Construit la réponse CoT avec signal dropout aléatoire.

    Retourne (signals_used, answer_dict) où signals_used est le dict après dropout
    (pour construire le prompt correspondant).

    Dropout : chaque signal est masqué indépendamment avec probabilité dropout_rate.
    → Force le modèle à utiliser la vision quand les signaux sont absents.
    """
    if signals is None:
        answer = {"percentage": round(gt, 4)}
        return None, answer

    DROPPABLE = ["blur_score", "hair_pct", "yaw_deg", "pose_occ", "orientation"]
    dropped = set()
    for key in DROPPABLE:
        if random.random() < dropout_rate:
            dropped.add(key)

    if len(dropped) == len(DROPPABLE):
        dropped.discard(random.choice(DROPPABLE))

    signals_for_prompt: Dict[str, Any] = {}
    for key in ["blur_score", "hair_pct", "yaw_deg", "pose_occ", "orientation"]:
        if key not in dropped:
            signals_for_prompt[key] = signals[key]
        else:

            neutral = {"blur_score": -1, "hair_pct": -1, "yaw_deg": 0.0,
                       "pose_occ": 0.0, "orientation": "unknown"}
            signals_for_prompt[key] = neutral[key]

    for key in dropped:
        if key in ("blur_score",):
            signals_for_prompt[key] = 999.0   
        elif key == "hair_pct":
            signals_for_prompt[key] = -1.0

  
    analysis: Dict[str, str] = {}

    if "blur_score" not in dropped:
        analysis["blur"] = _signal_blur_desc(signals["blur_score"])
    else:
        analysis["blur"] = "n/a (signal masqué)"

    if "hair_pct" not in dropped:
        analysis["hair"] = _signal_hair_desc(signals["hair_pct"])
    else:
        analysis["hair"] = "n/a (signal masqué)"

    if "yaw_deg" not in dropped and "pose_occ" not in dropped:
        analysis["pose"] = _signal_pose_desc(
            signals["yaw_deg"], signals["pose_occ"],
            signals.get("orientation", "unknown")
        )
    else:
        analysis["pose"] = "n/a (signal masqué)"

    available_signals = {k: signals[k] for k in signals if k not in dropped}
    dominant = _dominant_signal(available_signals if available_signals else signals, gt)

    answer = {
        "signal_analysis": analysis,
        "dominant": dominant,
        "percentage": round(gt, 4),
    }
    return signals_for_prompt, answer


def _testlike_sample(df: pd.DataFrame, n_per_bin: int, seed: int) -> pd.DataFrame:
    """Equal-weight sample across OCC_BINS (~1/3 each)."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, lo, hi in OCC_BINS:
        pool = df[(df["FaceOcclusion"] >= lo) & (df["FaceOcclusion"] < hi)]
        k = min(n_per_bin, len(pool))
        parts.append(pool.iloc[rng.choice(len(pool), k, replace=False)])
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

class OcclusionSFTDataset(Dataset):
    """
    Retourne un dict {image, label, gender, signals} par sample.
    signals est None si aucun CSV de signaux n'est disponible.
    Le formatage en tokens est fait dans le collator (nécessite le processor).
    """
    def __init__(self, df: pd.DataFrame, image_base: Path,
                 signals_map: Optional[Dict[str, Dict]] = None):
        self.df          = df.reset_index(drop=True)
        self.image_base  = image_base
        self.signals_map = signals_map  

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = self.image_base / row["filename"]
        image = Image.open(img_path).convert("RGB")
        label = float(row["FaceOcclusion"])
        gender = float(row.get("gender", 0.5))
        signals = self.signals_map.get(row["filename"]) if self.signals_map else None
        return {"image": image, "label": label, "gender": gender, "signals": signals}

class OcclusionSFTCollator:
    """
    Formate chaque sample en conversation Qwen, tokenize, masque le prompt dans les labels.

    Mode normal : assistant: {"percentage": 0.1234}
    Mode CoT    : assistant: {"signal_analysis": {...}, "dominant": "hair", "percentage": 0.1234}
                  + signal dropout aléatoire dans le prompt
    """
    def __init__(self, processor, max_length: int = 512,
                 cot: bool = False, signal_dropout: float = 0.0):
        self.processor      = processor
        self.max_length     = max_length
        self.cot            = cot
        self.signal_dropout = signal_dropout

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        all_input_ids      = []
        all_attention_mask = []
        all_labels         = []
        all_pixel_values   = []
        all_image_grid_thw = []
        all_genders        = []
        all_gt             = []

        for s in samples:
            image   = s["image"]
            label   = s["label"]
            gender  = s["gender"]
            signals = s.get("signals")

            if self.cot and signals is not None:
                signals_for_prompt, answer_dict = build_cot_answer(
                    signals, label, dropout_rate=self.signal_dropout
                )
                answer = json.dumps(answer_dict, ensure_ascii=False)
                prompt = build_prompt(signals_for_prompt, cot=True)
            else:
                answer = f'{{"percentage": {label:.4f}}}'
                prompt = build_prompt(signals, cot=False)

            full_conv = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": answer},
                ]},
            ]
            full_text = self.processor.apply_chat_template(
                full_conv, tokenize=False, add_generation_prompt=False
            )

            prompt_conv = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ]},
            ]
            prompt_text = self.processor.apply_chat_template(
                prompt_conv, tokenize=False, add_generation_prompt=True
            )

            full_enc = self.processor(
                text=[full_text],
                images=[image],
                padding=False,
                return_tensors="pt",
            )
            prompt_enc = self.processor(
                text=[prompt_text],
                images=[image],
                padding=False,
                return_tensors="pt",
            )

            input_ids      = full_enc["input_ids"][0]
            attention_mask = full_enc["attention_mask"][0]
            prefix_len     = prompt_enc["input_ids"].shape[1]

            input_ids      = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            prefix_len     = min(prefix_len, self.max_length)

            labels = input_ids.clone()
            labels[:prefix_len] = -100

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)
            all_pixel_values.append(full_enc["pixel_values"])
            all_image_grid_thw.append(full_enc["image_grid_thw"])
            all_genders.append(gender)
            all_gt.append(label)

        pad_id  = self.processor.tokenizer.pad_token_id or 0
        max_len = max(ids.shape[0] for ids in all_input_ids)

        def _pad(t: torch.Tensor, pad_val: int) -> torch.Tensor:
            n = max_len - t.shape[0]
            if n == 0:
                return t
            return torch.cat([t, torch.full((n,), pad_val, dtype=t.dtype)])

        input_ids_batch      = torch.stack([_pad(x, pad_id) for x in all_input_ids])
        attention_mask_batch = torch.stack([_pad(x, 0)      for x in all_attention_mask])
        labels_batch         = torch.stack([_pad(x, -100)   for x in all_labels])

        pixel_values_batch   = torch.cat(all_pixel_values,   dim=0)
        image_grid_thw_batch = torch.cat(all_image_grid_thw, dim=0)

        return {
            "input_ids":       input_ids_batch,
            "attention_mask":  attention_mask_batch,
            "labels":          labels_batch,
            "pixel_values":    pixel_values_batch,
            "image_grid_thw":  image_grid_thw_batch,
            "_gender":         torch.tensor(all_genders, dtype=torch.float32),
            "_gt":             torch.tensor(all_gt,      dtype=torch.float32),
        }


def _parse_pct(text: str) -> float:
    """Extrait le float percentage du JSON généré."""
    m = re.search(r'"percentage"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if m:
        return float(np.clip(float(m.group(1)), 0.0, 1.0))
    for tok in re.findall(r'\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b', text):
        try:
            v = float(tok)
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            pass
    return 0.1


class QwenOcclusionTrainer(Trainer):
    """
    Trainer HF standard avec :
    - compute_loss qui ignore les colonnes _gender/_gt
    - generate_predictions pour éval challenge score
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        clean = {k: v for k, v in inputs.items() if not k.startswith("_")}
        return super().compute_loss(model, clean, return_outputs=return_outputs, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        clean = {k: v for k, v in inputs.items() if not k.startswith("_")}
        loss, logits, labels = super().prediction_step(
            model, clean, prediction_loss_only, ignore_keys=ignore_keys
        )
        return loss, logits, labels

class ChallengeScorecallback(TrainerCallback):
    """
    Génère des prédictions en mode autoregressive sur le val set,
    logue le challenge score et applique l'early stopping.
    """
    def __init__(self, model, processor, val_dataset, image_base: Path,
                 max_new_tokens: int = 64, eval_batch: int = 1,
                 patience: int = 2, cot: bool = False):
        self.model          = model
        self.processor      = processor
        self.val_dataset    = val_dataset
        self.image_base     = image_base
        self.max_new_tokens = max_new_tokens
        self.eval_batch     = eval_batch
        self.cot            = cot
        self.patience       = patience
        self.best_score     = float("inf")
        self.no_improve     = 0

    def _generate_preds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        preds, gts, genders = [], [], []

        loader = DataLoader(self.val_dataset, batch_size=self.eval_batch,
                            shuffle=False, num_workers=0, collate_fn=_identity_collate)

        device = next(self.model.parameters()).device

        with torch.no_grad():
            for batch in loader:
                for s in batch:
                    image   = s["image"]
                    label   = s["label"]
                    gender  = s["gender"]
                    signals = s.get("signals")
                    prompt  = build_prompt(signals)

                    msgs = [{"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ]}]
                    text = self.processor.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                    enc = self.processor(
                        text=[text], images=[image], padding=True, return_tensors="pt"
                    )
                    enc = {k: v.to(device) for k, v in enc.items()}

                    gen = self.model.generate(
                        **enc,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                    )
                    raw = self.processor.decode(
                        gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
                    )
                    preds.append(_parse_pct(raw))
                    gts.append(label)
                    genders.append(gender)

        return np.array(preds), np.array(gts), np.array(genders)

    def on_evaluate(self, args, state, control, **kwargs):
        print("\n  [ChallengScore] Génération val predictions …")
        preds, gts, genders = self._generate_preds()
        m = compute_score(preds, gts, genders)
        score = m["challenge_score"]

        print(f"  challenge_score={score:.5f}  "
              f"MAE={m['mae']:.5f}  err_F={m['err_F']:.5f}  err_M={m['err_M']:.5f}  "
              f"|F-M|={m['err_diff']:.5f}")

        top_mode = pd.Series(np.round(preds, 1)).value_counts()
        if len(top_mode) and top_mode.iloc[0] / len(preds) > 0.2:
            pct = 100 * top_mode.iloc[0] / len(preds)
            print(f"ANCRAGE : {top_mode.iloc[0]}/{len(preds)} ({pct:.0f}%) preds ≈ {top_mode.index[0]:.1f}")
        else:
            print(f"Distribution : min={preds.min():.3f}  max={preds.max():.3f}"
                  f"  mean={preds.mean():.3f}  std={preds.std():.3f}")

        if score < self.best_score - 1e-5:
            self.best_score = score
            self.no_improve = 0
            print(f"Nouveau meilleur score : {score:.5f}")
        else:
            self.no_improve += 1
            print(f"  Pas d'amélioration ({self.no_improve}/{self.patience})"
                  f"  best={self.best_score:.5f}")
            if self.no_improve >= self.patience:
                print(f"  ⏹ Early stopping : {self.patience} évals sans amélioration.")
                control.should_training_stop = True

        if state.log_history is not None:
            state.log_history.append({
                "step":                 state.global_step,
                "eval_challenge_score": score,
                "eval_mae":             m["mae"],
                "eval_err_F":           m["err_F"],
                "eval_err_M":           m["err_M"],
                "eval_err_diff":        m["err_diff"],
            })


def _identity_collate(batch):
    return batch

def load_model_qlora(repo: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenCls
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as QwenCls  # type: ignore

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {repo} in 4-bit …")
    model = QwenCls.from_pretrained(
        repo,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager", 
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=[],
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model

def _get(cfg: dict, *keys: str, default: Any = None) -> Any:
    val = cfg
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
    return val if val is not None else default


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}

def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/qwen_finetune.yaml"
    print(f"Config : {config_path}")
    cfg = load_config(config_path)
    n_train      = int(_get(cfg, "data", "n_train",           default=5000))
    n_val        = int(_get(cfg, "data", "n_val",             default=500))
    seed         = int(_get(cfg, "data", "seed",              default=42))
    qwen_repo    = str(_get(cfg, "model", "qwen_repo",        default=QWEN_REPO))
    lora_r       = int(_get(cfg, "model", "lora_r",           default=32))
    lora_alpha   = int(_get(cfg, "model", "lora_alpha",       default=64))
    lora_dropout = float(_get(cfg, "model", "lora_dropout",   default=0.05))
    epochs       = int(_get(cfg, "training", "epochs",        default=3))
    batch_size   = int(_get(cfg, "training", "batch_size",    default=1))
    grad_accum   = int(_get(cfg, "training", "grad_accum",    default=8))
    lr           = float(_get(cfg, "training", "lr",          default=1e-4))
    max_length   = int(_get(cfg, "training", "max_length",    default=700))
    cot          = bool(_get(cfg, "training", "cot",          default=False))
    signal_drop  = float(_get(cfg, "training", "signal_dropout", default=0.30))
    output_dir   = Path(str(_get(cfg, "training", "output_dir",       default=str(OUTPUT_DIR))))
    lora_save    = Path(str(_get(cfg, "training", "lora_save_path",   default=str(output_dir / "lora_adapters"))))
    resume       = _get(cfg, "training", "resume_from_checkpoint",    default=False)
    eval_batch   = int(_get(cfg, "eval", "eval_gen_batch",        default=1))
    patience     = int(_get(cfg, "eval", "early_stopping_patience", default=2))
    no_eval      = bool(_get(cfg, "eval", "no_challenge_eval",   default=False))

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATA_CSV} …")
    df_full = pd.read_csv(DATA_CSV).dropna(subset=["filename", "FaceOcclusion", "gender"])
    df_full["gender"] = pd.to_numeric(df_full["gender"], errors="coerce").fillna(0.5)
    df_full = df_full[df_full["gender"].isin([0, 1])].reset_index(drop=True)
    print(f"  Total: {len(df_full)} samples")

    if n_train > 0:
        n_per_bin_train = n_train // 3
        df_train = _testlike_sample(df_full, n_per_bin_train, seed=seed)
        n_val_eff = n_val if n_val > 0 else max(100, n_train // 10)
        n_per_bin_val = n_val_eff // 3
        remaining = df_full[~df_full.index.isin(df_train.index)]
        df_val = _testlike_sample(remaining, n_per_bin_val, seed=seed + 1)
    else:
        from src.zero_shot_eval import stratified_sample
        df_val   = stratified_sample(df_full, n=min(2000, len(df_full) // 10), seed=seed)
        df_train = df_full[~df_full.index.isin(df_val.index)]

    print(f"  Train: {len(df_train)}  Val: {len(df_val)}")
    print(f"  Train occ: mean={df_train['FaceOcclusion'].mean():.3f}  "
          f"Val occ: mean={df_val['FaceOcclusion'].mean():.3f}")

    signals_map: Optional[Dict[str, Dict]] = None
    if SIGNALS_CSV.exists():
        sig_df = pd.read_csv(SIGNALS_CSV)
        signals_map = {
            row["filename"]: {
                "blur_score":  float(row.get("blur_score",  -1)),
                "hair_pct":    float(row.get("hair_pct",    -1)),
                "yaw_deg":     float(row.get("yaw_deg",      0)),
                "pose_occ":    float(row.get("pose_occ",     0)),
                "orientation": str(row.get("orientation", "unknown")),
            }
            for _, row in sig_df.iterrows()
        }
        n_with = sum(1 for f in df_train["filename"] if f in signals_map)
        print(f"  Signaux précalculés : {len(signals_map)} entrées  "
              f"({n_with}/{len(df_train)} du train set couverts)")
    else:
        print(f"  ⚠ {SIGNALS_CSV} absent — prompt sans signaux (run precompute_signals.py d'abord)")

    train_ds = OcclusionSFTDataset(df_train, IMAGE_BASE, signals_map=signals_map)
    val_ds   = OcclusionSFTDataset(df_val,   IMAGE_BASE, signals_map=signals_map)
    print(f"Loading processor from {qwen_repo} …")
    processor = AutoProcessor.from_pretrained(qwen_repo)

    model = load_model_qlora(
        qwen_repo,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    collator = OcclusionSFTCollator(
        processor,
        max_length=max_length,
        cot=cot,
        signal_dropout=signal_drop if cot else 0.0,
    )

    eff_batch = batch_size * grad_accum
    print(f"\n  Batch effectif : {eff_batch}  (bs={batch_size} × accum={grad_accum})")
    print(f"  LR={lr}  LoRA r={lora_r}  α={lora_alpha}  epochs={epochs}")
    print(f"  CoT={cot}  signal_dropout={signal_drop if cot else 0.0}"
          f"  early_stopping_patience={patience}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        logging_steps=10,
        report_to=[],
        dataloader_num_workers=2,
        remove_unused_columns=False,
        label_names=["labels"],
        seed=seed,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    callbacks = []
    if not no_eval:
        callbacks.append(ChallengeScorecallback(
            model=model,
            processor=processor,
            val_dataset=val_ds,
            image_base=IMAGE_BASE,
            eval_batch=eval_batch,
            patience=patience,
            cot=cot,
        ))

    trainer = QwenOcclusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=callbacks or None,
    )

    print("\n  Démarrage de l'entraînement …\n")
    if resume:
        print(f"  resume_from_checkpoint=True → reprise depuis {output_dir}")
    trainer.train(resume_from_checkpoint=resume if resume else None)

    lora_save.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(lora_save))
    processor.save_pretrained(str(lora_save))
    print(f"\nAdapteurs LoRA sauvegardés : {lora_save}")
    print("Pour inférence : uv run python src/qwen_inference.py config/qwen_inference.yaml")


if __name__ == "__main__":
    main()
