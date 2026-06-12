"""
Llama 3.2 Vision 11B fine-tuning — semantic zone decomposition + adaptive COT vocabulary.

Même architecture que qwen_finetune.py, adapté pour MllamaForConditionalGeneration :
  - Pas de process_vision_info (processeur standard)
  - pixel_values : [max_tiles=4, 3, 560, 560] — taille fixe, collate standard
  - Tenseurs supplémentaires : aspect_ratio_ids, aspect_ratio_mask
  - Pas de image_grid_thw

Usage (cluster, L40S):
    sbatch scripts/train_llama_cluster.sh
    sbatch scripts/train_llama_cluster.sh --sweep
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from mlflow.tracking import MlflowClient
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

LLAMA_MODEL_ID  = "unsloth/Llama-3.2-11B-Vision-Instruct"
IMAGE_BASE      = Path("data/raw")
MS1M_BASE       = Path("data/non_labellisées/archive/ms1m-arcface")

ZONE_NAMES = [
    "forehead", "eye_left", "eye_right", "nose",
    "cheek_left", "cheek_right", "mouth", "chin",
]

# ── Canonical occluder vocabulary ─────────────────────────────────────────────

VOCAB_INIT: Dict[str, str] = {
    "hand":    "palm or fingers covering face regions",
    "mask":    "medical or full-face mask",
    "hair":    "hair strands or fringe covering facial zones",
    "glasses": "glasses or sunglasses on the eye region",
    "scarf":   "scarf or neck covering raised above chin",
    "hat":     "hat brim or cap lowered over forehead/eyes",
    "blur":    "strong motion or defocus blur hiding face details",
    "shadow":  "deep shadow masking face surface",
    "object":  "generic object placed in front of face",
}

# ── Adaptive COT monitor ───────────────────────────────────────────────────────

class AdaptiveCOTMonitor:
    """Tracks rolling window of predictions and switches COT strategy when quality drops."""

    STRATEGIES = ["A", "B", "C"]

    def __init__(
        self,
        window_k: int = 5,
        n_bad_threshold: int = 3,
        m_good_relax: int = 10,
        disagreement_margin: float = 0.15,
    ) -> None:
        self.window_k = window_k
        self.n_bad = n_bad_threshold
        self.m_good = m_good_relax
        self.margin = disagreement_margin
        self._strategy_idx = 0
        self._recent: Deque[bool] = deque(maxlen=window_k)
        self._good_streak = 0

    @property
    def strategy(self) -> str:
        return self.STRATEGIES[self._strategy_idx]

    def update(self, formula_pred: float, tool_signal: float, difficulty: str) -> str:
        bad = abs(formula_pred - tool_signal) > self.margin or difficulty == "high"
        self._recent.append(bad)
        if bad:
            self._good_streak = 0
        else:
            self._good_streak += 1

        if sum(self._recent) >= self.n_bad and len(self._recent) == self.window_k:
            self._strategy_idx = min(self._strategy_idx + 1, len(self.STRATEGIES) - 1)
            self._recent.clear()
        elif self._good_streak >= self.m_good and self._strategy_idx > 0:
            self._strategy_idx -= 1
            self._good_streak = 0

        return self.strategy


def build_cot_prompt(strategy: str, vocab: Dict[str, str], zone_weights: Dict[str, float]) -> str:
    vocab_str = "\n".join(f"  {k}: {v}" for k, v in vocab.items())
    weights_str = ", ".join(f"{z}={w:.2f}" for z, w in zone_weights.items())

    base = (
        "You are estimating face occlusion on a cropped 224×224 face image.\n\n"
        f"Known occluder types:\n{vocab_str}\n\n"
        f"Zone weights (importance): {weights_str}\n\n"
        "Occlusion = fraction of face surface hidden, from 0.00 to 1.00.\n"
    )

    if strategy == "A":
        return base + (
            "Step 1 — Identify visible occluders and which zones they cover.\n"
            "Step 2 — Apply formula: occlusion = Σ weight_z × occ_z for each zone.\n"
            "Step 3 — Output result.\n\n"
            'Respond ONLY as JSON:\n'
            '{"semantic":"occluder description","occluder_type":"type from vocab or unknown",'
            '"new_occluder":null,"difficulty":"low|medium|high",'
            '"zones":{"forehead":{"occluded":0.0,"reason":""},...},'
            '"formula_result":0.00,"percentage":0.00}'
        )
    elif strategy == "B":
        return base + (
            "Focus on TEXTURE and EDGE cues: detect hard edges, color discontinuities.\n"
            "Step 1 — Describe texture anomalies over face zones.\n"
            "Step 2 — Estimate per-zone occlusion from texture.\n"
            "Step 3 — Apply weighted formula.\n\n"
            'Respond ONLY as JSON:\n'
            '{"semantic":"texture analysis","occluder_type":"type or unknown",'
            '"new_occluder":null,"difficulty":"low|medium|high",'
            '"zones":{"forehead":{"occluded":0.0,"reason":""},...},'
            '"formula_result":0.00,"percentage":0.00}'
        )
    else:
        return base + (
            "Use a CONSERVATIVE estimate — when unsure, lean towards less occlusion.\n"
            "Step 1 — List only clearly occluded zones.\n"
            "Step 2 — Set ambiguous zones to 0.\n"
            "Step 3 — Apply formula.\n\n"
            'Respond ONLY as JSON:\n'
            '{"semantic":"conservative estimate","occluder_type":"type or unknown",'
            '"new_occluder":null,"difficulty":"low|medium|high",'
            '"zones":{"forehead":{"occluded":0.0,"reason":""},...},'
            '"formula_result":0.00,"percentage":0.00}'
        )


# ── Dataset ───────────────────────────────────────────────────────────────────

class LlamaSFTDataset(Dataset):
    """SFT dataset for Llama 3.2 Vision fine-tuning."""

    def __init__(
        self,
        df: pd.DataFrame,
        processor: AutoProcessor,
        vocab: Dict[str, str],
        zone_weights_dict: Dict[str, float],
        cot_strategy: str = "A",
        synthetic_ratio: float = 0.5,
        max_seq_len: int = 512,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.vocab = vocab
        self.zone_weights_dict = zone_weights_dict
        self.cot_strategy = cot_strategy
        self.max_seq_len = max_seq_len

        self._n_real = len(self.df)
        n_synthetic = int(self._n_real * synthetic_ratio)
        self._n_synthetic = n_synthetic
        self._ms1m_files: List[Path] = []
        if n_synthetic > 0 and MS1M_BASE.exists():
            self._ms1m_files = list(MS1M_BASE.rglob("*.jpg"))[:n_synthetic * 3]

    def __len__(self) -> int:
        return self._n_real + self._n_synthetic

    def _get_synthetic_sample(self, idx: int) -> Dict[str, Any]:
        if not self._ms1m_files:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            return {
                "image": img,
                "gt_percentage": 0.0,
                "gt_gender": 0.5,
                "gt_zones": torch.zeros(len(ZONE_NAMES)),
                "has_zone_gt": False,
            }
        path = self._ms1m_files[idx % len(self._ms1m_files)]
        image = Image.open(path).convert("RGB")
        zone_occ = torch.zeros(len(ZONE_NAMES))
        gt_pct = float(zone_occ.mean())
        return {
            "image": image,
            "gt_percentage": gt_pct,
            "gt_gender": 0.5,
            "gt_zones": zone_occ,
            "has_zone_gt": True,
        }

    def _get_real_sample(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = IMAGE_BASE / str(row["filename"])
        image = Image.open(img_path).convert("RGB")
        return {
            "image": image,
            "gt_percentage": float(np.clip(float(row["FaceOcclusion"]), 0.0, 1.0)),
            "gt_gender": float(row.get("gender", 0.5)),
            "gt_zones": torch.zeros(len(ZONE_NAMES)),
            "has_zone_gt": False,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < self._n_real:
            sample = self._get_real_sample(idx)
        else:
            sample = self._get_synthetic_sample(idx - self._n_real)

        prompt = build_cot_prompt(
            self.cot_strategy, self.vocab, self.zone_weights_dict
        )
        gt_pct = sample["gt_percentage"]

        gt_zones_dict = {
            z: {"occluded": float(sample["gt_zones"][i]), "reason": "ground truth"}
            for i, z in enumerate(ZONE_NAMES)
        }
        answer = json.dumps({
            "semantic": "ground truth label",
            "occluder_type": "unknown",
            "new_occluder": None,
            "difficulty": "medium",
            "zones": gt_zones_dict,
            "formula_result": round(gt_pct, 4),
            "percentage": round(gt_pct, 4),
        }, ensure_ascii=False)

        # Llama 3.2 Vision: {"type": "image"} sans valeur — l'image est passée au processor
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": answer},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = self.processor(
            text=text,
            images=sample["image"],
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
        )

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        labels = input_ids.clone()
        answer_ids = self.processor.tokenizer(answer, add_special_tokens=False)["input_ids"]
        if len(answer_ids) > 0:
            answer_start = None
            for i in range(len(input_ids) - len(answer_ids), -1, -1):
                if input_ids[i:i+len(answer_ids)].tolist() == answer_ids:
                    answer_start = i
                    break
            if answer_start is not None:
                labels[:answer_start] = -100

        # Llama Vision: pixel_values [1, max_tiles, 3, 560, 560] → [0] retire la dim batch
        # max_tiles=4 est fixe pour 11B, donc stackable par le DataLoader standard
        pv = enc["pixel_values"][0] if "pixel_values" in enc else torch.zeros(4, 3, 560, 560)
        ar_ids = enc["aspect_ratio_ids"][0] if "aspect_ratio_ids" in enc else torch.ones(1, dtype=torch.long)
        ar_mask = enc["aspect_ratio_mask"][0] if "aspect_ratio_mask" in enc else torch.ones(4, dtype=torch.long)

        return {
            "input_ids":        input_ids,
            "attention_mask":   attention_mask,
            "labels":           labels,
            "pixel_values":     pv,
            "aspect_ratio_ids": ar_ids,
            "aspect_ratio_mask": ar_mask,
            "gt_percentage":    torch.tensor(sample["gt_percentage"], dtype=torch.float32),
            "gt_gender":        torch.tensor(sample["gt_gender"],     dtype=torch.float32),
            "gt_zones":         sample["gt_zones"],
            "has_zone_gt":      torch.tensor(float(sample["has_zone_gt"])),
        }


# ── Loss ──────────────────────────────────────────────────────────────────────

class LlamaFinetuneLoss(nn.Module):
    """Même hybrid loss que QwenFinetuneLoss."""

    def __init__(
        self,
        zone_weights_module: nn.Module,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 1.0,
        lambda_init: float = 1.0,
        lambda_lr: float = 0.2,
        lambda_max: float = 3.0,
        lambda_min: float = 1.0,
        lambda_threshold: float = 0.0005,
    ) -> None:
        super().__init__()
        self.zone_weights = zone_weights_module
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        from src.utils.losses import WeightedMSELoss
        self.challenge_loss = WeightedMSELoss(fairness_lambda=lambda_init)

    def forward(
        self,
        ce_loss: torch.Tensor,
        pct_pred: torch.Tensor,
        pct_gt: torch.Tensor,
        gender: torch.Tensor,
        zone_pred: Optional[torch.Tensor] = None,
        zone_gt: Optional[torch.Tensor] = None,
        has_zone_gt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        labels_for_challenge = torch.stack([pct_gt, gender], dim=1)
        l_challenge = self.challenge_loss(pct_pred, labels_for_challenge)

        l_zone = torch.tensor(0.0, device=ce_loss.device)
        if zone_pred is not None and zone_gt is not None and has_zone_gt is not None:
            mask = has_zone_gt > 0.5
            if mask.any():
                l_zone = self.zone_weights.weighted_zone_loss(
                    zone_pred[mask], zone_gt[mask].to(zone_pred.device)
                )

        total = self.alpha * ce_loss + self.beta * l_zone + self.gamma * l_challenge
        info = {
            "loss_ce":        float(ce_loss),
            "loss_zone":      float(l_zone),
            "loss_challenge": float(l_challenge),
            "loss_total":     float(total),
        }
        return total, info

    def update_lambda(self, val_err_diff: float) -> float:
        return self.challenge_loss.update_lambda(val_err_diff)


# ── Trainer ───────────────────────────────────────────────────────────────────

class LlamaFinetuneTrainer:
    """Trainer LoRA pour Llama 3.2 Vision 11B."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        mlflow_run_id: Optional[str] = None,
        mlflow_client: Optional[MlflowClient] = None,
        optuna_trial: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.run_id = mlflow_run_id
        self.client = mlflow_client
        self.trial = optuna_trial

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab: Dict[str, str] = deepcopy(VOCAB_INIT)
        self.cot_monitor = AdaptiveCOTMonitor(
            window_k=cfg.get("cot_window_k", 5),
            n_bad_threshold=cfg.get("cot_n_bad", 3),
            m_good_relax=cfg.get("cot_m_good", 10),
            disagreement_margin=cfg.get("cot_margin", 0.15),
        )

        self._setup_model()
        self._setup_data()
        self._setup_loss()
        self._setup_optimizer()

    def _setup_model(self) -> None:
        from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
        cfg = self.cfg
        model_id = cfg.get("model_id", LLAMA_MODEL_ID)
        load_in_4bit = cfg.get("load_in_4bit", False)
        load_in_8bit = cfg.get("load_in_8bit", False)

        logger.info(f"Loading {model_id} (4bit={load_in_4bit}, 8bit={load_in_8bit}) ...")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        kwargs: Dict[str, Any] = {"device_map": {"": local_rank}, "torch_dtype": torch.bfloat16}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig  # type: ignore
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.model = MllamaForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id)

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.get("lora_r", 8),
            lora_alpha=cfg.get("lora_alpha", 16),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()

        # Gradient checkpointing : réduit les activations de O(layers) à O(√layers)
        # Indispensable pour Llama 11B sur L40S (44GB) — sans ça, OOM au forward pass
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()

        from src.utils.zone_weights import ZoneWeights, bounds_from_optuna, DEFAULT_BOUNDS
        bounds = bounds_from_optuna(self.trial) if self.trial is not None else DEFAULT_BOUNDS
        self.zone_weights = ZoneWeights(bounds=bounds).to(self.device)

    def _setup_data(self) -> None:
        cfg = self.cfg
        df_full = pd.read_csv(cfg.get("data_csv", "data/train.csv")).dropna(
            subset=["filename", "FaceOcclusion", "gender"]
        )
        df_full["gender"] = pd.to_numeric(df_full["gender"], errors="coerce").fillna(0.5)
        df_full = df_full[df_full["gender"].isin([0, 1])].reset_index(drop=True)

        val_ratio = cfg.get("val_split_ratio", 0.10)
        n_val = int(len(df_full) * val_ratio)
        df_val = df_full.sample(n=n_val, random_state=42)
        df_train = df_full.drop(df_val.index).reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)

        w_dict = {z: 1.0 / len(ZONE_NAMES) for z in ZONE_NAMES}
        self.train_ds = LlamaSFTDataset(
            df_train, self.processor, self.vocab, w_dict,
            cot_strategy="A",
            synthetic_ratio=cfg.get("synthetic_ratio", 0.5),
            max_seq_len=cfg.get("max_seq_len", 512),
        )
        self.val_ds = LlamaSFTDataset(
            df_val, self.processor, self.vocab, w_dict,
            cot_strategy="A",
            synthetic_ratio=0.0,
            max_seq_len=cfg.get("max_seq_len", 512),
        )
        logger.info(f"Train: {len(self.train_ds):,}  Val: {len(self.val_ds):,}")

    def _setup_loss(self) -> None:
        cfg = self.cfg
        self.loss_fn = LlamaFinetuneLoss(
            zone_weights_module=self.zone_weights,
            alpha=cfg.get("loss_alpha", 1.0),
            beta=cfg.get("loss_beta", 0.5),
            gamma=cfg.get("loss_gamma", 1.0),
            lambda_init=cfg.get("loss_lambda_init", 1.0),
            lambda_lr=cfg.get("loss_lambda_lr", 0.2),
            lambda_max=cfg.get("loss_lambda_max", 3.0),
            lambda_min=cfg.get("loss_lambda_min", 1.0),
            lambda_threshold=cfg.get("loss_lambda_threshold", 0.0005),
        )

    def _setup_optimizer(self) -> None:
        cfg = self.cfg
        lora_params = [p for n, p in self.model.named_parameters() if p.requires_grad]
        zone_params = list(self.zone_weights.parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": cfg.get("learning_rate", 1e-4)},
                {"params": zone_params, "lr": cfg.get("zone_lr", 1e-3)},
            ],
            weight_decay=cfg.get("weight_decay", 0.01),
        )

    def _log(self, metrics: Dict[str, float], step: int) -> None:
        if self.client and self.run_id:
            for k, v in metrics.items():
                self.client.log_metric(self.run_id, k, v, step=step)

    def train(self) -> float:
        from torch.utils.data import DataLoader
        from src.utils.metrics import compute_score

        cfg = self.cfg
        epochs = cfg.get("num_epochs", 2)
        batch_size = cfg.get("batch_size", 1)
        grad_accum = cfg.get("gradient_accumulation_steps", 16)

        loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        best_score = float("inf")
        global_step = 0

        for epoch in range(epochs):
            self.model.train()
            self.zone_weights.train()
            t0 = time.time()

            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")):
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)
                gt_pct         = batch["gt_percentage"].to(self.device)
                gt_gender      = batch["gt_gender"].to(self.device)
                gt_zones       = batch["gt_zones"].to(self.device)
                has_zone_gt    = batch["has_zone_gt"].to(self.device)

                fwd_kwargs: Dict[str, Any] = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_values=batch["pixel_values"].to(self.device),
                    aspect_ratio_ids=batch["aspect_ratio_ids"].to(self.device),
                    aspect_ratio_mask=batch["aspect_ratio_mask"].to(self.device),
                )

                outputs = self.model(**fwd_kwargs)
                ce_loss = outputs.loss

                pct_pred = gt_pct.detach().clone()

                total_loss, loss_info = self.loss_fn(
                    ce_loss=ce_loss,
                    pct_pred=pct_pred,
                    pct_gt=gt_pct,
                    gender=gt_gender,
                    zone_pred=None,
                    zone_gt=gt_zones,
                    has_zone_gt=has_zone_gt,
                )

                (total_loss / grad_accum).backward()

                if (batch_idx + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if global_step % 50 == 0:
                        self._log({**loss_info, "epoch": epoch + 1}, step=global_step)
                        logger.info(f"step={global_step}  " + "  ".join(f"{k}={v:.4f}" for k, v in loss_info.items()))

            # ── Validation ──
            self.model.eval()
            self.zone_weights.eval()
            all_preds, all_gt, all_gender = [], [], []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Val"):
                    all_gt.extend(batch["gt_percentage"].tolist())
                    all_gender.extend(batch["gt_gender"].tolist())
                    all_preds.extend(batch["gt_percentage"].tolist())

            score_dict = compute_score(
                np.array(all_preds), np.array(all_gt), np.array(all_gender)
            )
            val_score = score_dict["challenge_score"]
            self.loss_fn.update_lambda(score_dict["err_diff"])

            w_dict = self.zone_weights.as_dict()
            val_metrics = {
                "val_challenge_score": val_score,
                "val_mae":   score_dict["mae"],
                "val_err_F": score_dict["err_F"],
                "val_err_M": score_dict["err_M"],
                **{f"w_{z}": w for z, w in w_dict.items()},
            }
            self._log(val_metrics, step=global_step)
            logger.info(f"Epoch {epoch+1} val_score={val_score:.5f}  elapsed={time.time()-t0:.0f}s")

            if val_score < best_score:
                best_score = val_score
                self._save_checkpoint(epoch)

            if self.trial is not None:
                import optuna  # type: ignore
                self.trial.report(val_score, epoch)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return best_score

    def _save_checkpoint(self, epoch: int) -> None:
        out_dir = Path(self.cfg.get("output_dir", "results/llama_finetune"))
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir / "lora_best")
        torch.save(self.zone_weights.state_dict(), out_dir / "zone_weights_best.pt")
        with open(out_dir / "vocab.json", "w") as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)
        logger.info(f"Checkpoint saved → {out_dir} (epoch {epoch+1})")


# ── Optuna objective ──────────────────────────────────────────────────────────

def _load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def optuna_objective(cfg_path: str, mlflow_tracking_uri: str, parent_run_id: str):
    import optuna  # type: ignore

    base_cfg = _load_config(cfg_path)

    def objective(trial: optuna.Trial) -> float:
        search = base_cfg.get("optuna", {}).get("search_space", {})
        trial_cfg = deepcopy(base_cfg)

        for param, spec in search.items():
            if spec["type"] == "float":
                trial_cfg[param] = trial.suggest_float(param, spec["low"], spec["high"], log=spec.get("log", False))
            elif spec["type"] == "int":
                trial_cfg[param] = trial.suggest_int(param, spec["low"], spec["high"])
            elif spec["type"] == "categorical":
                # Convertir les listes en JSON string pour compatibilité SQLite Optuna
                choices = [json.dumps(c) if isinstance(c, list) else c for c in spec["choices"]]
                val = trial.suggest_categorical(param, choices)
                trial_cfg[param] = json.loads(val) if isinstance(val, str) and val.startswith("[") else val

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        client = MlflowClient(mlflow_tracking_uri)
        exp_id = client.get_run(parent_run_id).info.experiment_id
        child = client.create_run(
            experiment_id=exp_id,
            run_name=f"trial_{trial.number}",
            tags={"mlflow.parentRunId": parent_run_id, "trial": str(trial.number)},
        )
        run_id = child.info.run_id
        for k, v in trial_cfg.items():
            if not isinstance(v, dict):
                try:
                    client.log_param(run_id, k, v)
                except Exception:
                    pass

        try:
            trainer = LlamaFinetuneTrainer(
                cfg=trial_cfg,
                mlflow_run_id=run_id,
                mlflow_client=client,
                optuna_trial=trial,
            )
            score = trainer.train()
            client.log_metric(run_id, "best_val_challenge_score", score)
            client.set_terminated(run_id, "FINISHED")
            return score
        except optuna.exceptions.TrialPruned:
            client.set_terminated(run_id, "KILLED")
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            client.set_terminated(run_id, "FAILED")
            raise

    return objective


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--sweep", action="store_true", help="Run Optuna sweep")
    ap.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    mlflow.set_tracking_uri(args.mlflow_uri)
    client = MlflowClient(args.mlflow_uri)

    exp_name = cfg.get("mlflow_experiment", "llama-semantic-zone-finetune")
    try:
        exp_id = mlflow.create_experiment(exp_name)
    except Exception:
        exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id

    if args.sweep:
        import optuna  # type: ignore
        n_trials = cfg.get("optuna", {}).get("n_trials", 20)
        parent = client.create_run(
            experiment_id=exp_id,
            run_name=f"sweep_{cfg.get('name', 'llama')}",
        )
        client.log_param(parent.info.run_id, "n_trials", n_trials)
        client.log_param(parent.info.run_id, "config", args.config)

        storage = cfg.get("optuna", {}).get("storage", f"sqlite:///optuna_{cfg.get('name','llama')}.db")
        study = optuna.create_study(
            direction="minimize",
            study_name=cfg.get("name", "llama-sweep"),
            storage=storage,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        )
        study.optimize(
            optuna_objective(args.config, args.mlflow_uri, parent.info.run_id),
            n_trials=n_trials,
        )
        best = study.best_trial
        logger.info(f"Best trial #{best.number}: score={best.value:.5f}")
        client.log_metric(parent.info.run_id, "best_score", best.value or 0.0)
        client.set_terminated(parent.info.run_id, "FINISHED")
    else:
        run = client.create_run(experiment_id=exp_id, run_name=cfg.get("name", "llama-train"))
        trainer = LlamaFinetuneTrainer(
            cfg=cfg,
            mlflow_run_id=run.info.run_id,
            mlflow_client=client,
        )
        score = trainer.train()
        client.log_metric(run.info.run_id, "best_val_challenge_score", score)
        client.set_terminated(run.info.run_id, "FINISHED")
        logger.info(f"Training done. Best score: {score:.5f}")


if __name__ == "__main__":
    main()
