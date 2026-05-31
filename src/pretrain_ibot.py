import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, IterableDataset

from src.data.dataset import IMAGE_EXTS
from src.models.dinov3_loader import get_image_processor, hidden_size_of, load_dinov3
from src.models.sapiens2_loader import (
    hidden_size_of as sapiens2_hidden_size_of,
    is_sapiens2,
    load_sapiens2,
)
from src.utils.environment import setup_environment

setup_environment()


CONFIG: Dict[str, Any] = {
    "arch": os.environ.get("FACE_OCC_PRETRAIN_ARCH", "dinov3_vith16plus"),
    "data_source": os.environ.get("FACE_OCC_PRETRAIN_SRC", "data/pretrain/"),
    "wds_pattern": None,
    "output_dir": os.environ.get("FACE_OCC_PRETRAIN_OUT", "./results/pretrain"),
    "tracking_uri": "sqlite:///mlflow.db",
    "mlflow_experiment": "face-occ-pretrain",
    "num_train_epochs": int(os.environ.get("FACE_OCC_PRETRAIN_EPOCHS", "30")),
    "per_device_train_batch_size": int(os.environ.get("FACE_OCC_PRETRAIN_BS", "32")),
    "gradient_accumulation_steps": int(os.environ.get("FACE_OCC_PRETRAIN_GA", "2")),
    "learning_rate": float(os.environ.get("FACE_OCC_PRETRAIN_LR", "5.0e-5")),
    "mask_ratio": float(os.environ.get("FACE_OCC_PRETRAIN_MASK_RATIO", "0.5")),
    "warmup_ratio": float(os.environ.get("FACE_OCC_PRETRAIN_WARMUP", "0.05")),
    "weight_decay": float(os.environ.get("FACE_OCC_PRETRAIN_WD", "0.05")),
    "seed": 42,
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": True,
    "image_size": int(os.environ.get("FACE_OCC_PRETRAIN_IMG_SIZE", "112")),
    "patch_size": 16,
    "max_steps": int(os.environ.get("FACE_OCC_PRETRAIN_MAX_STEPS", "100000")),
    "save_steps": int(os.environ.get("FACE_OCC_PRETRAIN_SAVE_STEPS",
        "12000" if is_sapiens2(os.environ.get("FACE_OCC_PRETRAIN_ARCH", "dinov3_vith16plus")) else "22000")),
    "logging_steps": 100,
    "teacher_frozen": os.environ.get("FACE_OCC_PRETRAIN_TEACHER_FROZEN", "1") != "0",
    "teacher_ema_decay": float(os.environ.get("FACE_OCC_PRETRAIN_EMA_DECAY", "0.999")),
}


class DinoV3IBoT(nn.Module):
    def __init__(
        self,
        arch: str,
        mask_ratio: float = 0.5,
        image_size: int = 224,
        patch_size: int = 16,
        teacher_frozen: bool = True,
        teacher_ema_decay: float = 0.999,
    ) -> None:
        super().__init__()
        self.student = load_dinov3(arch)
        self.teacher = load_dinov3(arch)
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        self.teacher_frozen = teacher_frozen
        self.teacher_ema_decay = teacher_ema_decay
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.hidden = hidden_size_of(arch)
        self._grad_ckpt_enabled = False

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        from torch.utils.checkpoint import checkpoint as ckpt
        if self._grad_ckpt_enabled or not hasattr(self.student, "blocks"):
            return
        for block in self.student.blocks:
            orig = block.forward

            def _wrap(orig_forward):
                def _ckpt_forward(*args, **kw):
                    return ckpt(orig_forward, *args, use_reentrant=False, **kw)
                return _ckpt_forward
            block.forward = _wrap(orig)
        self._grad_ckpt_enabled = True
        print(f"[DinoV3IBoT] gradient_checkpointing enabled on {len(self.student.blocks)} student blocks")

    def gradient_checkpointing_disable(self) -> None:
        self._grad_ckpt_enabled = False

    @torch.no_grad()
    def ema_update_teacher(self) -> None:
        if self.teacher_frozen:
            return
        d = self.teacher_ema_decay
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(d).add_(ps.data, alpha=1.0 - d)

    def _random_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        n_mask = max(1, int(self.num_patches * self.mask_ratio))
        noise = torch.rand(batch_size, self.num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :n_mask], True)
        return mask

    def forward(self, pixel_values: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        B = pixel_values.shape[0]
        mask = self._random_mask(B, pixel_values.device)

        student_out = self.student.forward_features(pixel_values, masks=mask)
        student_patches = student_out["x_norm_patchtokens"]

        with torch.no_grad():
            teacher_out = self.teacher.forward_features(pixel_values, masks=None)
            teacher_patches = teacher_out["x_norm_patchtokens"]

        s = F.normalize(student_patches.float(), dim=-1, eps=1e-6)
        t = F.normalize(teacher_patches.float(), dim=-1, eps=1e-6)
        cos = (s * t).sum(dim=-1)
        per_pos_loss = 1.0 - cos
        denom = mask.float().sum().clamp(min=1.0)
        loss = (per_pos_loss * mask.float()).sum() / denom

        with torch.no_grad():
            cls_s = F.normalize(student_out["x_norm_clstoken"].float(), dim=-1, eps=1e-6)
            cls_t = F.normalize(teacher_out["x_norm_clstoken"].float(), dim=-1, eps=1e-6)
            cls_drift = (1.0 - (cls_s * cls_t).sum(dim=-1)).mean()

        return {"loss": loss, "cls_drift": cls_drift.detach()}


class Sapiens2IBoT(nn.Module):
    """iBOT-style pretrain for Sapiens2 backbones.

    Unlike DINOv3 (which exposes `forward_features(masks=...)` for proper token-level
    masking), Sapiens2's standalone forward doesn't accept a mask argument. We use
    pixel-space masking instead: the student sees an image with the corresponding
    patch regions zeroed out, while the teacher sees the original image. The student
    must reconstruct the masked patch features by attending to surrounding context.

    Less clean than DINOv3's [MASK] token approach, but safer in terms of "ne pas
    abîmer le backbone" — no structural modification, the model just sees masked
    images as a strong cut-out augmentation.
    """

    def __init__(
        self,
        arch: str,
        mask_ratio: float = 0.4,
        image_size: int = 224,
        patch_size: int = 16,
        teacher_frozen: bool = True,
        teacher_ema_decay: float = 0.9995,
        drop_rate: float = 0.15,
    ) -> None:
        super().__init__()
        self.student = load_sapiens2(arch, image_size=image_size, drop_rate=drop_rate)
        self.teacher = load_sapiens2(arch, image_size=image_size, drop_rate=0.0)
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        self.teacher_frozen = teacher_frozen
        self.teacher_ema_decay = teacher_ema_decay
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_side = image_size // patch_size
        self.num_patches = self.num_patches_side ** 2
        self.hidden = sapiens2_hidden_size_of(arch)
        self._grad_ckpt_enabled = False

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        from torch.utils.checkpoint import checkpoint as ckpt
        if self._grad_ckpt_enabled:
            return
        blocks_attr = next((a for a in ("blocks", "layers") if hasattr(self.student, a)), None)
        if blocks_attr is None:
            print(f"[Sapiens2IBoT] WARNING: cannot enable gradient_checkpointing (no .blocks/.layers)")
            return
        blocks = getattr(self.student, blocks_attr)
        for block in blocks:
            orig = block.forward

            def _wrap(orig_forward):
                def _ckpt_forward(*args, **kw):
                    return ckpt(orig_forward, *args, use_reentrant=False, **kw)
                return _ckpt_forward
            block.forward = _wrap(orig)
        self._grad_ckpt_enabled = True
        print(f"[Sapiens2IBoT] gradient_checkpointing enabled on {len(blocks)} student blocks")

    def gradient_checkpointing_disable(self) -> None:
        self._grad_ckpt_enabled = False

    @torch.no_grad()
    def ema_update_teacher(self) -> None:
        if self.teacher_frozen:
            return
        d = self.teacher_ema_decay
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(d).add_(ps.data, alpha=1.0 - d)

    def _random_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        n_mask = max(1, int(self.num_patches * self.mask_ratio))
        noise = torch.rand(batch_size, self.num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :n_mask], True)
        return mask

    def _apply_pixel_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Zero out the pixel regions corresponding to masked patches.

        x:    (B, C, H, W)  pixel values
        mask: (B, num_patches)  True = masked
        """
        B, _, _, _ = x.shape
        P = self.patch_size
        S = self.num_patches_side
        mask_2d = mask.view(B, S, S).to(x.dtype)
        mask_full = (
            mask_2d.unsqueeze(1)
            .repeat_interleave(P, dim=2)
            .repeat_interleave(P, dim=3)
        )
        return x * (1.0 - mask_full)

    @staticmethod
    def _extract_tokens(out: Any) -> torch.Tensor:
        """Return (B, N, D). Sapiens2 returns either a Tensor, tuple/list, or dict."""
        if isinstance(out, (tuple, list)):
            return out[0]
        if isinstance(out, dict):
            for k in ("x", "tokens", "last_hidden_state", "x_norm_patchtokens"):
                if k in out:
                    return out[k]
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out

    def forward(self, pixel_values: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        B = pixel_values.shape[0]
        mask = self._random_mask(B, pixel_values.device)
        masked_pixel_values = self._apply_pixel_mask(pixel_values, mask)

        student_tokens = self._extract_tokens(self.student(masked_pixel_values))
        with torch.no_grad():
            teacher_tokens = self._extract_tokens(self.teacher(pixel_values))

        # Skip CLS token at index 0; the remaining tokens align with our patch mask
        student_patches = student_tokens[:, -self.num_patches:, :]
        teacher_patches = teacher_tokens[:, -self.num_patches:, :]

        s = F.normalize(student_patches.float(), dim=-1, eps=1e-6)
        t = F.normalize(teacher_patches.float(), dim=-1, eps=1e-6)
        cos = (s * t).sum(dim=-1)
        per_pos_loss = 1.0 - cos
        denom = mask.float().sum().clamp(min=1.0)
        loss = (per_pos_loss * mask.float()).sum() / denom

        with torch.no_grad():
            cls_s = F.normalize(student_tokens[:, 0, :].float(), dim=-1, eps=1e-6)
            cls_t = F.normalize(teacher_tokens[:, 0, :].float(), dim=-1, eps=1e-6)
            cls_drift = (1.0 - (cls_s * cls_t).sum(dim=-1)).mean()

        return {"loss": loss, "cls_drift": cls_drift.detach()}


def build_ibot_model(
    arch: str,
    mask_ratio: float,
    image_size: int,
    patch_size: int,
    teacher_frozen: bool,
    teacher_ema_decay: float,
    drop_rate: float = 0.15,
) -> nn.Module:
    """Dispatch the right iBOT wrapper based on the backbone arch."""
    if arch.startswith("dinov3"):
        return DinoV3IBoT(
            arch=arch, mask_ratio=mask_ratio, image_size=image_size, patch_size=patch_size,
            teacher_frozen=teacher_frozen, teacher_ema_decay=teacher_ema_decay,
        )
    if is_sapiens2(arch):
        return Sapiens2IBoT(
            arch=arch, mask_ratio=mask_ratio, image_size=image_size, patch_size=patch_size,
            teacher_frozen=teacher_frozen, teacher_ema_decay=teacher_ema_decay,
            drop_rate=drop_rate,
        )
    raise ValueError(f"Unknown arch for iBOT pretrain: {arch}")


class ImageOnlyDataset(Dataset):
    def __init__(self, paths: List[str], processor: Any) -> None:
        self.paths = paths
        self.processor = processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = Image.open(self.paths[idx]).convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return {"pixel_values": pixel_values}


class WebDatasetWrapper(IterableDataset):
    def __init__(self, pattern: str, processor: Any, shuffle_buffer: int = 1000) -> None:
        super().__init__()
        import glob

        import webdataset as wds
        shards = sorted(glob.glob(pattern))
        if not shards:
            raise FileNotFoundError(f"No tar shards matched: {pattern}")
        print(f"WebDataset: {len(shards)} shards (e.g. {Path(shards[0]).name} ... {Path(shards[-1]).name})")
        self.pattern = pattern
        self.processor = processor
        self._inner = (
            wds.WebDataset(shards, resampled=True, nodesplitter=wds.split_by_node, shardshuffle=False)
            .shuffle(shuffle_buffer)
            .decode("pil")
            .to_tuple("jpg")
        )

    def __iter__(self):
        for (img,) in self._inner:
            if img.mode != "RGB":
                img = img.convert("RGB")
            pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
            yield {"pixel_values": pixel_values}


def _scan_images(data_dir: str) -> List[str]:
    paths = sorted(str(p) for p in Path(data_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    print(f"Found {len(paths):,} images in {data_dir}")
    return paths


def _build_dataset(data_source: str, wds_pattern: Optional[str], processor: Any) -> Any:
    if wds_pattern:
        print(f"Using WebDataset pattern: {wds_pattern}")
        return WebDatasetWrapper(wds_pattern, processor)
    tar_shards = sorted(Path(data_source).rglob("*.tar"))
    if tar_shards:
        pattern = str(tar_shards[0].parent / "*.tar")
        print(f"Auto-detected {len(tar_shards)} WDS shards under {data_source} → pattern: {pattern}")
        return WebDatasetWrapper(pattern, processor)
    return ImageOnlyDataset(_scan_images(data_source), processor)


def pretrain_ibot(
    arch: str,
    data_source: str,
    wds_pattern: Optional[str],
    output_dir: str,
    tracking_uri: str,
    mlflow_experiment: str,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    mask_ratio: float,
    warmup_ratio: float,
    weight_decay: float,
    seed: int,
    bf16: bool,
    fp16: bool,
    gradient_checkpointing: bool,
    image_size: int,
    patch_size: int,
    max_steps: int,
    save_steps: int,
    logging_steps: int,
    teacher_frozen: bool,
    teacher_ema_decay: float,
) -> str:
    from transformers import Trainer, TrainerCallback, TrainingArguments

    processor = get_image_processor(arch)
    if hasattr(processor, "size"):
        processor.size = {"height": image_size, "width": image_size}

    model = build_ibot_model(
        arch=arch,
        mask_ratio=mask_ratio,
        image_size=image_size,
        patch_size=patch_size,
        teacher_frozen=teacher_frozen,
        teacher_ema_decay=teacher_ema_decay,
    )

    dataset = _build_dataset(data_source, wds_pattern, processor)
    is_iterable = isinstance(dataset, IterableDataset)

    if not torch.cuda.is_available():
        if bf16 or fp16:
            print(f"WARNING: non-CUDA device — disabling bf16/fp16 (was bf16={bf16}, fp16={fp16})")
        bf16 = False
        fp16 = False

    class EMACallback(TrainerCallback):
        def __init__(self, m: nn.Module) -> None:
            self.m = m

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.m.ema_update_teacher()

    class SpeedCallback(TrainerCallback):
        def __init__(self, target_steps: int) -> None:
            self.target = target_steps
            self._t0: Optional[float] = None
            self._step0: Optional[int] = None

        def on_log(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            import time
            if int(os.environ.get("RANK", "0")) != 0:
                return
            now = time.time()
            if self._t0 is None or self._step0 is None:
                self._t0, self._step0 = now, state.global_step
                return
            ds = max(1, state.global_step - self._step0)
            s_per_step = (now - self._t0) / ds
            remaining = max(0, self.target - state.global_step) * s_per_step
            print(
                f"[speed] step={state.global_step} | {s_per_step:.2f}s/step | "
                f"ETA_to_{self.target}={remaining / 3600:.1f}h",
                flush=True,
            )
            self._t0, self._step0 = now, state.global_step

    snapshot_env = os.environ.get("FACE_OCC_PRETRAIN_SNAPSHOT_STEPS", "").strip()
    snapshot_steps = sorted({int(s) for s in snapshot_env.split(",") if s.strip()})

    class EncoderSnapshotCallback(TrainerCallback):
        """Log the student encoder to MLflow at each listed step as artifact
        `encoder_<step>`. URI `runs:/<run_id>/encoder_<step>` plugs directly into
        `pretrained_source` of finetune yamls — lets us A/B test multiple pretrain
        checkpoints from a single run."""
        def __init__(self, m: nn.Module, steps: List[int]) -> None:
            import copy as _copy
            self._copy = _copy
            self.m = m
            self.steps = set(steps)
            self.done: set = set()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            step = state.global_step
            if step not in self.steps or step in self.done:
                return
            self.done.add(step)
            if int(os.environ.get("RANK", "0")) != 0:
                return
            snap = self._copy.deepcopy(self.m.student).to(torch.float32).cpu()
            info = mlflow.pytorch.log_model(snap, f"encoder_{step}")
            print(f"[snapshot] encoder_{step} logged: {info.model_uri}")
            del snap

    if is_iterable and max_steps <= 0:
        raise ValueError("WebDataset is iterable — set max_steps>0 (env FACE_OCC_PRETRAIN_MAX_STEPS).")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs if not is_iterable else 1,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        logging_steps=logging_steps,
        report_to=["mlflow"],
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=seed,
        max_steps=max_steps if (is_iterable or max_steps > 0) else -1,
    )

    rank = int(os.environ.get("RANK", "0"))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    run_id_file = Path(output_dir) / "mlflow_run_id.txt"
    existing_ckpts = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])) if Path(output_dir).exists() else []
    if run_id_file.exists() and existing_ckpts:
        resume_arg: Optional[str] = str(existing_ckpts[-1])
    else:
        resume_arg = None

    if rank == 0:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(mlflow_experiment)
        if resume_arg is not None:
            prev_run_id = run_id_file.read_text().strip()
            mlflow.start_run(run_id=prev_run_id)
            print(f"RESUMING run {prev_run_id} from {resume_arg}")
        else:
            run_name = f"pretrain-ibot-{arch}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            active = mlflow.start_run(run_name=run_name)
            run_id_file.write_text(active.info.run_id)
            mlflow.log_params({
                "method": "iBOT-light-frozen" if teacher_frozen else f"iBOT-light-ema-{teacher_ema_decay}",
                "arch": arch,
                "data_source": data_source,
                "wds_pattern": wds_pattern,
                "mask_ratio": mask_ratio,
                "image_size": image_size,
                "patch_size": patch_size,
                "is_iterable": is_iterable,
                "teacher_frozen": teacher_frozen,
                "teacher_ema_decay": teacher_ema_decay,
            })

    callbacks: List[TrainerCallback] = [] if teacher_frozen else [EMACallback(model)]
    if snapshot_steps:
        callbacks.append(EncoderSnapshotCallback(model, snapshot_steps))
        print(f"Encoder snapshots will be logged at steps: {snapshot_steps}")
    if max_steps > 0:
        callbacks.append(SpeedCallback(max_steps))
    trainer = Trainer(model=model, args=args, train_dataset=dataset, callbacks=callbacks)
    print(f"iBOT pretraining: {arch} @ {image_size}x{image_size} | mask_ratio={mask_ratio} | "
          f"teacher={'frozen' if teacher_frozen else f'EMA(decay={teacher_ema_decay})'} | "
          f"bs={per_device_train_batch_size}x{gradient_accumulation_steps} (per GPU) | "
          f"resume={resume_arg} | rank={rank}")
    trainer.train(resume_from_checkpoint=resume_arg)

    if rank != 0:
        return ""

    if hasattr(trainer, "accelerator") and trainer.accelerator:
        try:
            wrapped = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
        except TypeError:
            wrapped = trainer.accelerator.unwrap_model(trainer.model)
    else:
        wrapped = trainer.model
    student = wrapped.student.to(torch.float32).cpu()
    info = mlflow.pytorch.log_model(student, "encoder")
    encoder_uri = info.model_uri
    print(f"Student encoder saved: {encoder_uri}")
    mlflow.end_run()
    return encoder_uri


if __name__ == "__main__":
    pretrain_ibot(**CONFIG)
