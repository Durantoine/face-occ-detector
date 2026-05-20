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
from src.utils.environment import setup_environment

setup_environment()

CONFIG: Dict[str, Any] = {
    "arch": os.environ.get("FACE_OCC_PRETRAIN_ARCH", "dinov3_vith16plus"),
    "data_source": os.environ.get("FACE_OCC_PRETRAIN_SRC", "data/pretrain/"),
    "wds_pattern": None,
    "output_dir": "./results/pretrain",
    "tracking_uri": "sqlite:///mlflow.db",
    "mlflow_experiment": "face-occ-pretrain",
    "num_train_epochs": 30,
    "per_device_train_batch_size": 32,
    "gradient_accumulation_steps": 2,
    "learning_rate": 5.0e-5,
    "mask_ratio": 0.5,
    "warmup_ratio": 0.05,
    "weight_decay": 0.05,
    "seed": 42,
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": True,
    "image_size": 224,
    "patch_size": 16,
    "max_steps": -1,
    "save_steps": 5000,
    "logging_steps": 100,
    "teacher_frozen": True,
    "teacher_ema_decay": 0.999,
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
        import webdataset as wds
        self.pattern = pattern
        self.processor = processor
        self._inner = (
            wds.WebDataset(pattern, resampled=True, nodesplitter=wds.split_by_node, shardshuffle=True)
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

    model = DinoV3IBoT(
        arch=arch,
        mask_ratio=mask_ratio,
        image_size=image_size,
        patch_size=patch_size,
        teacher_frozen=teacher_frozen,
        teacher_ema_decay=teacher_ema_decay,
    )

    dataset = _build_dataset(data_source, wds_pattern, processor)
    is_iterable = isinstance(dataset, IterableDataset)

    class EMACallback(TrainerCallback):
        def __init__(self, m: DinoV3IBoT) -> None:
            self.m = m

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.m.ema_update_teacher()

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

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_experiment)
    run_name = f"pretrain-ibot-{arch}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.start_run(run_name=run_name)

    mlflow.log_params({
        "method": "iBOT-light-frozen" if teacher_frozen else f"iBOT-light-ema-{teacher_ema_decay}",
        "arch": arch,
        "data_source": data_source,
        "wds_pattern": wds_pattern,
        "mask_ratio": mask_ratio,
        "image_size": image_size,
        "patch_size": patch_size,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "bf16": bf16, "fp16": fp16,
        "gradient_checkpointing": gradient_checkpointing,
        "max_steps": max_steps,
        "is_iterable": is_iterable,
        "teacher_frozen": teacher_frozen,
        "teacher_ema_decay": teacher_ema_decay,
    })

    callbacks = [] if teacher_frozen else [EMACallback(model)]
    trainer = Trainer(model=model, args=args, train_dataset=dataset, callbacks=callbacks)
    print(f"iBOT pretraining: {arch} @ {image_size}x{image_size} | mask_ratio={mask_ratio} | "
          f"teacher={'frozen' if teacher_frozen else f'EMA(decay={teacher_ema_decay})'} | "
          f"bs={per_device_train_batch_size}x{gradient_accumulation_steps} (per GPU)")
    trainer.train()

    student = (
        trainer.accelerator.unwrap_model(trainer.model).student
        if hasattr(trainer, "accelerator") and trainer.accelerator
        else trainer.model.student
    ).cpu()
    info = mlflow.pytorch.log_model(student, "encoder")
    encoder_uri = info.model_uri
    print(f"Student encoder saved: {encoder_uri}")
    mlflow.end_run()
    return encoder_uri


if __name__ == "__main__":
    pretrain_ibot(**CONFIG)
