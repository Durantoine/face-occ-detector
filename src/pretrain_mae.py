from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.dataset import IMAGE_EXTS
from src.models.dinov3_loader import get_image_processor, load_dinov3
from src.utils.environment import setup_environment

setup_environment()

CONFIG: Dict[str, Any] = {
    "model_name": "facebook/vit-mae-large",
    "data_dir": "data/pretrain/",
    "output_dir": "./results/pretrain",
    "tracking_uri": "sqlite:///mlflow.db",
    "mlflow_experiment": "face-occ-pretrain",
    "num_train_epochs": 10,
    "per_device_train_batch_size": 8,
    "learning_rate": 1.5e-4,
    "mask_ratio": 0.75,
    "warmup_ratio": 0.05,
    "weight_decay": 0.05,
    "seed": 42,
}


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


def _scan_images(data_dir: str) -> List[str]:
    paths = sorted(str(p) for p in Path(data_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    print(f"Found {len(paths):,} images in {data_dir}")
    return paths


def _build_mae_from_hf(model_name: str, mask_ratio: float) -> torch.nn.Module:
    from transformers import ViTMAEForPreTraining
    model = ViTMAEForPreTraining.from_pretrained(model_name)
    model.config.mask_ratio = mask_ratio
    return model


def _build_mae_from_dinov3(arch: str, mask_ratio: float) -> torch.nn.Module:
    _ = (load_dinov3(arch), mask_ratio)
    raise NotImplementedError(
        "DINOv3 → ViTMAE weight mapping not yet implemented. "
        "Use model_name='facebook/vit-mae-large' for now, or write the mapping in this function."
    )


def pretrain_mae(
    model_name: str,
    data_dir: str,
    output_dir: str = "./results/pretrain",
    tracking_uri: str = "sqlite:///mlflow.db",
    mlflow_experiment: str = "face-occ-pretrain",
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 8,
    learning_rate: float = 1.5e-4,
    mask_ratio: float = 0.75,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.05,
    seed: int = 42,
) -> str:
    from transformers import Trainer, TrainingArguments

    processor = get_image_processor(model_name)
    if model_name.startswith("dinov3_"):
        model = _build_mae_from_dinov3(model_name, mask_ratio)
    else:
        model = _build_mae_from_hf(model_name, mask_ratio)

    dataset = ImageOnlyDataset(_scan_images(data_dir), processor)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        fp16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=seed,
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    print(f"MAE pretraining: {model_name} | {len(dataset):,} images | mask_ratio={mask_ratio}")
    trainer.train()

    encoder = (
        trainer.accelerator.unwrap_model(trainer.model).vit
        if hasattr(trainer, "accelerator") and trainer.accelerator
        else trainer.model.vit
    ).cpu()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_experiment)
    run_name = f"pretrain-{model_name.replace('/', '_')}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_name": model_name,
            "mask_ratio": mask_ratio,
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "num_images": len(dataset),
        })
        info = mlflow.pytorch.log_model(encoder, "encoder")
        encoder_uri = info.model_uri
        print(f"Encoder saved: {encoder_uri}")

    return encoder_uri


if __name__ == "__main__":
    pretrain_mae(
        model_name=CONFIG["model_name"],
        data_dir=CONFIG["data_dir"],
        output_dir=CONFIG["output_dir"],
        tracking_uri=CONFIG["tracking_uri"],
        mlflow_experiment=CONFIG["mlflow_experiment"],
        num_train_epochs=CONFIG["num_train_epochs"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        learning_rate=CONFIG["learning_rate"],
        mask_ratio=CONFIG["mask_ratio"],
        warmup_ratio=CONFIG["warmup_ratio"],
        weight_decay=CONFIG["weight_decay"],
        seed=CONFIG["seed"],
    )
