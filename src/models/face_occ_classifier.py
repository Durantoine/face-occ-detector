from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

from src.models.dinov3_loader import hidden_size_of, load_dinov3


def _is_dinov3(model_name: str) -> bool:
    return model_name.startswith("dinov3_")


def _build_backbone(model_name: str) -> Tuple[nn.Module, int]:
    if _is_dinov3(model_name):
        backbone = load_dinov3(model_name)
        return backbone, hidden_size_of(model_name)
    backbone = AutoModel.from_pretrained(model_name)
    return backbone, backbone.config.hidden_size


def _forward_backbone(backbone: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    if hasattr(backbone, "get_intermediate_layers"):
        return backbone.get_intermediate_layers(pixel_values, n=1)[0]
    return backbone(pixel_values=pixel_values).last_hidden_state


class FaceOccRegressor(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov3_vits16",
        output_dim: int = 1,
        hidden_dropout_prob: float = 0.1,
        pooling: str = "cls",
        projection_size: Optional[int] = None,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.hidden_dropout_prob = hidden_dropout_prob
        self.pooling = pooling
        self.output_activation = output_activation

        self.backbone, hidden_size = _build_backbone(model_name)

        if projection_size:
            self.projection: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(hidden_size, projection_size),
                nn.LayerNorm(projection_size),
                nn.GELU(),
                nn.Dropout(hidden_dropout_prob),
            )
            final_size = projection_size
        else:
            self.projection = None
            final_size = hidden_size

        self.attention_pool: Optional[nn.Linear] = nn.Linear(hidden_size, 1) if pooling == "attention" else None
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.head = nn.Linear(final_size, output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in (self.head, self.attention_pool):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        if self.projection is not None:
            for m in self.projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)

    def _pool(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return hidden.mean(dim=1)
        if self.pooling == "max":
            return hidden.max(dim=1)[0]
        if self.pooling == "attention":
            assert self.attention_pool is not None
            w = torch.softmax(self.attention_pool(hidden).squeeze(-1), dim=1).unsqueeze(-1)
            return (hidden * w).sum(dim=1)
        return hidden[:, 0, :]

    def _activate(self, logits: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "sigmoid":
            return torch.sigmoid(logits)
        if self.output_activation == "clamp":
            return logits.clamp(0.0, 1.0)
        return logits

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        hidden = _forward_backbone(self.backbone, pixel_values)
        pooled = self._pool(hidden)
        if self.projection is not None:
            pooled = self.projection(pooled)
        pooled = self.dropout(pooled)
        logits = self.head(pooled)
        pred = self._activate(logits).squeeze(-1) if self.output_dim == 1 else self._activate(logits)
        return {"logits": pred}

    @classmethod
    def load_from_mlflow(cls, model_uri: str, output_dim: Optional[int] = None) -> "FaceOccRegressor":
        import mlflow
        model = mlflow.pytorch.load_model(model_uri)
        if output_dim and hasattr(model, "output_dim") and output_dim != model.output_dim:
            new_model = cls(
                model_name=model.model_name,
                output_dim=output_dim,
                hidden_dropout_prob=model.hidden_dropout_prob,
                pooling=model.pooling,
                output_activation=getattr(model, "output_activation", "sigmoid"),
            )
            new_model.backbone.load_state_dict(model.backbone.state_dict(), strict=False)
            return new_model
        return model


