from pathlib import Path
from typing import Any, Dict, Union

import yaml

INT_KEYS = {
    "warmup_steps",
    "num_epochs",
    "num_train_epochs",
    "train_batch_size",
    "eval_batch_size",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "n_trials",
    "min",
    "max",
    "num_labels",
    "max_length",
    "image_size",
    "save_total_limit",
    "dataloader_num_workers",
    "seed",
    "synthetic_seed",
    "early_stopping_patience",
    "num_layers",
    "num_heads",
    "cross_attention_heads",
    "size",
    "projection_size",
    "extra_transformer_num_layers",
    "extra_transformer_num_heads",
    "min_aug_per_class",
}
FLOAT_KEYS = {
    "learning_rate",
    "weight_decay",
    "max_grad_norm",
    "adam_epsilon",
    "warmup_ratio",
    "hidden_dropout_prob",
    "attention_probs_dropout_prob",
    "label_smoothing_factor",
    "adam_beta1",
    "adam_beta2",
}


class Config:
    def __init__(self, config_source: Union[str, Dict[str, Any]]):
        if isinstance(config_source, str):
            config_file = Path(config_source)
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_source}")
            with open(config_file, "r") as f:
                config_dict = yaml.safe_load(f)
        elif isinstance(config_source, dict):
            config_dict = config_source
        else:
            raise TypeError(f"Expected str or dict, got {type(config_source)}")

        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                match key:
                    case k if k in INT_KEYS:
                        setattr(self, k, int(value))
                    case k if k in FLOAT_KEYS:
                        setattr(self, k, float(value))
                    case _:
                        setattr(self, key, value)

    def __repr__(self) -> str:
        items = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"Config({', '.join(items)})"

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"Config has no attribute '{name}'")

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_architecture_config(architecture_name: str) -> Config:
    config_path = Path(f"configs/architectures/{architecture_name}.yaml")

    if not config_path.exists():
        available = [f.stem for f in Path("configs/architectures").glob("*.yaml")]
        raise FileNotFoundError(
            f"Architecture config not found: {config_path}\nAvailable architectures: {', '.join(available)}"
        )

    return Config(str(config_path))
