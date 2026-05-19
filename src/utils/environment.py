import os
from functools import lru_cache
from typing import Optional

import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def setup_environment() -> None:
    _configure_device()
    _configure_mps()
    _configure_logging()


def _configure_device() -> None:
    device = get_device()
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    elif device == "mps":
        print("Using Apple Silicon GPU (MPS)")


def _configure_mps(default_watermark: str = "0.0") -> None:
    if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = default_watermark


def _configure_logging(level: Optional[int] = None) -> None:
    try:
        from transformers import logging as transformers_logging

        if level:
            transformers_logging.set_verbosity(level)
        else:
            transformers_logging.set_verbosity_info()
    except ImportError:
        pass
