
import torch


_HF_REPO_BY_ARCH = {
    "sapiens2_0.1b": "facebook/sapiens2-pretrain-0.1b",
    "sapiens2_0.4b": "facebook/sapiens2-pretrain-0.4b",
    "sapiens2_0.8b": "facebook/sapiens2-pretrain-0.8b",
    "sapiens2_1b":   "facebook/sapiens2-pretrain-1b",
    "sapiens2_5b":   "facebook/sapiens2-pretrain-5b",
}

_HIDDEN_SIZES = {
    "sapiens2_0.1b": 768,
    "sapiens2_0.4b": 1024,
    "sapiens2_0.8b": 1280,
    "sapiens2_1b":   1536,
    "sapiens2_5b":   2432,
}

_SAFETENSOR_FILENAME = {
    arch: f"{arch}_pretrain.safetensors" for arch in _HF_REPO_BY_ARCH
}


def is_sapiens2(model_name: str) -> bool:
    return model_name.startswith("sapiens2_")


def hidden_size_of(arch: str) -> int:
    if arch not in _HIDDEN_SIZES:
        raise ValueError(f"Unknown Sapiens2 arch '{arch}'. Available: {list(_HIDDEN_SIZES)}")
    return _HIDDEN_SIZES[arch]


def load_sapiens2(
    arch: str = "sapiens2_0.8b",
    image_size: int = 224,
    drop_rate: float = 0.0,
    pretrained: bool = True,
) -> torch.nn.Module:
    from sapiens.backbones.standalone.sapiens2 import Sapiens2

    if arch not in _HF_REPO_BY_ARCH:
        raise ValueError(f"Unknown Sapiens2 arch '{arch}'. Available: {list(_HF_REPO_BY_ARCH)}")

    backbone = Sapiens2(
        arch=arch,
        img_size=(image_size, image_size),
        patch_size=16,
        out_indices=-1,
        out_type="raw",
        with_cls_token=True,
        drop_rate=drop_rate,
    )
    # Expose patch_size for _forward_backbone token-standardization (avoids guessing
    # via loop — robust if Sapiens variants ever use non-16 patches).
    backbone.patch_size_used = 16
    if not pretrained:
        print(f"Sapiens2 {arch}: random init (pretrained=False)")
        return backbone

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # v22: try the local cache FIRST (local_files_only=True) → instant hit, no network.
    # Without this, hf_hub_download does a HEAD request to HF on EVERY trial to validate the
    # cache; on a compute node with slow/blocked internet that round-trip (with retries) adds
    # latency to each trial's model build. Only the first trial actually downloads.
    _hf_kw = dict(repo_id=_HF_REPO_BY_ARCH[arch], filename=_SAFETENSOR_FILENAME[arch])
    try:
        ckpt_path = hf_hub_download(local_files_only=True, **_hf_kw)
    except Exception:
        ckpt_path = hf_hub_download(**_hf_kw)  # first trial / cache miss → download once
    state = load_file(ckpt_path)
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    print(f"Loaded Sapiens2 {arch}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    return backbone


class _ConfigShim:
    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size


def make_config(arch: str) -> _ConfigShim:
    return _ConfigShim(hidden_size_of(arch))
