import os
from typing import Any

import torch
import torch.distributed as dist


def setup_distributed() -> int:
    rank = int(os.environ.get("LOCAL_RANK", -1))
    if rank != -1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)
    return rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main() -> bool:
    return int(os.environ.get("LOCAL_RANK", -1)) in (-1, 0)


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def broadcast(value: Any, src: int = 0) -> Any:
    if not dist.is_initialized():
        return value
    lst = [value]
    dist.broadcast_object_list(lst, src=src)
    return lst[0]
