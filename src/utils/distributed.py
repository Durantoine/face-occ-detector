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


def reinit_process_group() -> bool:
    """Destroy + recreate NCCL process group. Use between trials to clear stale
    communicator state from a previously failed/timed-out trial.

    Both ranks must call this together (it's a collective). If the group is already
    broken (e.g., one rank crashed), destroy is still safe but init may hang.

    Returns True if the group is initialized (re-created or was kept), False otherwise.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return False  # single-process, nothing to do
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"[Rank {local_rank}] WARNING: destroy_process_group failed: {e}", flush=True)
    try:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return True
    except Exception as e:
        print(f"[Rank {local_rank}] WARNING: init_process_group failed: {e}", flush=True)
        return False
