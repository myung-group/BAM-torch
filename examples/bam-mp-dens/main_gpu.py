"""Multi-GPU training launched with torchrun.

Usage (single GPU, plain Python):
    python main_gpu.py [input.json]

Usage (single-node, N GPUs):
    torchrun --standalone --nproc_per_node=N main_gpu.py [input.json]

Usage (multi-node, M nodes x N GPUs):
    torchrun --nnodes=M --nproc_per_node=N --node_rank=<i> \
             --master_addr=<IP> --master_port=29500 main_gpu.py [input.json]
"""
import os
import json

import torch
import torch.distributed as dist

from bam_torch.training import TRAINER_REGISTRY

import sys as _sys
import bam_torch.training.mp_trainer as _bt
print(f"[env] pid={os.getpid()} python={_sys.executable} trainer={_bt.__file__}",
      flush=True)
from bam_torch.utils.utils import date


def is_torchrun() -> bool:
    return "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed():
    """Initialize the process group using env vars set by torchrun."""
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")# timeout=timedelta(hours=2))
    return rank, local_rank, world_size


def run(rank, world_size, json_data):
    trainer_name = json_data.get("trainer", "mp")
    trainer_cls = TRAINER_REGISTRY[trainer_name]
    trainer = trainer_cls(json_data, rank, world_size)
    trainer.train()


if __name__ == '__main__':
    print(f"Start time: {date()}")
    input_json_path = _sys.argv[1] if len(_sys.argv) > 1 else "input.json"
    torch.cuda.empty_cache()

    with open(input_json_path) as f:
        json_data = json.load(f)

    if is_torchrun():
        rank, local_rank, world_size = setup_distributed()
        try:
            run(rank, world_size, json_data)
        finally:
            dist.destroy_process_group()
    else:
        if json_data.get('gpu-parallel'):
            raise RuntimeError(
                "gpu-parallel is enabled but the script was not launched with torchrun. "
                "Run with: torchrun --standalone --nproc_per_node=<N> main.py"
            )
        run(rank=0, world_size=1, json_data=json_data)

    print(f"End time: {date()}")
