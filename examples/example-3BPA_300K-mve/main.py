import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import os
import json

from bam_torch.training import TRAINER_REGISTRY
from bam_torch.utils.utils import find_input_json, date


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def run(rank, world_size, json_data):
    setup(rank, world_size)
    trainer_name = json_data.get("trainer", "base")
    trainer_cls = TRAINER_REGISTRY.get(trainer_name)
    trainer = trainer_cls(json_data, rank, world_size)
    trainer.train()
    
    
if __name__ == '__main__':
    print(date()) 
    input_json_path = find_input_json()
    torch.cuda.empty_cache()

    with open(input_json_path) as f:
        json_data = json.load(f)

    if not json_data['gpu-parallel']:
        rank = 0
        world_size = 1
        run(rank, world_size, json_data)
    else:
        world_size = torch.cuda.device_count()
        mp.spawn(run, args=(world_size, json_data), nprocs=world_size, join=True)
        dist.destroy_process_group()
    
    print(date())
