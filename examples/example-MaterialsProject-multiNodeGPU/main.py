import os
import json
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from bam_torch.training.mp_trainer import MPTrainer_V2 as MPTrainer
from bam_torch.utils.utils import find_input_json, date


def setup(rank, world_size):
    # MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE are expected to be set
    # either by the user or by the job scheduler (e.g., SLURM)
    # when using srun. init_process_group will automatically use them.
    # Pass rank and world_size for clarity, although auto-detection might work too.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # Use SLURM_LOCALID for proper local GPU assignment
    local_rank = int(os.environ.get('SLURM_LOCALID', rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)

def run(rank, world_size, json_data):
    setup(rank, world_size)
    # Each process now knows its rank and the total world size.
    # Ensure the MPTrainer uses the correct device (handled by setup and MPTrainer init)
    trainer = MPTrainer(json_data, rank, world_size)
    trainer.train()
    #base_trainer.check_parameter_sync()
    dist.destroy_process_group() # Clean up the process group

def run_single_node(rank, world_size, json_data):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # Or another port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    trainer = MPTrainer(json_data, rank, world_size)
    trainer.train()
    dist.destroy_process_group()


if __name__ == '__main__':
    print(f"Start time: {date()}")
    input_json_path = find_input_json()
    torch.cuda.empty_cache()

    with open(input_json_path) as f:
        json_data = json.load(f)

    # Check if running under SLURM
    if 'SLURM_PROCID' in os.environ and 'SLURM_NTASKS' in os.environ:
        # Get rank and world size from SLURM environment variables
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])

        print(f"Running in SLURM environment. Rank: {rank}, World Size: {world_size}")

        # Call run directly, srun handles launching processes
        run(rank, world_size, json_data)

    else:
        # Fallback or single-node execution (optional)
        # or just raise an error if not in a SLURM environment.
        print("Not running in SLURM environment or required variables not set.")
        print("Falling back to original single-node logic (if gpu-parallel is true)")
        if json_data.get('gpu-parallel', False):
             print("Using torch.multiprocessing.spawn for single-node multi-GPU.")
             world_size = torch.cuda.device_count()
             mp.spawn(run_single_node, args=(world_size, json_data), nprocs=world_size, join=True)
        else:
             print("Running non-distributed on a single device.")
             trainer = MPTrainer(json_data, rank=0, world_size=1)
             trainer.train()

    print(f"End time: {date()}")
