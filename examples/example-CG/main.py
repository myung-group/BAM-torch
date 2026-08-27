"""
Main script for Coarse-Grained (CG) model training with BAM-torch.

This script trains a CG model using the bottom-up approach:
- Atomistic trajectory is converted to CG representation
- Standard RACE model architecture is used with CG inputs
- Forces and energies are mapped from atomistic to CG level

Usage:
    python main.py                    # Uses input_cg.json in current directory
    python main.py --config my.json   # Uses specified config file
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import os
import sys
import json
import argparse

# Add parent path for imports when running from examples directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from bam_torch.training import TRAINER_REGISTRY
from bam_torch.utils.utils import find_input_json, date


def setup(rank, world_size):
    """Initialize distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def run(rank, world_size, json_data):
    """Run training on a single process/GPU."""
    if world_size > 1:
        setup(rank, world_size)

    # Get trainer class from registry
    trainer_name = json_data.get("trainer", "cg")
    trainer_cls = TRAINER_REGISTRY.get(trainer_name)

    if trainer_cls is None:
        raise ValueError(f"Unknown trainer: {trainer_name}. "
                        f"Available: {list(TRAINER_REGISTRY.keys())}")

    # Initialize and run trainer
    trainer = trainer_cls(json_data, rank, world_size)
    trainer.train()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='CG Model Training with BAM-torch')
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to JSON configuration file (default: input_cg.json)')
    args = parser.parse_args()

    print("="*60)
    print("BAM-torch Coarse-Grained (CG) Training")
    print("="*60)
    print(f"Start time: {date()}")
    print()

    # Find and load configuration
    if args.config:
        input_json_path = args.config
    else:
        # Look for input_cg.json first, then input.json
        if os.path.exists('input_cg.json'):
            input_json_path = 'input_cg.json'
        else:
            input_json_path = find_input_json()

    if input_json_path is None or not os.path.exists(input_json_path):
        print("Error: No configuration file found!")
        print("Please provide input_cg.json or use --config option.")
        sys.exit(1)

    print(f"Configuration: {input_json_path}")

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Load configuration
    with open(input_json_path) as f:
        json_data = json.load(f)

    # Validate CG configuration
    if json_data.get('trainer') not in ['cg', 'coarse_grained', 'coarse-grained', 'cg_multihead', 'cg_mh']:
        print(f"\nWarning: trainer is set to '{json_data.get('trainer')}', "
              f"changing to 'cg' for CG training.")
        json_data['trainer'] = 'cg'

    # Run training
    if not json_data.get('gpu-parallel', False):
        rank = 0
        world_size = 1
        run(rank, world_size, json_data)
    else:
        world_size = torch.cuda.device_count()
        print(f"Multi-GPU training with {world_size} GPUs")
        mp.spawn(run, args=(world_size, json_data), nprocs=world_size, join=True)
        dist.destroy_process_group()

    print()
    print("="*60)
    print(f"End time: {date()}")
    print("="*60)


if __name__ == '__main__':
    main()
