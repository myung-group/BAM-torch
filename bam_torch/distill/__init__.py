"""Knowledge-distillation utilities for BAM-torch.

Train a smaller `RACE` student against a larger pretrained teacher with a
hybrid energy/force loss:

    L = lambda_dft * L_DFT + (1 - lambda_dft) * L_teacher

See ``examples/example-distillation/`` for a complete pipeline.
"""
from .trainer import DistillTrainer
from .dataset import (
    DistillData,
    get_distill_dataloader,
    teacher_baselines_from_ckpt,
)

__all__ = [
    "DistillTrainer",
    "DistillData",
    "get_distill_dataloader",
    "teacher_baselines_from_ckpt",
]
