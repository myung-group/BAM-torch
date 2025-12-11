from .base_trainer import BaseTrainer
from .mp_trainer import MPTrainer, MPTrainer_V2
from .mve_trainer import MVETrainer
from .multihead_trainer import MultiheadTrainer
from bam_torch.group_averaging.training.ga_trainer import GATrainer


TRAINER_REGISTRY = {
    "base": BaseTrainer,
    "mve": MVETrainer,
    "mh": MultiheadTrainer,
    "multi_head": MultiheadTrainer,
    "mp": MPTrainer_V2,
    "materials_project": MPTrainer_V2,
    "mp_v1": MPTrainer,
    "mp_v2": MPTrainer_V2,
    "ga": GATrainer,
    "group_averaging": GATrainer,
    "frame_averaging": GATrainer,
    "probabilistic_symmetrization": GATrainer,
}