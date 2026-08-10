from .base_trainer import BaseTrainer
from .mp_trainer import MPTrainer, MPTrainer_V2
from .mve_trainer import MVETrainer
from .multihead_trainer import MultiheadTrainer

try:
    from bam_torch.group_averaging.training.ga_trainer import GATrainer
except ImportError:
    GATrainer = None

try:
    from bam_torch.distill import DistillTrainer
except ImportError:
    DistillTrainer = None


TRAINER_REGISTRY = {
    "base": BaseTrainer,
    "mve": MVETrainer,
    "mh": MultiheadTrainer,
    "multi_head": MultiheadTrainer,
    "mp": MPTrainer_V2,
    "materials_project": MPTrainer_V2,
    "mp_v1": MPTrainer,
    "mp_v2": MPTrainer_V2,
}
if GATrainer is not None:
    TRAINER_REGISTRY.update({
        "ga": GATrainer,
        "group_averaging": GATrainer,
        "frame_averaging": GATrainer,
        "probabilistic_symmetrization": GATrainer,
    })
if DistillTrainer is not None:
    TRAINER_REGISTRY.update({
        "distill": DistillTrainer,
        "distillation": DistillTrainer,
    })