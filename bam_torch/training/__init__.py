from .base_trainer import BaseTrainer
from .mp_trainer import MPTrainer, MPTrainer_V2
from .mve_trainer import MVETrainer
from .multihead_trainer import MultiheadTrainer

try:
    from bam_torch.group_averaging.training.ga_trainer import GATrainer
    _GA_AVAILABLE = True
except (ImportError, OSError):
    _GA_AVAILABLE = False


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

if _GA_AVAILABLE:
    TRAINER_REGISTRY["ga"] = GATrainer
    TRAINER_REGISTRY["group_averaging"] = GATrainer
    TRAINER_REGISTRY["frame_averaging"] = GATrainer
    TRAINER_REGISTRY["probabilistic_symmetrization"] = GATrainer

try:
    from bam_torch.charge_dependent.training.cd_trainer import CDTrainer
    TRAINER_REGISTRY["cd"] = CDTrainer
    TRAINER_REGISTRY["charge_dependent"] = CDTrainer
except ImportError:
    pass

try:
    from bam_torch.charge_dependent.training.cd_trainer_v3 import CDTrainerV3
    TRAINER_REGISTRY["cd_v3"] = CDTrainerV3
    TRAINER_REGISTRY["charge_dependent_v3"] = CDTrainerV3
    TRAINER_REGISTRY["cd_e"] = CDTrainerV3
except ImportError:
    pass