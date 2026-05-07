from .base_trainer import BaseTrainer
from .mp_trainer import MPTrainer, MPTrainer_V2
from .mve_trainer import MVETrainer
from .multihead_trainer import MultiheadTrainer
from bam_torch.group_averaging.training.ga_trainer import GATrainer
from bam_torch.group_averaging.training.ga_mp_trainer import GAMPTrainer
from bam_torch.group_averaging.training.simple_gnn_trainer import SimpleGNNTrainer
from bam_torch.group_averaging.training.df_trainer import DFTrainer
from bam_torch.group_averaging.training.dplr_trainer import DPLRTrainer


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
    "ga_mp": GAMPTrainer,
    "sgnn": SimpleGNNTrainer,
    "simple_gnn": SimpleGNNTrainer,
    "group_averaging": GATrainer,
    "frame_averaging": GATrainer,
    "probabilistic_symmetrization": GATrainer,
    "df": DFTrainer,
    "dplr_paper": DPLRTrainer,
}