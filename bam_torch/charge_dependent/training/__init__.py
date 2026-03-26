from .cd_trainer import CDTrainer
from .cd_trainer_v3 import CDTrainerV3

TRAINER_REGISTRY = {
    # Phase 2 (CENT2, U_CENT 에너지 포함)
    "cd": CDTrainer,
    "charge_dependent": CDTrainer,
    # Phase 3 (E) — CEP as pure charge predictor
    "cd_v3": CDTrainerV3,
    "charge_dependent_v3": CDTrainerV3,
    "cd_e": CDTrainerV3,
}
