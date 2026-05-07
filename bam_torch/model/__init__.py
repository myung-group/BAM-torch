from .models import RACE, MACE, RACEUnified
from .race_nnx import RACE_V2
from .race_nnx_7 import RACE_V2_7

MODEL_REGISTRY = {
    "race": RACE,
    "mace": MACE,
    "race_multihead": RACEUnified,
    "race_unified": RACEUnified,
    "race_v2": RACE_V2,
    "race_v2_7": RACE_V2_7
}

__all__ = ["RACE", "MACE", "RACEUnified", "MODEL_REGISTRY", "RACE_V2"]

