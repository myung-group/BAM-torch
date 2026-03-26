from .cd_model import ChargeRACE
from .cd_model_v3 import ChargeRACEv3

MODEL_REGISTRY = {
    # Phase 2 (CENT2 기반 CEP, U_CENT 에너지 포함)
    "charge_race": ChargeRACE,
    "cd_race": ChargeRACE,
    # Phase 3 (E) — CEP as pure charge predictor, E_total = E_SR only
    "charge_race_v3": ChargeRACEv3,
    "cd_race_v3": ChargeRACEv3,
    "charge_race_e": ChargeRACEv3,
}

__all__ = ["ChargeRACE", "ChargeRACEv3", "MODEL_REGISTRY"]
