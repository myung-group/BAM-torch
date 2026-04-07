from .models import RACE, MACE, RACEUnified


MODEL_REGISTRY = {
    "race": RACE,
    "mace": MACE,
    "race_multihead": RACEUnified,
    "race_unified": RACEUnified,
}


def _register_charge_dependent():
    """Lazy import to avoid circular dependency
    (cd_model → model.blocks → model.__init__ → cd_model)."""
    if "charge_race" not in MODEL_REGISTRY:
        from bam_torch.charge_dependent.model.cd_model import ChargeRACE
        MODEL_REGISTRY["charge_race"] = ChargeRACE
        MODEL_REGISTRY["cd_race"] = ChargeRACE
    if "charge_race_v3" not in MODEL_REGISTRY:
        from bam_torch.charge_dependent.model.cd_model_v3 import ChargeRACEv3
        MODEL_REGISTRY["charge_race_v3"] = ChargeRACEv3
        MODEL_REGISTRY["cd_race_v3"] = ChargeRACEv3
        MODEL_REGISTRY["charge_race_e"] = ChargeRACEv3


# Register on first access via try/except so it works even if
# charge_dependent subpackage is not installed.
try:
    _register_charge_dependent()
except ImportError:
    pass

__all__ = ["RACE", "MACE", "RACEUnified", "MODEL_REGISTRY"]

