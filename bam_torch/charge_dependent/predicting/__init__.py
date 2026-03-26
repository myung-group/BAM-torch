from .cd_evaluator import CDEvaluator
from .cd_evaluator_v3 import CDEvaluatorV3


EVALUATOR_REGISTRY = {
    "cd": CDEvaluator,
    "charge_dependent": CDEvaluator,
    "cd_v3": CDEvaluatorV3,
    "charge_dependent_v3": CDEvaluatorV3,
    "cd_e": CDEvaluatorV3,
}
