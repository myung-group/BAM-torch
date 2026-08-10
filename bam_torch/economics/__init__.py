"""Catalyst economics evaluation utilities for BAM-torch.

The module is intentionally non-trainable: it adds carbon-cost reporting to
BAM evaluation outputs without changing the energy/force/stress loss.
"""

from .catalyst_cost import (
    ACTIVE26,
    METALS,
    CatalystCostResult,
    catalyst_carbon_cost,
    summarize_catalyst_cost,
)
from .report import run_catalyst_economics
from .scc_rf import SCCDistributionResult, build_scc_distribution, load_scc_draws

__all__ = [
    "ACTIVE26",
    "METALS",
    "CatalystCostResult",
    "SCCDistributionResult",
    "build_scc_distribution",
    "catalyst_carbon_cost",
    "load_scc_draws",
    "run_catalyst_economics",
    "summarize_catalyst_cost",
]
