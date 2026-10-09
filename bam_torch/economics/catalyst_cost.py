from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

# Metal constants: atomic mass, point GWP, 2.5% GWP, 97.5% GWP, d-block.
# GWP source: Nuss & Eckelman (2014), PLoS ONE 9(7): e101298, Table S38
# supply-mix TOTAL. Units are kgCO2e/kg metal.
METALS: dict[str, tuple[float, float, float, float, str]] = {
    "Ti": (47.867, 8.1, 7.6, 8.7, "3d"),
    "V": (50.942, 33.1, 19.4, 53.0, "3d"),
    "Cr": (51.996, 2.4, 2.2, 2.7, "3d"),
    "Mn": (54.938, 1.0, 0.8, 1.3, "3d"),
    "Fe": (55.845, 1.5, 1.4, 1.7, "3d"),
    "Co": (58.933, 8.3, 6.0, 11.5, "3d"),
    "Ni": (58.693, 6.5, 5.9, 7.3, "3d"),
    "Cu": (63.546, 2.8, 2.2, 3.5, "3d"),
    "Zr": (91.224, 1.1, 0.9, 1.3, "4d"),
    "Nb": (92.906, 12.5, 10.1, 15.4, "4d"),
    "Mo": (95.95, 5.7, 4.5, 7.3, "4d"),
    "Ru": (101.07, 2110, 1660, 2690, "4d"),
    "Rh": (102.906, 35100, 26700, 45500, "4d"),
    "Pd": (106.42, 3880, 3090, 4860, "4d"),
    "Ag": (107.868, 196, 164, 234, "4d"),
    "Hf": (178.49, 131, 69, 252, "5d"),
    "Ta": (180.948, 260, 206, 331, "5d"),
    "W": (183.84, 12.6, 9.6, 16.3, "5d"),
    "Re": (186.207, 450, 213, 836, "5d"),
    "Os": (190.23, 4560, 3700, 5650, "5d"),
    "Ir": (192.217, 8860, 7000, 11200, "5d"),
    "Pt": (195.084, 12500, 9650, 16200, "5d"),
    "Au": (196.967, 12500, 10100, 15400, "5d"),
}

M_C = 12.011
M_N = 14.007
DEFAULT_SCC_USD_PER_KGCO2 = {
    "low_$56": 0.056,
    "central_$141": 0.141,
    "high_$248": 0.248,
}
ACTIVE26 = {
    "Ti2@2Na",
    "Mn2@2Na",
    "Fe2@2Na",
    "Cu2@2Na",
    "Rh2@2Na",
    "Zr2@2Na",
    "Zr2@2Nb",
    "Zr2@2Nc",
    "Nb2@2Nc",
    "Zr2@2Nd",
    "Mn2@2Ne",
    "Mn2@2Nf",
    "Ti2@3Na",
    "Au2@3Na",
    "Fe2@3Na",
    "Pd2@3Nb",
    "Rh2@3Nc",
    "Rh2@3Nd",
    "Au2@3Nd",
    "V2@4Na",
    "Ti2@4Nb",
    "Pd2@4Nb",
    "Ti2@4Nc",
    "Cr2@4Nd",
    "Ni2@4Nd",
    "Cu2@4Nd",
}


@dataclass(frozen=True)
class CatalystCostResult:
    candidates: object
    per_metal: object
    pt_c_5wt_embodied_co2: float
    pt_c_5wt_cost_central_usd_per_kg: float


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError(
            "Catalyst economics tables require pandas. Install BAM-torch with "
            "the economics extra, or install pandas manually."
        ) from exc
    return pd


def catalyst_carbon_cost(
    scc_draws_usd_per_tco2,
    catalyst_csv: str | Path,
    *,
    metals: Mapping[str, tuple[float, float, float, float, str]] = METALS,
    scc_usd_per_kgco2: Mapping[str, float] = DEFAULT_SCC_USD_PER_KGCO2,
    active_candidates: set[str] = ACTIVE26,
    out_dir: str | Path | None = None,
    random_seed: int = 7,
) -> CatalystCostResult:
    """Compute catalyst carbon social cost from SCC draws.

    This is the BAM-torch version of ``reproduce_scc.py`` stage 2:
    ``SCC_D + catalysts_formationE.csv -> catalyst carbon cost``.
    """

    pd = _require_pandas()
    catalyst_csv = Path(catalyst_csv)
    if not catalyst_csv.exists():
        raise FileNotFoundError(f"Catalyst CSV not found: {catalyst_csv}")

    scc_draws = np.asarray(scc_draws_usd_per_tco2, dtype=float)
    n_draws = len(scc_draws)
    if n_draws == 0:
        raise ValueError("scc_draws_usd_per_tco2 is empty")
    scc_kg = scc_draws / 1000.0

    ef = pd.read_csv(catalyst_csv)
    required = {"metal", "N_number", "coordination"}
    missing_columns = required - set(ef.columns)
    if missing_columns:
        raise ValueError(f"Catalyst CSV missing required columns: {sorted(missing_columns)}")

    rng = np.random.RandomState(random_seed)
    gwp_draw: dict[str, np.ndarray] = {}
    for metal, (_amu, gwp, lo95, hi95, _block) in metals.items():
        sigma = (np.log(hi95) - np.log(lo95)) / (2 * 1.96)
        gwp_draw[metal] = np.exp(np.log(gwp) + sigma * rng.standard_normal(n_draws))

    rows = []
    for _, row in ef.iterrows():
        metal = row["metal"]
        if metal not in metals:
            raise KeyError(f"No built-in metal GWP entry for {metal!r}")
        n_number = int(row["N_number"])
        motif = row["coordination"]
        amu, gwp, _lo95, _hi95, block = metals[metal]
        molecular_weight = 2 * amu + (52 - n_number) * M_C + n_number * M_N
        metal_weight_fraction = 2 * amu / molecular_weight
        embodied = metal_weight_fraction * gwp
        candidate = f"{metal}2@{motif}"
        mc = metal_weight_fraction * gwp_draw[metal] * scc_kg

        rec = {
            "candidate": candidate,
            "metal": metal,
            "motif": motif,
            "block": block,
            "embodied_CO2": round(float(embodied), 3),
            "is_active26": candidate in active_candidates,
        }
        for name, scc in scc_usd_per_kgco2.items():
            rec[f"cost_{name}"] = round(float(embodied * scc), 4)
        rec["MC_lo95"] = round(float(np.percentile(mc, 2.5)), 4)
        rec["MC_hi95"] = round(float(np.percentile(mc, 97.5)), 4)
        rows.append(rec)

    candidates = pd.DataFrame(rows)
    per_metal = (
        candidates.groupby("metal")
        .agg(
            block=("block", "first"),
            embodied_CO2=("embodied_CO2", "mean"),
            cost_central=("cost_central_$141", "mean"),
            MC_lo95=("MC_lo95", "mean"),
            MC_hi95=("MC_hi95", "mean"),
        )
        .reset_index()
    )
    per_metal["cost_low_$56"] = (per_metal["embodied_CO2"] * 0.056).round(3)
    per_metal["cost_high_$248"] = (per_metal["embodied_CO2"] * 0.248).round(3)
    per_metal = per_metal.sort_values("cost_central").reset_index(drop=True)

    pt_c_5wt_embodied = metals["Pt"][1] * 0.05
    result = CatalystCostResult(
        candidates=candidates,
        per_metal=per_metal,
        pt_c_5wt_embodied_co2=float(pt_c_5wt_embodied),
        pt_c_5wt_cost_central_usd_per_kg=float(pt_c_5wt_embodied * 0.141),
    )

    if out_dir is not None:
        write_catalyst_cost_outputs(result, out_dir)
    return result


def write_catalyst_cost_outputs(result: CatalystCostResult, out_dir: str | Path) -> None:
    pd = _require_pandas()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out / "carbon_social_cost.xlsx", engine="openpyxl") as xw:
        result.per_metal.to_excel(xw, sheet_name="per_metal", index=False)
        result.candidates[result.candidates.is_active26].sort_values("cost_central_$141").to_excel(
            xw, sheet_name="26_candidates", index=False
        )
        result.candidates.sort_values("cost_central_$141").to_excel(xw, sheet_name="all_candidates", index=False)

    result.per_metal[
        [
            "metal",
            "block",
            "embodied_CO2",
            "cost_low_$56",
            "cost_central",
            "cost_high_$248",
            "MC_lo95",
            "MC_hi95",
        ]
    ].to_csv(out / "montecarlo_ci.csv", index=False, encoding="utf-8-sig")


def summarize_catalyst_cost(result: CatalystCostResult) -> dict[str, object]:
    candidates = result.candidates
    per_metal = result.per_metal
    lo = per_metal.iloc[0]
    hi = per_metal.iloc[-1]
    active = candidates[candidates["is_active26"]]
    return {
        "n_candidates": int(len(candidates)),
        "n_active26": int(len(active)),
        "n_metals": int(len(per_metal)),
        "lowest_cost_metal": str(lo["metal"]),
        "lowest_cost_central_usd_per_kg": float(lo["cost_central"]),
        "highest_cost_metal": str(hi["metal"]),
        "highest_cost_central_usd_per_kg": float(hi["cost_central"]),
        "highest_to_lowest_ratio": float(hi["cost_central"] / lo["cost_central"]),
        "pt_c_5wt_embodied_co2": result.pt_c_5wt_embodied_co2,
        "pt_c_5wt_cost_central_usd_per_kg": result.pt_c_5wt_cost_central_usd_per_kg,
        "active26_cost_central_usd_per_kg": {
            "min": float(active["cost_central_$141"].min()) if len(active) else None,
            "median": float(active["cost_central_$141"].median()) if len(active) else None,
            "max": float(active["cost_central_$141"].max()) if len(active) else None,
        },
    }
