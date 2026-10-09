from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .catalyst_cost import CatalystCostResult, catalyst_carbon_cost, summarize_catalyst_cost
from .scc_rf import build_scc_distribution, load_scc_draws, save_scc_draws


def run_catalyst_economics(
    *,
    scc_csv: str | Path,
    catalyst_csv: str | Path,
    out_dir: str | Path,
    scc_draws_path: str | Path | None = None,
    save_scc_draws_path: str | Path | None = None,
    n_draws: int = 20_000,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run RF SCC + catalyst carbon-cost reporting.

    This function is designed to be called after BAM validation/test metrics are
    computed. It does not touch the model, optimizer, gradients, or loss.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if scc_draws_path is not None:
        scc_draws = load_scc_draws(scc_draws_path)
        if verbose:
            print(f"[1] SCC draws loaded: {len(scc_draws):,} from {scc_draws_path}")
    else:
        scc_draws = build_scc_distribution(scc_csv, out_dir=out_dir, n_draws=n_draws, verbose=verbose)
        if save_scc_draws_path is not None:
            save_scc_draws(save_scc_draws_path, scc_draws)
            if verbose:
                print(f"[1b] SCC draws saved: {save_scc_draws_path}")

    result: CatalystCostResult = catalyst_carbon_cost(
        scc_draws,
        catalyst_csv=catalyst_csv,
        out_dir=out_dir,
    )
    economics_summary = summarize_catalyst_cost(result)
    summary = {
        "scc_csv": str(scc_csv),
        "catalyst_csv": str(catalyst_csv),
        "out_dir": str(out_dir),
        "scc_draws": {
            "n": int(len(scc_draws)),
            "source": str(scc_draws_path) if scc_draws_path is not None else "rf_synthetic_scenario_D",
        },
        "catalyst_economics": economics_summary,
        "outputs": {
            "excel": str(out_dir / "carbon_social_cost.xlsx"),
            "per_metal_csv": str(out_dir / "montecarlo_ci.csv"),
            "summary_json": str(out_dir / "catalyst_economics_summary.json"),
        },
    }
    write_catalyst_economics_summary(summary, out_dir / "catalyst_economics_summary.json")

    if verbose:
        print(
            "[3] Catalyst carbon social cost: "
            f"{economics_summary['n_candidates']:,} candidates | "
            f"{economics_summary['n_metals']:,} metals | "
            f"active26 {economics_summary['n_active26']:,}"
        )
        print(
            "    central $141/tCO2: "
            f"lowest {economics_summary['lowest_cost_metal']} "
            f"${economics_summary['lowest_cost_central_usd_per_kg']:.4g}/kg | "
            f"highest {economics_summary['highest_cost_metal']} "
            f"${economics_summary['highest_cost_central_usd_per_kg']:.4g}/kg"
        )
        print(
            "    Pt/C 5wt benchmark: "
            f"${economics_summary['pt_c_5wt_cost_central_usd_per_kg']:.4g}/kg"
        )
        print(f"[4] Catalyst economics outputs -> {out_dir}")

    return summary


def write_catalyst_economics_summary(summary: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_model_and_economics_report(
    *,
    model_metrics: dict[str, Any],
    economics_summary: dict[str, Any],
    output_json: str | Path,
) -> dict[str, Any]:
    """Merge BAM test metrics with catalyst economics into one JSON report."""

    report = {
        "model_metrics": model_metrics,
        "catalyst_economics": economics_summary.get("catalyst_economics", economics_summary),
        "economics_outputs": economics_summary.get("outputs", {}),
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
