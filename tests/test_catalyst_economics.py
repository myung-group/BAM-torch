from __future__ import annotations

from pathlib import Path

import numpy as np

from bam_torch.economics import catalyst_carbon_cost, summarize_catalyst_cost


def test_catalyst_carbon_cost_smoke(tmp_path: Path):
    catalyst_csv = tmp_path / "catalysts_formationE.csv"
    catalyst_csv.write_text(
        "metal,N_number,coordination\n"
        "Fe,2,2Na\n"
        "Pt,4,3Na\n",
        encoding="utf-8",
    )
    draws = np.linspace(56.0, 248.0, 128)

    result = catalyst_carbon_cost(draws, catalyst_csv)
    summary = summarize_catalyst_cost(result)

    assert len(result.candidates) == 2
    assert len(result.per_metal) == 2
    assert summary["n_candidates"] == 2
    assert summary["n_metals"] == 2
    assert summary["pt_c_5wt_cost_central_usd_per_kg"] > 0
    assert set(result.candidates["metal"]) == {"Fe", "Pt"}
    assert (result.candidates["cost_central_$141"] > 0).all()
