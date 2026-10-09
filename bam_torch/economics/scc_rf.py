from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SCCDistributionResult:
    """RF-derived SCC distribution and diagnostics.

    ``draws_usd_per_tco2`` are scenario-D social-cost-of-carbon draws in
    USD/tCO2. They are intended to be multiplied by embodied kgCO2/kg material
    after dividing by 1000.
    """

    draws_usd_per_tco2: np.ndarray
    validation_factor_median: float
    top_feature_importances: list[tuple[str, float]]
    estimates: list[dict[str, float | int | str]]


def _require_rf_deps():
    try:
        import pandas as pd  # type: ignore
        from sklearn.ensemble import RandomForestRegressor  # type: ignore
        from sklearn.model_selection import KFold  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError(
            "Catalyst economics SCC RF requires optional dependencies: "
            "pandas and scikit-learn. Install BAM-torch with the economics "
            "extra, or install pandas scikit-learn manually."
        ) from exc
    return pd, RandomForestRegressor, KFold


def load_scc_draws(path: str | Path) -> np.ndarray:
    """Load SCC draws in USD/tCO2 from .npy or CSV/text."""

    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    return np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None)


def save_scc_draws(path: str | Path, draws: Sequence[float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npy":
        np.save(path, np.asarray(draws, dtype=float))
    else:
        np.savetxt(path, np.asarray(draws, dtype=float), delimiter="," if path.suffix == ".csv" else " ")


def build_scc_distribution(
    scc_csv: str | Path,
    *,
    out_dir: str | Path | None = None,
    n_draws: int = 20_000,
    rf_seed: int = 0,
    synth_seed: int = 42,
    n_estimators: int = 500,
    min_samples_leaf: int = 5,
    save_estimates: bool = True,
    verbose: bool = True,
    return_result: bool = False,
) -> np.ndarray | SCCDistributionResult:
    """Train the RF SCC metamodel and synthesize scenario-D SCC draws.

    This is the BAM-torch version of ``reproduce_scc.py`` stage 1:
    ``socialcostcarbon.csv -> RF -> SCC_D``. The returned/default draw array is
    in USD per metric ton CO2.
    """

    pd, RandomForestRegressor, KFold = _require_rf_deps()
    scc_csv = Path(scc_csv)
    if not scc_csv.exists():
        raise FileNotFoundError(f"SCC CSV not found: {scc_csv}")

    df = pd.read_csv(scc_csv)
    df["SCC_CO2"] = df["SCC"] / 3.667  # $/tC -> $/tCO2
    d = df[(df["SCC_CO2"] > 0) & df["PRTP"].notna()].copy()
    d["Risk"] = d["Risk"].replace([np.inf, -np.inf], np.nan)
    d["Risk_miss"] = d["Risk"].isna().astype(int)
    d["Risk"] = d["Risk"].fillna(d["Risk"].median())
    d["EIS"] = d["EIS"].fillna(d["EIS"].median())

    topfun = d["Function"].value_counts().head(4).index.tolist()
    for fn in topfun:
        d[f"fun_{fn}"] = (d["Function"] == fn).astype(int)

    feats = [
        "PRTP",
        "year",
        "EIS",
        "Risk",
        "Risk_miss",
        "Equity",
        "Uncertainty",
        "Hope",
        "Nordhaus",
        "Tol",
        "Ploeg",
        "Traeger",
        "Tax",
    ] + [f"fun_{fn}" for fn in topfun]

    x = d[feats].values
    y = np.log10(d["SCC_CO2"].values)
    w = (d["quality"] * d["censor"]).clip(lower=1e-6).values

    if verbose:
        print(f"[1] SCC training rows {len(d):,} | features {len(feats)} | target log10(SCC $/tCO2)")

    def mk(seed: int):
        return RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )

    factors = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(x):
        model = mk(rf_seed).fit(x[tr], y[tr], sample_weight=w[tr])
        factors.append(10 ** np.abs(model.predict(x[te]) - y[te]))
    factors = np.concatenate(factors)

    rf = mk(rf_seed).fit(x, y, sample_weight=w)
    imp = sorted(zip(feats, rf.feature_importances_), key=lambda t: -t[1])[:10]
    if verbose:
        print(
            f"    5-fold median factor error {np.median(factors):.2f}x | top features "
            + " ".join(f"{k}={v:.2f}" for k, v in imp[:3])
        )

    rng = np.random.RandomState(synth_seed)
    prtp_i = feats.index("PRTP")
    fun_idx = {fn: feats.index(f"fun_{fn}") for fn in topfun}
    growth = "growth" if "growth" in topfun else None
    recent = np.where(d["year"].values >= 2016)[0]

    def drupp_delta(n: int):
        # Drupp et al. (2018) expert pure rate of time preference, truncated 0-8%.
        s = rng.gamma(0.62, 1.10 / 0.62, size=int(n * 1.4))
        out = s[s <= 8.0]
        while len(out) < n:
            extra = rng.gamma(0.62, 1.10 / 0.62, size=int(n * 0.2) + 1)
            out = np.concatenate([out, extra[extra <= 8.0]])
        return out[:n]

    def synth(pool, *, drupp: bool = False, persistent: bool = False):
        ww = w[pool] / w[pool].sum()
        xq = x[rng.choice(pool, size=n_draws, replace=True, p=ww)].copy()
        if drupp:
            xq[:, prtp_i] = drupp_delta(n_draws)
        if persistent and growth:
            for fn, j in fun_idx.items():
                xq[:, j] = 1.0 if fn == growth else 0.0
        return 10 ** rf.predict(xq)

    all_idx = np.arange(len(d))
    ladder = [
        ("(A) all literature", synth(all_idx)),
        ("(B) recent 2016+", synth(recent)),
        ("(C) +Drupp discounting", synth(recent, drupp=True)),
        ("(D) +persistent damages = headline", synth(recent, drupp=True, persistent=True)),
    ]

    estimates: list[dict[str, float | int | str]] = []
    for label, arr in ladder:
        p = np.percentile(arr, [5, 50, 95])
        estimates.append(
            {
                "query": label,
                "mean": round(float(arr.mean())),
                "p5": round(float(p[0])),
                "median": round(float(p[1])),
                "p95": round(float(p[2])),
            }
        )

    draws = ladder[-1][1]
    if out_dir is not None and save_estimates:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(estimates).to_csv(out / "our_scc_estimates.csv", index=False, encoding="utf-8-sig")

    if verbose:
        print(
            f"[2] Synthetic SCC scenario D: mean ${draws.mean():.0f} | "
            f"5-95% [${np.percentile(draws, 5):.0f} ~ ${np.percentile(draws, 95):.0f}]"
        )
        print("    paper adopted values: central $141/tCO2, 5-95% $56-$248")

    result = SCCDistributionResult(
        draws_usd_per_tco2=draws,
        validation_factor_median=float(np.median(factors)),
        top_feature_importances=[(str(k), float(v)) for k, v in imp],
        estimates=estimates,
    )
    return result if return_result else draws
