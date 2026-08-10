from __future__ import annotations

import argparse
from pathlib import Path

from .report import run_catalyst_economics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run catalyst carbon social cost reporting with the RF SCC metamodel."
    )
    parser.add_argument("--scc-data", required=True, help="Path to socialcostcarbon.csv.")
    parser.add_argument("--catalyst-data", required=True, help="Path to catalysts_formationE.csv.")
    parser.add_argument("--out-dir", default="outputs/catalyst_economics", help="Output directory.")
    parser.add_argument("--scc-draws", help="Optional precomputed SCC draws in USD/tCO2 (.npy, .csv, or text).")
    parser.add_argument("--save-scc-draws", help="Optional path to save RF-generated SCC draws.")
    parser.add_argument("--n-draws", type=int, default=20_000, help="Number of synthetic SCC draws when training RF.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_catalyst_economics(
        scc_csv=Path(args.scc_data),
        catalyst_csv=Path(args.catalyst_data),
        out_dir=Path(args.out_dir),
        scc_draws_path=Path(args.scc_draws) if args.scc_draws else None,
        save_scc_draws_path=Path(args.save_scc_draws) if args.save_scc_draws else None,
        n_draws=args.n_draws,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
