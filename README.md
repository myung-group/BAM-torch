## Summary

This branch adds catalyst carbon-social-cost reporting to BAM-torch evaluation.

The implementation is based on the catalyst sustainability workflow from the accompanying SCC manuscript. It keeps BAM's physical prediction task unchanged, while adding an optional economics reporting path for TM2@Nx-Gr catalyst screening.

## Workflow

```text
BAM prediction / evaluation
        |
        v
Energy / Force / Stress prediction
        |
        v
Energy MAE + Force MAE + Stress MAE
        |
        +-------------------------------+
                                        |
                                        v
                          Optional catalyst economics
                                        |
                                        v
socialcostcarbon.csv ──> RF SCC metamodel ──> SCC_D distribution
                                        |
catalysts_formationE.csv ──────────────+
                                        |
                                        v
                       catalyst embodied CO2 + carbon social cost
                                        |
                                        v
                         combined BAM + economics report
```

## What was added

```text
bam_torch/economics/
  ├── scc_rf.py          # RF SCC metamodel: socialcostcarbon.csv -> SCC_D
  ├── catalyst_cost.py   # Catalyst carbon cost: SCC_D + catalysts_formationE.csv
  ├── report.py          # Economics summary + combined report utilities
  ├── cli.py             # Standalone catalyst economics CLI
  └── init.py
```

Also updated:

```text
bam_torch/predicting/evaluator.py
  └── optionally appends catalyst economics reporting after BAM evaluation
pyproject.toml
  └── adds optional economics dependencies
tests/test_catalyst_economics.py
  └── smoke test for catalyst carbon-cost calculation
```

## Evaluation behavior

The economics model is not used as a training loss.

```text
Training/evaluation loss:
  energy
  forces
  stress
Post-evaluation report:
  embodied CO2
  carbon social cost
  SCC uncertainty interval
  Pt/C 5 wt% benchmark
```

This means the physical BAM model remains focused on energy/force/stress prediction, while catalyst economics is attached as a non-trainable reporting layer.

## Example config

```json
{
  "economics": {
    "enabled": true,
    "type": "catalyst",
    "scc_data": "/path/to/socialcostcarbon.csv",
    "catalyst_data": "/path/to/catalysts_formationE.csv",
    "out_dir": "outputs/catalyst_economics",
    "combined_report": "outputs/bam_catalyst_economics_report.json"
  }
}
```

If precomputed SCC draws are available:

```json
{
  "economics": {
    "enabled": true,
    "type": "catalyst",
    "scc_data": "/path/to/socialcostcarbon.csv",
    "catalyst_data": "/path/to/catalysts_formationE.csv",
    "scc_draws": "/path/to/scc_draws.npy",
    "out_dir": "outputs/catalyst_economics",
    "combined_report": "outputs/bam_catalyst_economics_report.json"
  }
}
```

## Expected output

```text
MEAN_LOSS(E): ...
MEAN_LOSS(F): ...
MAE(E):       ...
MAE(F):       ...
MAE(S):       ...
[3] Catalyst carbon social cost: 460 candidates | 23 metals | active26 26
    central $141/tCO2: lowest Mn $0.02094/kg | highest Rh $1218/kg
    Pt/C 5wt benchmark: $88.12/kg
[ECONOMICS] Combined BAM + catalyst report -> outputs/bam_catalyst_economics_report.json
```

## Validation

```text
python3 -m pytest tests/test_catalyst_economics.py -q
# 1 passed
```

Standalone CLI smoke test:

```text
[1] SCC draws loaded: 20,000
[2] Catalyst carbon social cost: 460 candidates | 23 metals | active26 26
    central $141/tCO2: lowest Mn $0.02094/kg | highest Rh $1218/kg
    Pt/C 5wt benchmark: $88.12/kg
```
