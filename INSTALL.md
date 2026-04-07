# BAM-torch Charge-Dependent Subpackage — Installation Guide

## Overview

Charge-dependent RACE models (Phase 2/3) with CENT2-based CEP
(Charge Equilibration Process) for hard charge conservation.

This is a **subpackage** of BAM-torch — not an independent package.
It inherits from `BaseTrainer`, `RACECalculator`, and registers into
`MODEL_REGISTRY` / `TRAINER_REGISTRY` via lazy imports.

## File Manifest

### New files (copy into BAM-torch root)

```
bam_torch/charge_dependent/          # Core subpackage
  __init__.py
  calculator/
    __init__.py
    cd_calculator.py                  # CDRACECalculator (ASE Calculator)
  model/
    __init__.py
    cd_model.py                       # ChargeRACE (Phase 2: E_SR + U_CENT)
    cd_model_v3.py                    # ChargeRACEv3 (Phase 3: E_SR only)
    cep_block.py                      # CEPBlock (shared, CENT2 Lagrange)
  training/
    __init__.py
    cd_trainer.py                     # CDTrainer (Phase 2)
    cd_trainer_v3.py                  # CDTrainerV3 (Phase 3)
  predicting/
    __init__.py
    cd_evaluator.py                   # CDEvaluator (Phase 2)
    cd_evaluator_v3.py                # CDEvaluatorV3 (Phase 3)
  utils/
    __init__.py
    cd_utils.py                       # Charge-aware dataloader
    qm9star_preprocessor.py           # SQL → extended XYZ converter

bam_torch/utils/
  model_config.py                     # Centralized config parser (NEW)
  zbl.py                              # ZBL repulsive prior (NEW)

examples/example-QM9star-charge/
  input.json                          # Phase 2 config example
  input_v3.json                       # Phase 3 config example
  input_v4.json                       # Phase 4+ optimized config

tests/
  test_cd_v3.py                       # CPU unit test (Phase 3)
```

### Modified files (replace existing)

```
bam_torch/model/__init__.py          # Added lazy charge_dependent registration
bam_torch/training/__init__.py       # Added lazy CD trainer registration
                                     # + GATrainer OSError guard
pyproject.toml                        # Added charge_dependent = [] to
                                     # [project.optional-dependencies]
```

### .gitignore additions (append to existing)

```
# Charge-dependent development artifacts
qm9star/
_archive/
_upload/
Wiggle150/
*.zip
bam_torch/legacy_phases/
tests/*_output/
```

## Installation Steps

```bash
# 1. From this _upload/ directory, copy everything into BAM-torch root:
cp -r bam_torch/ examples/ tests/ pyproject.toml /path/to/BAM-torch/

# 2. Append .gitignore additions (see above)

# 3. Verify
cd /path/to/BAM-torch
python -c "from bam_torch.model import MODEL_REGISTRY; print(MODEL_REGISTRY.keys())"
# Should include: charge_race, cd_race, charge_race_v3, cd_race_v3, charge_race_e

python tests/test_cd_v3.py
# Should print: ALL TESTS PASSED
```

## Registry Keys

| Key                 | Model/Trainer  | Phase |
|---------------------|----------------|-------|
| `charge_race`       | ChargeRACE     | P2    |
| `cd_race`           | ChargeRACE     | P2    |
| `charge_race_v3`    | ChargeRACEv3   | P3    |
| `cd_race_v3`        | ChargeRACEv3   | P3    |
| `charge_race_e`     | ChargeRACEv3   | P3    |
| `cd` / `charge_dependent`       | CDTrainer    | P2 |
| `cd_v3` / `charge_dependent_v3` | CDTrainerV3  | P3 |
| `cd_e`                           | CDTrainerV3  | P3 |

## Architecture

```
BaseTrainer ──> CDTrainer ──> CDTrainerV3
                    │              │
                CDEvaluator   CDEvaluatorV3

RACECalculator ──> CDRACECalculator (+ ZBL prior)

RACE model ──> ChargeRACE (P2: E = E_SR + U_CENT)
           ──> ChargeRACEv3 (P3: E = E_SR, CEP = charge predictor only)
                    └── CEPBlock (shared, CENT2 Lagrange hard conservation)
```

## Quick Start (Phase 3 training)

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m bam_torch.training.run_train --config examples/example-QM9star-charge/input_v3.json
```
