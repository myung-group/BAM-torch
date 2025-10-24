# BAM (Bayesian Atoms Modeling)

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-arXiv-green)](https://arxiv.org/abs/2510.03046)

## Overview

<p align="center">
  <img src="./bam.png" width="25%" title="BAM" alt="BAM"/>
</p>

BAM (Bayesian Atoms Modeling) is an implementation of **Bayesian E(3) Equivariant Machine Learning Potentials (MLPs)** with iterative restratification of many-body message passing (RACE). Our framework enables uncertainty-aware atomistic simulations at scale by combining ab initio-level accuracy with robust uncertainty quantification.

## Key Features

- 🎯 **Joint Energy-Force NLL Loss**: Novel loss function explicitly modeling uncertainty in both energies and interatomic forces, yielding superior accuracy compared to conventional NLL losses
- 🔬 **Uncertainty Quantification**: Comprehensive uncertainty estimation for active learning, calibration, and out-of-distribution detection
- ⚡ **E(3) Equivariance**: Maintains rotational and translational symmetry for physically consistent predictions
- 🏗️ **RACE Architecture**: Iterative Restratification of Atoms with Combined Encoding for improved message passing
- 🎲 **Multiple Bayesian Approaches**: Implementation of various uncertainty quantification methods:
  - Deep Ensembles with Mean-Variance Estimation (MVE)
  - Stochastic Weight Averaging Gaussian (SWAG)
  - Improved Variational Online Newton (IVON)
  - Laplace Approximation
- 📊 **Active Learning**: Bayesian Active Learning by Disagreement (BALD) for efficient data acquisition
- 🚀 **Scalable**: Designed for large-scale atomistic simulations with computational efficiency
- ⚙️ **GPU Acceleration**: Support for both single-GPU and multi-GPU training with DistributedDataParallel

## Installation

### Prerequisites
- Python 3.11
- CUDA 12.4+ (for GPU support)
- PyTorch 2.5.1+

### Step 1: Create Conda Environment
```bash
$ conda create --name bam_torch python=3.11
$ conda activate bam_torch
```

### Step 2: Install Core Dependencies
```bash
$ pip install numpy scipy matscipy torch
```

### Step 3: Check PyTorch and CUDA Version
```bash
$ python -c "import torch; print(torch.__version__)"  
>>> 2.5.1+cu124
```

### Step 4: Install PyTorch Geometric
```bash
$ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

For example:
```bash
$ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

Then:
```bash
$ pip install pytorch_warmup torch_geometric
```

### Step 5: Install Laplace Approximation (Optional)
```bash
$ pip install laplace-torch
```

### Step 6: Install BAM
```bash
$ git clone https://github.com/myung-group/BAM-torch
$ cd bam-torch
$ pip install -e .
```

## Quick Start

### Basic Training Example
```python
from bam import BayesianE3Model
from bam.losses import JointEnergyForceNLL
from bam.data import load_dataset

# Load configuration
config = {
    "model": {
        "hidden_dim": 128,
        "num_layers": 4,
        "uncertainty_method": "ensemble",
        "race_iterations": 3
    },
    "training": {
        "lr": 1e-3,
        "epochs": 500,
        "batch_size": 32
    }
}

# Initialize model
model = BayesianE3Model(**config["model"])

# Load data
train_loader, val_loader = load_dataset("qm9", batch_size=32)

# Training
criterion = JointEnergyForceNLL()
optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

for epoch in range(config["training"]["epochs"]):
    for batch in train_loader:
        energy_pred, force_pred, energy_unc, force_unc = model(batch)
        loss = criterion(energy_pred, force_pred, batch.energy, batch.force, 
                        energy_unc, force_unc)
        loss.backward()
        optimizer.step()
```

## Running Examples

There are examples in `examples/example-*/`

### Single-GPU Training
Using environment variable:
```bash
$ CUDA_VISIBLE_DEVICES=0 python main.py
```

Or set `"gpu-parallel": false` in `input.json`, then:
```bash
$ python main.py
```

### Multi-GPU Training (DistributedDataParallel)
Using environment variables:
```bash
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
```

Or set `"gpu-parallel": "data"` (or `true`) in `input.json`, then:
```bash
$ python main.py
```
This automatically detects all available GPUs and uses them for computation.

### Evaluation
```bash
$ python evaluate.py --checkpoint path/to/checkpoint.pth --dataset test_data
```

### Active Learning Pipeline
```bash
$ python active_learning.py \
    --config configs/active_learning.yaml \
    --acquisition bald \
    --budget 1000 \
    --iterations 10
```

## Project Structure

```
bam-torch/
├── bam/
│   ├── models/           # Model architectures (RACE, E3-equivariant layers)
│   ├── losses/           # Loss functions (Joint Energy-Force NLL, MVE)
│   ├── uncertainty/      # Uncertainty quantification methods
│   ├── active_learning/  # BALD and other acquisition strategies
│   ├── data/            # Dataset loaders and preprocessing
│   ├── ood/             # Out-of-distribution detection
│   └── utils/           # Utility functions
├── examples/
│   ├── example-qm9/     # QM9 benchmark example
│   ├── example-md17/    # MD17 benchmark example
│   └── example-ani/     # ANI-1 benchmark example
├── configs/             # Configuration files
├── scripts/             # Training and evaluation scripts
└── tests/               # Unit tests
```

## Benchmarks & Results

### Uncertainty Quantification Performance
#### Table 4: RMSE on the 3BPA test dataset. We present the errors in energy ($E$, meV) and force ($F$, meV/Å) for models trained on ID (T=300 K) configurations and tested on both ID (T=300 K) and OOD (T=600 K, 1200 K) configurations of the flexible drug-like molecule 3BPA.
| Method        | $E$ (300 K) | $F$ (300 K) | $E$ (600 K) | $F$ (600 K) | $E$ (1200 K) | $F$ (1200 K) |
|---------------|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|:------------:|
| RACE          | 3.4         | 12.1        | 11.7        | 31.8        | 37.5         | 115.3        |
| RACE-Ensemble | 2.9         | 10.2        | 11.1        | 30.4        | 37.7         | 119.2        |
| RACE-DE-E     | 17.5        | 52.7        | 43.7        | 98.0        | 171.9        | 232.8        |
| RACE-DE-JEF   | 5.0         | 14.8        | 14.6        | 37.0        | 51.1         | 120.8        |
| RACE-LA       | 4.8         | 18.2        | 15.3        | 51.0        | 60.8         | 171.8        |

#### Table 5: Evaluation results on the oBN25 test dataset using different UQ methods. Values are reported for RMSE and CE of energy and force, and AUROC.
| Method    | Energy RMSE<br>(ID/OOD) | Force RMSE<br>(ID/OOD)  | Energy CE<br>(ID/OOD) | Force CE<br>(ID/OOD) | AUROC | 
|-----------|:-----------------------:|:-----------------------:|:---------------------:|:--------------------:|:-----:|
| RACE-MVE  | 0.20/7.39               | 0.62/0.53               | 0.01/0.33             | 0.06×10⁻³/1.25×10⁻²  | 0.54  |
| RACE-DE   | 0.14/6.94               | 0.53/0.37               | 0.03/0.21             | 8.49×10⁻³/5.17×10⁻²  | 1.00  |
| RACE-SWAG | 0.23/8.18               | 0.62/0.50               | 0.02/0.33             | 0.03×10⁻³/0.74×10⁻²  | 0.58  |
| RACE_IVON | 0.97/10.79              | 0.62/0.52               | 0.02/0.33             | 2.14×10⁻³/0.36×10⁻²  | 1.00  |

### Active Learning Efficiency
- **50% reduction** in required training data compared to random sampling
#### Table 6: Active learning results with increasing numbers of selected data (random vs. high BALD score) on the 3BPA test dataset. RMSE for energy ($E$, meV) and force ($F$, meV/Å) on 600 K and 1200 K test sets.
| Temp (Quantity)  | Baseline | +10 R | +10 B    | +20 R | +20 B    | +50 R | +50 B    | +100 R | +100 B   | +200 R   | +200 B   |
|:----------------:|:--------:|:-----:|:--------:|:-----:|:--------:|:-----:|:--------:|:------:|:--------:|:--------:|:--------:|
| **600 K** ($E$)  | 14.6     | 13.9  | **12.2** | 13.5  | **11.5** | 12.2  | **11.1** | 11.0   | **10.1** | **8.2**  | 8.8      |
| **600 K** ($F$)  | 37.0     | 33.3  | **30.1** | 31.0  | **28.8** | 27.5  | **25.8** | 23.8   | **23.3** | **20.0** | 20.0     |
| **1200 K** ($E$) | 51.1     | 40.8  | **36.3** | 36.2  | **31.9** | 30.5  | **26.0** | 25.3   | **21.9** | 20.7     | **18.8** |
| **1200 K** ($F$) | 120.8    | 102.4 | **85.2** | 87.4  | **78.3** | 70.7  | **60.4** | 55.6   | **49.7** | 45.6     | **41.3** |

R: Random samples B: High BALD score samples.

- The Balanced active learning strategy proved to be the most effective
#### Table 7: Active learning results with increasing numbers of selected data (random vs. high BALD score) on the 3BPA test dataset. RMSE for energy ($E$, meV) and force ($F$, meV/Å) on 600 K and 1200 K test sets.
| Temp (Quantity)  | Baseline | +10 BALD-E | +10 BALD-F | +10 BALD-EF | +10 Random |
|:----------------:|:--------:|:----------:|:----------:|:-----------:|:----------:|
| **600 K** ($E$)  | 14.6     | **12.2**   | 12.5       | 12.2        | 13.9       |
| **600 K** ($F$)  | 37.0     | 32.6       | 30.7       | **30.1**    | 33.3       |
| **1200 K** ($E$) | 51.1     | 40.2       | 36.9       | **36.3**    | 40.8       |
| **1200 K** ($F$) | 120.8    | 103.4      | 87.7       | **85.2**    | 102.4      |

## Configuration

### Model Configuration (`input.json`)
```json
{
    "device" : "gpu",
    "gpu-parallel" : false,
    "model" : "race",
    "cueq_config" : false,
    "regress_forces" : true,
    "trainer" : "base",
    "fname_traj" : "train_300K.xyz",
    "ntrain" : 450,
    "nvalid" : 50,
    "element" : "auto",
    "cutoff" : 6.0,
    "avg_num_neighbors": 26,
    "num_species" : 4,
    "max_ell" : 2,
    "num_radial_basis" : 8,
    "hidden_channels" : "64x0e+64x1o+64x2e",
    "output_channels" : "1x0e",
    "nbatch" : 2,
    "nlayers" : 3,
    "features_dim" : 128,
    "active_fn": "identity",
    "pbc" : true,
    "NN" : {
            "data_seed" :  10,
            "init_seed" :  11,
            "learning_rate" : 0.01,
            "weight_decay" : 0,
            "nepoch" : 50000,
            "nsave": 5,
            "restart" : false,
            "fname_pkl" : "model.pkl",
            "loss_config" : {"energy_loss": "mse", "force_loss": "mse"},
            "frc_lambda" : 100,
            "enr_lambda" : 1,
            "energy_grad_loss" : true,
            "energy_grad_mult" : 10,
            "l2_lambda" : 0.0,
            "cosine_sim" : false
    },
    "scheduler": {
            "scheduler" : "ReduceLROnPlateau",
            "lr_gamma" : 0.1,
            "decay_factor" : 0.9,
            "max_steps" : 30,
            "warmup_steps" : 10,
            "warmup_factor" : 0.2,
            "patience" : 50,
            "threshold" : 0.0001
    },
    "log_length": "simple",
    "log_interval": 2,
    "log_config": {
            "step":["date", "epoch"],
            "train": ["loss", "loss_e", "loss_f"],
            "valid": ["loss", "loss_e", "loss_f"],
            "lr": ["lr"]
    },
    "train" : {
            "fname_log" : "loss_train.out"
    },
    "predict" : {
            "evaluate_tag" : false,
            "fname_traj" : "test_1200K.xyz",
            "ndata" : "test_1200K.xyz",
            "model" : "model.pkl",
            "fname_plog" : "predict.out"
    }
}
```

## Datasets

Supported datasets:
- **QM9**: Small organic molecules and their quantum properties
- **rMD17**: Molecular dynamics trajectories of organic molecules
- **3BPA**: Flexible drug-like molecule at 300–1200 K
- **Materials Project**: Inorganic materials from high-throughput DFT
- **Custom datasets(oBN25)**: Solid/liquid boron nitride dataset for uncertainty benchmarking


## Citation

If you use BAM in your research, please cite our paper:

```bibtex
@article{bam2025,
  title={Bayesian E(3)-Equivariant Interatomic Potential with Iterative Restratification of Many-body Message Passing},
  author={Soohaeng Yoo Willow, Tae Hyeon Park, Gi Beom Sim, Sung Wook Moon, Seung Kyu Min, D. ChangMo Yang, Hyun Woo Kim, Juho Lee, Chang Woo Myung},
  journal={arXiv:2510.03046},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built upon [e3nn](https://github.com/e3nn/e3nn) for E(3) equivariant operations
- Main Development Group at [SKKU](https://www.myung.skku.edu/) 

%## Contact

%- **Lead Author**: [Your Name] - [email@example.com]
%- **Project Link**: [https://github.com/yourusername/bam-torch](https://github.com/yourusername/bam-torch)
%- **Issues**: [GitHub Issues](https://github.com/yourusername/bam-torch/issues)
%- **Discussions**: [GitHub Discussions](https://github.com/yourusername/bam-torch/discussions)
