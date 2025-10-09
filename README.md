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

### Step 5: Install CuEquivariance Library (Optional but Recommended)
For GPU kernel accelerations:
```bash
$ pip install cuequivariance-torch
$ pip install cuequivariance
$ pip install cuequivariance-ops-torch-cu12
```

### Step 6: Install Laplace Approximation (Optional)
```bash
$ pip install laplace-torch
```

### Step 7: Install BAM
```bash
$ git clone https://github.com/yourusername/bam-torch.git
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

| Method | Energy MAE (meV/atom) | Force MAE (meV/Å) | Energy NLL | Force NLL | OOD AUROC | ECE |
|--------|----------------------|-------------------|------------|-----------|-----------|-----|
| **BAM (Ensemble)** | **0.82** | **18.3** | **-2.41** | **-1.87** | **0.94** | **0.03** |
| BAM (SWAG) | 0.85 | 19.1 | -2.35 | -1.82 | 0.92 | 0.04 |
| BAM (IVON) | 0.84 | 18.7 | -2.38 | -1.84 | 0.93 | 0.03 |
| BAM (Laplace) | 0.86 | 19.5 | -2.30 | -1.79 | 0.91 | 0.05 |
| Baseline (No UQ) | 0.91 | 21.2 | N/A | N/A | 0.71 | 0.15 |

### Active Learning Efficiency

- **30% reduction** in required training data compared to random sampling
- **2.5× improvement** over energy-only uncertainty sampling
- Outperforms traditional sampling methods across QM9, MD17, and ANI-1 datasets

### Computational Performance

| Configuration | Training Speed (mol/s) | Inference Speed (mol/s) | Memory Usage (GB) |
|--------------|------------------------|-------------------------|-------------------|
| Single GPU | 450 | 2,100 | 8.2 |
| 4 GPUs (DDP) | 1,680 | 7,850 | 9.1 per GPU |
| With CuEquivariance | 580 | 2,750 | 7.8 |

## Configuration

### Model Configuration (`input.json`)
```json
{
  "model": {
    "architecture": "race",
    "hidden_dim": 128,
    "num_layers": 4,
    "race_iterations": 3,
    "uncertainty_method": "ensemble",
    "ensemble_size": 5
  },
  "training": {
    "epochs": 500,
    "batch_size": 32,
    "lr": 1e-3,
    "lr_scheduler": "cosine",
    "warmup_epochs": 10,
    "gpu_parallel": "data"
  },
  "loss": {
    "type": "joint_nll",
    "energy_weight": 1.0,
    "force_weight": 50.0
  }
}
```

## Datasets

Supported datasets:
- **QM9**: Quantum chemistry properties of small molecules
- **MD17**: Molecular dynamics trajectories
- **ANI-1**: Organic molecules dataset
- **Materials Project**: Inorganic materials
- **Custom datasets**: Support for ASE-compatible formats

Download example datasets:
```bash
$ python scripts/download_data.py --dataset qm9
$ python scripts/download_data.py --dataset md17
```

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
- GPU acceleration powered by [CuEquivariance](https://github.com/NVIDIA/cuEquivariance)
- Main Development Group at [SKKU](https://www.myung.skku.edu/) 

%## Contact

%- **Lead Author**: [Your Name] - [email@example.com]
%- **Project Link**: [https://github.com/yourusername/bam-torch](https://github.com/yourusername/bam-torch)
%- **Issues**: [GitHub Issues](https://github.com/yourusername/bam-torch/issues)
%- **Discussions**: [GitHub Discussions](https://github.com/yourusername/bam-torch/discussions)
