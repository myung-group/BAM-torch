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

Equivalent environments have also been tested with [virtualenv](https://virtualenv.pypa.io/) and [venv](https://docs.python.org/library/venv.html).

### Step 2: Install Core Dependencies

As of October 2025, [PyTorch Geometric](https://pyg.org/) (PyG) supports PyTorch up to version 2.8.0.  
Check their [wheels page](https://data.pyg.org/whl/) for any updates (near the bottom).  
We should therefore manually install a version of PyTorch between 2.5.1 and 2.8.0.  
Running `install_deps.py` will automatically detect the version of CUDA linked to PyTorch, and then install the version of PyG compatible with that combination of CUDA and PyTorch.

```bash
$ git clone https://github.com/myung-group/BAM-torch
$ cd BAM-torch
$ pip install "torch<=2.8"
$ python install_deps.py
```

### Step 3: Install BAM

```bash
$ pip install -e .
```

### Step 4: Install optional components as needed

For low-level acceleration of equivariant neural networks, use [cuEquivariance](https://github.com/NVIDIA/cuEquivariance).

```bash
$ pip install -e ".[cueq]"
```

To enable Laplace approximations for neural networks, use [laplace-torch](https://aleximmer.com/Laplace/).

```bash
$ pip install -e ".[laplace]"
```

## Quick Start

### Basic Training Example

```python
import os
import json

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from bam_torch.training.base_trainer import BaseTrainer
from bam_torch.utils.utils import find_input_json, date

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def run(rank, world_size, json_data):
    setup(rank, world_size)
    base_trainer = BaseTrainer(json_data, rank, world_size)
    base_trainer.train()

if __name__ == '__main__':
    input_json_path = find_input_json()
    with open(input_json_path) as f:
        json_data = json.load(f)

    if not json_data['gpu-parallel'] or json_data['device'] == 'cpu':
        rank = 0
        world_size = 1
        run(rank, world_size, json_data)
    else:
        world_size = torch.cuda.device_count()
        mp.spawn(run, args=(world_size, json_data), nprocs=world_size, join=True)
        dist.destroy_process_group()
```

## Running Examples

There are examples in `examples/example-*/`

### Single-GPU Training
Run on a single node without using a job scheduler, by setting the environment variable:
```bash
$ CUDA_VISIBLE_DEVICES=0 python main.py
```

Alternatively, set `"gpu-parallel": false` in `input.json`:
```bash
$ python main.py
```

### Multi-GPU Training (DistributedDataParallel)
Run on a single node with multiple GPUs by setting the environment variables:
```bash
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
```

Alternatively, set `"gpu-parallel": true` in `input.json`:
```bash
$ python main.py
```
This automatically detects all available GPUs and utilizes them for distributed computation.

For an example of using multiple nodes and multiple GPUs with the SLURM scheduler, please refer to `examples/example-3BPA_300K-multiGPU/`

### Evaluation
```python
import json
import torch

from bam_torch.predicting.evaluator import Evaluator
from bam_torch.utils import find_input_json, date

input_json_path = find_input_json()
with open(input_json_path) as f:
    json_data = json.load(f)

evaluator = Evaluator(json_data)
evaluator.evaluate()
```

## Project Structure

```
bam-torch/
├── bam_torch/
│   ├── lammps/          # LAMMPS interface for ML potentials
|   |   ├── README.md    # Installation and usage guide for LAMMPS interface
│   ├── laplace/         # Laplace approximation and Bayesian inference utilities
│   ├── model/           # Core neural network architectures (RACE, E(3)-equivariant layers)
│   ├── predicting/      # Inference and prediction pipelines (Model evaluation)
│   ├── tase/            # ASE interface for ML potentials
│   ├── training/        # Training loop, loss computation, and optimization routines
│   └── utils/           # Common utility functions and helper tools
├── examples/
│   ├── example-3BPA_300K/    # QM9 benchmark example
├── README.md            # Project documentation
├── install_deps.py      # Dependency installation script
├── pyproject.toml       # Build configuration
└── LICENSE              # License information
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

