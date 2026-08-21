## Installation for GPU
Load modules
```
$ module load intel-22.3.1/icc-22.3.1 intel-22.3.1/fftw-3.3.10 cuda/cuda-11.8
```

Install libtorch
```
$ wget https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.2.0%2Bcu121.zip
$ unzip libtorch-shared-with-deps-2.2.0+cu121.zip
$ rm libtorch-shared-with-deps-2.2.0+cu121.zip
```

Install LAMMPS modified for BAM Package
```
$ git clone https://github.com/myung-group/lammps.git
$ cd lammps
$ mkdir build
$ cd build
$ cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=$(pwd) \
    -D CMAKE_CXX_STANDARD=17 \
    -D CMAKE_CXX_STANDARD_REQUIRED=ON \
    -D BUILD_MPI=ON \
    -D BUILD_SHARED_LIBS=ON \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D CMAKE_CXX_COMPILER=$(pwd)/../lib/kokkos/bin/nvcc_wrapper \
    -D Kokkos_ARCH_AMDAVX=ON \
    -D Kokkos_ARCH_AMPERE100=ON \
    -D CMAKE_PREFIX_PATH=$(pwd)/../../libtorch \
    -D PKG_ML-BAM=ON \
    ../cmake
$ make -j N     (N: integer)
    ex) $ make -j 8
$ make install
```

Then, you will see a sentence like the one below.
```
[ 85%] Building CXX object CMakeFiles/lammps.dir/home/gbsim/prog/lammps/src/ML-BAM/pair_bam.cpp.o
```

You can check like below,
```
$ lmp -h | grep bam

>>> bam             born            buck            buck/coul/cut   buck/coul/cut/kk 
```

## Installation for CPU (an easy way to install it on your local PC)
Load modules
```
$ source /opt/intel/oneapi/setvars.sh
```

Install libtorch (same as on the GPU)
```
$ wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.2.0%2Bcpu.zip
$ unzip libtorch-shared-with-deps-2.2.0+cu121.zip
$ rm libtorch-shared-with-deps-2.2.0+cu121.zip
```

Install LAMMPS modified for BAM Package
```
$ git clone https://github.com/myung-group/lammps.git
$ cd lammps
$ mkdir build
$ cd build
$ cmake \
      -D CMAKE_INSTALL_PREFIX=$(pwd) \
      -D CMAKE_CXX_STANDARD=17 \
      -D CMAKE_CXX_STANDARD_REQUIRED=ON \
      -D BUILD_MPI=ON \
      -D BUILD_OMP=ON \
      -D PKG_OPENMP=ON \
      -D PKG_ML-BAM=ON \
      -D CMAKE_PREFIX_PATH=$(pwd)/../../libtorch \
      ../cmake
$ make -j N     (N: integer)
    ex) $ make -j 8
$ make install
```

Then, you will see a sentence like the one below.
```
[ 89%] Building CXX object CMakeFiles/lammps.dir/home/gbsim/prog/lammps/src/ML-BAM/pair_bam.cpp.o
```

You can check like below,
```
$ lmp -h | grep bam

>>> bam             born            buck            buck/coul/cut 
```

If the above output does not appear, specify the path in your ```.bashrc``` as shown below.
Then check again.
```
$ vi ~/.bashrc
```
```
export PATH=/your_path/lammps/build:$PATH
    ex) export PATH=$HOME/prog/lammps/build:$PATH
```
```
$ source ~/.bashrc
$ lmp -h | grep bam
```


## Quick Start

If you only have ```model.pkl``` and not ```model.pt```, please follow the instructions below to generate "model.pt" from the existing checkpoint, ```model.pkl```.
If model.pt is already available, you can safely skip this step.
```
$ python make_pt.py
```

Once ```model.pt``` is prepared, generate the RACE model for LAMMPS (```model-lammps.pt```) using the process described below.
```
$ python create_lammps.py
```

Then, you can configure the LAMMPS input files (e.g.,```race.in```) to use the RACE-based Machine Learning Potential as shown below.
```
units metal
newton on
atom_style atomic
atom_modify map yes

read_data     3bpa_300K.data
timestep 0.0005  # 1 fs

pair_style bam #no_domain_decomposition
pair_coeff    * * model-lammps.pt C H N O

dump myDump all custom 1 dump.lammpstrj id type element x y z fx fy fz
dump_modify myDump element C H N O
thermo 1
thermo_style custom step temp pe ke etotal press

mass 1 1.008   # H
mass 2 12.011  # C
mass 3 14.007  # N
mass 4 15.999  # O

# MD run
fix 1 all nvt temp 300.0 300.0 100.0
run 1000
```

After that, you can run LAMMPS with the RACE-MLP.
```
$ lmp -in race.in
```

## ML-IAP (python) route — kernel-accelerated LAMMPS (OEQ/cueq capable)

`pair_style bam` loads a TorchScript file, which cannot contain JIT kernel
libraries (OpenEquivariance etc.). The ML-IAP unified route embeds python in
LAMMPS and runs the eager model instead, so those accelerations work in MD
(measured ~9x over `pair_style bam` e3nn on an A100, identical energies and
forces; see `lammps_mliap_bam.py` docstring for conventions).

### Build

Use **upstream** LAMMPS here, not the `myung-group/lammps` fork used above -
the fork is pinned to 29 Aug 2024 and predates `forward_exchange`.

```
$ git clone --branch stable https://github.com/lammps/lammps.git lammps-mliap
$ cd lammps-mliap
$ grep '#define LAMMPS_VERSION' src/version.h                                 # >= 22 Jul 2025
$ grep -c forward_exchange src/KOKKOS/mliap_unified_couple_kokkos.pyx         # must be > 0
$ mkdir build && cd build
```

Python env: `torch`, `e3nn`, `torch_ema`, `ase`, `cython`, `cupy`
(+ `openequivariance` for the OEQ backend).

The two configurations below have both been built and run.

|                  | aarch64 / GH200            | x86_64 / A100                       |
|------------------|----------------------------|-------------------------------------|
| host gcc         | 11.5 (system)              | system 8.5 too old -> `gcc-toolset-12` |
| CUDA             | 12.8                       | 12.6                                |
| MPI              | OpenMPI 5.0.6              | none installed -> `BUILD_MPI=OFF`   |
| `Kokkos_ARCH_*`  | `HOPPER90`                 | `AMPERE80`                          |

aarch64 / GH200:
```
$ export PATH=/usr/local/cuda-12.8/bin:/path/to/openmpi/bin:$PATH
$ cmake \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_CXX_STANDARD=17 \
      -D CMAKE_CXX_COMPILER=$(pwd)/../lib/kokkos/bin/nvcc_wrapper \
      -D BUILD_MPI=ON -D BUILD_OMP=ON \
      -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_HOPPER90=ON \
      -D PKG_ML-IAP=ON -D PKG_ML-SNAP=ON -D MLIAP_ENABLE_PYTHON=ON \
      -D PKG_PYTHON=ON -D Python_EXECUTABLE=$(which python) \
      ../cmake
$ make -j 16
```

x86_64 / A100 (RHEL/Rocky 8 - the system compiler is too old for CUDA 12 + C++17):
```
$ source /opt/rh/gcc-toolset-12/enable        # or gcc-toolset-13
$ export PATH=/usr/local/cuda-12.6/bin:$PATH
$ export CUDA_HOME=/usr/local/cuda-12.6
$ cmake \
      ... same as above, except ...
      -D BUILD_MPI=OFF \
      -D Kokkos_ARCH_AMPERE80=ON \
      ../cmake
$ make -j 24
```

`PKG_ML-SNAP` is mandatory - ML-IAP does not configure without it.
The KOKKOS coupling is required for multi-layer models (ghost feature
exchange) and needs `cupy` in the python env.

Pick `Kokkos_ARCH_*` from the GPU compute capability: 7.0 `VOLTA70`,
8.0 `AMPERE80`, 8.6 `AMPERE86`, 8.9 `ADA89`, 9.0 `HOPPER90`. There is no
`AMPERE100`; CMake silently ignores an unknown `Kokkos_ARCH_*`, so a typo
builds for the wrong GPU with no error.

Verify:
```
$ ./lmp -h | grep mliap        # expect: mliap  mliap/kk
```

**LAMMPS version requirement**: multi-rank (MPI) runs need LAMMPS
**stable 22 Jul 2025 or newer** - the ghost feature exchange API
(`forward_exchange` in the KOKKOS ML-IAP coupling) does not exist in
older releases (e.g. 29 Aug 2024), which work correctly on a single
rank only.

**Runtime flags**: `-pk kokkos neigh half newton on` (pair mliap requires
newton on; neigh half reconciles it with the KOKKOS check). The exchange
buffers must stay on the GPU - do **not** use `gpu/aware off` with mliap
(the host pack path is unimplemented). On fabrics whose UCX lacks CUDA
support, bypass UCX with TCP instead:
`--mca pml ob1 --mca btl self,sm,tcp --mca btl_tcp_if_include <subnet>`
plus explicit `-x` forwarding of PYTHONPATH/PATH/TORCH_FORCE_... to remote
ranks. Verified rank-independent (1 vs 2 ranks: energy 0.05 meV/atom,
identical pressure, force corr 1.000000). See
`examples/example-LAMMPS-mliap/` for complete SLURM job templates.

Export:
```
python -m bam_torch.lammps.create_lammps_mliap --pkl model.pkl --backend oeq \
    [--zbased | --elements Li P S Cl] --output bam_mliap_oeq.pt
```

Run:
```
export PYTHONPATH=<lammps>/python:<BAM-torch>:$PYTHONPATH
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1        # torch>=2.6
lmp -k on g 1 -sf kk -in in.run
# pair_style mliap unified /path/bam_mliap_oeq.pt 0
# pair_coeff * * H C O
```
Multi-GPU: standard MPI domain decomposition (`mpirun -np N`, one GPU per
rank); the adapter exchanges ghost features between message-passing layers
through the KOKKOS coupling.

### Troubleshooting

| symptom | cause |
|---|---|
| `No module named 'lammps'` | `PYTHONPATH` missing `<lammps>/python` |
| `... no forward_exchange()` | LAMMPS < 22 Jul 2025, or not launched with `-k on ... -sf kk` |
| `Cannot use -kokkos on without KOKKOS installed` | binary built without `PKG_KOKKOS` |
| `Loading mliappy unified module failure` | `MLIAP_ENABLE_PYTHON=OFF`, or the adapter is not importable |
| CMake cannot find MPI | no MPI on the host - use `BUILD_MPI=OFF` |
| nvcc rejects C++17 / Kokkos | host gcc too old - enable `gcc-toolset-12` |

Notes: torch<2.6 with OEQ needs a no-op shim for
`torch.library.register_autocast` (e.g. in sitecustomize.py); checkpoints
trained on 0-based z-table species (omol/opoly datasets) must be exported
with `--zbased`.
