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
    -D Kokkos_ARCH_AMPERE80=ON \      # match your GPU; there is no AMPERE100
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

One build serves all three run layouts on a given machine. Only `BUILD_MPI`
and the MPI it links against change:

| | single GPU | multi GPU (one node) | multi node |
|---|---|---|---|
| MPI needed | no | yes, **CUDA-aware** | yes, **CUDA-aware** |
| cmake | `BUILD_MPI=OFF` | `BUILD_MPI=ON` | `BUILD_MPI=ON` |
| build differs? | — | identical to multi node | identical to multi GPU |
| launch | `lmp -k on g 1 ...` | `mpirun -np N ./gpu_bind.sh lmp ...` | `mpirun -np N <MCA> ./gpu_bind.sh lmp ...` |

So there are really two builds per architecture: an MPI-less one, and one
that covers both multi-GPU and multi-node. If in doubt, build the MPI one -
it still runs single-rank.

Use **upstream** LAMMPS here, not the `myung-group/lammps` fork used above -
the fork is pinned to 29 Aug 2024 and predates `forward_exchange`.

```
$ git clone --branch stable https://github.com/lammps/lammps.git lammps-mliap
$ cd lammps-mliap
$ grep '#define LAMMPS_VERSION' src/version.h                                 # >= 22 Jul 2025
$ grep -c forward_exchange src/KOKKOS/mliap_unified_couple_kokkos.pyx         # must be > 0
```

#### Toolchain and python environment

Both configurations below have been built and run, including multi-rank GPU
MD. Package versions are the ones actually used, not lower bounds.

|                  | aarch64 / GH200                | x86_64 / A100                          |
|------------------|--------------------------------|----------------------------------------|
| host gcc         | 11.5 (system)                  | system 8.5 too old -> `gcc-toolset-12` (12.2.1) |
| CUDA             | 12.8                           | 12.6                                   |
| cmake            | 3.26.5                         | 3.26.5                                 |
| MPI              | OpenMPI 5.0.6 (system)         | OpenMPI 5.0.6, **self-built with `--with-cuda`** |
| `Kokkos_ARCH_*`  | `HOPPER90`                     | `AMPERE80`                             |
| python           | 3.11.15                        | 3.11.12                                |
| torch            | 2.12.1+cu126                   | 2.5.1+cu121                            |
| e3nn             | 0.6.0                          | 0.4.4                                  |
| ase              | 3.29.0                         | 3.25.0                                 |
| numpy            | 1.26.4                         | 1.26.4                                 |
| cupy             | 13.6.0                         | 13.6.0                                 |
| openequivariance | 0.6.8                          | 0.6.8                                  |
| torch-ema        | 0.3                            | 0.3                                    |
| cython           | 3.2.9                          | 3.2.9                                  |
| ninja            | required (see below)           | required (see below)                   |

```
$ conda create -n bam-mlmd python=3.11 && conda activate bam-mlmd
$ pip install torch --index-url https://download.pytorch.org/whl/cu126
$ pip install e3nn ase numpy torch_ema cython cupy-cuda12x ninja
$ pip install openequivariance          # OEQ backend
```

`cupy` is not optional: the KOKKOS coupling that multi-layer models need for
ghost feature exchange goes through it.

**ninja.** OpenEquivariance ships a precompiled extension only for
torch >= 2.10. On older torch it JIT-compiles at first use and needs the
`ninja` executable on `PATH` (`pip install ninja` puts it in the env's `bin`).
Without it every rank dies with `Could not import DeviceProp: Ninja is
required to load C++ extensions`, surfacing as
`ERROR: Running mliappy unified module failure`. Warm the cache with a
single-rank run before launching multi-rank, so ranks do not compile
concurrently into the same directory.

#### MPI - only for multi GPU and multi node

Any run with more than one rank needs a **CUDA-aware** MPI: the ghost feature
exchange hands device pointers straight to MPI and has no host-staging path.

```
$ ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
```

`:true` -> use it. `:false` -> multi-rank segfaults inside the exchange
(`invalid permissions for mapped object`), and `-pk kokkos ... gpu/aware off`
does **not** work around it. A stock distribution OpenMPI is usually not
CUDA-aware; RHEL/Rocky 8 ships 4.1.1 without it. Build one:

```
$ wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.6.tar.bz2
$ tar xf openmpi-5.0.6.tar.bz2 && cd openmpi-5.0.6
$ ./configure --prefix=$HOME/prog/openmpi-cuda \
      --with-cuda=/usr/local/cuda-12.6 \
      --with-cuda-libdir=/usr/local/cuda-12.6/lib64/stubs
$ make -j 24 && make install
$ export PATH=$HOME/prog/openmpi-cuda/bin:$PATH
$ export LD_LIBRARY_PATH=$HOME/prog/openmpi-cuda/lib:$LD_LIBRARY_PATH
```

Put its `bin` first on `PATH` **before** configuring LAMMPS, so cmake picks it
up rather than the system one.

#### cmake + make: aarch64 / GH200

```
$ export PATH=/usr/local/cuda-12.8/bin:/path/to/openmpi/bin:$PATH
$ mkdir build && cd build
```

single GPU:
```
$ cmake \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_CXX_STANDARD=17 \
      -D CMAKE_CXX_COMPILER=$(pwd)/../lib/kokkos/bin/nvcc_wrapper \
      -D BUILD_MPI=OFF -D BUILD_OMP=ON \
      -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_HOPPER90=ON \
      -D PKG_ML-IAP=ON -D PKG_ML-SNAP=ON -D MLIAP_ENABLE_PYTHON=ON \
      -D PKG_PYTHON=ON -D Python_EXECUTABLE=$(which python) \
      ../cmake
$ make -j 16
```

multi GPU / multi node (one build, covers both):
```
$ cmake \
      ... same as above, except ...
      -D BUILD_MPI=ON \
      -D MPI_CXX_COMPILER=$(which mpicxx) \
      ../cmake
$ make -j 16
```

#### cmake + make: x86_64 / A100

RHEL/Rocky 8 - the system compiler is too old for CUDA 12 + C++17:

```
$ source /opt/rh/gcc-toolset-12/enable        # or gcc-toolset-13
$ export PATH=$HOME/prog/openmpi-cuda/bin:/usr/local/cuda-12.6/bin:$PATH
$ export LD_LIBRARY_PATH=$HOME/prog/openmpi-cuda/lib:$LD_LIBRARY_PATH
$ export CUDA_HOME=/usr/local/cuda-12.6
$ mkdir build && cd build
```

single GPU:
```
$ cmake \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_CXX_STANDARD=17 \
      -D CMAKE_CXX_COMPILER=$(pwd)/../lib/kokkos/bin/nvcc_wrapper \
      -D BUILD_MPI=OFF -D BUILD_OMP=ON \
      -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_AMPERE80=ON \
      -D PKG_ML-IAP=ON -D PKG_ML-SNAP=ON -D MLIAP_ENABLE_PYTHON=ON \
      -D PKG_PYTHON=ON -D Python_EXECUTABLE=$(which python) \
      ../cmake
$ make -j 24
```

multi GPU / multi node (one build, covers both):
```
$ cmake \
      ... same as above, except ...
      -D BUILD_MPI=ON \
      -D MPI_CXX_COMPILER=$(which mpicxx) \
      ../cmake
$ make -j 24
```

Confirm it linked against the CUDA-aware MPI, not the system one:
```
$ ldd lmp | grep libmpi        # must point into $HOME/prog/openmpi-cuda/lib
```

#### Notes on the flags

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

### Export

```
python -m bam_torch.lammps.create_lammps_mliap --pkl model.pkl --backend oeq \
    [--zbased | --elements Li P S Cl] --output bam_mliap_oeq.pt
```

### Run

```
export PYTHONPATH=<lammps>/python:<BAM-torch>:$PYTHONPATH
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1        # torch>=2.6
```

```
# in.run
pair_style mliap unified /path/bam_mliap_oeq.pt 0
pair_coeff * * H C O
```

`-pk kokkos neigh half newton on` is required: pair mliap needs `newton on`,
and `neigh half` reconciles it with the KOKKOS check. `-k on g 1` means one
GPU **per rank** - do not raise it to the node's GPU count.

single GPU:
```
lmp -k on g 1 -sf kk -pk kokkos neigh half newton on -in in.run
```

multi GPU, one node (`gpu_bind.sh` gives each rank its own device;
`GPUS=<list>` skips devices other users occupy):
```
GPUS=0,1 mpirun -np 2 ./gpu_bind.sh \
    lmp -k on g 1 -sf kk -pk kokkos neigh half newton on -in in.run
```

multi node:
```
mpirun -np <N> $MCA ./gpu_bind.sh \
    lmp -k on g 1 -sf kk -pk kokkos neigh half newton on -in in.run
```

`$MCA` is empty on a fabric with working GPUDirect. Where UCX lacks CUDA
support, bypass it with TCP and forward the environment to remote ranks,
which start without a login shell:

```
MCA="--mca pml ob1 --mca btl self,sm,tcp --mca btl_tcp_if_include <subnet> \
     -x PYTHONPATH -x PATH -x LD_LIBRARY_PATH -x TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
```

Do **not** use `gpu/aware off` with mliap - the host pack path is
unimplemented.

Results are rank-count and architecture independent. On a 5,232-atom
amorphous H/C/O cell, single-point forces (|F| RMS 1.39 eV/A) agreed to a
relative RMSE of 3.0e-07 between 1 and 2 ranks, and 2.8e-07 between
aarch64/GH200 and x86_64/A100 (regression slope 1.0000000054), with total
energy identical to every printed digit.

Extra ranks halve GPU memory per rank but do not speed MD up: the per-layer
ghost exchange scales with ghost count, which barely falls when the cell is
split. Measured 1.05x on two GPUs in one node, and 0.10x across two nodes on
a TCP fabric. Use extra GPUs to fit a system that does not fit on one, and
run independent jobs at one GPU each for throughput.

Complete SLURM and PBS job templates for all three layouts:
`examples/example-LAMMPS-mliap/`.

### Troubleshooting

| symptom | cause |
|---|---|
| `No module named 'lammps'` | `PYTHONPATH` missing `<lammps>/python` |
| `... no forward_exchange()` | LAMMPS < 22 Jul 2025, or not launched with `-k on ... -sf kk` |
| `Cannot use -kokkos on without KOKKOS installed` | binary built without `PKG_KOKKOS` |
| `Loading mliappy unified module failure` | `MLIAP_ENABLE_PYTHON=OFF`, or the adapter is not importable |
| CMake cannot find MPI | point `-D MPI_CXX_COMPILER=$(which mpicxx)` at your MPI, or use `BUILD_MPI=OFF` for single-rank builds |
| Segfault in the ghost exchange, `invalid permissions for mapped object` | MPI is not CUDA-aware - rebuild it with `--with-cuda` |
| `Could not import DeviceProp: Ninja is required` | `ninja` not on `PATH`; OEQ JIT-compiles on torch < 2.10 |
| nvcc rejects C++17 / Kokkos | host gcc too old - enable `gcc-toolset-12` |

Notes: torch<2.6 with OEQ needs a no-op shim for
`torch.library.register_autocast` (e.g. in sitecustomize.py); checkpoints
trained on 0-based z-table species (i.e. species = Z-1) must be exported
with `--zbased`.
