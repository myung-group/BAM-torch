# LAMMPS ML-IAP example — KOKKOS + OpenEquivariance

Runs a BAM-RACE model inside LAMMPS through `pair_style mliap unified` with the
eager PyTorch stack, so OpenEquivariance kernel acceleration works in MD (~9x
over the libtorch `pair_style bam` route on identical systems, with energies,
forces and pressure matching to float32 ordering noise).

Job templates are provided for SLURM and PBS, in single-GPU, multi-GPU
(one node) and multi-node layouts.

## Requirements

| Item | Requirement |
|---|---|
| LAMMPS | **stable 22 Jul 2025 or newer** — older releases lack the ghost feature exchange API that multi-layer models need, and work single-rank only |
| packages | `-D PKG_ML-IAP=ON -D PKG_ML-SNAP=ON -D PKG_PYTHON=ON -D MLIAP_ENABLE_PYTHON=ON -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_<GPU>=ON -D Python_EXECUTABLE=<env>/bin/python` |
| MPI | **CUDA-aware**, for any run with more than one rank (see below) |
| python env | torch, e3nn, ase, torch-ema, cython, **cupy** (KOKKOS coupling), **openequivariance** (OEQ backend) |
| env vars | `PYTHONPATH=<lammps>/python:<BAM-torch>` ; torch>=2.6: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` |

Architecture-specific build flags (host compiler, CUDA, `Kokkos_ARCH_*`) are in
[`bam_torch/lammps/README.md`](../../bam_torch/lammps/README.md). The job
scripts here differ by *scheduler* and *GPUs per node*, not by CPU
architecture — the same script works on aarch64 and x86_64.

## 1. Export the model

```bash
python -m bam_torch.lammps.create_lammps_mliap --pkl model.pkl --backend oeq \
    --output bam_mliap_oeq.pt          # --zbased if the checkpoint encodes species as Z-1
```

## 2. Prepare a data file

`make_data.py` converts any ASE-readable structure to a LAMMPS data file with a
fixed element order (types must match `pair_coeff * * <elements>`).

## 3. Run

| Layout | SLURM | PBS |
|---|---|---|
| 1 GPU | `job_single_gpu.slurm` | `job_single_gpu.pbs` |
| N GPUs, one node | `job_multi_gpu.slurm` | `job_multi_gpu.pbs` |
| N nodes | `job_multi_node.slurm` | `job_multi_node.pbs` |

> **Validation status.** The runtime flags and the `gpu_bind.sh` launch line
> were exercised on aarch64 (1 GPU/node, 1-4 nodes) and on x86_64 (2 GPUs in
> one node). `job_multi_node.pbs` has **not** been run — no multi-node PBS
> cluster was available — so its resource request and `$PBS_NODEFILE` handling
> follow the standard idiom rather than a measured configuration.

All of them launch **one rank per GPU** with the same runtime flags:

```
-k on g 1 -sf kk -pk kokkos neigh half newton on
```

`newton on` is required by pair mliap; `neigh half` reconciles it with the
KOKKOS package check. `-k on g 1` means *one GPU per rank* — do not raise it to
the node's GPU count. `gpu_bind.sh` gives each rank a distinct device:

```bash
GPUS=1,2 mpirun -np 2 ./gpu_bind.sh lmp -k on g 1 -sf kk ...
```

Set `GPUS` to skip devices other users occupy.

## MPI requirements

The adapter exchanges ghost features between message-passing layers through
LAMMPS' KOKKOS coupling, and those buffers stay on the GPU — there is no
host-staging path. Multi-rank runs therefore need an MPI that accepts device
pointers:

```bash
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
```

Two distinct failure modes, with different fixes:

| Symptom | Cause | Fix |
|---|---|---|
| Segfault (`invalid permissions for mapped object`) in the ghost exchange, backtrace through `THPFunction_apply` | `mpi_built_with_cuda_support:false` — the MPI treats device pointers as host memory | Rebuild MPI with CUDA support: `./configure --with-cuda=<cuda> --with-cuda-libdir=<cuda>/lib64/stubs`, then rebuild LAMMPS against it. `-pk kokkos ... gpu/aware off` does **not** help |
| Crash inside `uct_*` frames | MPI is CUDA-aware but UCX has no CUDA transport | Bypass UCX: `--mca pml ob1 --mca btl self,sm,tcp` and pin the subnet with `--mca btl_tcp_if_include` |

Remote ranks start without a login shell, so forward the environment
explicitly: `-x PYTHONPATH -x PATH -x LD_LIBRARY_PATH
-x TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`.

## Correctness across ranks and architectures

Same input, 5,232-atom amorphous H/C/O cell, single-point forces
(|F| RMS 1.39 eV/A):

| Comparison | max abs diff | RMSE | relative RMSE | regression slope |
|---|---|---|---|---|
| 1 vs 2 ranks (same host) | 0.0022 meV/A | 0.00028 meV/A | 3.0e-07 | — |
| aarch64/GH200 vs x86_64/A100 | 0.0015 meV/A | 0.00026 meV/A | 2.8e-07 | 1.0000000054 |

Total energy agreed to every printed digit in both comparisons, and the
residual force sum stayed at ~1e-14 eV/A. The differences are float32
accumulation ordering, five orders of magnitude below the model's own error.

Note that at |E| ~ 1e6 eV a float32 total energy is quantised at ~0.25 eV
(~0.05 meV/atom for 5k atoms); compare forces rather than absolute energies
when validating.

## Scaling: multi-GPU buys memory, not speed

Measured on the cell above, 200 steps NVT:

| Setup | 1 GPU | 2 GPUs | speedup |
|---|---|---|---|
| One node, CUDA IPC | 2.26 steps/s | 2.38 steps/s | 1.05x |
| Two nodes, TCP fabric | 4.48 steps/s | 0.43 steps/s | 0.10x |

The two rows are different hosts; compare within a row, not across.

Splitting the cell halves the local atom count but barely reduces the ghost
count (1.9 -> 2.8 ghosts per local atom here), and the per-layer feature
exchange scales with ghosts. Within a node the extra traffic roughly cancels
the compute gain; across a TCP fabric it dominates by an order of magnitude.
The cost is invisible in the LAMMPS timing table — the exchange happens inside
the pair style, so it is reported under `Pair` (>99%), not `Comm`.

What multi-GPU does deliver is memory: GPU memory per rank halved from 33.9 GB
to 17.1/17.4 GB. Use extra GPUs to fit a system that does not fit on one, and
run independent jobs at one GPU each when you want throughput.

## Files

| File | Purpose |
|---|---|
| `make_data.py` | ASE structure -> LAMMPS data file |
| `in.singlepoint` | run-0 energy/force check (compare against an ASE reference) |
| `in.equil` | staged NPT equilibration template (min -> hot NVT -> NPT -> cool -> target NPT) |
| `gpu_bind.sh` | binds one MPI rank to one GPU; honours `GPUS=<list>` |
| `job_single_gpu.{slurm,pbs}` | 1 GPU |
| `job_multi_gpu.{slurm,pbs}` | N GPUs inside one node |
| `job_multi_node.{slurm,pbs}` | N nodes, with the MPI settings for non-CUDA-aware fabrics |
