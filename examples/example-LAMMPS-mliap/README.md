# LAMMPS ML-IAP example — KOKKOS + OpenEquivariance, single-GPU & multi-node

Runs a BAM-RACE model inside LAMMPS through `pair_style mliap unified` with
the eager PyTorch stack, so OpenEquivariance kernel acceleration works in MD
(~9x over the libtorch `pair_style bam` route on identical systems, with
energies/forces/pressure matching to float32 ordering noise).

## Requirements

| Item | Requirement |
|---|---|
| LAMMPS | **stable 22 Jul 2025 or newer** (older releases lack the ghost feature exchange API needed by multi-layer models; they work single-rank only). Build with `-D PKG_ML-IAP=ON -D PKG_PYTHON=ON -D MLIAP_ENABLE_PYTHON=ON -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_<GPU>=ON -D Python_EXECUTABLE=<env>/bin/python` |
| python env | torch, e3nn, ase, torch-ema, cython, **cupy** (KOKKOS coupling), **openequivariance** (OEQ backend) |
| env vars | `PYTHONPATH=<lammps>/python:<BAM-torch>` ; torch>=2.6: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` ; torch<2.6 + OEQ: a `torch.library.register_autocast` no-op shim |

## 1. Export the model

```bash
python -m bam_torch.lammps.create_lammps_mliap --pkl model.pkl --backend oeq \
    --output bam_mliap_oeq.pt          # add --zbased for omol/opoly-style checkpoints
```

## 2. Prepare a data file

`make_data.py` converts any ASE-readable structure to a LAMMPS data file with
a fixed element order (types must match `pair_coeff * * <elements>`).

## 3. Run

- Single GPU: `sbatch job_single_gpu.slurm` (or run the `lmp` line directly)
- Multi-node (1 GPU/node): `sbatch job_multi_node.slurm`

Key runtime flags (all validated):

```
-k on g 1 -sf kk -pk kokkos neigh half newton on
```
`newton on` is required by pair mliap; `neigh half` reconciles it with the
KOKKOS package check. Exchange buffers must stay on the GPU (do not stage
them to host).

### Multi-node MPI notes

The adapter exchanges ghost features between message-passing layers through
LAMMPS' KOKKOS coupling. On clusters whose UCX has no CUDA support (GPU-aware
MPI crashes in `uct_*` frames), bypass UCX with plain TCP and pin the
interconnect subnet (container/virtual interfaces like `docker0` otherwise
break peer matching):

```
mpirun -np <N> \
  --mca pml ob1 --mca btl self,sm,tcp \
  --mca btl_tcp_if_include <node-subnet e.g. 10.0.0.0/24> \
  -x TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD -x PYTHONPATH -x PATH \
  lmp -k on g 1 -sf kk -pk kokkos neigh half newton on -in in.equil
```

Correctness was verified rank-count independent (1 vs 2 ranks: energy within
0.05 meV/atom, pressure identical, force corr 1.000000). Note that with
TCP-only fabrics the per-step communication latency can outweigh the
domain-decomposition speedup for systems below ~10k atoms — running
independent jobs at 1 node each often gives better total throughput.

## Files

| File | Purpose |
|---|---|
| `make_data.py` | ASE structure -> LAMMPS data file |
| `in.singlepoint` | run-0 energy/force check (compare against an ASE reference) |
| `in.equil` | staged NPT equilibration template (min -> hot NVT -> NPT -> cool -> target NPT) |
| `job_single_gpu.slurm` | 1-node/1-GPU SLURM template |
| `job_multi_node.slurm` | multi-node SLURM template with the TCP MPI settings |
