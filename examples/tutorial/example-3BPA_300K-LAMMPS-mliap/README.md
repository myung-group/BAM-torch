# 3BPA LAMMPS MD example (`pair_style mliap unified`, OEQ-accelerated)

Runs the RACE model trained in `../example-3BPA_300K` inside LAMMPS through
the ML-IAP unified (python) interface. Unlike the TorchScript
`pair_style bam` route (`../example-3BPA_300K-LAMMPS`), the model runs as
eager PyTorch, so OpenEquivariance kernel acceleration works in MD
(~9x over `pair_style bam` on identical systems on an A100).

## Files

- `lammps_mliap_bam.py`      - ML-IAP unified adapter (self-contained copy;
                               auto-detects species- vs Z-keyed E0 tables)
- `create_lammps_mliap.py`   - exports model.pkl -> bam_mliap_<backend>.pt
- `in.mliap`                 - LAMMPS input: same MD protocol as ../example-3BPA_300K-LAMMPS/race.in
- `3bpa_300K.data`           - same structure/type order as the pair_bam example

## Requirements

**Upstream** LAMMPS (stable >= 22 Jul 2025 - the myung-group fork is too old
for this route) built with:

    -D PKG_ML-IAP=ON -D PKG_ML-SNAP=ON -D MLIAP_ENABLE_PYTHON=ON
    -D PKG_PYTHON=ON -D Python_EXECUTABLE=$(which python3)
    # GPU (recommended; required for --backend oeq):
    -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_<GPU>=ON
    -D CMAKE_CXX_COMPILER=<lammps>/lib/kokkos/bin/nvcc_wrapper

Python env: bam_torch (pip install -e), cython (build), cupy (KOKKOS
coupling), openequivariance (OEQ backend). Multi-rank runs additionally need
a CUDA-aware MPI - see the BAM-LAMMPS branch (`bam_torch/lammps/README.md`,
`examples/example-LAMMPS-mliap/`) for HPC job templates.

## Steps

1. Export (from this folder; --backend e3nn for a CPU-only fallback):

       python create_lammps_mliap.py --pkl ../example-3BPA_300K/model.pkl \
           --backend oeq --output bam_mliap_oeq.pt

2. Run (PYTHONPATH must include LAMMPS' python/ dir AND this folder - the
   exported .pt reloads the adapter module by name at run time):

       PYTHONPATH=<lammps>/python:$(pwd) TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
           lmp -in in.mliap -k on g 1 -sf kk -pk kokkos neigh half newton on

   (drop the -k/-sf/-pk flags for a CPU, non-KOKKOS build; multi-layer models
   then require a ghost-free system, e.g. a centered isolated molecule)

3. Post-process (same conventions as the pair_bam example):

       python /path/to/BAM-torch/bam_torch/lammps/lammpsout_to_traj.py
