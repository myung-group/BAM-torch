#!/bin/bash
#SBATCH -J mliap_md_mpi
#SBATCH -p <partition>
#SBATCH --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH -o %x.o%j -e %x.e%j
cd ${SLURM_SUBMIT_DIR}

# --- adjust to your installation ---
LMP=$HOME/prog/lammps/build/lmp                    # LAMMPS >= 22 Jul 2025 (ghost exchange API)
export PATH=/usr/local/cuda/bin:<mpi>/bin:<env>/bin:$PATH
export PYTHONPATH=<lammps>/python:<BAM-torch>:$PYTHONPATH
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1          # torch >= 2.6

# MPI settings for fabrics WITHOUT CUDA-capable UCX:
#  - bypass UCX with plain TCP (pml ob1)
#  - pin the internode subnet so container/virtual interfaces don't break peers
#  - forward the environment to remote ranks explicitly
MCA="--mca pml ob1 --mca btl self,sm,tcp \
     --mca btl_tcp_if_include <node-subnet e.g. 10.0.0.0/24> \
     -x TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD -x PYTHONPATH -x PATH"
# On CUDA-aware fabrics (working GPUDirect) drop the MCA block entirely.

mpirun -np ${SLURM_NTASKS} $MCA \
  $LMP -k on g 1 -sf kk -pk kokkos neigh half newton on -in in.equil -log log.equil
