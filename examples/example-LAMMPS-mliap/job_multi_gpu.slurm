#!/bin/bash
# Several GPUs inside ONE node (SLURM). One rank per GPU.
#SBATCH -J mliap_mgpu
#SBATCH -p <partition>
#SBATCH --nodes=1 --ntasks=4 --ntasks-per-node=4 --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH -o %x.o%j -e %x.e%j
cd ${SLURM_SUBMIT_DIR}

# --- adjust to your installation ---
LMP=$HOME/prog/lammps/build/lmp                    # LAMMPS >= 22 Jul 2025
export PATH=/usr/local/cuda/bin:<mpi>/bin:<env>/bin:$PATH
export PYTHONPATH=<lammps>/python:<BAM-torch>:$PYTHONPATH
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1          # torch >= 2.6

# The MPI must be CUDA-aware; check with
#   ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
# Intra-node transfers then go over CUDA IPC and need no extra MCA flags.

# GPUS=... restricts the wrapper to specific devices on a shared node.
mpirun -np ${SLURM_NTASKS} \
  ./gpu_bind.sh \
  $LMP -k on g 1 -sf kk -pk kokkos neigh half newton on -in in.equil -log log.equil
