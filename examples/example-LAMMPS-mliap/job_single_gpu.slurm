#!/bin/bash
#SBATCH -J mliap_md
#SBATCH -p <partition>
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH -o %x.o%j -e %x.e%j
cd ${SLURM_SUBMIT_DIR}

# --- adjust to your installation ---
LMP=$HOME/prog/lammps/build/lmp                    # LAMMPS >= 22 Jul 2025
export PATH=/usr/local/cuda/bin:<mpi>/bin:<env>/bin:$PATH
export PYTHONPATH=<lammps>/python:<BAM-torch>:$PYTHONPATH
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1          # torch >= 2.6

$LMP -k on g 1 -sf kk -pk kokkos neigh half newton on -in in.equil -log log.equil
