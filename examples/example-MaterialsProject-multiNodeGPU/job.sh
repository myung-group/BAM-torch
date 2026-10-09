#!/bin/sh
#SBATCH -J mp_train
#SBATCH -p cas_v100_2
#SBATCH --nodes=3
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=2
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --comment etc

cd ${SLURM_SUBMIT_DIR}

source ${HOME}/anaconda3/bin/activate
conda activate bam_torch
module load cuda/12.4 # cudatoolkit/12.2

export OMP_NUM_THREADS=1

# Get the hostname of the first node in the allocation
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Choose a random port or a fixed one (ensure it's free)
export MASTER_PORT=29555

srun python main.py

