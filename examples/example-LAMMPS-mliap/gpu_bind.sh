#!/bin/bash
# Bind one MPI rank to one GPU.
#
# mliap runs one LAMMPS/KOKKOS rank per GPU: launch every rank with `-k on g 1`
# and let this wrapper choose the device, so ranks that share a node do not all
# land on GPU 0.
#
#   mpirun -np 2 ./gpu_bind.sh lmp -k on g 1 -sf kk -pk kokkos neigh half newton on -in in.equil
#
# GPUS lists the devices to use, in local-rank order. Set it to skip GPUs that
# other users already occupy:
#
#   GPUS=1,2 mpirun -np 2 ./gpu_bind.sh lmp ...
#
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
IFS=',' read -ra DEV <<< "$GPUS"

LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-${MV2_COMM_WORLD_LOCAL_RANK:-0}}}
if [ -z "${DEV[$LOCAL_RANK]}" ]; then
    echo "gpu_bind: local rank $LOCAL_RANK has no device in GPUS=$GPUS" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES=${DEV[$LOCAL_RANK]}

# OpenEquivariance and cupy JIT-compile at startup; give each rank its own
# cache directory so concurrent first runs do not race on the same files.
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}_rank${LOCAL_RANK}

exec "$@"
