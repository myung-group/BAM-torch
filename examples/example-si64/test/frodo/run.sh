#!/bin/bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# train
CUDA_VISIBLE_DEVICES=0 lmp -k on g 1 -sf kk -in in.lammps
