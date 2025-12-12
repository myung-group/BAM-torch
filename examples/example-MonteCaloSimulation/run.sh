#!/bin/bash
# Monte Carlo Simulation Runner
# Includes folder separation by Na count

echo "========================================"
echo "Flexible TM Swap Monte Carlo Runner"
echo "========================================"

# Set CUDA visible devices (Specify GPU ID)
# Single GPU: export CUDA_VISIBLE_DEVICES=0
# Multi GPU: export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=0

# Enable MultiGPU parallel execution (run each seed on a different GPU)
# If false, run sequentially
export USE_MULTIGPU=false

# Set Python path (tutorial directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR

# Activate conda environment
echo "🔧 Activating conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate main_git_251101
# Check available GPU count
echo "🖥️  Available GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"

# Execute
echo "Starting Monte Carlo Simulation..."
echo "PYTHONPATH: $PYTHONPATH"
cd "$SCRIPT_DIR"
python MCs.py

echo ""
echo "Flexible MC Simulation completed!"
echo ""
