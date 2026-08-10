#!/usr/bin/env bash
# Run the lambda-ablation: three sequential training runs with
# lambda_dft in {0.0, 0.5, 1.0}. Each run writes to runs/ldft_<lambda>/
# and a separate stdout log.
#
# Usage:
#   ./run_ablation.sh                 # nepoch from input.json (typical: 200)
#   ./run_ablation.sh --nepoch 50     # quicker check
#
# Make sure the conda env is active and bam_torch is editable-installed:
#   conda activate bam_torch
#   pip install -e .

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

EXTRA_ARGS=("$@")

for L in 0.0 0.5 1.0; do
    NAME="ldft_${L}"
    LOG="runs/${NAME}/stdout.log"
    mkdir -p "runs/${NAME}"
    echo "============================================================"
    echo "  starting run: lambda_dft=${L}  -> runs/${NAME}/"
    echo "============================================================"
    python main.py --lambda-dft "$L" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
done

echo "============================================================"
echo "  ablation finished. Compare with evaluate_student.py:"
echo "    python evaluate_student.py --student-ckpt runs/ldft_<L>/student_runtime.pkl ..."
echo "============================================================"
