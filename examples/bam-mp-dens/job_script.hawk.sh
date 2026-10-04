#!/bin/sh
#SBATCH -J mp-train-v2-260928
#SBATCH -p a100_40g
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -o test_mp_train.out
#SBATCH -e test_mp_train.err
#SBATCH --time 12:00:00
#SBATCH --gres=gpu:4

source /etc/profile.d/modules.sh
module load cuda/13.0

# Activate explicitly rather than inheriting the submitting shell, so the job
# cannot silently run under the wrong python.
source /apps/applications/miniconda3/etc/profile.d/conda.sh
conda activate /home/hg23h05/hgpark/.conda/envs/bam || exit 1

# A previous segment ran 12h against a stale copy of mp_trainer.py and logged
# nan for every Huber metric. Fail fast instead of discovering that later.
python - <<'PY' || { echo "ABORT: loaded module is stale"; exit 1; }
import inspect, sys
from bam_torch.training import TRAINER_REGISTRY
src = inspect.getsource(TRAINER_REGISTRY["mp_v1"].compute_loss_valid)
missing = [k for k in ("loss_e_h", "loss_f_h", "loss_s_h", "loss_f_raw") if k not in src]
if missing:
    sys.exit(f"missing from loaded compute_loss_valid: {missing}")
import bam_torch
print("module OK:", bam_torch.__path__)

# inspect.getsource reads the .py from disk and says nothing about the bytecode
# that will actually execute. Check the loaded code object instead.
for name in ("compute_loss_valid", "compute_loss_train"):
    code = getattr(TRAINER_REGISTRY["mp_v1"], name).__code__
    consts = {c for c in code.co_consts if isinstance(c, str)}
    print(f"bytecode {name}: loss_e_h={'loss_e_h' in consts}")
import bam_torch.training.mp_trainer as _m
print("module file:", _m.__file__)
if "sy7willow" in _m.__file__ or "sy7willow" in sys.executable:
    sys.exit(f"ABORT: resolved into sy7willow ({sys.executable}, {_m.__file__})")
print("cached pyc :", getattr(_m, "__cached__", None))
PY

export OMP_NUM_THREADS=1
echo "host=$(hostname) start=$(date)"
echo "python: $(which python)"
python -c "import torch; print('torch', torch.__version__, '| gpus', torch.cuda.device_count())"

# NOT `torchrun`: conda console scripts hardcode the interpreter path in
# their shebang, and this env was rsync-copied from sy7willow, so
# bin/torchrun pointed at sy7willow's python -> workers imported
# sy7willow's unpatched bam_torch and every Huber metric logged nan.
# `python -m` inherits the activated interpreter by construction.
python -m torch.distributed.run --standalone --nproc-per-node=4 main_gpu.py
echo "exit=$? end=$(date)"
