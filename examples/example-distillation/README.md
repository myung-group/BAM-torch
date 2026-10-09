# Distillation example

Distill a small RACE student from a larger pretrained `BAM-torch` teacher
using a hybrid energy + force loss:

```
L = lambda_dft       * (e_lambda * loss_e_dft  + f_lambda * loss_f_dft)
  + (1 - lambda_dft) * (e_lambda * loss_e_t    + f_lambda * loss_f_t)
```

The trainer is registered as `"distill"` in `bam_torch.training.TRAINER_REGISTRY`
and lives at `bam_torch.distill.DistillTrainer`. The eight scripts in this
directory are a complete reproducible pipeline from raw MPtrj JSON to
ablation plots.

## Layout

| File | Purpose |
|---|---|
| `build_subset.py` | Stream-parse MPtrj JSON, mp-id-hash buckets, write train/valid/test trajs |
| `precompute_teacher.py` | Forward teacher through a traj, save raw residual energies + forces |
| `input.json` | Student config template (edit paths to your data) |
| `main.py` | Entry point — `--lambda-dft X` reroutes outputs to `runs/ldft_X/` |
| `run_ablation.sh` | Three sequential runs at `lambda_dft ∈ {0.0, 0.5, 1.0}` |
| `evaluate_student.py` | Student vs teacher per-atom energy/force MAE + speed bench |
| `plot_eval.py` | 2x2 parity + error-distribution figure from `evaluate_student.py` output |

## Prerequisites

Beyond the standard `bam_torch` install:

```bash
pip install ijson      # for streaming MPtrj JSON
pip install matplotlib # for plot_eval.py
```

You also need:

* A pretrained teacher checkpoint (a `.pkl` saved by `BAM-torch`).
* The MPtrj dataset (`MPtrj_2022.9_full.json`, ~12 GB) from
  [Figshare/CHGNet](https://figshare.com/articles/dataset/23713842).

## Pipeline

```bash
# 0) edit paths inside input.json so it points at your data + teacher ckpt

# 1) build train/valid/test splits (~5-10 min on a fast disk)
python build_subset.py \
    --json /path/to/MPtrj_2022.9_full.json \
    --out-dir /path/to/data/splits

# 2) precompute teacher predictions (~30 min on RTX 4090)
python precompute_teacher.py \
    --traj /path/to/data/splits/train.traj \
    --teacher-ckpt /path/to/teacher.pkl \
    --out /path/to/data/splits/teacher_train.pt
python precompute_teacher.py \
    --traj /path/to/data/splits/valid.traj \
    --teacher-ckpt /path/to/teacher.pkl \
    --out /path/to/data/splits/teacher_valid.pt

# 3) train a single student
python main.py --lambda-dft 0.5    # outputs land in runs/ldft_0.5/

# OR run the full lambda-ablation (3x training time)
./run_ablation.sh

# 4) evaluate one trained student against the teacher on test split
python evaluate_student.py \
    --student-ckpt runs/ldft_0.5/student_runtime.pkl \
    --teacher-ckpt /path/to/teacher.pkl \
    --test-traj /path/to/data/splits/test.traj \
    --out-dir runs/ldft_0.5/eval

# 5) plot parity + error distributions
python plot_eval.py \
    --eval-dir runs/ldft_0.5/eval \
    --test-traj /path/to/data/splits/test.traj
```

## Design notes

**Splits are mp-id-hash deterministic** so a structure (and all its
relaxation frames) is guaranteed never to leak across train/valid/test:

```python
bucket = int(hashlib.sha1(mp_id).hexdigest()[:8], 16) % 100
0..79  -> train  (80%)
80..94 -> valid  (15%)
95..99 -> test   (5%)
```

**Teacher predictions are precomputed offline** (one forward pass,
stored as per-frame residual energy + per-atom forces in a `.pt`
sidecar). The student then trains in the same time per epoch as a
non-distilled run because the teacher's GPU cost is amortized.

**Per-element baselines** (`enr_avg_per_element`, `uniq_element`) are
read directly from the teacher checkpoint and reused — *not*
recomputed on the subset — so DFT residual labels and teacher
predictions live in the same reference frame.

**Default student** (in `input.json`): `hidden_channels="64x0e+64x1o"`,
`nlayers=2`, `max_ell=1`, `features_dim=32`. Roughly 13x fewer params
than a `128x0e+128x1o+128x2e, nlayers=3, max_ell=2` teacher.

## Known limitations

* If the teacher saw the entire MPtrj training pool, its predictions on
  in-distribution frames are very close to DFT, so the soft-loss term
  may add little new gradient signal. The lambda-ablation is the
  cleanest test for whether distillation is buying anything; expand to
  teacher-labeled OOD augmentation (perturbed structures, MD frames)
  if it isn't.
* `nbatch` defaults to 8. RACE handles ragged batches via `data["ptr"]`
  natively; raise `nbatch` if you have GPU headroom.
