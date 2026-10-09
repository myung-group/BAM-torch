# RACE with DeNS on Materials Project trajectories

This example trains RACE with DeNS through the `dens` trainer. Start from
`input.minimal.json`, a generic configuration that needs no energy-reference
file. The full list of `dens` options is in the top-level README, section
[Using DeNS (`input.json`)](../../README.md#using-dens-inputjson).
Checkpoints and training datasets are supplied separately.

## Provenance

The folder started as a Materials Project experiment run on the Hawk cluster.
`input.json` and `input.fresh.json` are kept unchanged as that experiment's
settings. They name the trainer `mp_v1`, which is now an alias of `dens`, so
they keep working. RACE keeps the layout of `main` unless
`species_embedding_dim` is set, so checkpoints trained on `main` load. The
experiment's checkpoints use a different architecture and load only when the
model is built with the experiment's settings from `input.json`.

- `input.minimal.json`: generic DeNS configuration, the same as the minimal
  block in the top-level README.
- `input.json`: original experiment settings, with checkpoint restart enabled.
- `input.fresh.json`: the same settings with `NN.restart` disabled.
- `main_gpu.py`: entrypoint with an optional config-file argument.
- `job_script.hawk.sh`: archived Hawk submission script, kept as a record.
  It has cluster-specific paths and resource settings, and its guard that
  checks the loaded `compute_loss_valid` source for `loss_f_raw` fails on the
  current code. Don't resubmit it unchanged; use the launch commands below.
- `get_enr_avg_per_element.txt`: the experiment's elemental energy reference,
  only needed to reproduce the experiment.

Set `"trainer": "dens"` in your config. `main_gpu.py` falls back to `mp`
(`MPTrainer_V2`, no DeNS) when the `trainer` key is missing, without a warning.
`mp_v1` is an alias of `dens`, and `mp_pkl` is the old pickle-based trainer.

The original experiment used Python 3.13 and PyTorch 2.11.0+cu130 on NVIDIA A100
GPUs. See the repository's dependency metadata for supported Python versions.
Use a CUDA-enabled PyTorch installation compatible with your driver for the
OEQ-enabled settings in these configs.

The archived `grad_checkpoint: true` setting is retained for provenance but
does not activate gradient checkpointing in this experiment's RACE class.

## Installation and data

Install the repository in an isolated environment, including the OEQ extra:

```bash
git clone https://github.com/myung-group/BAM-torch.git
cd BAM-torch
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e '.[oeq]'
cd examples/bam-mp-dens
```

Create `train_data/` and `valid_data/` and provide extxyz shards containing
positions, atomic numbers, cell/PBC information, DFT energies, forces, and stress.
These are local dataset directories, not committed links to Hawk storage.

You don't need a reference file to get the element list. Leave out
`enr_avg_per_element` and set `"element": "auto"`, as `input.minimal.json`
does: the elements and their reference energies are then taken from the train
and valid data. `input.json` and `input.fresh.json` point to
`get_enr_avg_per_element.txt`, which matches the experiment's species mapping.

The original run used 40 training and 40 validation shards. Validation is
scheduled every 20 training shards (`valid_interval`), so use a smaller interval
when trying a smaller dataset.

## Start training

For a single-GPU run with the generic configuration:

```bash
python main_gpu.py input.minimal.json
```

To rerun the experiment settings from scratch:

```bash
python main_gpu.py input.fresh.json
```

For four GPUs on one node:

```bash
python -m torch.distributed.run \
  --standalone --nproc-per-node=4 main_gpu.py input.fresh.json
```

Request the corresponding GPUs through your site's scheduler before running
these commands. Keep the active Python environment on the launcher and workers.
The entrypoint defaults to `input.json` when no config argument is supplied.

## Resume

Supply a compatible DeNS checkpoint as `model_fd128.pkl`, or change
`NN.fname_pkl` to its path, then run:

```bash
python main_gpu.py input.json
```

Checkpoint compatibility depends on model architecture, species mapping, and
DeNS settings. The source-only branch does not include the experiment's model
weights. Work in a separate output directory when comparing runs so checkpoints
and logs do not overwrite another experiment.

## DeNS behavior

`dens.enabled` enables the force-conditioning encoder and vector noise head.
For each training shard, `dens.probability` selects whether to use perturbed
coordinates. The experiment uses probability `0.4`, with a per-structure noise
scale sampled uniformly between `0.01` and `0.1`.

With `corrupt_ratio: 0.4`, each atom is independently selected for corruption.
Selected atoms receive coordinate noise and their original DFT forces as
conditioning input after the first interaction layer. Unselected atoms have
zero conditioning input and retain the ordinary force regression task.
Neighbor graphs are rebuilt from the perturbed coordinates.

On a perturbed batch, the objective combines:

- Original-frame energy targets, weighted by `dens.energy_lambda`.
- Noise-vector regression on corrupted atoms, weighted by
  `dens.denoise_lambda`.
- Force regression on uncorrupted atoms, weighted by `NN.frc_lambda`.

Stress regression is omitted on perturbed batches because stress labels refer
to the original cell geometry. Ordinary batches retain energy, force, and
stress losses. `corrupt_ratio: null` selects full-atom corruption instead.

Set `dens.enabled` to `false` for ordinary training. For inference from a
DeNS-trained model, omit the conditioning fields and use evaluation mode;
the physical energy/force route does not require input DFT forces.

## Logs and outputs

The experiment uses task-specific Huber thresholds: energy `0.05`, force
`0.25`, and stress `0.05`. `loss_*_h` reports Huber metrics and `loss_e`,
`loss_f`, and `loss_s` report absolute errors. The training log's top-level
`loss` follows the energy reporting metric, not the full weighted backward
objective. The ordinary validation selection uses the energy Huber metric.
Do not compare those reporting fields as if they were the same objective.

Some optional metrics are absent for perturbed batches; in particular a
missing stress metric there is not itself a non-finite training objective.
Checkpoints, logs, datasets, and runtime segment state are ignored within this
example directory.
