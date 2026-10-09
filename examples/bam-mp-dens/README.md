# RACE with DeNS on Materials Project trajectories

This example publishes the RACE+DeNS implementation used for the Hawk
`mp-train-willow-260729-v2-260928` experiment. It includes the experiment settings
and a fresh-start variant; checkpoints and training datasets are supplied
separately.

## Provenance

The branch starts from BAM-torch commit
`c8b563b6cea1c67e08820c29459b934763148642`, the baseline of the experimental
working tree. Required model, extxyz training, and batching changes are included
alongside DeNS. The experiment itself was run on that baseline; this branch has
since been merged with upstream `main`, and the legacy `example-*` folders are
kept. Note that the default `RACE` model now includes the species embedding and
per-layer scales, so checkpoints trained on earlier `main` revisions do not load.

- `input.json`: original experiment settings, with checkpoint restart enabled.
- `input.fresh.json`: the same settings with `NN.restart` disabled.
- `main_gpu.py`: experiment entrypoint with an optional config-file argument.
- `job_script.hawk.sh`: archived Hawk submission script. It contains
  cluster-specific paths, resource settings, and historical diagnostic guards;
  do not submit it unchanged. Use the launch commands below.
- `get_enr_avg_per_element.txt`: the experiment's elemental energy reference.

The original experiment used Python 3.13 and PyTorch 2.11.0+cu130 on NVIDIA A100
GPUs. See the repository's dependency metadata for supported Python versions.
Use a CUDA-enabled PyTorch installation compatible with your driver for the
OEQ-enabled settings in these configs.

The archived `grad_checkpoint: true` setting is retained for provenance but
does not activate gradient checkpointing in this experiment's RACE class.

## Installation and data

Install this branch in an isolated environment, including the OEQ extra:

```bash
git clone --branch BAM_DeNS https://github.com/myung-group/BAM-torch.git
cd BAM-torch
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e '.[oeq]'
cd examples/bam-mp-dens
```

Create `train_data/` and `valid_data/` and provide extxyz shards containing
positions, atomic numbers, cell/PBC information, DFT energies, forces, and stress.
These are local dataset directories, not committed links to Hawk storage.
The elemental reference provided here covers the experiment's species mapping;
use a matching reference when changing the dataset/species.

The original run used 40 training and 40 validation shards. Validation is
scheduled every 20 training shards (`valid_interval`), so use a smaller interval
when trying a smaller dataset.

## Start training

For a fresh single-GPU run:

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
