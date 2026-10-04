import ast
import gc
import os
import re
import pickle
from contextlib import nullcontext
from pathlib import Path
from time import time
from copy import deepcopy

import numpy as np
import torch
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from matscipy.neighbours import neighbour_list
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from bam_torch.training.base_trainer import BaseTrainer
from bam_torch.training.loss import (
    HuberLoss,
    L1Loss,
    MSELoss,
    RMSELoss,
    l2_regularization,
    resolve_huber_delta,
)
from bam_torch.utils.sampler import (
    BucketedDataLoader,
    DistributedBalancedAtomCountBatchSampler,
)
from bam_torch.utils.utils import data_to_dict, date

import atexit
from torch.utils.data import Dataset, DataLoader as TorchLoader
from torch.utils.data.distributed import DistributedSampler


def copy_frame(atoms):
    """Deep-enough copy of an ASE frame for preprocess_graph.

    DeNS preprocessing mutates the frame (rattles positions, replaces the
    calculator), which must not corrupt the shared parsed-frame cache.
    Atoms.copy() drops the calculator; the results are reattached directly
    rather than through SinglePointCalculator kwargs (which reject
    nonstandard keys such as 'virial'), with the arrays deep-copied so no
    in-place mutation downstream can ever reach the cached originals.
    """
    a = atoms.copy()
    calc = atoms.calc
    if calc is not None:
        a.calc = SinglePointCalculator(a)
        a.calc.results = {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in calc.results.items()
        }
    return a


def preprocess_graph(
    atoms,
    cutoff,
    ATOM_ENERGIES,
    ATOMIC_NUMBER_TO_INDEX,
    mode,
    dens_sigma=None,
    dens_corrupt_ratio=None,
):
    # DeNS: rattle the positions BEFORE the neighbour list so edges match
    # the noisy geometry. The noise is the regression target; the original
    # frame's DFT forces become the conditioning input. Energy label stays
    # the original frame's energy (predicted from the noisy geometry).
    #
    # With `dens_corrupt_ratio` set, only a random subset of atoms is
    # corrupted (DeNS paper Sec. 3.3, `corrupt_ratio` in the reference
    # implementation). Every label then still refers to the ORIGINAL frame:
    # corrupted atoms supply the noise target, uncorrupted atoms keep their
    # positions and their force label and stay on the ordinary force task.
    dens_noise = None
    dens_mask = None
    if dens_sigma is not None:
        dens_noise = np.random.normal(0.0, dens_sigma, (len(atoms), 3))
        if dens_corrupt_ratio is not None:
            # Independent per-atom draw, not an exact count — matches the
            # reference (`torch.rand(natoms) < corrupt_ratio`). Left None for
            # full corruption, whose downstream code needs no mask.
            dens_mask = np.random.random(len(atoms)) < dens_corrupt_ratio
            dens_noise = dens_noise * dens_mask[:, None]
        # Snapshot the DFT results before moving atoms: ASE getters refuse
        # to serve cached calculator results once positions change.
        results = dict(atoms.calc.results) if atoms.calc is not None else {}
        atoms.positions = atoms.get_positions() + dens_noise
        keep = {k: v for k, v in results.items()
                if k in ('energy', 'free_energy', 'forces', 'stress')}
        if keep:
            atoms.calc = SinglePointCalculator(atoms, **keep)

    iatoms, jatoms, Sij = neighbour_list("ijS", atoms, cutoff)

    if len(iatoms) == 0:
        #print(f"Warning: 0 neighbours found for {atoms.get_chemical_formula()}")
        return None

    crds = atoms.get_positions()
    species = np.array(
                [ATOMIC_NUMBER_TO_INDEX[n] for n in atoms.get_atomic_numbers()],
            ).astype(np.int32)
    # Subtract the per-element baseline for validation as well: the model
    # learns baseline-subtracted energies, so validating against raw totals
    # would offset loss_e by ~the mean atomic energy and bias checkpoint
    # selection toward more-negative predictions.
    if mode in ("train", "valid"):
        graph_energy = (ATOM_ENERGIES[species]).sum()
    else:
        graph_energy = 0.0
    # Formation Energy
    if 'mp2020_corrected_energy' in atoms.info.keys():
        energy = atoms.info['mp2020_corrected_energy'] - graph_energy
    else:
        energy = atoms.get_potential_energy() - graph_energy

    cell = atoms.get_cell()
    if np.all(np.asarray(cell) == 0.0):
        cell = np.diag([30., 30., 30.])
        atoms.set_cell(cell)

    calc_results = atoms.calc.results if atoms.calc is not None else {}
    if 'forces' in calc_results:
        forces = atoms.get_forces()
    else:
        forces = np.zeros ( crds.shape )

    if 'stress' in calc_results:
        stress = atoms.get_stress()
        volume = np.array([atoms.get_volume()])
    elif 'virial' in calc_results:
        # Convert virial -> stress (Voigt 6). Sign convention: stress = -virial / V.
        virial = np.asarray(calc_results['virial']).reshape(3, 3)
        vol = float(atoms.get_volume())
        s33 = -virial / vol if vol > 0.0 else np.zeros((3, 3))
        stress = np.array([s33[0, 0], s33[1, 1], s33[2, 2],
                           s33[1, 2], s33[0, 2], s33[0, 1]])
        volume = np.array([vol])
    else:
        stress = np.zeros(6)
        volume = np.array([1.0])

    num_nodes = len(atoms)
    num_edges = len(iatoms)

    graph_kwargs = {}
    if dens_noise is not None:
        graph_kwargs["dens_noise"] = torch.tensor(
            dens_noise, dtype=torch.float32
        )
        # Conditioning input: the ORIGINAL (unrattled) frame's DFT forces. On
        # partial corruption, only on corrupted atoms — an uncorrupted atom is
        # still being trained to predict its own force, so encoding that force
        # would hand the model the label. Zeroing here is equivalent to the
        # reference's in-model `force_sh * noise_mask`: ForceEncodingBlock
        # emits exactly zero for a zero-force node. Full corruption (no mask)
        # encodes every atom, as before.
        cond_forces = forces if dens_mask is None else forces * dens_mask[:, None]
        graph_kwargs["dens_forces"] = torch.tensor(
            cond_forces, dtype=torch.float32
        )
        if dens_mask is not None:
            graph_kwargs["dens_mask"] = torch.tensor(dens_mask, dtype=torch.bool)

    return Data(
        positions=torch.tensor(crds, dtype=torch.float32),
        species=torch.tensor(species, dtype=torch.long),
        forces=torch.tensor(forces, dtype=torch.float32),
        edges=torch.tensor(Sij, dtype=torch.float32),
        num_nodes=num_nodes,
        num_edges=num_edges,
        energy=torch.tensor([energy], dtype=torch.float32),
        cell=torch.tensor(np.array(cell), dtype=torch.float32).view(1, 3, 3),
        edge_index=torch.tensor(np.array([iatoms, jatoms]), dtype=torch.long),
        stress=torch.tensor(stress, dtype=torch.float32),
        volume=torch.tensor(volume, dtype=torch.float32),
        **graph_kwargs,
    )


def move_to_device(data, device):
    if isinstance(data, dict):
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in data.items()}
    elif hasattr(data, "to"):
        return data.to(device)
    else:
        return data


class MPTrainer(BaseTrainer):
    def __init__(self, json_data, rank=0, world_size=1):
        super().__init__(json_data, rank, world_size)

    def _dens_enabled(self):
        """DeNS on? The force encoder / noise head only receive gradients on
        denoising batches, so DDP needs find_unused_parameters then."""
        dens_config = self.json_data.get('dens', {})
        return bool(dens_config.get('enabled', False)
                    if isinstance(dens_config, dict) else dens_config)


    def configure_dataloader(self):
        with open(self.json_data['enr_avg_per_element'], 'r', encoding='utf-8') as file:
            content = file.read()
        enr_avg_per_element, uniq_element = ast.literal_eval(content)
        self.ATOM_ENERGIES = np.array ([
            enr for n, enr in enr_avg_per_element.items()
        ])
        #print ('ATOM_ENERGIES', self.ATOM_ENERGIES)
        # BaseTrainer.__init__ unpacks as (train_loader, valid_loader, uniq_element, enr_avg_per_element)
        return None, None, uniq_element, enr_avg_per_element

    def train(self):
        """Main training loop for BAM models.
        """
        #if self.model_ckpt is not None and 'loss' in self.model_ckpt:
        #    self.loss_valid_min = torch.tensor(float(self.model_ckpt['loss']['valid']))
        #else:
        self.loss_valid_min = torch.tensor(torch.inf)

        # loss_valid_min above resets on every 12h restart, and
        # prepare_segment.sh promotes whichever checkpoint is *newest* -- that
        # is model_train.pkl, written every epoch -- over fname_pkl. So
        # fname_pkl holds at best the current segment's best and is clobbered
        # by the latest state at every segment boundary; the all-time best
        # weights were being lost. Track them in a file the chain never
        # touches, seeding the bar from that file so it survives the restart.
        self.global_best = torch.tensor(torch.inf)
        if self.rank == 0 and os.path.exists('model_best.pkl'):
            try:
                _prev = torch.load('model_best.pkl', map_location='cpu',
                                   weights_only=False)
                self.global_best = torch.tensor(float(_prev['loss']['valid']))
                print(f"[best] bar resumed from model_best.pkl: "
                      f"valid={float(self.global_best):.6f}", flush=True)
                del _prev
            except Exception as exc:
                print(f"[best] model_best.pkl unreadable ({exc}); "
                      f"starting the bar at inf", flush=True)

        # Print logger head
        #if self.rank == 0:
        #    self.logger.print_logger_head()

        # Main training loop
        train_files, valid_files = self.get_xyz_data_path()
        #print('train_files', train_files)
        #print('valid_files', valid_files)

        nepoch = self.json_data['NN']['nepoch']
        for epoch in range(nepoch):
            try:
                self.model.update_criterion_value(epoch+self.start_epoch+1)
            except:
                pass

            train_loss_log_config = self.log_config['train']
            self.ckpt['train_scale_shift'] = {
                k: [] for k in self.enr_avg_per_element.keys()
            }
            self.ckpt['train_scale_shift_origin'] = []
            # {metric: [running_sum, running_count]} accumulated across all
            # train batches in this epoch, mirroring valid_acc.
            train_acc = {}

            n_xyz = len(train_files)
            for ixyz, filename in enumerate(train_files):
                ##################################
                #### TRAIN
                ##################################
                #print ('train', ixyz, filename)
                mode='train'
                self.model.train()
                # DeNS: decide per (epoch, file) whether this loader runs the
                # denoising task. Seeded so every DDP rank draws the same
                # flag (each rank still rattles with its own noise, which is
                # fine — BucketedDataLoader gives each graph to one rank).
                dens_config = self.json_data.get('dens', {})
                dens_file = False
                if dens_config.get('enabled', False):
                    rng = np.random.default_rng(
                        self.json_data['NN']['data_seed']
                        + 100_000 * (epoch + self.start_epoch) + ixyz
                    )
                    dens_file = bool(
                        rng.random() < dens_config.get('probability', 0.5)
                    )
                data_loader = self.configure_dataloader_from_xyz(
                    filename, mode=mode, dens=dens_file
                )
                if hasattr(data_loader, 'set_epoch'):
                    data_loader.set_epoch(epoch)

                for idata, data in enumerate(data_loader):
                    #print ('idata', idata)
                    data.to(self.device)
                    data = data_to_dict(data)  # This is for torch.jit compile

                    # Forward + loss + backward for the current batch.
                    self.optimizer.zero_grad()
                    loss_dict = self._train_backward(data)
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(),
                        clip_value=1.0)
                    self.optimizer.step()

                    if self.ema is not None:
                        self.ema.update()
                    # done backprop

                    for key, sc in loss_dict['sums'].items():
                        if key not in train_acc:
                            train_acc[key] = [
                                torch.zeros((), dtype=torch.float64),
                                torch.zeros((), dtype=torch.float64),
                            ]
                        train_acc[key][0] += sc['sum'].detach().to(torch.float64).cpu()
                        train_acc[key][1] += sc['count'].detach().to(torch.float64).cpu()

                    if self.ddp:
                        torch.distributed.barrier()

                    if (idata+1)%500 == 0:
                        curr_train_loss_dict = self._reduce_train_metrics(
                            train_acc, train_loss_log_config
                        )
                        if self.rank == 0:
                            train_losses = ' '.join(f"{k}={float(v):.6f}"
                                    for k, v in curr_train_loss_dict.items())
                            print(f"[per-xyz] epoch={epoch+1} xyz={ixyz+1}/{n_xyz} "
                                  f"{min(idata/len(data_loader)*100, 100.0):5.2f}% "
                                  f"train: {train_losses}", flush=True,
                                  file=self.logger.fout)
                    data.clear()
                    del data, loss_dict


                curr_train_loss_dict = self._reduce_train_metrics(
                    train_acc, train_loss_log_config
                )
                if self.rank == 0:

                    self.update_check_point(
                        epoch,
                        curr_train_loss_dict,
                        {'loss': self.loss_valid_min}
                    )
                    ckpt_to_save = dict(self.ckpt)
                    torch.save(ckpt_to_save, 'model_train.pkl')

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()
                del data_loader

                # Validate (and checkpoint) every `valid_interval` train files,
                # but always on the last file of the epoch — otherwise runs
                # with fewer train files than the interval would never
                # validate and never save a checkpoint.
                valid_interval = self.json_data.get('valid_interval', 5)
                if (ixyz+1) % valid_interval != 0 and (ixyz+1) != n_xyz:
                    continue

                ##################################
                #### VALID
                ##################################
                # valid
                loss_log_config = self.log_config['valid']
                self.ckpt['valid_scale_shift'] = {
                    k: [] for k in self.enr_avg_per_element.keys()
                }
                self.ckpt['valid_scale_shift_origin'] = []
                # Accumulate sum / count separately per metric so the final
                # mean is Σ sums / Σ counts — invariant to batch size.
                valid_acc = {}
                param_context = (
                    self.ema.average_parameters() if self.ema is not None else nullcontext()
                )
                self.model.eval()
                mode='valid'
                # No torch.no_grad(): force/stress need autograd through energy,
                # even in eval mode for energy-conserving force fields.
                with param_context:
                    for valid_filename in valid_files:

                        data_loader = self.configure_dataloader_from_xyz(
                            valid_filename,
                            mode=mode
                        )
                        for data in data_loader:
                            data.to(self.device)
                            data = data_to_dict(data)  # This is for torch.jit compile
                            preds = self.model(data, backprop=False)
                            #preds = self.scale_shift(preds, data, mode)

                            partial = self.compute_loss_valid(preds, data)
                            for key, sc in partial.items():
                                if key not in valid_acc:
                                    valid_acc[key] = [
                                        torch.zeros((), dtype=torch.float64),
                                        torch.zeros((), dtype=torch.float64),
                                    ]
                                valid_acc[key][0] += sc['sum'].detach().to(torch.float64).cpu()
                                valid_acc[key][1] += sc['count'].detach().to(torch.float64).cpu()

                            data.clear()
                            del data, preds, partial
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        gc.collect()
                        del data_loader

                    # Reduce per-rank sums/counts so every rank divides the
                    # same global totals. Each rank sees only its shard from
                    # BucketedDataLoader, so without this the reported metric
                    # is per-shard, not global.
                    if self.ddp:
                        for key in sorted(valid_acc.keys()):
                            for i in range(2):
                                t = valid_acc[key][i].to(self.device)
                                torch.distributed.all_reduce(
                                    t, op=torch.distributed.ReduceOp.SUM
                                )
                                valid_acc[key][i] = t.cpu()

                    # Single division at the end gives the unbiased global metric.
                    metrics = {
                        key: (s / c).to(torch.float32)
                        for key, (s, c) in valid_acc.items()
                    }
                    # Keep the controller metric invariant to task-weight changes.
                    metrics['loss'] = metrics['loss_e_h']
                    # Keep 'loss' regardless of log_config: the should_save
                    # comparison below and update_check_point require it
                    # (same coupling as in _reduce_train_metrics).
                    valid_keys = list(loss_log_config)
                    if 'loss' not in valid_keys:
                        valid_keys.append('loss')
                    last_valid_loss_dict = {
                        key: metrics.get(key, torch.tensor(float('nan')))
                        for key in valid_keys
                    }

                    if self.ddp:
                        torch.distributed.barrier()

                # last_valid_loss_dict is now identical across ranks (valid
                # accumulators were all-reduced), so this comparison agrees
                # everywhere. Compute + all-reduce train metrics on every rank
                # so the value written to the checkpoint reflects the global
                # average, then guard only the actual print/save on rank 0.
                should_save = bool(last_valid_loss_dict['loss'] < self.loss_valid_min)
                curr_train_loss_dict = None
                if should_save:
                    curr_train_loss_dict = self._reduce_train_metrics(
                        train_acc, train_loss_log_config
                    )
                    self.loss_valid_min = last_valid_loss_dict['loss']

                if self.rank == 0:
                    valid_losses = ' '.join(f"{k}={float(v):.6f}"
                        for k, v in last_valid_loss_dict.items())
                    print(f"[per-xyz] epoch={epoch+1} xyz={ixyz+1}/{n_xyz} "
                          f"valid: {valid_losses}",
                          flush=True,
                          file=self.logger.fout)

                    if should_save:
                        self.update_check_point(
                            epoch,
                            curr_train_loss_dict,
                            last_valid_loss_dict
                        )
                        # Build a save-only copy so the per-epoch list buffers
                        # in self.ckpt stay appendable for subsequent xyz files.
                        ckpt_to_save = dict(self.ckpt)
                        ckpt_to_save['train_scale_shift'] = {
                            k: (
                                torch.stack(v).mean()
                                if len(v) > 0
                                else torch.tensor(0.0, device=self.device)
                            )
                            for k, v in self.ckpt['train_scale_shift'].items()
                        }
                        ckpt_to_save['valid_scale_shift'] = {
                            k: (
                                torch.stack(v).mean()
                                if len(v) > 0
                                else torch.tensor(0.0, device=self.device)
                            )
                            for k, v in self.ckpt['valid_scale_shift'].items()
                        }
                        ckpt_to_save['valid_scale_shift_origin'] = torch.tensor(
                            self.ckpt['valid_scale_shift_origin']
                        ).mean()
                        torch.save(ckpt_to_save, self.json_data['NN']['fname_pkl'])

                        # All-time best, kept outside the chain's promotion
                        # path. Written to a temp file and atomically renamed:
                        # the 12h wall clock SIGKILLs this process, and a torn
                        # 156MB pickle would be unrecoverable.
                        if last_valid_loss_dict['loss'] < self.global_best:
                            self.global_best = last_valid_loss_dict['loss']
                            torch.save(ckpt_to_save, 'model_best.pkl.tmp')
                            os.replace('model_best.pkl.tmp', 'model_best.pkl')
                            print(f"[best] model_best.pkl updated: "
                                  f"valid={float(self.global_best):.6f} "
                                  f"epoch={epoch+1} xyz={ixyz+1}/{n_xyz}",
                                  flush=True, file=self.logger.fout)

                torch.cuda.empty_cache()
                gc.collect()

                # Update scheduler (learning rate)
                scheduler_cfg = self.json_data.get("scheduler", {})
                is_plateau = (
                    isinstance(scheduler_cfg, dict)
                    and scheduler_cfg.get("scheduler") == "ReduceLROnPlateau"
                )
                if is_plateau:
                    self.scheduler.step(last_valid_loss_dict['loss'])
                else:
                    self.scheduler.step()

                # log_config declares an "lr" entry but nothing ever emitted
                # it, so reading the schedule meant opening a 156MB
                # checkpoint. Log it right after the step, next to the metric
                # that drives it, so a decay and the patience counter filling
                # up are both visible as they happen.
                if self.rank == 0:
                    _inner = getattr(self.scheduler, 'scheduler', None)
                    _best = getattr(_inner, 'best', float('nan'))
                    print(f"[lr] epoch={epoch+1} xyz={ixyz+1}/{n_xyz} "
                          f"lr={self.scheduler.get_lr():.4e} "
                          f"bad={getattr(_inner, 'num_bad_epochs', '?')}/"
                          f"{getattr(_inner, 'patience', '?')} "
                          f"cooldown={getattr(_inner, 'cooldown_counter', '?')} "
                          f"best={float(_best):.7f}",
                          flush=True, file=self.logger.fout)


    def get_xyz_data_path(self):
        def _list_xyzs(path, key):
            if path is None:
                raise ValueError(
                    f"'{key}' must be set to a .extxyz file or a directory of .extxyz files"
                )
            if os.path.isdir(path):
                return sorted(
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.endswith(".extxyz")
                )
            return [path]

        train_files = _list_xyzs(self.json_data.get('ntrain'), 'ntrain')
        valid_files = _list_xyzs(self.json_data.get('nvalid'), 'nvalid')
        return train_files, valid_files

    def _read_frames_cached(self, xyz_file_path, cache=True):
        """Parse an xyz file once per process and reuse the frames.

        DeNS file-visits re-preprocess every time (fresh noise needs new
        neighbour lists) but must not re-READ from disk: on a shared
        filesystem the ASE parse is minutes per file and per-rank, and rank
        skew inside that window is what trips the NCCL watchdog (the next
        collective is the batch-count sync in BucketedDataLoader.__iter__).
        Set `cache_parsed_frames: false` (top level) to trade the ~few-KB/
        frame host memory back for re-reads.
        """
        if cache and self.json_data.get('cache_parsed_frames', True):
            store = getattr(self, '_xyz_frames_cache', None)
            if store is None:
                store = {}
                self._xyz_frames_cache = store
            frames = store.get(xyz_file_path)
            if frames is None:
                frames = read(xyz_file_path, index=slice(None))
                store[xyz_file_path] = frames
            return frames
        return read(xyz_file_path, index=slice(None))

    def configure_dataloader_from_xyz(self, xyz_file_path, mode, dens=False):
        # DeNS loaders bypass the graph cache: the rattling noise must be
        # fresh on every visit, and the neighbour list must be rebuilt for
        # the rattled geometry (preprocess_graph handles both).
        if dens:
            dens_config = self.json_data.get('dens', {})
            sigma_min = dens_config.get('sigma_min', 0.05)
            sigma_max = dens_config.get('sigma_max', 0.30)
            # "uniform": sigma ~ U(min, max). "log": log-uniform, matching
            # the DeNS paper's geometric ladder of noise scales — most
            # rattles are small (median = sqrt(min*max)), large ones are the
            # rare tail. Large sigma helps energy robustness but degrades
            # force accuracy monotonically (paper, Sec. 4 ablations), so
            # "log" is the recommended production setting.
            sigma_sampling = dens_config.get('sigma_sampling', 'uniform')
            if sigma_sampling == 'log':
                def draw_sigma():
                    return float(np.exp(np.random.uniform(
                        np.log(sigma_min), np.log(sigma_max))))
            elif sigma_sampling == 'uniform':
                def draw_sigma():
                    return float(np.random.uniform(sigma_min, sigma_max))
            else:
                raise ValueError(
                    f"dens.sigma_sampling must be 'uniform' or 'log', "
                    f"got {sigma_sampling!r}"
                )
            # Fraction of atoms corrupted per structure. null/None restores
            # the legacy all-atom corruption (which also drops every force
            # label for the batch); the paper's ablations favour partial.
            corrupt_ratio = dens_config.get('corrupt_ratio', 0.5)
            if corrupt_ratio is not None:
                corrupt_ratio = float(corrupt_ratio)
                if not 0.0 < corrupt_ratio <= 1.0:
                    raise ValueError(
                        f"dens.corrupt_ratio must be in (0, 1] or null, "
                        f"got {corrupt_ratio!r}"
                    )
            dataset = self._read_frames_cached(xyz_file_path)
            cutoff = self.json_data.get('cutoff', 6.0)
            # copy_frame: rattling mutates positions/calculator in place —
            # the cached originals must stay pristine for future visits.
            graphs = [
                preprocess_graph(
                    copy_frame(atoms),
                    cutoff,
                    self.ATOM_ENERGIES,
                    self.uniq_element,
                    mode,
                    dens_sigma=draw_sigma(),
                    dens_corrupt_ratio=corrupt_ratio,
                )
                for atoms in dataset
            ]
            for i, g in enumerate(graphs):
                if g is not None:
                    g.structure_id = torch.tensor([i], dtype=torch.long)
            safe_graphs = [g for g in graphs if g is not None]
            if self.rank == 0:
                print(
                    f"[dens] rattled {len(safe_graphs)} graphs from "
                    f"{xyz_file_path} (sigma ~ {sigma_sampling}"
                    f"({sigma_min}, {sigma_max}), corrupt_ratio="
                    f"{'all atoms' if corrupt_ratio is None else corrupt_ratio})",
                    flush=True,
                    file=self.logger.fout
                )
            return self.get_dataloader_from_data(safe_graphs, mode)

        cache = getattr(self, '_xyz_graph_cache', None)
        if cache is None:
            cache = {}
            self._xyz_graph_cache = cache

        # The energy baseline subtraction depends on mode, so the cache must
        # be keyed by (path, mode) — path alone would silently reuse targets
        # preprocessed under a different convention.
        cache_key = (xyz_file_path, mode)
        safe_graphs = cache.get(cache_key)
        if safe_graphs is None:
            # Frame-cache only what a later DeNS visit could reuse (train
            # files while dens is on) — valid files and dens-off runs would
            # pay the memory for nothing (their graphs are already cached).
            dataset = self._read_frames_cached(
                xyz_file_path,
                cache=self._dens_enabled() and mode == 'train',
            )
            cutoff = self.json_data.get('cutoff', 6.0)
            graphs = [
                preprocess_graph(
                    atoms,
                    cutoff,
                    self.ATOM_ENERGIES,
                    self.uniq_element,
                    mode
                )
                for atoms in dataset
            ]
            # Tag each graph with its position in the source file (before
            # dropping empties) so downstream consumers can trace bucketed,
            # reordered batches back to the original structures. Must not be
            # named *index*: PyG collation offsets 'index'-suffixed attributes
            # by num_nodes per graph.
            for i, g in enumerate(graphs):
                if g is not None:
                    g.structure_id = torch.tensor([i], dtype=torch.long)
            safe_graphs = [g for g in graphs if g is not None]
            cache[cache_key] = safe_graphs
            if self.rank == 0:
                print(
                    f"[xyz cache] preprocessed {len(safe_graphs)} graphs from "
                    f"{xyz_file_path} (dropped {len(graphs) - len(safe_graphs)} empty)",
                    flush=True,
                    file=self.logger.fout
                )

        return self.get_dataloader_from_data(safe_graphs, mode)

    def get_dataloader_from_data(self, graphset, mode):

        shuffle = mode == 'train'

        data_loader = BucketedDataLoader(
            dataset=graphset,
            batch_size=self.json_data['nbatch'],
            n_buckets=self.json_data.get('n_buckets', 8),
            shuffle=shuffle,
            drop_last=False,
            max_edges_per_batch=self.json_data.get('max_edges_per_batch', 16384),
            seed=self.json_data['NN']['data_seed'],
            reference='edges',
            rank=self.rank,
            world_size=self.world_size,
        )
        return data_loader

    def configure_loss(self, reduction='mean'):
        nn_config = self.json_data.get("NN")
        loss_config = nn_config.get("loss_config")
        if loss_config is None:
            if self.json_data["regress_forces"]:
                loss_config = {'energy_loss': 'huber',
                               'force_loss': 'huber',
                               'stress_loss' : 'huber'}
            else:
                loss_config = {'energy_loss': 'huber'}

        loss_fn = {}
        loss_fn['energy_loss'] = loss_config.get('energy_loss')
        loss_fn['force_loss'] = loss_config.get('force_loss')
        loss_fn['stress_loss'] = loss_config.get('stress_loss')

        for loss, loss_name in loss_fn.items():
            if loss_name in ['l1', 'L1', 'mae', 'MAE']:
                loss_fn[loss] = L1Loss(reduction=reduction)
            elif loss_name in ['mse', 'MSE']:
                loss_fn[loss] = MSELoss(reduction=reduction)
            elif loss_name in ['rmse', 'RMSE']:
                loss_fn[loss] = RMSELoss(reduction=reduction)
            elif loss_name in ['huber', 'HUBER', 'h', 'H']:
                loss_fn[loss] = HuberLoss(
                    huber_delta=resolve_huber_delta(loss_config, loss)
                )

        return loss_fn, loss_config

    def _reduce_train_metrics(self, train_acc, log_keys):
        # All-reduce running [sum, count] pairs across ranks (without mutating
        # the per-rank accumulators), divide once for the unbiased global mean,
        # and project onto the keys the logger asks for. Mirrors how
        # last_valid_loss_dict is built in the validation block.
        # Partial corruption can leave one rank with no clean atoms and
        # another with no corrupted atoms. Reduce the same keys on every
        # rank; absent local metrics contribute zero sum and zero count.
        metric_keys = (
            'loss_e', 'loss_f', 'loss_s', 'loss_dn',
            'loss_e_h', 'loss_f_h', 'loss_s_h', 'loss_dn_h',
        )
        snapshot = {}
        for key in metric_keys:
            if key in train_acc:
                snapshot[key] = [t.clone() for t in train_acc[key]]
            else:
                snapshot[key] = [
                    torch.zeros((), dtype=torch.float64),
                    torch.zeros((), dtype=torch.float64),
                ]
        if self.ddp:
            for key in sorted(snapshot.keys()):
                for i in range(2):
                    t = snapshot[key][i].to(self.device)
                    torch.distributed.all_reduce(
                        t, op=torch.distributed.ReduceOp.SUM
                    )
                    snapshot[key][i] = t.cpu()
        metrics = {
            key: (s / c).to(torch.float32) for key, (s, c) in snapshot.items()
        }
        e_lambda = self.json_data["NN"].get('enr_lambda', 1)
        metrics['loss'] = e_lambda * metrics['loss_e']
        # 'loss' must survive the projection even when the user's log_config
        # omits it: update_check_point and the best-valid comparison read it
        # unconditionally. Omitting it from log_config used to crash rank 0
        # at the first file boundary (KeyError) and strand the other ranks
        # in the next batch-count all_reduce until the NCCL watchdog fired.
        keys = list(log_keys)
        if 'loss' not in keys:
            keys.append('loss')
        return {
            key: metrics.get(key, torch.tensor(float('nan')))
            for key in keys
        }

    def _train_backward(self, data):
        preds = self.model(data, backprop=True)
        loss_dict = self.compute_loss_train(preds, data)
        loss_dict['loss'].backward()
        loss_dict['loss'] = loss_dict['loss'].detach()
        return loss_dict

    def compute_loss_train(self, preds, data):
        # Backprop scalar uses the configured loss_fn (Huber/MSE/L1/RMSE).
        # `sums` returns per-batch |Δ| sums + denominators so the training
        # logger can do Σ sums / Σ counts across batches and ranks — same
        # global-MAE convention as compute_loss_valid.
        lambda_config = self.json_data["NN"]
        e_lambda = lambda_config.get('enr_lambda', 1)
        f_lambda = lambda_config.get('frc_lambda', 1)
        s_lambda = lambda_config.get('str_lambda', 1)

        # PyG collapses `num_nodes` to the batch total on collation, so
        # per-graph atom counts must be rebuilt from ptr. Using
        # data['num_nodes'] here would divide every graph's energy by the
        # whole batch's atom count.
        num_atoms = data["ptr"][1:] - data["ptr"][:-1]

        # DeNS batch: train the energy of the noisy geometry toward the
        # original frame's energy, plus the noise-vector regression on the
        # corrupted atoms. Under partial corruption the uncorrupted atoms are
        # unmoved and keep the ordinary force task — both targets describe the
        # ORIGINAL frame, so they are consistent with each other and with the
        # energy label. Stress is skipped either way: it is a whole-cell
        # property of the unrattled geometry.
        if "noise_pred" in preds and "dens_noise" in data:
            dens_config = self.json_data.get("dens", {})
            de_lambda = dens_config.get('energy_lambda', 1.0)
            dn_lambda = dens_config.get('denoise_lambda', 10.0)

            # Full corruption (corrupt_ratio: null) stores no mask.
            if "dens_mask" in data:
                dens_mask = data["dens_mask"].view(-1)
            else:
                dens_mask = torch.ones(
                    preds["noise_pred"].shape[0], dtype=torch.bool,
                    device=preds["noise_pred"].device)
            n_corrupt = int(dens_mask.sum())
            n_clean = int(dens_mask.numel()) - n_corrupt

            loss = {"loss": []}
            energy_target = data["energy"].flatten()
            loss["loss_e"] = self.loss_fn["energy_loss"](
                preds["energy"].flatten(), energy_target,
                tag="energy", num_atoms=num_atoms)
            if n_corrupt == 0:
                # All-clean draw (tiny graph/chunk at low corrupt_ratio):
                # loss_dn is dropped below, but noise_pred is the ONLY path
                # from denoise_decoder into the loss graph on dens batches —
                # without it the head gets no grad and DDP stalls (the
                # 2026-07 unused-parameter failure mode). Tie it into loss_e
                # with zero weight; both the unchunked sum and the chunked
                # reconstruction consume loss_e, so the guarantee holds in
                # either path. Adds exactly 0.0 to the value.
                loss["loss_e"] = (
                    loss["loss_e"] + 0.0 * preds["noise_pred"].sum()
                )
            loss["loss"].append(de_lambda * loss["loss_e"])

            # Each term is a mean over its own atoms rather than the
            # reference's single all-atom mean with per-atom coefficients.
            # That keeps loss_dn/loss_f on the same scale as the full-
            # corruption and standard batches, so the lambdas stay meaningful
            # when corrupt_ratio changes. A small graph can land all-corrupted
            # or all-clean, hence the guards. Dropping loss_f is DDP-safe
            # (autograd forces share every parameter with loss_e); dropping
            # loss_dn needs the zero-tie above.
            if n_corrupt > 0:
                loss["loss_dn"] = self.loss_fn["force_loss"](
                    preds["noise_pred"][dens_mask].flatten(),
                    data["dens_noise"][dens_mask].flatten(),
                    tag="forces")
                loss["loss"].append(dn_lambda * loss["loss_dn"])
            if n_clean > 0 and "forces" in preds:
                loss["loss_f"] = self.loss_fn["force_loss"](
                    preds["forces"][~dens_mask].flatten(),
                    data["forces"][~dens_mask].flatten(),
                    tag="forces")
                loss["loss"].append(f_lambda * loss["loss_f"])
            loss["loss"] = sum(loss["loss"])

            n_graphs = num_atoms.numel()
            abs_dE = (preds['energy'].flatten() - energy_target).abs().detach()
            sums = {
                'loss_e': {
                    'sum': (abs_dE / num_atoms).sum(),
                    'count': torch.as_tensor(
                        float(n_graphs), dtype=abs_dE.dtype,
                        device=abs_dE.device),
                },
            }
            if n_corrupt > 0:
                sum_abs_dn = (
                    preds['noise_pred'][dens_mask].flatten()
                    - data['dens_noise'][dens_mask].flatten()
                ).abs().detach().sum()
                sums['loss_dn'] = {
                    'sum': sum_abs_dn,
                    'count': torch.as_tensor(
                        float(n_corrupt) * 3.0, dtype=sum_abs_dn.dtype,
                        device=sum_abs_dn.device),
                }
            if n_clean > 0 and "forces" in preds:
                sum_abs_f = (
                    preds['forces'][~dens_mask].flatten()
                    - data['forces'][~dens_mask].flatten()
                ).abs().detach().sum()
                sums['loss_f'] = {
                    'sum': sum_abs_f,
                    'count': torch.as_tensor(
                        float(n_clean) * 3.0, dtype=sum_abs_f.dtype,
                        device=sum_abs_f.device),
                }
            # Huber counterparts, mirroring the non-DeNS path below so the
            # log shows the minimised quantity on dens batches too. Without
            # this the dens shards logged nan for every *_h metric, since the
            # branch returns before the standard block.
            lc = self.json_data['NN'].get('loss_config', {})
            hf = torch.nn.functional.huber_loss
            sums['loss_e_h'] = {
                'sum': hf(preds['energy'].flatten() / num_atoms,
                          energy_target / num_atoms,
                          reduction='sum',
                          delta=resolve_huber_delta(lc, 'energy_loss')).detach(),
                'count': torch.as_tensor(
                    float(n_graphs), dtype=abs_dE.dtype, device=abs_dE.device),
            }
            if n_corrupt > 0:
                sum_h_dn = hf(
                    preds['noise_pred'][dens_mask].flatten(),
                    data['dens_noise'][dens_mask].flatten(),
                    reduction='sum',
                    delta=resolve_huber_delta(lc, 'force_loss')).detach()
                sums['loss_dn_h'] = {
                    'sum': sum_h_dn,
                    'count': torch.as_tensor(
                        float(n_corrupt) * 3.0, dtype=sum_h_dn.dtype,
                        device=sum_h_dn.device),
                }
            if n_clean > 0 and "forces" in preds:
                sum_h_f = hf(
                    preds['forces'][~dens_mask].flatten(),
                    data['forces'][~dens_mask].flatten(),
                    reduction='sum',
                    delta=resolve_huber_delta(lc, 'force_loss')).detach()
                sums['loss_f_h'] = {
                    'sum': sum_h_f,
                    'count': torch.as_tensor(
                        float(n_clean) * 3.0, dtype=sum_h_f.dtype,
                        device=sum_h_f.device),
                }
            loss['sums'] = sums
            return loss

        loss = {"loss": []}
        energy_target = data["energy"].flatten()
        loss["loss_e"] = self.loss_fn["energy_loss"](preds["energy"].flatten(),
                                                     energy_target,
                                                     tag="energy",
                                                     num_atoms=num_atoms)
        loss["loss"].append(e_lambda * loss["loss_e"])

        if "forces" in preds:
            force_target = data["forces"].flatten()
            loss["loss_f"] = self.loss_fn["force_loss"](preds["forces"].flatten(),
                                                        force_target,
                                                        tag="forces")
            loss["loss"].append(f_lambda * loss["loss_f"])
        if "stress" in preds:
            stress_target = data["stress"].flatten()
            loss["loss_s"] = self.loss_fn["stress_loss"](preds["stress"].flatten(),
                                                         stress_target,
                                                         tag="stress")
            loss["loss"].append(s_lambda * loss["loss_s"])

        loss["loss"] = sum(loss["loss"])

        n_graphs = num_atoms.numel()
        n_atoms_total = num_atoms.sum()
        abs_dE = (preds['energy'].flatten() - energy_target).abs().detach()
        sums = {
            'loss_e': {
                'sum': (abs_dE / num_atoms).sum(),
                'count': torch.as_tensor(
                    float(n_graphs), dtype=abs_dE.dtype, device=abs_dE.device
                ),
            },
        }
        if 'forces' in preds:
            sum_abs_f = (
                preds['forces'].flatten() - data['forces'].flatten()
            ).abs().detach().sum()
            sums['loss_f'] = {
                'sum': sum_abs_f,
                'count': n_atoms_total.to(sum_abs_f.dtype) * 3.0,
            }
        if 'stress' in preds:
            sum_abs_s = (
                preds['stress'].flatten() - data['stress'].flatten()
            ).abs().detach().sum()
            sums['loss_s'] = {
                'sum': sum_abs_s,
                'count': torch.as_tensor(
                    float(n_graphs) * 6.0,
                    dtype=sum_abs_s.dtype, device=sum_abs_s.device,
                ),
            }

        # Huber counterparts of the MAE metrics above, so the training log shows
        # the quantity that is actually minimised alongside the interpretable
        # MAE. loss["loss_*"] built earlier are per-batch means from self.loss_fn
        # and cannot be averaged across BucketedDataLoader's variable batch
        # sizes; recompute as sums to fit this function's sum/count scheme.
        lc = self.json_data['NN'].get('loss_config', {})
        hf = torch.nn.functional.huber_loss

        sums['loss_e_h'] = {
            'sum': hf(preds['energy'].flatten() / num_atoms,
                      energy_target / num_atoms,
                      reduction='sum',
                      delta=resolve_huber_delta(lc, 'energy_loss')).detach(),
            'count': torch.as_tensor(
                float(n_graphs), dtype=abs_dE.dtype, device=abs_dE.device
            ),
        }
        if 'forces' in preds:
            sum_h_f = hf(preds['forces'].flatten(), data['forces'].flatten(),
                         reduction='sum',
                         delta=resolve_huber_delta(lc, 'force_loss')).detach()
            sums['loss_f_h'] = {
                'sum': sum_h_f,
                'count': n_atoms_total.to(sum_h_f.dtype) * 3.0,
            }
        if 'stress' in preds:
            sum_h_s = hf(preds['stress'].flatten(), data['stress'].flatten(),
                         reduction='sum',
                         delta=resolve_huber_delta(lc, 'stress_loss')).detach()
            sums['loss_s_h'] = {
                'sum': sum_h_s,
                'count': torch.as_tensor(
                    float(n_graphs) * 6.0,
                    dtype=sum_h_s.dtype, device=sum_h_s.device,
                ),
            }

        loss['sums'] = sums
        return loss

    def compute_loss_valid(self, preds, data):
        # Per-batch absolute-error sums and denominators. The validation loop
        # accumulates these across all batches and divides once at the end:
        #   loss_e = Σ_g |ΔE_g|/N_atoms_g  /  N_graphs
        #   loss_f = Σ |ΔF|                /  (3 · Σ_g N_atoms_g)
        #   loss_s = Σ_g |ΔS_g|            /  (6 · N_graphs)
        # This is invariant to BucketedDataLoader's variable batch sizes.
        # Per-graph atom counts from ptr — data['num_nodes'] is the batch
        # total after PyG collation, not a per-graph vector.
        num_atoms = data["ptr"][1:] - data["ptr"][:-1]
        n_graphs = num_atoms.numel()
        n_atoms_total = num_atoms.sum()

        abs_dE = (preds['energy'].flatten() - data['energy'].flatten()).abs()
        out = {
            'loss_e': {
                'sum': (abs_dE / num_atoms).sum(),
                'count': torch.as_tensor(
                    float(n_graphs), dtype=abs_dE.dtype, device=abs_dE.device
                ),
            },
        }
        if 'forces' in preds:
            # This function was written for WBM relaxed structures, where forces
            # and stress are zero by definition, so |pred - 0| = |pred| and the
            # target was never subtracted. MPTrj val frames are mid-relaxation
            # snapshots with non-zero forces, so the target must be subtracted
            # to get a real error. Fall back to the old form when the validation
            # set carries no force data (WBM extxyz has positions only).
            if 'forces' in data:
                sum_abs_f = (preds['forces'].flatten() - data['forces'].flatten()).abs().sum()
            else:
                sum_abs_f = (preds['forces'].flatten()).abs().sum()
            out['loss_f'] = {
                'sum': sum_abs_f,
                'count': n_atoms_total.to(sum_abs_f.dtype) * 3.0,
            }
        if 'stress' in preds:
            # Same reasoning as forces above.
            if 'stress' in data:
                sum_abs_s = (preds['stress'].flatten() - data['stress'].flatten()).abs().sum()
            else:
                sum_abs_s = (preds['stress'].flatten()).abs().sum()
            out['loss_s'] = {
                'sum': sum_abs_s,
                'count': torch.as_tensor(
                    float(n_graphs) * 6.0,
                    dtype=sum_abs_s.dtype, device=sum_abs_s.device
                ),
            }

        # Huber counterparts of the MAE metrics above, so training's objective
        # and the reported validation number are the same quantity. Denominators
        # match the MAE keys so the two can be read side by side.
        #
        # Computed with reduction='sum' rather than by calling self.loss_fn:
        # HuberLoss returns a per-batch mean, and averaging per-batch means is
        # wrong under BucketedDataLoader's variable batch sizes -- the sum/count
        # scheme this function uses exists precisely for that reason.
        lc = self.json_data['NN'].get('loss_config', {})
        hf = torch.nn.functional.huber_loss

        out['loss_e_h'] = {
            'sum': hf(preds['energy'].flatten() / num_atoms,
                      data['energy'].flatten() / num_atoms,
                      reduction='sum',
                      delta=resolve_huber_delta(lc, 'energy_loss')),
            'count': torch.as_tensor(
                float(n_graphs), dtype=abs_dE.dtype, device=abs_dE.device
            ),
        }

        if 'forces' in preds and 'forces' in data:
            sum_h_f = hf(preds['forces'].flatten(), data['forces'].flatten(),
                         reduction='sum',
                         delta=resolve_huber_delta(lc, 'force_loss'))
            out['loss_f_h'] = {
                'sum': sum_h_f,
                'count': n_atoms_total.to(sum_h_f.dtype) * 3.0,
            }

        if 'stress' in preds and 'stress' in data:
            sum_h_s = hf(preds['stress'].flatten(), data['stress'].flatten(),
                         reduction='sum',
                         delta=resolve_huber_delta(lc, 'stress_loss'))
            out['loss_s_h'] = {
                'sum': sum_h_s,
                'count': torch.as_tensor(
                    float(n_graphs) * 6.0,
                    dtype=sum_h_s.dtype, device=sum_h_s.device
                ),
            }

        return out


class MPTrainerPkl(BaseTrainer):
    def __init__(self, json_data, rank=0, world_size=1):
        super().__init__(json_data, rank, world_size)

    def configure_dataloader(self):
        #with open(self.json_data['enr_avg_per_element'], 'r', encoding='utf-8') as file:
        #    content = file.read()
        #enr_avg_per_element, uniq_element = ast.literal_eval(content)

        #return None, None, enr_avg_per_element, uniq_element
        return None, None, None, None

    def load_pickle_files_with_progress(self, filename, folder_path):
        combined_list = []
        #files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
        #for filename in tqdm(files, desc=f"Loading files from {folder_path}"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, list):
                combined_list.extend(data)
            else:
                combined_list.append(data)
        return combined_list

    def train_one_epoch(self, mode='train', data_loader=None):
        train_files, valid_files = self.get_pkl_data_path()
        if mode == 'train':
            self.model.train()
            backprop = True
            loss_log_config = self.log_config['train']
            self.ckpt['train_scale_shift'] = {
                    k: [] for k in self.enr_avg_per_element.keys()
            }
            data_files = train_files
        else:  # test or valid
            self.model.eval()
            backprop = False
            loss_log_config = self.log_config['valid']
            self.ckpt['valid_scale_shift'] = {
                    k: [] for k in self.enr_avg_per_element.keys()
                }
            self.ckpt['valid_scale_shift_origin'] = []

        epoch_loss_dict = {key: [] for key in loss_log_config}
        for filename in data_files:
            data_loader = self.configure_dataloader_from_pkl(filename, mode=mode)

            for data in data_loader:
                data.to(self.device)
                data = data_to_dict(data)  # This is for torch.jit compile
                preds = self.model(data, backprop)
                preds = self.scale_shift(preds, data, mode)

                loss_dict = self.compute_loss(preds, data)
                loss = loss_dict['loss']
                if backprop:
                    self.optimizer.zero_grad()
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)
                    self.optimizer.step()

                    if self.ema is not None:
                        self.ema.update()

                for l in loss_log_config:
                    val = loss_dict.get(l, torch.nan)
                    epoch_loss_dict[l].append(val.detach().cpu() if isinstance(val, torch.Tensor) else val)

                data.clear()
                del data, preds, loss_dict
                torch.cuda.empty_cache()
                gc.collect()

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            del data_loader

        epoch_loss_dict = {key: torch.mean(torch.tensor(value)) \
                           for key, value in epoch_loss_dict.items()}
        return epoch_loss_dict

    def get_pkl_data_path(self):
        dir_path = self.json_data.get('fname_traj')
        ntrain = self.json_data.get('ntrain')
        nvalid = self.json_data.get('nvalid')
        if type(ntrain) == str:
            train_dir_path = ntrain
            if os.path.isdir(train_dir_path):
                train_files = [
                    os.path.join(train_dir_path, f)
                    for f in os.listdir(train_dir_path)
                    if f.endswith(".pkl")
                ]
            else:
                train_files = [train_dir_path]

            valid_dir_path = nvalid
            if os.path.isdir(valid_dir_path):
                valid_files = [
                    os.path.join(valid_dir_path, f)
                    for f in os.listdir(valid_dir_path)
                    if f.endswith(".pkl")
                ]
            else:
                valid_files = [valid_dir_path]
        else:
            train_dir_path = dir_path
            if os.path.isdir(train_dir_path):
                train_files = [
                    os.path.join(train_dir_path, f)
                    for f in os.listdir(train_dir_path)
                    if f.endswith(".pkl")
                ]
            else:
                train_files = [train_dir_path]
            valid_files = deepcopy(train_files)

        return train_files, valid_files

    def configure_dataloader_from_pkl(self, file_path, mode):
        file_number = 0
        match = re.search(r"_(\d+)\.pkl$", file_path)
        if match:
            file_number = int(match.group(1))

        sampled_dataset_save_folder = Path(f"./{mode}_datasets-{self.rank}")
        sampled_dataset_file_name = f"{mode}-{file_number}.pkl"
        sampled_dataset_file_path = sampled_dataset_save_folder / sampled_dataset_file_name

        if sampled_dataset_file_path.exists():
            t1 = time()
            file_path = sampled_dataset_file_path
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, list):
                data = [data]
            data_loader = self.get_dataloader_from_data(data)
        else:
            t1 = time()
            os.makedirs(sampled_dataset_save_folder, exist_ok=True)

            with open(file_path, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, list):
                data = [data]

            ntrain = self.json_data.get('ntrain')
            nvalid = self.json_data.get('nvalid')
            ntest = self.json_data.get('ntest')

            if type(ntrain) == float or ntrain < 1.0:
                ntrain = round(ntrain * len(data))
                nvalid = round(nvalid * len(data))
                if ntrain + nvalid > len(data):
                    nvalid = nvalid - (nvalid + ntrain - len(data))
                if type(ntest) == float:
                    ntest = round(ntest * len(data))
                    if ntrain + nvalid + ntest > len(data):
                        ntest = ntest - (ntrain + nvalid + ntest  - len(data))

            if type(ntrain) != str:
                if ntest == None:
                    ntest = 0
                    if ntrain + nvalid < len(data):
                        ntest = len(data) - (nvalid + ntrain)
                        assert ntrain + nvalid + ntest == len(data)

                idx = torch.arange(ntrain + nvalid + ntest)
                idx = idx[torch.randperm(ntrain + nvalid + ntest)]
                idx_train = idx[:ntrain]
                idx_valid = idx[ntrain:ntrain+nvalid]
                idx_test = idx[-ntest:]
                if mode == 'train':
                    test_data = [data[i] for i in idx_test]
                    data = [data[i] for i in idx_train]
                    sampled_test_dataset_save_folder = Path(f"./test_datasets-{self.rank}")
                    sampled_test_dataset_file_name = f"test-{file_number}.pkl"
                    sampled_test_dataset_file_path = sampled_test_dataset_save_folder / sampled_test_dataset_file_name
                    with open(sampled_test_dataset_file_path, "wb") as f:
                        pickle.dump(test_data, f)
                else:
                    data = [data[i] for i in idx_valid]

            if mode != 'test':
                with open(sampled_dataset_file_path, "wb") as f:
                    pickle.dump(data, f)

            data_loader = self.get_dataloader_from_data(data)
        return data_loader

    def get_dataloader_from_data(self, graphset):
        data_sampler = DistributedBalancedAtomCountBatchSampler(
            dataset=graphset,
            batch_size=self.json_data['nbatch'],
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            seed=self.json_data['NN']['data_seed'],
            drop_last=False,
            reference='edges'
        )
        data_loader = DataLoader(
            graphset,
            self.json_data['nbatch'],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=0,
            collate_fn=None,
            sampler=data_sampler
        )
        return data_loader

    def configure_loss(self, reduction='mean'):
        nn_config = self.json_data.get("NN")
        loss_config = nn_config.get("loss_config")
        if loss_config == None:
            if self.json_data["regress_forces"]:
                loss_config = {'energy_loss': 'huber',
                               'force_loss': 'huber',
                               'stress_loss' : 'huber'}
            else:
                loss_config = {'energy_loss': 'huber'}

        loss_fn = {}
        loss_fn['energy_loss'] = loss_config.get('energy_loss')
        loss_fn['force_loss'] = loss_config.get('force_loss')
        loss_fn['stress_loss'] = loss_config.get('stress_loss')
        huber_delta = loss_config.get('huber_delta')

        for loss, loss_name in loss_fn.items():
            if loss_name in ['l1', 'L1', 'mae', 'MAE']:
                loss_fn[loss] = torch.nn.L1Loss(reduction=reduction)
            elif loss_name in ['mse', 'MSE']:
                loss_fn[loss] = torch.nn.MSELoss(reduction=reduction)
            elif loss_name in ['rmse', 'RMSE']:
                loss_fn[loss] = RMSELoss(reduction=reduction)
            elif loss_name in ['huber', 'HUBER', 'h', 'H']:
                loss_fn[loss] = HuberLoss(huber_delta=huber_delta)

        return loss_fn, loss_config

    def compute_loss(self, preds, data):
        lambda_config = self.json_data["NN"]
        e_lambda = lambda_config.get('enr_lambda')
        f_lambda = lambda_config.get('frc_lambda')
        s_lambda = lambda_config.get('str_lambda')
        lambd = lambda_config.get('l2_lambda')
        if e_lambda == None:
            e_lambda = 1
        if f_lambda == None:
            f_lambda = 1
        if s_lambda == None:
            s_lambda = 1
        if lambd == None:
            lambd == 0

        loss = {"loss": []}
        energy_target = data["energy"].flatten()
        loss["loss_e"] = self.loss_fn["energy_loss"](preds["energy"].flatten(),
                                                     energy_target,
                                                     tag="energy",
                                                     num_atoms=data["num_nodes"])
        loss["loss"].append(e_lambda * loss["loss_e"])

        if "forces" in preds:
            force_target = data["forces"].flatten()
            loss["loss_f"] = self.loss_fn["force_loss"](preds["forces"].flatten(),
                                                        force_target,
                                                        tag="forces")
            loss["loss"].append(f_lambda * loss["loss_f"])
        if "stress" in preds:
            stress_target = data["stress"].flatten()
            loss["loss_s"] = self.loss_fn["stress_loss"](preds["stress"].flatten(),
                                                         stress_target,
                                                         tag="stress")
            loss["loss"].append(s_lambda * loss["loss_s"])

        if lambd != 0:
            params = self.model.parameters()
            loss["loss_l2"] = l2_regularization(params)
            loss["loss"].append(lambd * loss["loss_l2"])

        loss["loss"] = sum(loss["loss"])
        return loss


class DataBatchDataset(Dataset):
    def __init__(self, list_of_data_batch):
        self.data = list_of_data_batch

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_identity(x):
    return x[0]

class MPTrainer_V2(BaseTrainer):
    def __init__(self, json_data, rank=0, world_size=1):
        # multi-node version
        # Get node and local rank info from SLURM env vars
        node_id = os.environ.get('SLURM_NODEID', 'unknown')
        local_rank = os.environ.get('SLURM_LOCALID', rank)

        # multi-node version
        log_filename = f'node{node_id}_gpu{local_rank}_global{rank}.log'
        self.gpu_test_log = open(log_filename, 'w')
        atexit.register(self.close_log_file)

        self.epoch = 0
        super().__init__(json_data, rank, world_size)

    # multi-node version
    def close_log_file(self):
        if self.gpu_test_log and not self.gpu_test_log.closed:
            print(f"[{date()}] Closing log file for Rank {self.rank} (Node: {os.environ.get('SLURM_NODEID', 'unknown')}, "
                    f"Local GPU: {os.environ.get('SLURM_LOCALID', self.rank)})", flush=True)
            self.gpu_test_log.close()

    # multi-node version
    def configure_dataloader(self):
        with open(self.json_data['enr_avg_per_element'], 'r', encoding='utf-8') as file:
            content = file.read()
        enr_avg_per_element, uniq_element = ast.literal_eval(content)

        return None, None, uniq_element, enr_avg_per_element

    # multi-node version
    def load_pickle_files(self, filename, folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data if isinstance(data, list) else [data]

    def configure_model(self):
        """ Configure model using model configuration dictionary.
        Override BaseTrainer to fix multi-node DDP device_ids issue.
        """
        from torch.nn.parallel import DistributedDataParallel as DDP
        import sys

        model = self.set_model() # Set self.model
        model.to(self.device)

        # Set criterion for direct force training
        criterion = self.json_data.get("criterion")
        criterion_value = self.json_data.get("criterion_value")
        try:
            model.set_criterion(criterion, criterion_value)
        except:
            pass

        model_config = self.json_data['NN']
        restart = model_config.get('restart')

        evaluate_config = self.json_data['predict']
        evaluate = evaluate_config.get('evaluate_tag')  # True or False(None)

        if restart:
            rank = self.rank
            evaluate = False
            self.json_data['predict']['evaluate_tag'] = False
            model_ckpt = torch.load(model_config["fname_pkl"])
            start_epoch = model_ckpt['loss']['epoch']
            try:
                model.load_state_dict(model_ckpt['params'])
                if self.ddp:
                    # Use local rank for device_ids in multi-node setup
                    local_rank = int(os.environ.get('SLURM_LOCALID', self.rank % torch.cuda.device_count()))
                    model = DDP(model, device_ids=[local_rank])
            except RuntimeError as e:
                input_json = model_ckpt['input.json']
                fname = self.save_input_parameters(input_json, 'input_json_from_trained_model.txt')
                print(e)
                print(f'\n\033[31mSome of the parameter dimensions in the trained model you are trying to use')
                print(f'do not match the current input parameters.')
                print(f' -- Please check the ```{fname}``` file\033[0m\n')
                sys.exit(1)
            self.msg = f'\n\033[32mrestarting training from the {model_config["fname_pkl"]}\033[0m\n'
            self.msg += f' -- restarting from the step where the loss was {model_ckpt["loss"]}\n'

        if evaluate:  # True or False(None)
            rank = 0
            start_epoch = 0
            model_ckpt = torch.load(evaluate_config["model"], map_location=self.device)
            model.load_state_dict(model_ckpt['params'])
            model.eval()
            self.msg = f'\n\033[32mevaluating the {evaluate_config["model"]}\033[0m\n'

        if not restart and not evaluate: # initial train case
            rank = self.rank
            start_epoch = 0
            model_ckpt = None
            if self.ddp:
                # Use local rank for device_ids in multi-node setup
                local_rank = int(os.environ.get('SLURM_LOCALID', self.rank % torch.cuda.device_count()))
                model = DDP(model, device_ids=[local_rank])
            self.msg = f'\n\033[32minitializing training, results will be saved in the {model_config["fname_pkl"]}\033[0m\n'

        # Check the number of parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.msg += f'\nnumber of parameters:\n\033[33m -- model ({self.json_data["model"]})  {n_params}\033[0m\n'

        if rank == 0:
            print(self.msg)

        return model, n_params, model_ckpt, start_epoch

    def train_one_epoch(self, mode='train'):
        if mode == 'train':
            self.model.train()
            backprop = True
            loss_log_config = self.log_config['train']
            self.ckpt['train_scale_shift'] = {
                    k: [] for k in self.enr_avg_per_element.keys()
            }
            # data_files = train_files
            folder_path = self.json_data["ntrain"]
        else:
            self.model.eval()
            backprop = False
            loss_log_config = self.log_config['valid']
            folder_path = self.json_data["nvalid"]
            self.ckpt['valid_scale_shift'] = {
                    k: [] for k in self.enr_avg_per_element.keys()
                }
            self.ckpt['valid_scale_shift_origin'] = []

        self.gpu_test_log.flush()

        epoch_loss_dict = {key: [] for key in loss_log_config}

        files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
        print(f"Found {len(files)} files in {folder_path}", file=self.gpu_test_log)

        if self.world_size > 1:
            num_files = len(files)
            if num_files < self.world_size:
                rank_files = files
                use_internal_sampler = True
            else:
                files_per_rank = max(1, num_files // self.world_size)
                rank_start = self.rank * files_per_rank
                rank_end = min(rank_start + files_per_rank, num_files)
                rank_files = files[rank_start:rank_end]
                use_internal_sampler = False
            files = rank_files
        else:
            use_internal_sampler = False

        if not files:
            print(f"WARNING: RANK {self.rank} has no files to process!", file=self.gpu_test_log)
            if self.world_size > 1:
                torch.distributed.barrier()
            return {key: torch.tensor(0.0, device=self.device) for key in loss_log_config}
        print(f"RANK {self.rank}: Processing {len(files)} files: {files}", file=self.gpu_test_log)
        self.gpu_test_log.flush()

        # with torch.set_grad_enabled(mode == 'train'):
        for filename in files:
            try:
                data_batch = self.load_pickle_files(filename, folder_path)
            except Exception as e:
                print(f"ERROR: Failed to load {filename}: {e}", file=self.gpu_test_log)
                self.gpu_test_log.flush()
                continue
            if not data_batch:
                print(f"WARNING: Empty data batch for {filename}", file=self.gpu_test_log)
                self.gpu_test_log.flush()
                continue

            dataset = DataBatchDataset(data_batch)

            if use_internal_sampler and self.world_size > 1:
                data_sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=(mode == 'train'))
                data_sampler.set_epoch(self.epoch)
                shuffle = False
            else:
                data_sampler = None
                shuffle = (mode == 'train')

            data_loader = TorchLoader(
                dataset, batch_size=1, shuffle=shuffle, drop_last=False,
                pin_memory=True, num_workers=0, sampler=data_sampler,
                collate_fn=collate_identity
            )

            for data in data_loader:
                data.to(self.device)
                data = data_to_dict(data)  # This is for torch.jit compile
                self.gpu_test_log.flush()

                preds = self.model(data, backprop)
                preds = self.scale_shift(preds, data, mode)

                loss_dict = self.compute_loss(preds, data)
                loss = loss_dict['loss']

                if backprop:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)
                    self.optimizer.step()

                    if self.ema is not None:
                        self.ema.update()

                for l in loss_log_config:
                    epoch_loss_dict[l].append(loss_dict.get(l, torch.nan).detach().cpu())

                data.clear()
                del data, preds, loss_dict
                torch.cuda.empty_cache()
                gc.collect()

            data_batch.clear()
            del data_batch, dataset, data_loader
            if 'data_sampler' in locals() and data_sampler is not None:
                del data_sampler
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

        if self.world_size > 1:
            try:
                torch.distributed.barrier()
            except Exception as e:
                print(f"ERROR in barrier: {e}", file=self.gpu_test_log)
                self.gpu_test_log.flush()

        final_epoch_loss_dict = {}
        for key in epoch_loss_dict:
            tensor_list = epoch_loss_dict[key]

            if len(tensor_list) > 0:
                local_loss_sum = torch.sum(torch.stack([t.clone().detach().to(self.device) for t in tensor_list]))
                local_count = torch.tensor(len(tensor_list), device=self.device, dtype=torch.float)
            else:
                local_loss_sum = torch.tensor(0.0, device=self.device)
                local_count = torch.tensor(0.0, device=self.device)

            global_loss_sum = local_loss_sum.clone()
            global_count = local_count.clone()

            if self.world_size > 1:
                try:
                    torch.distributed.all_reduce(global_loss_sum, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(global_count, op=torch.distributed.ReduceOp.SUM)
                except Exception as e:
                    print(f"ERROR during all_reduce for key {key}: {e}", file=self.gpu_test_log)
                    self.gpu_test_log.flush()
                    global_loss_sum = torch.tensor(0.0, device=self.device)
                    global_count = torch.tensor(0.0, device=self.device)

            if global_count > 0:
                final_avg_loss = global_loss_sum / global_count
            else:
                final_avg_loss = torch.tensor(float('nan'), device=self.device)
                print(f"WARNING: No data for {key}!", file=self.gpu_test_log)

            final_epoch_loss_dict[key] = final_avg_loss
        self.gpu_test_log.flush()

        return final_epoch_loss_dict

    def configure_loss(self, reduction='mean'):
        nn_config = self.json_data.get("NN")
        loss_config = nn_config.get("loss_config")
        if loss_config == None:
            if self.json_data["regress_forces"]:
                loss_config = {'energy_loss': 'huber',
                               'force_loss': 'huber',
                               'stress_loss' : 'huber'}
            else:
                loss_config = {'energy_loss': 'huber'}
        loss_fn = {}
        loss_fn['energy_loss'] = loss_config.get('energy_loss')
        loss_fn['force_loss'] = loss_config.get('force_loss')
        loss_fn['stress_loss'] = loss_config.get('stress_loss')
        huber_delta = loss_config.get('huber_delta')
        compute_tag = "basic"
        for loss, loss_name in loss_fn.items():
            if loss_name in ['l1', 'L1', 'mae', 'MAE']:
                loss_fn[loss] = torch.nn.L1Loss(reduction=reduction)
            elif loss_name in ['mse', 'MSE']:
                loss_fn[loss] = torch.nn.MSELoss(reduction=reduction)
            elif loss_name in ['rmse', 'RMSE']:
                loss_fn[loss] = RMSELoss(reduction=reduction)
            elif loss_name in ['huber', 'HUBER', 'h', 'H']:
                loss_fn[loss] = HuberLoss(huber_delta=huber_delta)
                compute_tag = "huber"
        if compute_tag == "huber":
            self.compute_loss = self.compute_loss_huber
        else:
            self.compute_loss = self.compute_loss_basic

        return loss_fn, loss_config

    def compute_loss_huber(self, preds, data):
        lambda_config = self.json_data["NN"]
        e_lambda = lambda_config.get('enr_lambda', 1)
        f_lambda = lambda_config.get('frc_lambda', 1)
        s_lambda = lambda_config.get('str_lambda', 1)
        lambd = lambda_config.get('l2_lambda', 0)

        loss = {"loss": []}
        energy_target = data["energy"].flatten()
        loss["loss_e"] = self.loss_fn["energy_loss"](preds["energy"].flatten(),
                                                     energy_target,
                                                     tag="energy",
                                                     num_atoms=data["num_nodes"])
        loss["loss"].append(e_lambda * loss["loss_e"])

        if "forces" in preds:
            force_target = data["forces"].flatten()
            loss["loss_f"] = self.loss_fn["force_loss"](preds["forces"].flatten(),
                                                        force_target,
                                                        tag="forces")
            loss["loss"].append(f_lambda * loss["loss_f"])
        if "stress" in preds:
            stress_target = data["stress"].flatten()
            loss["loss_s"] = self.loss_fn["stress_loss"](preds["stress"].flatten(),
                                                         stress_target,
                                                         tag="stress")
            loss["loss"].append(s_lambda * loss["loss_s"])

        if lambd != 0:
            params = self.model.parameters()
            loss["loss_l2"] = l2_regularization(params)
            loss["loss"].append(lambd * loss["loss_l2"])

        loss["loss"] = sum(loss["loss"])
        return loss

    def compute_loss_basic(self, preds, data):
        lambda_config = self.json_data["NN"]
        e_lambda = lambda_config.get('enr_lambda', 1)
        f_lambda = lambda_config.get('frc_lambda', 1)
        s_lambda = lambda_config.get('str_lambda', 1)
        lambd = lambda_config.get('l2_lambda', 0)

        loss = {"loss": []}
        energy_target = data["energy"].flatten()
        loss["loss_e"] = self.loss_fn["energy_loss"](
            preds["energy"].flatten(), energy_target
        )
        loss["loss"].append(e_lambda * loss["loss_e"])

        if "forces" in preds and self.loss_fn.get("force_loss") is not None:
            force_target = data["forces"].flatten()
            loss["loss_f"] = self.loss_fn["force_loss"](
                preds["forces"].flatten(), force_target
            )
            loss["loss"].append(f_lambda * loss["loss_f"])

        if "stress" in preds and self.loss_fn.get("stress_loss") is not None:
            stress_target = data["stress"].flatten()
            loss["loss_s"] = self.loss_fn["stress_loss"](
                preds["stress"].flatten(), stress_target
            )
            loss["loss"].append(s_lambda * loss["loss_s"])
        elif (hasattr(self.model, "training_mode_for_lammps") \
                and self.model.training_mode_for_lammps):
            loss["loss_s"] = torch.tensor(
                0.0, device=preds["stress"].device, requires_grad=True
            )

        if lambd != 0:
            params = self.model.parameters()
            loss["loss_l2"] = l2_regularization(params)
            loss["loss"].append(lambd * loss["loss_l2"])

        # Get loss:
        # loss = (e_lambda * loss_e) + (f_lambda * loss_f)
        #        + (s_lambda * loss_s) + (lambd * loss_l2)
        loss["loss"] = sum(loss["loss"])
        return loss
