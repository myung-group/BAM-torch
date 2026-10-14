"""Serial CPU equivariant E/F/S MVE trainer; optional controls are train-only."""

from copy import deepcopy
from pathlib import Path
import math
import pickle

import numpy as np
import torch
from ase.io import read
from ase.stress import full_3x3_to_voigt_6_stress
from matscipy.neighbours import neighbour_list
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .base_trainer import BaseTrainer, mve_config_digest, validate_mve_config
from .loss import (
    MVE_UNITS, assemble_effective_covariances, detached_beta_weights,
    gaussian_observation_nll, stress_voigt_to_irrep,
)
from .target_norm import SCHEMA as STATS_SCHEMA, fit_train_stats, geometry_digest, stats_digest

GPA_TO_EV_PER_A3 = 1 / 160.21766208
TASKS = {'energy': 'e', 'forces': 'f', 'stress': 's'}


def mve_graphs(frames, cutoff, species, references, frame_ids=None, stress_mask_gpa=None):
    """Read original ASE results/virial before adding graph placeholders."""
    if frame_ids is not None and len(frame_ids) != len(frames):
        raise ValueError('MVE original frame IDs must match the frame count')
    graphs = []
    for index, atoms in enumerate(frames):
        frame_id = index if frame_ids is None else frame_ids[index]
        results = atoms.calc.results if atoms.calc is not None else {}
        has = {key: results.get(key) is not None for key in TASKS}
        virial_value = results.get('virial')
        if virial_value is None:
            virial_value = atoms.info.get('virial')
        has['stress'] = has['stress'] or virial_value is not None
        numbers = [int(z) for z in atoms.numbers]
        if any(z not in species for z in numbers):
            raise ValueError(f'MVE frame {frame_id} contains species absent from training split')
        cell = np.asarray(atoms.cell, dtype=float)
        if not np.isfinite(cell).all():
            raise ValueError(f'MVE frame {frame_id} cell must be finite')
        original_volume = float(abs(np.linalg.det(cell)))
        has_physical_volume = np.isfinite(original_volume) and original_volume > 0
        if has['stress'] and not has_physical_volume:
            raise ValueError(f'MVE frame {frame_id} observed stress/virial requires positive finite original cell volume')
        if not has_physical_volume:
            cell = np.eye(3) * 30  # Neighbor geometry only; stress export is unavailable.
        geometry = atoms.copy()
        geometry.set_cell(cell)
        i, j, shifts = neighbour_list('ijS', geometry, float(cutoff))
        if len(i) == 0:
            raise ValueError(f'MVE frame {frame_id} has no edges inside cutoff')
        energy = float(results['energy']) if has['energy'] else 0.0
        if has['energy'] and not np.isfinite(energy):
            raise ValueError(f'MVE frame {frame_id} observed energy must be finite')
        if has['energy']:
            energy -= sum(references[species[z]] for z in numbers)
        forces = np.asarray(results['forces'], dtype=float) if has['forces'] else np.zeros((len(atoms), 3))
        if results.get('stress') is not None:
            stress = np.asarray(atoms.get_stress(), dtype=float)
        elif virial_value is not None:
            virial = np.asarray(virial_value, dtype=float)
            if virial.shape == (3, 3):
                virial = full_3x3_to_voigt_6_stress(virial)
            stress = -virial / original_volume
        else:
            stress = np.zeros(6)
        if forces.shape != (len(atoms), 3) or stress.shape != (6,):
            raise ValueError(f'MVE frame {frame_id} requires forces[A,3] and ASE Voigt stress[6]')
        for name, observed, value in (('energy', has['energy'], energy),
                                      ('forces', has['forces'], forces),
                                      ('stress', has['stress'], stress)):
            if observed and not np.isfinite(value).all():
                raise ValueError(f'MVE frame {frame_id} observed {name} must be finite')
        selected = has['stress'] and (stress_mask_gpa is None or
                    np.max(np.abs(stress)) <= stress_mask_gpa * GPA_TO_EV_PER_A3)
        graphs.append(Data(
            positions=torch.tensor(atoms.positions, dtype=torch.float32),
            species=torch.tensor([species[z] for z in numbers], dtype=torch.long),
            forces=torch.tensor(forces, dtype=torch.float32),
            stress=torch.tensor(stress, dtype=torch.float32).reshape(1, 6),
            energy=torch.tensor([energy], dtype=torch.float32),
            energy_valid=torch.tensor([has['energy']], dtype=torch.bool),
            force_valid=torch.full((len(atoms),), has['forces'], dtype=torch.bool),
            stress_original_valid=torch.tensor([has['stress']], dtype=torch.bool),
            stress_valid=torch.tensor([selected], dtype=torch.bool),
            stress_geometry_valid=torch.tensor([has_physical_volume], dtype=torch.bool),
            frame_id=torch.tensor([frame_id], dtype=torch.long),
            edges=torch.tensor(np.asarray(shifts), dtype=torch.float32).reshape(-1, 3),
            edge_index=torch.tensor(np.stack((i, j)), dtype=torch.long),
            num_edges=len(i), cell=torch.tensor(cell, dtype=torch.float32).reshape(1, 3, 3),
            num_nodes=len(atoms),
        ))
    return graphs


def _contract(config):
    """Only the output location, epoch horizon and prediction paths are runtime controls."""
    copy = deepcopy(config)
    copy['NN'].pop('fname_pkl', None)
    copy['NN'].pop('nepoch', None)
    copy.pop('predict', None)
    return copy


class MVETrainer(BaseTrainer):
    def __init__(self, json_data, rank=0, world_size=1, restart_from=None):
        if world_size != 1 or rank != 0:
            raise ValueError('MVE streaming/DDP unsupported until Phase 3; serial only')
        validate_mve_config(json_data)
        output = Path(json_data['NN']['fname_pkl'])
        if restart_from is not None and Path(restart_from).resolve() == output.resolve():
            raise ValueError('MVE restart input/output must be distinct paths')
        if output.exists():
            raise FileExistsError('MVE refuses to overwrite an existing checkpoint output')
        self.canonical_config = deepcopy(json_data)
        self.json_data = deepcopy(json_data)
        self.rank, self.world_size, self.ddp = 0, 1, False
        if json_data['device'] == 'cuda' and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
            raise RuntimeError('MVE CUDA requires exactly one visible, available GPU; no CPU fallback')
        self.device, self.msg = torch.device(json_data['device']), ''
        torch.manual_seed(json_data['NN']['init_seed'])
        self.model = self.set_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=json_data['NN']['learning_rate'])
        controller = json_data['mve'].get('controller', {'factor': .5, 'patience': 1, 'min_lr': 1e-8})
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=controller['factor'],
            patience=controller['patience'], min_lr=controller['min_lr'])
        self.floors = deepcopy(json_data['mve']['floors'])
        self.config_digest = mve_config_digest(self.canonical_config)
        self.train_loader, self.valid_loader, self.uniq_element, self.enr_avg_per_element = self.configure_dataloader()
        epsilon = json_data['mve'].get('epsilon', {'enabled': False})
        self.next_adaptive = deepcopy(epsilon['min']) if epsilon['enabled'] else {key: 0.0 for key in TASKS}
        self.measured_adaptive = deepcopy(self.next_adaptive)
        self.validation_step = 0
        self.epoch = 0
        self.best_residual = float('inf')
        self.best_params = None
        self.best_measured = None
        self.best_validation_step = 0
        self.best_epoch = 0
        self.owned_output = output.resolve()
        if restart_from is not None:
            self.restore(restart_from)

    def configure_dataloader(self):
        config = self.json_data
        if (type(config['ntrain']) is not int or type(config['nvalid']) is not int or
                min(config['ntrain'], config['nvalid']) < 1):
            raise ValueError('MVE requires positive integer ntrain and nvalid')
        separate = 'fname_train' in config or 'fname_valid' in config
        if separate:
            if ('fname_traj' in config or not config.get('fname_train') or not config.get('fname_valid') or
                    Path(config['fname_train']).resolve() == Path(config['fname_valid']).resolve()):
                raise ValueError('MVE separate train/valid files must be distinct and exclude fname_traj')
            train = read(config['fname_train'], index=':')
            valid = read(config['fname_valid'], index=':')
            if len(train) != config['ntrain'] or len(valid) != config['nvalid']:
                raise ValueError('MVE separate train/valid files must match exact ntrain/nvalid counts')
            if {geometry_digest(atoms) for atoms in train} & {geometry_digest(atoms) for atoms in valid}:
                raise ValueError('MVE separate train/valid files contain overlapping geometries')
            train_ids, valid_ids = list(range(len(train))), list(range(len(valid)))
        else:
            if not config.get('fname_traj'):
                raise ValueError('MVE requires fname_traj or distinct fname_train/fname_valid')
            frames = read(config['fname_traj'], index=':')
            if len(frames) < config['ntrain'] + config['nvalid']:
                raise ValueError('MVE trajectory has fewer frames than ntrain+nvalid')
            indices = torch.randperm(len(frames), generator=torch.Generator().manual_seed(config['NN']['data_seed'])).tolist()
            train_ids = indices[:config['ntrain']]
            valid_ids = indices[config['ntrain']:config['ntrain'] + config['nvalid']]
            train = [frames[i] for i in train_ids]
            valid = [frames[i] for i in valid_ids]
        norm = config['mve'].get('target_norm', {'enabled': False})
        self.train_stats = fit_train_stats(train, train_ids, norm['enabled'], norm.get('external'))
        self.train_ids = train_ids
        species, references = self.train_stats['species'], self.train_stats['references']
        mask = config['mve'].get('mask', {'enabled': False})
        threshold = mask['max_gpa'] if mask['enabled'] else None
        train_graphs = mve_graphs(train, config['cutoff'], species, references, train_ids, threshold)
        valid_graphs = mve_graphs(valid, config['cutoff'], species, references, valid_ids, threshold)
        batch_size = config['nbatch']
        return (DataLoader(train_graphs, batch_size=batch_size, shuffle=False),
                DataLoader(valid_graphs, batch_size=batch_size, shuffle=False), species, references)

    def _batch_terms(self, preds, data, adaptive):
        counts_graph = torch.bincount(data['batch'], minlength=len(preds['energy'])).to(preds['energy'].dtype)
        effective = assemble_effective_covariances(
            {'energy': preds['energy_var'], 'forces': preds['forces_var'], 'stress': preds['stress_var']},
            self.floors, MVE_UNITS, adaptive, self.train_stats['scales'], counts_graph)
        q = stress_voigt_to_irrep(dtype=preds['stress'].dtype, device=preds['stress'].device)
        residuals = {
            'energy': (data['energy'].reshape(-1) - preds['energy'].reshape(-1)).reshape(-1, 1),
            'forces': data['forces'] - preds['forces'],
            'stress': (data['stress'].reshape(-1, 6) - preds['stress'].reshape(-1, 6)) @ q.T,
        }
        eligibility = {'energy': data['energy_valid'].reshape(-1),
                       'forces': data['force_valid'].reshape(-1),
                       'stress': data['stress_valid'].reshape(-1)}
        normalizing = self.json_data['mve'].get('target_norm', {'enabled': False})['enabled']
        terms = {}
        for key in TASKS:
            mask = eligibility[key]
            if mask.dtype != torch.bool or len(mask) != len(residuals[key]):
                raise ValueError(f'MVE {key} eligibility must match observation count')
            covariance = effective[key][mask]
            residual = residuals[key][mask]
            physical = covariance.reshape(-1, 1, 1) if key == 'energy' else covariance
            if normalizing:
                scale = self.train_stats['scales'][key]
                divisor = counts_graph[mask] * scale if key == 'energy' else scale
                residual = residual / (divisor[:, None] if key == 'energy' else divisor)
                normalized = physical / (divisor[:, None, None].square() if key == 'energy' else divisor ** 2)
            else:
                normalized = physical
            physical_nll = gaussian_observation_nll(residuals[key][mask], physical)
            train_nll = gaussian_observation_nll(residual, normalized) if normalizing else physical_nll
            sigma = self.train_stats['scales'][key] if normalizing else 1.0
            tracking_divisor = counts_graph[mask, None] * sigma if key == 'energy' else sigma
            tracking_residual = residuals[key][mask].detach() / tracking_divisor
            terms[key] = {
                'nll': train_nll, 'physical_nll': physical_nll,
                'cov': covariance, 'atom_counts': counts_graph[mask] if key == 'energy' else None,
                'residual_sum': residuals[key][mask].detach().square().sum(),
                'residual_components': int(mask.sum()) * residuals[key].shape[-1],
                'whitened_sum': tracking_residual.square().sum(),
                'zero': (preds[key].sum() + preds[key + '_var'].sum()) * 0,
                'unmasked_stress': int(data['stress_original_valid'].sum()) if key == 'stress' else None,
            }
        return terms

    def compute_loss(self, preds, data, training=True, adaptive=None):
        """Public single-step objective; beta denominators span the serial step batch."""
        terms = self._batch_terms(preds, data, self.next_adaptive if adaptive is None else adaptive)
        return self._reduce([terms], training)[0]

    def _reduce(self, batches, training):
        metrics, total = {}, sum(t[key]['zero'] for t in batches for key in TASKS)
        mse, whitened = {}, {}
        counts = {}
        for key, short in TASKS.items():
            parts = [batch[key] for batch in batches]
            count = sum(len(part['nll']) for part in parts)
            counts[short] = count
            metrics['count_' + short] = count
            if key == 'stress':
                unmasked = sum(part['unmasked_stress'] for part in parts)
                metrics['count_s_unmasked'] = unmasked
                metrics['count_s_masked_out'] = unmasked - count
            if count:
                nll = torch.cat([part['nll'] if training else part['physical_nll'] for part in parts])
                if training and self.json_data['mve'].get('beta', 0):
                    covariance = torch.cat([part['cov'] for part in parts])
                    atom_counts = torch.cat([part['atom_counts'] for part in parts]) if key == 'energy' else None
                    weights = detached_beta_weights(covariance, self.json_data['mve']['beta'], atom_counts)
                    value = (weights * nll).sum() / count
                else:
                    value = nll.mean()
                mse[key] = sum(part['residual_sum'] for part in parts) / sum(part['residual_components'] for part in parts)
                whitened[key] = sum(part['whitened_sum'] for part in parts) / sum(part['residual_components'] for part in parts)
                metrics['loss_' + short] = value
            else:
                value = sum(part['zero'] for part in parts)
                mse[key], whitened[key] = None, None
                metrics['loss_' + short] = None
            total = total + self.json_data['NN'].get({'e':'enr_lambda','f':'frc_lambda','s':'str_lambda'}[short], 1) * value
        if not any(counts.values()):
            raise ValueError('MVE epoch has no eligible E/F/S observations')
        metrics['loss'] = total
        metrics['counts'] = counts
        metrics['residual_selection'] = sum(self.json_data['NN'].get({'e':'enr_lambda','f':'frc_lambda','s':'str_lambda'}[short], 1) * mse[key]
                                             for key, short in TASKS.items() if mse[key] is not None)
        metrics['whitened_mse'] = whitened
        return metrics, counts

    def train_one_epoch(self, mode='train', data_loader=None):
        if mode not in ('train', 'valid'):
            raise ValueError('MVE epoch supports train or valid only')
        training = mode == 'train'
        self.model.train(training)
        loader = data_loader or (self.train_loader if training else self.valid_loader)
        adaptive = deepcopy(self.next_adaptive)
        sums = {key: 0.0 for key in TASKS}
        counts = {key: 0 for key in TASKS}
        residual_sums = {key: 0.0 for key in TASKS}
        whitened_sums = {key: 0.0 for key in TASKS}
        component_counts = {key: 0 for key in TASKS}
        unmasked_stress = 0
        for data in loader:
            preds = self.model(data.to(self.device), backprop=training)
            terms = self._batch_terms(preds, data, adaptive)
            unmasked_stress += terms['stress']['unmasked_stress']
            if not any(len(terms[key]['nll']) for key in TASKS):
                continue
            result, _ = self._reduce([terms], training)
            if training:
                self.optimizer.zero_grad()
                result['loss'].backward()
                self.optimizer.step()
            for key, short in TASKS.items():
                count = result['count_' + short]
                counts[key] += count
                if count:
                    sums[key] += float(result['loss_' + short].detach()) * count
                    residual_sums[key] += float(terms[key]['residual_sum'])
                    whitened_sums[key] += float(terms[key]['whitened_sum'])
                    component_counts[key] += terms[key]['residual_components']
            # The next batch must not retain the previous batch's autograd graph.
            del result, terms, preds
        if not any(counts.values()):
            raise ValueError(f'MVE {mode} epoch has no eligible E/F/S observations')
        if not training:
            self.measured_adaptive = adaptive
            self.validation_step += 1
        loss = {key: sums[key] / counts[key] if counts[key] else None for key in TASKS}
        residual = {key: residual_sums[key] / component_counts[key] if counts[key] else None for key in TASKS}
        whitened = {key: whitened_sums[key] / component_counts[key] if counts[key] else None for key in TASKS}
        return {**{'loss_' + short: loss[key] for key, short in TASKS.items()},
                **{'count_' + short: counts[key] for key, short in TASKS.items()},
                'loss': sum(self.json_data['NN'].get({'e':'enr_lambda','f':'frc_lambda','s':'str_lambda'}[short], 1) * loss[key]
                            for key, short in TASKS.items() if loss[key] is not None),
                'counts': {short: counts[key] for key, short in TASKS.items()},
                'count_s_unmasked': unmasked_stress,
                'count_s_masked_out': unmasked_stress - counts['stress'],
                'residual_selection': sum(self.json_data['NN'].get({'e':'enr_lambda','f':'frc_lambda','s':'str_lambda'}[short], 1) * residual[key]
                                          for key, short in TASKS.items() if residual[key] is not None),
                'whitened_mse': whitened}

    def _update_epsilon(self, valid):
        epsilon = self.json_data['mve'].get('epsilon', {'enabled': False})
        if not epsilon['enabled']:
            return
        for key, short in TASKS.items():
            measured = valid['whitened_mse'][key]
            if valid['count_' + short] and measured is not None:
                self.next_adaptive[key] = max(epsilon['min'][key], epsilon['c'][key] * measured)

    def train(self):
        for epoch in range(self.epoch + 1, self.json_data['NN']['nepoch'] + 1):
            training = self.train_one_epoch('train')
            valid = self.train_one_epoch('valid')
            selection = valid['residual_selection']
            if selection < self.best_residual:
                self.best_residual = selection
                self.best_params = deepcopy(self.model.state_dict())
                self.best_measured = deepcopy(self.measured_adaptive)
                self.best_validation_step = self.validation_step
                self.best_epoch = epoch
            self.scheduler.step(selection)
            self._update_epsilon(valid)  # AFTER full validation on measured floors.
            self.epoch = epoch
            self.save_checkpoint(self.json_data['NN']['fname_pkl'], epoch, training, valid)
        return self.best_residual

    def save_checkpoint(self, path, epoch, training, valid):
        if Path(path).resolve() != self.owned_output:
            raise ValueError('MVE may save only to the new output owned by this run')
        best_params = self.best_params if self.best_params is not None else self.model.state_dict()
        best_measured = self.best_measured if self.best_measured is not None else self.measured_adaptive
        best_step = self.best_validation_step if self.best_params is not None else self.validation_step
        best_epoch = self.best_epoch if self.best_params is not None else epoch
        manifest = {'schema': 'bam_torch.equivariant_mve.v2', 'version': 2,
                    'config_digest': self.config_digest, 'parameter_policy': 'raw-best',
                    'basis': 'e3nn-0e+2e', 'units': MVE_UNITS, 'floors': self.floors,
                    'stats_digest': stats_digest(self.train_stats)}
        torch.save({'manifest': manifest, 'input.json': deepcopy(self.canonical_config),
                    'params': best_params,
                    'uniq_element': deepcopy(self.uniq_element),
                    'enr_avg_per_element': deepcopy(self.enr_avg_per_element),
                    'train_stats': deepcopy(self.train_stats),
                    'adaptive': {'measured': deepcopy(best_measured),
                                 'next_train': deepcopy(self.next_adaptive),
                                 'validation_step': self.validation_step,
                                 'measured_validation_step': best_step},
                    'training_state': {'epoch': epoch, 'best_residual': self.best_residual,
                                       'best_epoch': best_epoch,
                                       'latest_params': self.model.state_dict(),
                                       'latest_measured': deepcopy(self.measured_adaptive),
                                       'optimizer': self.optimizer.state_dict(),
                                       'scheduler': self.scheduler.state_dict(),
                                       'torch_rng': torch.get_rng_state(),
                                       'cuda_rng': torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None,
                                       'numpy_rng': np.random.get_state()[1].tolist(),
                                       'numpy_rng_kind': np.random.get_state()[0],
                                       'numpy_rng_pos': int(np.random.get_state()[2]),
                                       'numpy_rng_gauss': int(np.random.get_state()[3]),
                                       'numpy_rng_cached': float(np.random.get_state()[4])},
                    'loss': {'epoch': epoch, 'train': training, 'valid': valid}}, path)

    def restore(self, path):
        checkpoint = load_mve_checkpoint(path)
        if _contract(checkpoint['input.json']) != _contract(self.canonical_config):
            raise ValueError('MVE restart canonical physical/model/statistics contract mismatch')
        if self.json_data['NN']['nepoch'] <= checkpoint['training_state']['epoch']:
            raise ValueError('MVE restart requires a larger total epoch horizon')
        if checkpoint['train_stats'] != self.train_stats:
            raise ValueError('MVE restart train source/split/statistics provenance mismatch')
        state = checkpoint['training_state']
        self.model.load_state_dict(state['latest_params'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.epoch, self.best_residual = state['epoch'], state['best_residual']
        self.measured_adaptive = deepcopy(state['latest_measured'])
        self.next_adaptive = deepcopy(checkpoint['adaptive']['next_train'])
        self.validation_step = checkpoint['adaptive']['validation_step']
        self.best_params = deepcopy(checkpoint['params'])
        self.best_measured = deepcopy(checkpoint['adaptive']['measured'])
        self.best_validation_step = checkpoint['adaptive']['measured_validation_step']
        self.best_epoch = state['best_epoch']
        torch.set_rng_state(state['torch_rng'])
        if self.device.type == 'cuda':
            if not isinstance(state['cuda_rng'], torch.Tensor):
                raise ValueError('MVE CUDA restart requires a saved CUDA RNG state')
            torch.cuda.set_rng_state(state['cuda_rng'], self.device)
        np.random.set_state((state['numpy_rng_kind'], np.asarray(state['numpy_rng'], dtype=np.uint32),
                             state['numpy_rng_pos'], state['numpy_rng_gauss'], state['numpy_rng_cached']))


def load_mve_checkpoint(path, device='cpu'):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError('MVE requires a safe weights-only checkpoint; legacy/executable pickle rejected') from exc
    if not isinstance(checkpoint, dict):
        raise ValueError('MVE checkpoint must be a mapping')
    manifest = checkpoint.get('manifest')
    if not isinstance(manifest, dict) or manifest.get('schema') != 'bam_torch.equivariant_mve.v2':
        raise ValueError('MVE checkpoint manifest missing or legacy; no migration supported')
    config = checkpoint.get('input.json')
    if not isinstance(config, dict):
        raise ValueError('MVE checkpoint requires canonical input.json')
    validate_mve_config(config)
    stats = checkpoint.get('train_stats')
    if (not isinstance(stats, dict) or stats.get('schema') != STATS_SCHEMA or
        stats.get('version') != 1 or stats.get('units') != MVE_UNITS or
        stats.get('normalization_enabled') != config['mve'].get('target_norm', {'enabled': False})['enabled'] or
        any(not isinstance(stats.get(key), str) or len(stats[key]) != 64 or
            any(char not in '0123456789abcdef' for char in stats[key])
            for key in ('split_digest', 'source_digest')) or
        not isinstance(stats.get('scales'), dict) or set(stats['scales']) != set(TASKS) or
        any(isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value <= 0 for value in stats['scales'].values())):
        raise ValueError('MVE checkpoint train statistics/provenance invalid')
    expected = {'schema': 'bam_torch.equivariant_mve.v2', 'version': 2,
                'config_digest': mve_config_digest(config), 'parameter_policy': 'raw-best',
                'basis': 'e3nn-0e+2e', 'units': MVE_UNITS, 'floors': config['mve']['floors'],
                'stats_digest': stats_digest(stats)}
    if manifest != expected or any(key not in checkpoint for key in
        ('params', 'uniq_element', 'enr_avg_per_element', 'train_stats', 'adaptive', 'training_state')):
        raise ValueError('MVE checkpoint manifest/config/raw policy or state mismatch')
    if (stats.get('species') != checkpoint['uniq_element'] or
        stats.get('references') != checkpoint['enr_avg_per_element'] or
        stats.get('units') != MVE_UNITS):
        raise ValueError('MVE checkpoint statistics/reference mismatch')
    state = checkpoint['training_state']
    if (not isinstance(state, dict) or
        any(key not in state for key in ('epoch', 'best_epoch', 'best_residual', 'latest_params',
            'latest_measured', 'optimizer', 'scheduler', 'torch_rng', 'cuda_rng', 'numpy_rng',
            'numpy_rng_kind', 'numpy_rng_pos', 'numpy_rng_gauss', 'numpy_rng_cached')) or
        type(state['epoch']) is not int or type(state['best_epoch']) is not int or
        not 0 <= state['best_epoch'] <= state['epoch'] or
        not isinstance(state['best_residual'], (int, float)) or
        math.isnan(state['best_residual']) or
        not isinstance(checkpoint['params'], dict) or
        not isinstance(state['latest_params'], dict) or
        (config['device'] == 'cuda' and not isinstance(state['cuda_rng'], torch.Tensor)) or
        (config['device'] == 'cpu' and state['cuda_rng'] is not None)):
        raise ValueError('MVE checkpoint optimizer/selection/restart state invalid')
    adaptive = checkpoint['adaptive']
    epsilon = config['mve'].get('epsilon', {'enabled': False})
    if (not isinstance(adaptive, dict) or
        set(adaptive) != {'measured', 'next_train', 'validation_step', 'measured_validation_step'} or
        type(adaptive['validation_step']) is not int or adaptive['validation_step'] < 0 or
        type(adaptive['measured_validation_step']) is not int or
        not 0 <= adaptive['measured_validation_step'] <= adaptive['validation_step']):
        raise ValueError('MVE checkpoint validation floor state invalid')
    # A caller may run an explicit validation before the epoch loop; steps
    # then legitimately exceed completed epochs, including manual saves.
    if adaptive['measured_validation_step'] < state['best_epoch'] or adaptive['validation_step'] < state['epoch']:
        raise ValueError('MVE checkpoint epoch and measured floor steps disagree')
    for phase, values in (('measured', adaptive['measured']),
                          ('next_train', adaptive['next_train']),
                          ('latest_measured', state['latest_measured'])):
        if not isinstance(values, dict) or set(values) != set(TASKS):
            raise ValueError('MVE checkpoint adaptive floor keys invalid')
        for key, value in values.items():
            minimum = epsilon['min'][key] if epsilon['enabled'] else 0.0
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < minimum or (not epsilon['enabled'] and value != 0)):
                raise ValueError('MVE checkpoint adaptive floor value invalid')
    return checkpoint
