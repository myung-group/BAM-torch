import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from torch_geometric.loader import DataLoader

from test_equivariant_mve_e2e import ROOT, config, fixture
from bam_torch.predicting.evaluator import Evaluator
from bam_torch.training.base_trainer import mve_config_digest, validate_mve_config
from bam_torch.training.loss import (
    MVE_FLOORS, assemble_effective_covariances, detached_beta_weights,
    gaussian_observation_nll, stress_voigt_to_irrep,
)
from bam_torch.training.mve_trainer import MVETrainer, load_mve_checkpoint, mve_graphs
from bam_torch.training.target_norm import fit_train_stats


class PhaseTwoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def two_atoms():
        atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]], cell=[8, 8, 8], pbc=True)
        atoms.calc = SinglePointCalculator(atoms, energy=-.4,
                                           forces=np.array([[.1, 0, 0], [-.1, 0, 0]]),
                                           stress=np.array([.01, .02, .03, 0, 0, 0]))
        return atoms

    def test_serial_cuda_contract_and_distinct_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            train_path, valid_path = directory/'train.traj', directory/'valid.traj'
            write(train_path, [fixture(), self.two_atoms()], format='traj')
            validation = fixture(zero=True)
            labels = dict(validation.calc.results)
            validation.positions[2, 1] = 1.25
            validation.calc = SinglePointCalculator(validation, **labels)
            write(valid_path, [validation], format='traj')
            settings = config('unused.xyz', directory/'unused.pt')
            settings.pop('fname_traj')
            settings.pop('predict')
            settings.update(fname_train=str(train_path), fname_valid=str(valid_path), ntrain=2, nvalid=1)
            trainer = MVETrainer(settings)
            self.assertEqual(trainer.train_ids, [0, 1])
            self.assertEqual([int(graph.frame_id[0]) for graph in trainer.train_loader.dataset], [0, 1])
            self.assertEqual([int(graph.frame_id[0]) for graph in trainer.valid_loader.dataset], [0])
            self.assertEqual(sorted(trainer.train_stats['species']), [1])
            for changes in ({'fname_traj': str(train_path)}, {'fname_valid': str(train_path)},
                            {'fname_valid': ''}, {'nvalid': 2}):
                invalid = copy.deepcopy(settings)
                invalid.update(changes)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    MVETrainer(invalid)
            write(valid_path, [fixture()], format='traj')
            with self.assertRaisesRegex(ValueError, 'overlapping geometries'):
                MVETrainer(settings)
            cuda = copy.deepcopy(settings)
            cuda['device'] = 'cuda'
            validate_mve_config(cuda)
            with patch('torch.cuda.is_available', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'no CPU fallback'):
                    MVETrainer(cuda)
            for change in ({'gpu-parallel': True}, {'dens': {'enabled': True}},
                           {'device': 'gpu'}, {'cueq_config': True}, {'oeq_config': True}):
                bad = copy.deepcopy(cuda)
                bad.update(change)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    validate_mve_config(bad)

    def test_cpu_prediction_reads_cuda_canonical_checkpoint_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            trajectory = directory/'data.extxyz'
            prediction = directory/'geometry.extxyz'
            checkpoint = directory/'cpu.pt'
            write(trajectory, [fixture(), fixture(zero=True)], format='extxyz')
            write(prediction, [self.two_atoms()], format='extxyz')
            trainer = MVETrainer(config(trajectory, checkpoint))
            trainer.train()
            saved = load_mve_checkpoint(checkpoint)
            saved['input.json']['device'] = 'cuda'
            saved['manifest']['config_digest'] = mve_config_digest(saved['input.json'])
            saved['training_state']['cuda_rng'] = torch.get_rng_state().clone()
            encoded = directory/'cuda-canonical.pt'
            torch.save(saved, encoded)
            # This exercises safe CPU reconstruction of a CUDA configuration, not CUDA execution.
            with patch('torch.cuda.is_available', side_effect=AssertionError('CPU inference queried CUDA')):
                result = Evaluator({'trainer': 'mve', 'predict': {'model': str(encoded),
                    'fname_traj': str(prediction), 'uncertainty_output': str(directory/'out.json')}}).evaluate()
            self.assertEqual(result['config_digest'], saved['manifest']['config_digest'])
            self.assertTrue(np.isfinite(result['predictions'][0]['effective']['energy']).all())

    def test_train_only_stats_and_external_provenance(self):
        frames = [fixture(), fixture(zero=True)]
        stats = fit_train_stats([frames[0]], [17], True)
        changed_valid = fixture(zero=True)
        changed_valid.calc.results['energy'] = 999.
        original_valid = frames[1]
        frames[1] = changed_valid
        self.assertEqual(stats, fit_train_stats([frames[0]], [17], True))
        frames[1] = original_valid
        self.assertTrue(all(value > 0 for value in stats['scales'].values()))
        missing = fixture(missing=('stress', 'forces'))
        zeros = fixture(zero=True)
        eligible = fit_train_stats([missing, zeros], [1, 2], True)
        self.assertEqual(eligible['scales']['stress'], 1e-12)
        self.assertEqual(eligible['scales']['forces'], 1e-12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stats.json'
            path.write_text(json.dumps(stats))
            self.assertEqual(fit_train_stats([frames[0]], [17], True, path), stats)
            for source, ids in (([frames[0]], [18]), ([frames[0], changed_valid], [17, 18])):
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    fit_train_stats(source, ids, True, path)
            corrupted = copy.deepcopy(stats)
            corrupted['scales']['energy'] = -1
            path.write_text(json.dumps(corrupted))
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                fit_train_stats([frames[0]], [17], True, path)
            ordered = fit_train_stats([frames[0], frames[1]], [17, 18], True)
            path.write_text(json.dumps(ordered))
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                fit_train_stats([frames[1], frames[0]], [18, 17], True, path)
            changed_pbc = frames[0].copy()
            changed_pbc.set_pbc([False, True, True])
            changed_pbc.calc = SinglePointCalculator(changed_pbc, **frames[0].calc.results)
            path.write_text(json.dumps(stats))
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                fit_train_stats([changed_pbc], [17], True, path)

    def test_validation_label_perturbation_does_not_change_fitted_state(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            frames = [fixture(), fixture(zero=True), fixture()]
            original = directory / 'original.extxyz'
            changed = directory / 'changed.extxyz'
            write(original, frames, format='extxyz')
            train_ids = torch.randperm(3, generator=torch.Generator().manual_seed(2)).tolist()[:2]
            valid_id = next(i for i in range(3) if i not in train_ids)
            perturbed = copy.deepcopy(frames)
            perturbed[valid_id].calc.results['energy'] = 987.0
            perturbed[valid_id].calc.results['forces'][:] = 58.0
            write(changed, perturbed, format='extxyz')
            first = config(original, directory / 'first.pt')
            second = config(changed, directory / 'second.pt')
            first.update(ntrain=2, nvalid=1)
            second.update(ntrain=2, nvalid=1)
            first['mve']['target_norm'] = second['mve']['target_norm'] = {'enabled': True}
            self.assertEqual(MVETrainer(first).train_stats, MVETrainer(second).train_stats)

    def test_fixed_plus_adaptive_once_and_detached_beta(self):
        raw = {'energy': torch.zeros(2, dtype=torch.float64),
               'forces': torch.zeros(2, 3, 3, dtype=torch.float64),
               'stress': torch.zeros(2, 6, 6, dtype=torch.float64)}
        scales = {'energy': 2., 'forces': 3., 'stress': 5.}
        atoms = torch.tensor([2., 4.], dtype=torch.float64)
        adaptive = {'energy': .2, 'forces': .3, 'stress': .4}
        fixed = assemble_effective_covariances(raw, MVE_FLOORS)
        effective = assemble_effective_covariances(raw, MVE_FLOORS, adaptive=adaptive,
                                                   scales=scales, atom_counts=atoms)
        torch.testing.assert_close(effective['energy'] - fixed['energy'], .2 * (atoms * 2).square())
        torch.testing.assert_close(effective['forces'] - fixed['forces'],
                                   (.3 * 9 * torch.eye(3, dtype=torch.float64)).expand(2, -1, -1))
        torch.testing.assert_close(effective['stress'] - fixed['stress'],
                                   (.4 * 25 * torch.eye(6, dtype=torch.float64)).expand(2, -1, -1))
        self.assertTrue((fixed['energy'] > 0).all())
        covariance = torch.diag_embed(torch.tensor([[1.,1.,1.],[9.,9.,9.]], requires_grad=True))
        weight = detached_beta_weights(covariance, .5)
        torch.testing.assert_close(weight, torch.tensor([.5, 1.5]))
        self.assertFalse(weight.requires_grad)
        torch.testing.assert_close(detached_beta_weights(covariance, 0), torch.ones(2))
        energy = torch.tensor([4., 64.], requires_grad=True)
        expected = torch.tensor([4. / 4., 64. / 16.])
        torch.testing.assert_close(detached_beta_weights(energy, 1, atom_counts=torch.tensor([2.,4.])),
                                   expected / expected.mean())
        self.assertFalse(detached_beta_weights(energy, 1, atom_counts=torch.tensor([2.,4.])).requires_grad)
        with self.assertRaisesRegex(ValueError, 'positive graph atom counts'):
            detached_beta_weights(energy, 1)

    def test_mask_and_original_eligibility(self):
        high = fixture()
        high.calc.results['stress'] = np.array([.4, 0, 0, 0, 0, 0])
        zeros = fixture(zero=True)
        absent = fixture(missing=('stress',))
        graphs = mve_graphs([high, zeros, absent], 4, {1:0}, {0:0.}, [7, 8, 9], 50)
        self.assertEqual([int(g.frame_id[0]) for g in graphs], [7, 8, 9])
        self.assertEqual([bool(g.stress_original_valid[0]) for g in graphs], [True, True, False])
        self.assertEqual([bool(g.stress_valid[0]) for g in graphs], [False, True, False])
        self.assertTrue(all(bool(g.energy_valid[0]) for g in graphs))
        self.assertTrue(all(g.force_valid.all() for g in graphs))

    def test_all_masked_stress_preserves_energy_force_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            frames = [fixture(), fixture(zero=True)]
            for frame in frames:
                frame.calc.results['stress'] = np.array([.5, 0, 0, 0, 0, 0])
            trajectory = directory/'masked.extxyz'
            write(trajectory, frames, format='extxyz')
            settings = config(trajectory, directory/'masked.pt')
            settings['mve']['mask'] = {'enabled': True, 'max_gpa': 50}
            trainer = MVETrainer(settings)
            train = trainer.train_one_epoch('train')
            valid = trainer.train_one_epoch('valid')
            for metrics in (train, valid):
                self.assertEqual(metrics['counts'], {'e': 1, 'f': 3, 's': 0})
                self.assertEqual(metrics['count_s_unmasked'], 1)
                self.assertEqual(metrics['count_s_masked_out'], 1)
                self.assertIsNone(metrics['loss_s'])
                self.assertIsNone(metrics['whitened_mse']['stress'])

    def test_beta_affects_only_train_nll_not_validation_or_residual_controller(self):
        trainer = object.__new__(MVETrainer)
        trainer.floors = dict(MVE_FLOORS)
        trainer.train_stats = {'scales': {key: 1.0 for key in MVE_FLOORS}}
        trainer.next_adaptive = {key: 0.0 for key in MVE_FLOORS}
        trainer.json_data = {'mve': {'beta': 1, 'target_norm': {'enabled': False}},
                             'NN': {'enr_lambda': 1, 'frc_lambda': 1, 'str_lambda': 1}}
        preds = {'energy': torch.tensor([.2, .5], requires_grad=True),
                 'energy_var': torch.tensor([1., 9.], requires_grad=True),
                 'forces': torch.tensor([[.2,0,0],[.7,0,0]], requires_grad=True),
                 'forces_var': torch.diag_embed(torch.tensor([[1.,1.,1.],[9.,9.,9.]], requires_grad=True)),
                 'stress': torch.tensor([[.2,0,0,0,0,0],[.7,0,0,0,0,0]], requires_grad=True),
                 'stress_var': torch.diag_embed(torch.tensor([[1.]*6,[9.]*6], requires_grad=True))}
        labels = {'batch': torch.tensor([0, 1]), 'energy': torch.zeros(2),
                  'forces': torch.zeros(2,3), 'stress': torch.zeros(2,6),
                  'energy_valid': torch.ones(2, dtype=torch.bool),
                  'force_valid': torch.ones(2, dtype=torch.bool),
                  'stress_original_valid': torch.ones(2, dtype=torch.bool),
                  'stress_valid': torch.ones(2, dtype=torch.bool)}
        train = trainer.compute_loss(preds, labels, training=True)
        valid = trainer.compute_loss(preds, labels, training=False)
        self.assertNotAlmostEqual(float(train['loss'].detach()), float(valid['loss'].detach()), places=5)
        torch.testing.assert_close(train['residual_selection'], valid['residual_selection'])
        torch.testing.assert_close(train['whitened_mse']['energy'], valid['whitened_mse']['energy'])
        train['loss'].backward()
        self.assertTrue(torch.isfinite(preds['energy'].grad).all())
        trainer.json_data['mve']['beta'] = 0
        zero_beta = trainer.compute_loss(preds, labels, training=True)
        torch.testing.assert_close(zero_beta['loss'], valid['loss'])
        trainer.next_adaptive = {key: .5 for key in MVE_FLOORS}
        with_adaptive = trainer.compute_loss(preds, labels, training=False)
        torch.testing.assert_close(with_adaptive['residual_selection'], valid['residual_selection'])
        self.assertNotAlmostEqual(float(with_adaptive['loss'].detach()), float(valid['loss'].detach()), places=5)

    def test_energy_size_exponent_and_max_logvar_config(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            trajectory = directory/'train.extxyz'
            write(trajectory, [fixture(), fixture(zero=True)], format='extxyz')
            settings = config(trajectory, directory/'model.pt')
            validate_mve_config(settings)
            for bad in ({'energy_size_exponent': {'init': 1.0}},
                        {'energy_size_exponent': {'init': True, 'learn': True}},
                        {'energy_size_exponent': {'init': float('nan'), 'learn': True}},
                        {'energy_size_exponent': {'init': 1.0, 'learn': 1}},
                        {'max_logvar': 0}, {'max_logvar': True}, {'max_logvar': float('inf')}):
                broken = copy.deepcopy(settings)
                broken['mve'].update(bad)
                with self.assertRaises(ValueError):
                    validate_mve_config(broken)
            for key, value in (('max_logvar', 12.0), ('logvar_n_alpha_init', 1.0),
                               ('logvar_n_alpha_learn', True)):
                for nested in (False, True):
                    aliased = copy.deepcopy(settings)
                    (aliased['NN'] if nested else aliased)[key] = value
                    with self.assertRaisesRegex(ValueError, 'canonical mve'):
                        validate_mve_config(aliased)
            default = MVETrainer(copy.deepcopy(settings))
            self.assertEqual(float(default.model.logvar_n_alpha.detach()), 1.0)
            self.assertTrue(default.model.logvar_n_alpha.requires_grad)
            self.assertEqual(default.model.max_logvar, 8.0)
            settings['NN']['fname_pkl'] = str(directory/'other.pt')
            settings['mve'].update(energy_size_exponent={'init': 0.5, 'learn': False}, max_logvar=12.0)
            custom = MVETrainer(settings)
            self.assertEqual(float(custom.model.logvar_n_alpha.detach()), 0.5)
            self.assertFalse(custom.model.logvar_n_alpha.requires_grad)
            self.assertEqual(custom.model.max_logvar, 12.0)

    def test_selection_uses_lambda_weighted_whitened_not_physical_mse(self):
        trainer = object.__new__(MVETrainer)
        trainer.floors = dict(MVE_FLOORS)
        trainer.train_stats = {'scales': {'energy': 2., 'forces': 4., 'stress': 5.}}
        trainer.next_adaptive = {key: 0. for key in MVE_FLOORS}
        trainer.json_data = {'mve': {'target_norm': {'enabled': True}, 'beta': 0},
                             'NN': {'enr_lambda': 2, 'frc_lambda': 3, 'str_lambda': 5}}
        q = stress_voigt_to_irrep()
        target = {'batch': torch.tensor([0, 0, 1, 1, 1]),
                  'energy': torch.tensor([8., 30.]),
                  'forces': torch.tensor([[4., 0, 0]] * 5),
                  'stress': torch.tensor([[5., 0, 0, 0, 0, 0], [0, 0, 0, 0, 5., 0]]),
                  'energy_valid': torch.tensor([True, True]),
                  'force_valid': torch.ones(5, dtype=torch.bool),
                  'stress_original_valid': torch.tensor([True, True]),
                  'stress_valid': torch.tensor([True, True])}
        preds = {'energy': torch.zeros(2), 'forces': torch.zeros(5, 3),
                 'stress': torch.zeros(2, 6), 'energy_var': torch.tensor([1., 3.]),
                 'forces_var': torch.eye(3).expand(5, -1, -1).clone(),
                 'stress_var': torch.eye(6).expand(2, -1, -1).clone()}
        valid = trainer.compute_loss(preds, target, training=False)
        whitened = (((torch.tensor([8., 30.]) / torch.tensor([2., 3.]) / 2.) ** 2).mean(),
                    ((target['forces'] / 4.) ** 2).mean(),
                    ((target['stress'] @ q.T / 5.) ** 2).mean())
        physical = ((torch.tensor([8., 30.])) ** 2).mean(), (target['forces'] ** 2).mean(), \
            ((target['stress'] @ q.T) ** 2).mean()
        torch.testing.assert_close(valid['residual_selection'],
                                   2 * whitened[0] + 3 * whitened[1] + 5 * whitened[2])
        for key, expected in zip(MVE_FLOORS, physical):
            torch.testing.assert_close(valid['physical_mse'][key], expected)
        self.assertGreater(abs(float(valid['residual_selection']) -
                               float(2 * physical[0] + 3 * physical[1] + 5 * physical[2])), 1.0)

    def test_tracking_uses_data_scales_and_validation_physical_nll(self):
        trainer = object.__new__(MVETrainer)
        trainer.floors = dict(MVE_FLOORS)
        trainer.train_stats = {'scales': {'energy': 2., 'forces': 4., 'stress': 5.}}
        trainer.next_adaptive = {key: 0. for key in MVE_FLOORS}
        trainer.measured_adaptive = dict(trainer.next_adaptive)
        trainer.json_data = {'mve': {'target_norm': {'enabled': True}, 'beta': 0,
            'epsilon': {'enabled': True, 'c': {key: .5 for key in MVE_FLOORS},
                        'min': {key: .01 for key in MVE_FLOORS}}}, 'NN': {}}
        q = stress_voigt_to_irrep()
        target = {'batch': torch.tensor([0, 0, 1, 1, 1]),
                  'energy': torch.tensor([8., 30.]),
                  'forces': torch.tensor([[4., 0, 0]] * 5),
                  'stress': torch.tensor([[5., 0, 0, 0, 0, 0], [0, 0, 0, 0, 5., 0]]),
                  'energy_valid': torch.tensor([True, True]),
                  'force_valid': torch.ones(5, dtype=torch.bool),
                  'stress_original_valid': torch.tensor([True, True]),
                  'stress_valid': torch.tensor([True, True])}
        preds = {'energy': torch.zeros(2), 'forces': torch.zeros(5, 3),
                 'stress': torch.zeros(2, 6), 'energy_var': torch.tensor([1., 3.]),
                 'forces_var': torch.eye(3).expand(5, -1, -1).clone(),
                 'stress_var': torch.eye(6).expand(2, -1, -1).clone()}
        valid = trainer.compute_loss(preds, target, training=False)
        raw = assemble_effective_covariances({'energy': preds['energy_var'],
                 'forces': preds['forces_var'], 'stress': preds['stress_var']}, trainer.floors)
        expected_e = gaussian_observation_nll(target['energy'][:, None], raw['energy'][:, None, None]).mean()
        expected_f = gaussian_observation_nll(target['forces'], raw['forces']).mean()
        expected_s = gaussian_observation_nll(target['stress'] @ q.T, raw['stress']).mean()
        for short, expected in zip('efs', (expected_e, expected_f, expected_s)):
            torch.testing.assert_close(valid['loss_' + short], expected)
        train = trainer.compute_loss(preds, target, training=True)
        self.assertNotAlmostEqual(float(train['loss_e']), float(valid['loss_e']))
        expected_e_mse = ((torch.tensor([8., 30.]) / torch.tensor([2., 3.]) / 2.) ** 2).mean()
        expected_f_mse = ((target['forces'] / 4.) ** 2).mean()
        expected_s_mse = ((target['stress'] @ q.T / 5.) ** 2).mean()
        for key, expected in zip(MVE_FLOORS, (expected_e_mse, expected_f_mse, expected_s_mse)):
            torch.testing.assert_close(valid['whitened_mse'][key], expected)
        baseline = {key: float(valid['whitened_mse'][key]) for key in MVE_FLOORS}
        measured = {'whitened_mse': baseline, 'count_e': 2, 'count_f': 5, 'count_s': 2}
        trainer._update_epsilon(measured)
        expected_next = dict(trainer.next_adaptive)
        changed = {**preds, 'energy_var': preds['energy_var'] * 30,
                   'forces_var': preds['forces_var'] * 40,
                   'stress_var': preds['stress_var'] * 50}
        trainer.next_adaptive = {key: 1. for key in MVE_FLOORS}
        changed_valid = trainer.compute_loss(changed, target, training=False)
        for key in MVE_FLOORS:
            torch.testing.assert_close(changed_valid['whitened_mse'][key], valid['whitened_mse'][key])
        torch.testing.assert_close(changed_valid['residual_selection'], valid['residual_selection'])
        trainer._update_epsilon({'whitened_mse': {key: float(changed_valid['whitened_mse'][key])
                                                   for key in MVE_FLOORS}, **{f'count_{short}': n
                                                   for short, n in zip('efs', (2, 5, 2))}})
        self.assertEqual(trainer.next_adaptive, expected_next)
        trainer.json_data['mve']['target_norm']['enabled'] = False
        disabled = trainer.compute_loss(preds, target, training=False)
        torch.testing.assert_close(disabled['whitened_mse']['energy'],
            ((target['energy'] / torch.tensor([2., 3.])) ** 2).mean())
        torch.testing.assert_close(disabled['whitened_mse']['forces'], target['forces'].square().mean())
        torch.testing.assert_close(disabled['whitened_mse']['stress'], (target['stress'] @ q.T).square().mean())

    def test_disabled_options_match_phase_one_adam_per_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            trajectory = directory / 'multibatch.extxyz'
            frames = [fixture(), fixture(zero=True), fixture(), fixture()]
            selected_id = torch.randperm(4, generator=torch.Generator().manual_seed(2)).tolist()[0]
            frames[selected_id] = self.two_atoms()
            write(trajectory, frames, format='extxyz')
            setting = config(trajectory, directory / 'unused.pt')
            setting.update(ntrain=3, nvalid=1, nbatch=1)
            actual = MVETrainer(setting)
            reference_setting = copy.deepcopy(setting)
            reference_setting['NN']['fname_pkl'] = str(directory / 'reference.pt')
            reference = MVETrainer(reference_setting)
            self.assertEqual(len(actual.train_loader), 3)
            self.assertEqual(sorted(len(graph.force_valid) for graph in actual.train_loader.dataset), [2, 3, 3])
            with patch.object(actual.optimizer, 'step', wraps=actual.optimizer.step) as steps:
                metrics = actual.train_one_epoch('train')
            self.assertEqual(steps.call_count, 3)
            self.assertEqual(metrics['counts'], {'e': 3, 'f': 8, 's': 3})
            q = stress_voigt_to_irrep()
            for data in reference.train_loader:
                prediction = reference.model(data, backprop=True)
                effective = assemble_effective_covariances({key: prediction[key + '_var']
                                    for key in ('energy', 'forces', 'stress')}, reference.floors)
                residual_e = (data.energy.reshape(-1) - prediction['energy'].reshape(-1))[:, None]
                residual_f = data.forces - prediction['forces']
                residual_s = (data.stress - prediction['stress']) @ q.T
                objective = (gaussian_observation_nll(residual_e, effective['energy'][:, None, None]).mean() +
                             gaussian_observation_nll(residual_f, effective['forces']).mean() +
                             gaussian_observation_nll(residual_s, effective['stress']).mean())
                reference.optimizer.zero_grad()
                objective.backward()
                reference.optimizer.step()
            for name, value in actual.model.state_dict().items():
                torch.testing.assert_close(value, reference.model.state_dict()[name], rtol=0, atol=0)
            for index, state in actual.optimizer.state_dict()['state'].items():
                counterpart = reference.optimizer.state_dict()['state'][index]
                self.assertEqual(int(state['step']), 3)
                for key in ('exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(state[key], counterpart[key], rtol=0, atol=0)

    def test_fully_unlabelled_batch_never_steps_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            trajectory = Path(directory)/'minimal.extxyz'
            write(trajectory, [fixture(), fixture(zero=True)], format='extxyz')
            trainer = MVETrainer(config(trajectory, Path(directory)/'result.pt'))
            unlabeled = mve_graphs([fixture(missing=('energy', 'forces', 'stress'))],
                                   4, trainer.uniq_element, trainer.enr_avg_per_element)[0]
            labelled = trainer.train_loader.dataset[0]
            loader = DataLoader([unlabeled, labelled], batch_size=1, shuffle=False)
            with patch.object(trainer.optimizer, 'step', wraps=trainer.optimizer.step) as steps:
                result = trainer.train_one_epoch('train', loader)
            self.assertEqual(steps.call_count, 1)
            self.assertEqual(result['counts'], {'e': 1, 'f': 3, 's': 1})
            with patch.object(trainer.optimizer, 'step', wraps=trainer.optimizer.step) as steps:
                with self.assertRaisesRegex(ValueError, 'no eligible'):
                    trainer.train_one_epoch('train', DataLoader([unlabeled], batch_size=1))
            self.assertEqual(steps.call_count, 0)

    def test_config_canonical_and_epsilon_zero_count(self):
        with tempfile.TemporaryDirectory() as directory:
            c = config('unused.extxyz', Path(directory) / 'out.pt')
            for key in ('target_norm', 'beta', 'epsilon', 'mask'):
                bad = copy.deepcopy(c)
                bad[key] = True
                with self.assertRaisesRegex(ValueError, 'canonical'):
                    validate_mve_config(bad)
            c['mve']['epsilon'] = {'enabled': True,
                'c': {key: .5 for key in MVE_FLOORS},
                'min': {key: .01 for key in MVE_FLOORS}}
            validate_mve_config(c)
            trainer = object.__new__(MVETrainer)
            trainer.json_data = c
            trainer.measured_adaptive = {key: .1 for key in MVE_FLOORS}
            trainer.next_adaptive = dict(trainer.measured_adaptive)
            trainer._update_epsilon({'whitened_mse': {'energy': 3., 'forces': .5, 'stress': None},
                                     'count_e': 2, 'count_f': 3, 'count_s': 0})
            self.assertAlmostEqual(trainer.next_adaptive['energy'], 1.5)
            self.assertAlmostEqual(trainer.next_adaptive['forces'], .25)
            self.assertEqual(trainer.next_adaptive['stress'], .1)

    def test_best_snapshot_uses_its_own_measured_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            trajectory = directory/'tiny.extxyz'
            write(trajectory, [fixture(), fixture(zero=True)], format='extxyz')
            c = config(trajectory, directory/'best.pt')
            c['NN']['nepoch'] = 2
            c['mve']['epsilon'] = {'enabled': True,
                                  'c': {key: .5 for key in MVE_FLOORS},
                                  'min': {key: .01 for key in MVE_FLOORS}}
            trainer = MVETrainer(c)
            validation = [1., 2.]
            def controlled_epoch(mode):
                if mode == 'train':
                    with torch.no_grad():
                        trainer.model.readouts[0].bias.add_(1.)
                    return {'loss': 0.}
                trainer.measured_adaptive = copy.deepcopy(trainer.next_adaptive)
                trainer.validation_step += 1
                return {'residual_selection': validation.pop(0),
                        'whitened_mse': {key: 3. for key in MVE_FLOORS},
                        'count_e': 1, 'count_f': 3, 'count_s': 1}
            with patch.object(trainer, 'train_one_epoch', side_effect=controlled_epoch):
                trainer.train()
            saved = load_mve_checkpoint(c['NN']['fname_pkl'])
            self.assertEqual(saved['training_state']['best_epoch'], 1)
            self.assertEqual(saved['adaptive']['measured_validation_step'], 1)
            self.assertEqual(saved['adaptive']['validation_step'], 2)
            self.assertEqual(saved['adaptive']['measured']['energy'], .01)
            self.assertEqual(saved['training_state']['latest_measured']['energy'], 1.5)
            self.assertEqual(saved['adaptive']['next_train']['energy'], 1.5)
            torch.testing.assert_close(saved['training_state']['latest_params']['readouts.0.bias'],
                                       saved['params']['readouts.0.bias'] + 1)

    def test_restart_and_measured_export(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            trajectory = directory / 'train.extxyz'
            prediction = directory / 'predict.extxyz'
            original_frames = [fixture(), self.two_atoms(), fixture(zero=True), fixture()]
            write(trajectory, original_frames, format='extxyz')
            write(prediction, [fixture(missing=('energy', 'forces', 'stress'))], format='extxyz')
            opts = {'target_norm': {'enabled': True}, 'beta': .5,
                    'epsilon': {'enabled': True, 'c': {k: .25 for k in MVE_FLOORS},
                                'min': {k: .01 for k in MVE_FLOORS}},
                    'mask': {'enabled': True, 'max_gpa': 50},
                    'controller': {'factor': .5, 'patience': 0, 'min_lr': 1e-8}}
            full_cfg = config(trajectory, directory/'full.pt')
            full_cfg.update(ntrain=3, nvalid=1)
            full_cfg['NN']['nepoch'] = 3
            full_cfg['mve'].update(opts)
            full = MVETrainer(full_cfg)
            full.train()
            partial_cfg = copy.deepcopy(full_cfg)
            partial_cfg['NN'].update(nepoch=1, fname_pkl=str(directory/'partial.pt'))
            partial = MVETrainer(partial_cfg)
            partial.train()
            resumed_cfg = copy.deepcopy(full_cfg)
            resumed_cfg['NN']['fname_pkl'] = str(directory/'resumed.pt')
            resumed = MVETrainer(resumed_cfg, restart_from=partial_cfg['NN']['fname_pkl'])
            resumed.train()
            a = load_mve_checkpoint(full_cfg['NN']['fname_pkl'])
            b = load_mve_checkpoint(resumed_cfg['NN']['fname_pkl'])
            for key in a['params']:
                torch.testing.assert_close(a['params'][key], b['params'][key], rtol=0, atol=0)
            self.assertEqual(a['training_state']['scheduler'], b['training_state']['scheduler'])
            self.assertEqual(a['training_state']['best_residual'], b['training_state']['best_residual'])
            self.assertEqual(a['adaptive'], b['adaptive'])
            self.assertEqual(a['training_state']['optimizer']['param_groups'],
                             b['training_state']['optimizer']['param_groups'])
            for parameter, first in a['training_state']['optimizer']['state'].items():
                second = b['training_state']['optimizer']['state'][parameter]
                for name, value in first.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, second[name], rtol=0, atol=0)
                    else:
                        self.assertEqual(value, second[name])
            torch.testing.assert_close(a['training_state']['torch_rng'],
                                       b['training_state']['torch_rng'], rtol=0, atol=0)
            self.assertEqual(a['adaptive']['validation_step'], 3)
            with self.assertRaisesRegex(ValueError, 'distinct'):
                MVETrainer(partial_cfg, restart_from=partial_cfg['NN']['fname_pkl'])
            invalid = copy.deepcopy(resumed_cfg)
            invalid['NN']['fname_pkl'] = str(directory/'bad.pt')
            invalid['mve']['floors']['forces'] *= 2
            with self.assertRaisesRegex(ValueError, 'canonical'):
                MVETrainer(invalid, restart_from=partial_cfg['NN']['fname_pkl'])
            modified_frames = copy.deepcopy(original_frames)
            modified_frames[partial.train_ids[0]].calc.results['energy'] = -22.0
            write(trajectory, modified_frames, format='extxyz')
            source_mismatch = copy.deepcopy(resumed_cfg)
            source_mismatch['NN']['fname_pkl'] = str(directory/'source-mismatch.pt')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                MVETrainer(source_mismatch, restart_from=partial_cfg['NN']['fname_pkl'])
            write(trajectory, original_frames, format='extxyz')
            pbc_frames = copy.deepcopy(original_frames)
            pbc_frames[partial.train_ids[0]].set_pbc([False, True, True])
            write(trajectory, pbc_frames, format='extxyz')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                MVETrainer(source_mismatch, restart_from=partial_cfg['NN']['fname_pkl'])
            corrupted_path = directory/'corrupted.pt'
            corrupted = copy.deepcopy(a)
            corrupted['train_stats']['scales']['energy'] *= 2
            torch.save(corrupted, corrupted_path)
            with self.assertRaisesRegex(ValueError, 'manifest'):
                load_mve_checkpoint(corrupted_path)
            corrupted = copy.deepcopy(a)
            corrupted['adaptive']['measured']['energy'] = -1
            torch.save(corrupted, corrupted_path)
            with self.assertRaisesRegex(ValueError, 'adaptive floor'):
                load_mve_checkpoint(corrupted_path)
            request = {'trainer': 'mve', 'predict': {'model': full_cfg['NN']['fname_pkl'],
                       'fname_traj': str(prediction), 'uncertainty_output': str(directory/'result.json')}}
            payload = Evaluator(request).evaluate()
            self.assertEqual(payload['adaptive']['measured'], a['adaptive']['measured'])
            self.assertNotEqual(payload['adaptive_floor_policy'], 'next-train')
            self.assertEqual(payload['train_stats_provenance']['source_digest'], a['train_stats']['source_digest'])
            records = payload['predictions'][0]
            self.assertEqual(len(records['effective']['energy']), 1)
            atom_count = 3
            expected = records['raw']['energy'][0] + full_cfg['mve']['floors']['energy'] + a['adaptive']['measured']['energy'] * (atom_count * a['train_stats']['scales']['energy']) ** 2
            self.assertAlmostEqual(records['effective']['energy'][0], expected, places=5)
            fresh_output = directory/'fresh-phase2.json'
            command = [sys.executable, '-S', str(ROOT/'examples/example-3BPA_300K-mve/evaluate.py'),
                       str(full_cfg['NN']['fname_pkl']), str(prediction), '--output', str(fresh_output)]
            proc = subprocess.run(command, cwd=directory, env=os.environ.copy(),
                                  capture_output=True, text=True, timeout=90)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            fresh = json.loads(fresh_output.read_text())
            self.assertEqual(fresh['adaptive'], payload['adaptive'])
            self.assertEqual(fresh['predictions'][0]['effective'], records['effective'])


if __name__ == '__main__':
    unittest.main()
