import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from bam_torch.predicting.evaluator import Evaluator
from bam_torch.training.base_trainer import BaseTrainer, validate_mve_config
from bam_torch.training.mve_trainer import MVETrainer, load_mve_checkpoint, mve_graphs
from bam_torch.training.loss import MVE_FLOORS, MVE_UNITS


ROOT = Path(__file__).resolve().parents[1]


class ExecutableLegacyValue:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.system, (f"printf executed > {shlex.quote(str(self.marker))}",)


def fixture(zero=False, missing=()):
    atoms = Atoms('H3', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], cell=[8, 8, 8], pbc=True)
    labels = {'energy': 0.0 if zero else -0.5,
              'forces': np.zeros((3, 3)) if zero else np.array([[.1,0,0],[-.1,0,0],[0,0,0]]),
              'stress': np.zeros(6) if zero else np.array([.01,.02,.03,0,0,0])}
    atoms.calc = SinglePointCalculator(atoms, **{key: val for key, val in labels.items() if key not in missing})
    return atoms


def config(path, checkpoint):
    return {'trainer':'mve', 'mve':{'enabled':True,'version':1,'floors':dict(MVE_FLOORS),'units':dict(MVE_UNITS)},
            'device':'cpu','gpu-parallel':False,'model':'race','regress_forces':'auto',
            'interaction_block':'slow','cueq_config':False,'oeq_config':False,
            'fname_traj':str(path),'ntrain':1,'nvalid':1,'nbatch':1,
            'cutoff':4.,'num_species':1,'avg_num_neighbors':2,'max_ell':1,
            'num_radial_basis':4,'hidden_channels':'4x0e+4x1o+4x2e',
            'output_channels':'2x0e+3x1o+6x2e','nlayers':1,'features_dim':4,
            'active_fn':'identity','NN':{'data_seed':2,'init_seed':2026,'learning_rate':.001,
                'nepoch':1,'restart':False,'ema':False,'fname_pkl':str(checkpoint),
                'enr_lambda':1,'frc_lambda':1,'str_lambda':1},
            'predict':{'evaluate_tag':False,'model':str(checkpoint),'fname_traj':str(path)}}


class SerialE2ETests(unittest.TestCase):
    def test_missing_tasks_do_not_duplicate_loss_or_gradients(self):
        trainer = object.__new__(MVETrainer)
        trainer.floors = dict(MVE_FLOORS)
        trainer.json_data = {'mve': {'target_norm': {'enabled': False}, 'beta': 0},
                             'NN': {'enr_lambda': 2.5, 'frc_lambda': 3.5, 'str_lambda': 4.5}}
        trainer.train_stats = {'scales': {'energy': 1., 'forces': 1., 'stress': 1.}}
        trainer.next_adaptive = {'energy': 0., 'forces': 0., 'stress': 0.}

        def run(energy_valid, force_valid, stress_valid):
            energy = torch.tensor([.4], requires_grad=True)
            forces = torch.full((2, 3), .2, requires_grad=True)
            stress = torch.full((1, 6), .1, requires_grad=True)
            force_var = torch.eye(3).expand(2, -1, -1).clone().requires_grad_()
            stress_var = torch.eye(6).unsqueeze(0).requires_grad_()
            preds = {'energy': energy, 'energy_var': torch.ones(1, requires_grad=True),
                     'forces': forces, 'forces_var': force_var,
                     'stress': stress, 'stress_var': stress_var}
            target = {'energy': torch.zeros(1), 'forces': torch.zeros(2, 3),
                      'stress': torch.zeros(1, 6),
                      'batch': torch.tensor([0, 0]),
                      'energy_valid': torch.tensor([energy_valid]),
                      'force_valid': torch.tensor([force_valid] * 2),
                      'stress_valid': torch.tensor([stress_valid]),
                      'stress_original_valid': torch.tensor([stress_valid])}
            metrics = trainer.compute_loss(preds, target)
            metrics['loss'].backward()
            return metrics, preds

        only_energy, e = run(True, False, False)
        self.assertEqual([only_energy[f'count_{k}'] for k in 'efs'], [1, 0, 0])
        self.assertIsNone(only_energy['loss_f'])
        self.assertIsNone(only_energy['loss_s'])
        torch.testing.assert_close(only_energy['loss'], 2.5 * only_energy['loss_e'])
        self.assertNotEqual(float(e['energy'].grad[0]), 0)
        self.assertTrue(torch.equal(e['forces'].grad, torch.zeros_like(e['forces'])))
        self.assertTrue(torch.equal(e['stress'].grad, torch.zeros_like(e['stress'])))

        no_stress, es = run(True, True, False)
        torch.testing.assert_close(no_stress['loss'], 2.5 * no_stress['loss_e'] + 3.5 * no_stress['loss_f'])
        self.assertNotEqual(float(es['forces'].grad.abs().sum()), 0)
        self.assertTrue(torch.equal(es['stress'].grad, torch.zeros_like(es['stress'])))

        no_force, fs = run(True, False, True)
        torch.testing.assert_close(no_force['loss'], 2.5 * no_force['loss_e'] + 4.5 * no_force['loss_s'])
        self.assertTrue(torch.equal(fs['forces'].grad, torch.zeros_like(fs['forces'])))
        self.assertNotEqual(float(fs['stress'].grad.abs().sum()), 0)

        with self.assertRaisesRegex(ValueError, 'no eligible'):
            run(False, False, False)

    def test_factory_never_autoselects_acceleration(self):
        canonical = config('not-needed.xyz', 'not-needed.pt')
        canonical.pop('cueq_config')
        canonical.pop('oeq_config')
        factory = object.__new__(BaseTrainer)
        factory.json_data, factory.msg = canonical, ''
        with patch('bam_torch.training.base_trainer.CUET_AVAILABLE', True), \
             patch('bam_torch.training.base_trainer.OEQ_AVAILABLE', True), \
             patch('bam_torch.training.base_trainer.CuEquivarianceConfig', side_effect=AssertionError('CUET selected')):
            model = factory.set_model()
        self.assertTrue(model.mve)
        self.assertIn('e3nn', factory.msg)

    def test_original_missing_vs_true_zero_labels(self):
        virial = fixture(missing=('stress',))
        virial.calc.results['virial'] = -512 * np.array([[.02,.04,0],[.06,.02,0],[0,0,.02]])
        graphs = mve_graphs([fixture(zero=True), fixture(missing=('energy', 'forces', 'stress')), virial],
                            4, {1:0}, {0:0.}, [13, 2, 57])
        self.assertEqual([int(graph.frame_id[0]) for graph in graphs], [13, 2, 57])
        self.assertEqual(graphs[0].force_valid.tolist(), [True] * 3)
        self.assertEqual(graphs[0].stress_valid.tolist(), [True])
        self.assertEqual(graphs[0].energy_valid.tolist(), [True])
        self.assertEqual(graphs[1].force_valid.tolist(), [False] * 3)
        self.assertEqual(graphs[1].energy_valid.tolist(), [False])
        self.assertEqual(graphs[1].stress_valid.tolist(), [False])
        self.assertEqual(graphs[1].forces.tolist(), [[0.] * 3] * 3)
        self.assertEqual(graphs[2].stress_valid.tolist(), [True])
        torch.testing.assert_close(graphs[2].stress, torch.tensor([[.02,.02,.02,0,0,.05]]))

    def test_extxyz_info_virial_and_original_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'virial.extxyz'
            frame = fixture(missing=('stress',))
            frame.info['virial'] = -512 * np.array([[.02,.04,0],[.06,.02,0],[0,0,.02]])
            write(path, frame, format='extxyz')
            restored = read(path, format='extxyz')
            self.assertIn('virial', restored.info)
            self.assertNotIn('virial', restored.calc.results)
            self.assertNotIn('stress', restored.calc.results)
            graph = mve_graphs([restored], 4, {1:0}, {0:0.})[0]
            self.assertEqual(graph.stress_valid.tolist(), [True])
            torch.testing.assert_close(graph.stress, torch.tensor([[.02,.02,.02,0,0,.05]]))

    def test_invalid_original_labels_and_volume_fail_closed(self):
        for key, value in (('energy', float('nan')),
                           ('forces', np.full((3,3), float('inf'))),
                           ('stress', np.array([0,0,0,0,0,float('nan')]))):
            bad = fixture()
            bad.calc.results[key] = value
            with self.subTest(label=key), self.assertRaisesRegex(ValueError, f'observed {key} must be finite'):
                mve_graphs([bad], 4, {1:0}, {0:0.})
        bad_virial = fixture(missing=('stress',))
        bad_virial.info['virial'] = np.full((3,3), float('inf'))
        with self.assertRaisesRegex(ValueError, 'observed stress must be finite'):
            mve_graphs([bad_virial], 4, {1:0}, {0:0.})
        for label in ('stress', 'virial'):
            unphysical = fixture(missing=('stress',)) if label == 'virial' else fixture()
            if label == 'virial':
                unphysical.info['virial'] = np.eye(3)
            unphysical.set_cell(np.zeros((3,3)))
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'original cell volume'):
                mve_graphs([unphysical], 4, {1:0}, {0:0.})
        molecule = fixture(missing=('stress',))
        molecule.set_cell(np.zeros((3,3)))
        graph = mve_graphs([molecule], 4, {1:0}, {0:0.})[0]
        self.assertEqual(graph.stress_valid.tolist(), [False])
        self.assertEqual(graph.stress_geometry_valid.tolist(), [False])

    def test_nonfinite_training_data_rejected_before_fit_or_step(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'never-created.pt'
            c = config('unused.extxyz', checkpoint)
            bad_energy = fixture()
            bad_energy.calc.results['energy'] = float('nan')
            with patch('bam_torch.training.mve_trainer.read', return_value=[bad_energy, bad_energy]), \
                 patch('bam_torch.training.target_norm.get_enr_avg_per_element', side_effect=AssertionError('fit called')):
                with self.assertRaisesRegex(ValueError, 'observed energy must be finite'):
                    MVETrainer(c)
            bad_forces = fixture()
            bad_forces.calc.results['forces'] = np.full((3, 3), float('inf'))
            with patch('bam_torch.training.mve_trainer.read', return_value=[bad_forces, bad_forces]), \
                 patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('optimizer stepped')):
                with self.assertRaisesRegex(ValueError, 'observed forces must be finite'):
                    MVETrainer(c)
            self.assertFalse(checkpoint.exists())

    def test_legacy_and_unsupported_configs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            c = config('not-needed.xyz', Path(directory) / 'checkpoint.pt')
            for key, value in (("mve", {'enabled':True}), ('output_channels','8x0e'),
                               ('dens',{'enabled':True}), ('device','gpu')):
                bad = json.loads(json.dumps(c))
                bad[key] = value
                with self.assertRaises(ValueError):
                    validate_mve_config(bad)
            for path, value in (('mve', {'enabled':True,'version':1,'floors':dict(MVE_FLOORS),
                                          'units':dict(MVE_UNITS),'beta':-1}),
                                ('NN', {**c['NN'], 'tracked_epsilon':True})):
                bad = json.loads(json.dumps(c))
                bad[path] = value
                with self.assertRaises(ValueError):
                    validate_mve_config(bad)
            torch.save({'input.json': c, 'params': {}}, c['NN']['fname_pkl'])
            with self.assertRaisesRegex(ValueError, 'manifest'):
                load_mve_checkpoint(c['NN']['fname_pkl'])
            marker = Path(directory) / 'executed'
            torch.save({'manifest': ExecutableLegacyValue(marker)}, c['NN']['fname_pkl'])
            with self.assertRaisesRegex(ValueError, 'safe weights-only'):
                load_mve_checkpoint(c['NN']['fname_pkl'])
            self.assertFalse(marker.exists())

    def test_energy_force_only_train_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ef.extxyz'
            checkpoint = Path(directory) / 'ef.pt'
            write(path, [fixture(missing=('stress',)), fixture(zero=True, missing=('stress',))], format='extxyz')
            trainer = MVETrainer(config(path, checkpoint))
            training = trainer.train_one_epoch('train')
            valid = trainer.train_one_epoch('valid')
            for metrics in (training, valid):
                self.assertEqual(metrics['counts'], {'e':1, 'f':3, 's':0})
                self.assertIsNone(metrics['loss_s'])
                self.assertAlmostEqual(metrics['loss'], metrics['loss_e'] + metrics['loss_f'], places=5)
            self.assertTrue(np.isfinite(trainer.train()))
            self.assertIsNone(load_mve_checkpoint(checkpoint)['loss']['valid']['loss_s'])

    def test_train_valid_checkpoint_fresh_process_prediction(self):
        torch.manual_seed(2026)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            trajectory, prediction = directory/'train.extxyz', directory/'prediction.extxyz'
            checkpoint, output = directory/'model.pt', directory/'predictions.json'
            write(trajectory, [fixture(), fixture(zero=True)], format='extxyz')
            write(prediction, [Atoms('H3', positions=[[0,0,0],[1,0,0],[0,1,0]], cell=[8,8,8], pbc=True),
                               Atoms('H3', positions=[[0,0,0],[1,0,0],[0,1,0]])], format='extxyz')
            c = config(trajectory, checkpoint)
            trainer = MVETrainer(c)
            self.assertNotEqual(int(trainer.train_loader.dataset[0].frame_id[0]),
                                int(trainer.valid_loader.dataset[0].frame_id[0]))
            before = trainer.model.readouts[0].bias.detach().clone()
            training = trainer.train_one_epoch('train')
            self.assertFalse(torch.equal(before, trainer.model.readouts[0].bias.detach()))
            validation = trainer.train_one_epoch('valid')
            self.assertEqual(training['counts'], {'e':1, 'f':3, 's':1})
            self.assertEqual(validation['counts'], {'e':1, 'f':3, 's':1})
            self.assertTrue(np.isfinite(training['loss']) and np.isfinite(validation['loss']))
            trainer.save_checkpoint(checkpoint, 1, training, validation)
            self.assertTrue(checkpoint.is_file())
            stored = load_mve_checkpoint(checkpoint)
            self.assertEqual(stored['manifest']['parameter_policy'], 'raw-best')
            self.assertTrue(all(type(k) is int and type(v) is float
                                for k, v in stored['enr_avg_per_element'].items()))
            with self.assertRaisesRegex(FileExistsError, 'overwrite'):
                MVETrainer(c)
            request = {'trainer':'mve', 'predict':{'model':str(checkpoint),
                        'fname_traj':str(prediction), 'uncertainty_output':str(output)}}
            with self.assertRaisesRegex(ValueError, 'unexpected'):
                Evaluator({'trainer':'mve', 'predict': {**request['predict'], 'ndata': 1}})
            with self.assertRaisesRegex(ValueError, 'canonical setting output_channels'):
                Evaluator({**request, 'output_channels': '8x0e'})
            unchanged = json.loads(json.dumps(request))
            payload = Evaluator(request).evaluate()
            self.assertEqual(request, unchanged)
            self.assertEqual(payload['schema'], 'bam_torch.equivariant_mve.v2')
            self.assertEqual(len(payload['predictions']), 2)
            self.assertEqual(np.asarray(payload['predictions'][0]['effective']['stress']).shape, (1,6,6))
            self.assertFalse(payload['predictions'][1]['stress_geometry_valid'])
            self.assertIsNone(payload['predictions'][1]['stress_voigt'])
            self.assertIsNone(payload['predictions'][1]['raw']['stress'])
            self.assertIsNone(payload['predictions'][1]['effective']['stress'])
            self.assertIsNone(payload['predictions'][1]['stress_voigt_effective'])
            # Independent process uses ONLY checkpoint plus geometries, never train.extxyz.
            trajectory.unlink()
            fresh = directory/'fresh.json'
            command = [sys.executable, '-S', str(ROOT/'examples/example-3BPA_300K-mve/evaluate.py'),
                       str(checkpoint), str(prediction), '--output', str(fresh)]
            proc = subprocess.run(command, cwd=directory, env=os.environ.copy(),
                                  capture_output=True, text=True, timeout=90)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertEqual(json.loads(fresh.read_text())['config_digest'], stored['manifest']['config_digest'])
            altered = torch.load(checkpoint, weights_only=True)
            altered['manifest']['basis'] = 'wrong'
            torch.save(altered, checkpoint)
            with self.assertRaisesRegex(ValueError, 'manifest'):
                Evaluator(request)


if __name__ == '__main__':
    unittest.main()
