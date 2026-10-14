import unittest
import math

import torch
from e3nn import o3

from bam_torch.model.models import RACE
from bam_torch.training.loss import MVE_FLOORS, assemble_effective_covariances, gaussian_observation_nll, stress_voigt_to_irrep


def tiny_model(mve=True, **kwargs):
    return RACE(cutoff=4., avg_num_neighbors=2, num_species=1, max_ell=1,
                num_basis_func=4, hidden_irreps=o3.Irreps('4x0e+4x1o+4x2e'),
                nlayers=1, features_dim=4,
                output_irreps=o3.Irreps('2x0e+3x1o+6x2e' if mve else '1x0e'),
                radial_MLP=[8], regress_forces='auto', interaction_block='slow',
                mve=mve, **kwargs)


def tiny_graph():
    from torch_geometric.data import Data
    from torch_geometric.data import Batch
    return Batch.from_data_list([Data(
        positions=torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
        species=torch.zeros(3, dtype=torch.long),
        edge_index=torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]),
        edges=torch.zeros(6, 3), num_edges=6, num_nodes=3,
        cell=torch.eye(3).reshape(1, 3, 3) * 8,
    )])


class ModelTests(unittest.TestCase):
    def test_energy_variance_size_exponent_and_max_logvar(self):
        torch.manual_seed(7)
        fixed = tiny_model(energy_size_exponent_init=0.0, energy_size_exponent_learn=False)
        sized = tiny_model()
        self.assertEqual(float(sized.logvar_n_alpha.detach()), 1.0)  # default alpha=1, learned
        sized.load_state_dict({**fixed.state_dict(), 'logvar_n_alpha': torch.tensor(1.0)})
        base = fixed(tiny_graph(), backprop=False)['energy_var']
        scaled = sized(tiny_graph(), backprop=False)['energy_var']
        torch.testing.assert_close(scaled, 3.0 * base)  # n=3 atoms, alpha=1
        self.assertFalse(fixed.logvar_n_alpha.requires_grad)
        self.assertTrue(sized.logvar_n_alpha.requires_grad)
        sized(tiny_graph(), backprop=True)['energy_var'].sum().backward()
        self.assertIsNotNone(sized.logvar_n_alpha.grad)
        self.assertNotEqual(float(sized.logvar_n_alpha.grad), 0.0)
        clamped = tiny_model(max_logvar=1e-3, energy_size_exponent_init=50.0)
        self.assertLessEqual(float(clamped(tiny_graph(), backprop=False)["energy_var"].detach().max()),
                             math.exp(1e-3) + 1e-6)
        self.assertNotIn('logvar_n_alpha', tiny_model(mve=False).state_dict())

    def test_raw_covariance_and_autograd_means_and_decoder_gradients(self):
        torch.manual_seed(2026)
        model = tiny_model()
        graph = tiny_graph()
        preds = model(graph, backprop=True)
        self.assertEqual(tuple(preds['energy_var'].shape), (1,))
        self.assertEqual(tuple(preds['forces_var'].shape), (3, 3, 3))
        self.assertEqual(tuple(preds['stress_var'].shape), (1, 6, 6))
        effective = assemble_effective_covariances({
            'energy': preds['energy_var'],
            'forces': preds['forces_var'], 'stress': preds['stress_var']}, MVE_FLOORS)
        self.assertTrue(all(torch.isfinite(c).all() for c in effective.values()))
        self.assertTrue(all(torch.linalg.eigvalsh(c).min() > 0 for key, c in effective.items() if key != 'energy'))
        exact_forces = -torch.autograd.grad(preds['energy'].sum(), graph.positions, retain_graph=True)[0]
        torch.testing.assert_close(preds['forces'], exact_forces, rtol=1e-6, atol=1e-7)
        # An independent derivative through cell reproduces the force/stress helper.
        from bam_torch.utils.output_utils import get_outputs
        _, _, exact_stress, _ = get_outputs(preds['energy'], graph.positions, None,
                                            graph.cell, graph.batch, 1, training=True,
                                            compute_force=True, compute_virials=True,
                                            compute_stress=True)
        torch.testing.assert_close(preds['stress'], exact_stress, rtol=1e-6, atol=1e-7)
        gaussian = sum((
            gaussian_observation_nll(torch.full((1, 1), .3), effective['energy'].reshape(-1, 1, 1)).mean(),
            gaussian_observation_nll(torch.full((3, 3), .2), effective['forces']).mean(),
            gaussian_observation_nll(torch.full((1, 6), .2), effective['stress']).mean(),
        ))
        gaussian.backward()
        for decoders in (model.force_cov_decoders, model.stress_cov_decoders):
            gradients = [p.grad for p in decoders.parameters()]
            self.assertTrue(gradients and all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertTrue(any(torch.count_nonzero(g) for g in gradients))

    def test_invalid_head_and_missing_irreps_fail(self):
        with self.assertRaisesRegex(ValueError, 'Legacy scalar MVE'):
            RACE(output_irreps=o3.Irreps('8x0e'), mve=False)
        with self.assertRaisesRegex(ValueError, 'output_channels'):
            RACE(output_irreps=o3.Irreps('8x0e'), mve=True, regress_forces='auto')
        with self.assertRaisesRegex(ValueError, '1o and 2e'):
            RACE(output_irreps=o3.Irreps('2x0e+3x1o+6x2e'),
                 hidden_irreps=o3.Irreps('4x0e'), mve=True, regress_forces='auto')
        with self.assertRaisesRegex(ValueError, 'DeNS'):
            RACE(output_irreps=o3.Irreps('2x0e+3x1o+6x2e'), mve=True,
                 regress_forces='auto', dens=True)

    def test_model_rotation_of_force_and_stress_covariance(self):
        torch.manual_seed(2026)
        model = tiny_model().eval()
        graph = tiny_graph()
        original = model(graph, backprop=False)
        q = stress_voigt_to_irrep()
        for rotation in (
            torch.tensor([[math.sqrt(.5),-math.sqrt(.5),0],
                          [math.sqrt(.5),math.sqrt(.5),0],[0,0,1.]]),
            o3.rand_matrix(),
        ):
            turned = tiny_graph()
            turned.positions = turned.positions @ rotation.T
            turned.cell = turned.cell @ rotation.T
            rotated = model(turned, backprop=False)
            torch.testing.assert_close(rotated['energy'], original['energy'], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(rotated['forces'], original['forces'] @ rotation.T, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(rotated['forces_var'], rotation @ original['forces_var'] @ rotation.T, rtol=1e-5, atol=1e-6)
            d = torch.block_diag(torch.ones(1, 1), o3.Irrep('2e').D_from_matrix(rotation))
            torch.testing.assert_close(rotated['stress_var'], d @ original['stress_var'] @ d.T, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(rotated['stress'] @ q.T, original['stress'] @ q.T @ d.T, rtol=1e-5, atol=1e-6)

    def test_non_mve_mean_api_unchanged(self):
        torch.manual_seed(2026)
        preds = tiny_model(mve=False)(tiny_graph(), backprop=True)
        self.assertEqual(tuple(preds['energy'].shape), (1,))
        self.assertEqual(tuple(preds['forces'].shape), (3, 3))
        self.assertEqual(tuple(preds['stress'].shape), (1, 6))
        self.assertEqual(tuple(preds['forces_var'].shape), (3, 6))


if __name__ == '__main__':
    unittest.main()
