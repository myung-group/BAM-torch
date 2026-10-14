import math
import unittest

import torch
from e3nn import o3

from bam_torch.training.loss import (
    MVE_FLOORS, MVE_UNITS, assemble_effective_covariances,
    gaussian_observation_nll, stress_voigt_to_irrep,
)


class GaussianTests(unittest.TestCase):
    def test_zero_factors_positive_fixed_floor_and_validation(self):
        raw = {'energy': torch.zeros(2, dtype=torch.float64),
               'forces': torch.zeros(3, 3, 3, dtype=torch.float64),
               'stress': torch.zeros(2, 6, 6, dtype=torch.float64)}
        effective = assemble_effective_covariances(raw, MVE_FLOORS, MVE_UNITS)
        for key, expected in MVE_FLOORS.items():
            self.assertTrue(torch.allclose(
                torch.linalg.eigvalsh(effective[key].reshape(-1, 1, 1) if key == 'energy' else effective[key]),
                torch.full((len(raw[key]), 1 if key == 'energy' else raw[key].shape[-1]), expected, dtype=torch.float64),
            ))
        for invalid in ({**MVE_FLOORS, 'stress': 0}, {**MVE_FLOORS, 'forces': float('nan')}):
            with self.assertRaises(ValueError):
                assemble_effective_covariances(raw, invalid)
        with self.assertRaises(ValueError):
            assemble_effective_covariances({**raw, 'stress': torch.full((2, 6, 6), float('nan'))}, MVE_FLOORS)
        with self.assertRaises(ValueError):
            assemble_effective_covariances(raw, MVE_FLOORS, {'energy':'wrong'})
        with self.assertRaisesRegex(ValueError, 'energy_var'):
            assemble_effective_covariances({**raw, 'energy': raw['energy'][:, None]}, MVE_FLOORS)

    def test_analytic_nll_observation_counts(self):
        for dimension, observations in ((1, 2), (3, 4), (6, 2)):
            residual = torch.arange(1, observations * dimension + 1, dtype=torch.float64).reshape(observations, dimension) / 10
            covariance = torch.eye(dimension, dtype=torch.float64).expand(observations, -1, -1).clone() * 2
            actual = gaussian_observation_nll(residual, covariance)
            expected = 0.5 * (residual.square().sum(dim=-1) / 2 + dimension * math.log(2) + dimension * math.log(2 * math.pi))
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(actual.mean(), expected.sum() / observations, rtol=1e-6, atol=1e-7)
        with self.assertRaisesRegex(ValueError, 'finite'):
            gaussian_observation_nll(torch.tensor([[float('nan')]]), torch.ones(1, 1, 1))

    def test_non_axis_stress_basis_rotation_and_derived_voigt(self):
        torch.manual_seed(2026)
        q = stress_voigt_to_irrep(dtype=torch.float64)
        b = torch.linalg.inv(q)
        torch.testing.assert_close(q.T @ q, torch.diag(torch.tensor([1.,1.,1.,2.,2.,2.], dtype=torch.float64)), rtol=1e-6, atol=1e-7)
        rotation = o3.rand_matrix().to(torch.float64)
        diagonal = torch.tensor([2.,1.,3.], dtype=torch.float64)
        r45 = torch.tensor([[math.sqrt(.5), -math.sqrt(.5), 0], [math.sqrt(.5), math.sqrt(.5), 0], [0, 0, 1]], dtype=torch.float64)
        for r in (r45, rotation):
            stress = r45 @ torch.diag(diagonal) @ r45.T
            rotated = r @ stress @ r.T
            def voigt(t):
                return torch.stack((t[0,0],t[1,1],t[2,2],t[1,2],t[0,2],t[0,1]))
            residual = voigt(stress) @ q.T
            rotated_residual = voigt(rotated) @ q.T
            d = torch.block_diag(torch.ones(1, 1, dtype=torch.float64), o3.Irrep('2e').D_from_matrix(r))
            torch.testing.assert_close(rotated_residual, d @ residual, rtol=1e-5, atol=1e-6)
            factors = torch.randn(6, 6, dtype=torch.float64)
            raw = factors @ factors.T
            covariance = raw + .1 * torch.eye(6, dtype=torch.float64)
            rotated_covariance = d @ covariance @ d.T
            torch.testing.assert_close(rotated_covariance, d @ raw @ d.T + .1 * torch.eye(6, dtype=torch.float64), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(gaussian_observation_nll(residual[None], covariance[None]),
                                       gaussian_observation_nll(rotated_residual[None], rotated_covariance[None]), rtol=1e-5, atol=1e-6)
            exported = b @ covariance @ b.T
            torch.testing.assert_close(q @ exported @ q.T, covariance, rtol=1e-5, atol=1e-6)
            force = torch.randn(3, dtype=torch.float64)
            f_cov = torch.randn(3,3, dtype=torch.float64)
            f_cov = f_cov @ f_cov.T + .1 * torch.eye(3, dtype=torch.float64)
            torch.testing.assert_close(gaussian_observation_nll(force[None], f_cov[None]),
                                       gaussian_observation_nll((r @ force)[None], (r @ f_cov @ r.T)[None]), rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
