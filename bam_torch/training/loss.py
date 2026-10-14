import torch
import math
from e3nn.io import CartesianTensor
import torch.distributed as dist
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from bam_torch.utils.utils import apply_along_axis


class RMSELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super(RMSELoss,self).__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)
        self.eps = 1e-7

    def forward(self,y,y_hat):
        return torch.sqrt(self.mse(y,y_hat) + self.eps)


def l2_regularization(params):
    wgt = torch.cat([p.view(-1) for p in params if p.requires_grad])
    return (wgt * wgt).mean()


def is_ddp_enabled():
    return dist.is_initialized() and dist.get_world_size() > 1


def reduce_loss(raw_loss: torch.Tensor, ddp: Optional[bool] = None) -> torch.Tensor:
    """
    Reduces an element-wise loss tensor.

    If ddp is True and distributed is initialized, the function computes:

        loss = (local_sum * world_size) / global_num_elements

    Otherwise, it returns the regular mean.
    """
    ddp = is_ddp_enabled() if ddp is None else ddp
    if ddp and dist.is_initialized():
        world_size = dist.get_world_size()
        n_local = raw_loss.numel()
        loss_sum = raw_loss.sum()
        total_samples = torch.tensor(
            n_local, device=raw_loss.device, dtype=raw_loss.dtype
        )
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        return loss_sum * world_size / total_samples
    return raw_loss.mean()


class _TagAwareElementwiseLoss(torch.nn.Module):
    """Tag-aware wrapper around an elementwise loss, matching HuberLoss's
    interface: energies are normalized per atom when num_atoms is given
    (a per-graph atom-count vector); forces/stress are compared as-is.
    """

    def __init__(self, loss_fn: Callable, reduction: str = "mean"):
        super().__init__()
        self.loss_fn = loss_fn
        self.reduction = reduction

    def forward(
        self, pred, target, tag, num_atoms=None, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        if tag == "energy" and num_atoms is not None:
            pred = pred / num_atoms
            target = target / num_atoms
        raw = self.loss_fn(pred, target, reduction="none")
        if ddp:
            return reduce_loss(raw, ddp)
        if self.reduction == "sum":
            return raw.sum()
        return raw.mean()


class L1Loss(_TagAwareElementwiseLoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(torch.nn.functional.l1_loss, reduction)


class MSELoss(_TagAwareElementwiseLoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(torch.nn.functional.mse_loss, reduction)


# Per-target Huber deltas.
#
# One shared delta cannot suit all three targets: they live on different
# scales (energy ~1e-2 eV/atom, forces ~1e-1 eV/A, stress ~1e-2 eV/A^3), and
# delta is an absolute-error threshold. Set it far above a target's typical
# error and that whole term stays in the quadratic branch, so the Huber
# degenerates to plain MSE and its outlier robustness never engages; set it
# far below and the term is effectively L1. A single value therefore cannot
# put more than one target in the intended regime.
#
# Falls back to the shared `huber_delta`, so configs that do not set the
# per-target keys behave exactly as before.
HUBER_DELTA_KEYS = {
    'energy_loss': 'huber_delta_energy',
    'force_loss': 'huber_delta_force',
    'stress_loss': 'huber_delta_stress',
}


def resolve_huber_delta(
    loss_config: Dict[str, Any], loss_key: str, default: float = 0.01
) -> float:
    """Huber delta for one target.

    Precedence: `huber_delta_<target>` -> shared `huber_delta` -> `default`.
    The default also guards the case where neither key is present, which
    previously passed delta=None straight into torch's huber_loss.
    """
    specific = HUBER_DELTA_KEYS.get(loss_key)
    if specific is not None and loss_config.get(specific) is not None:
        return loss_config[specific]
    shared = loss_config.get('huber_delta')
    return default if shared is None else shared


class HuberLoss(torch.nn.Module):
    def __init__(self, huber_delta=0.01) -> None:
        super().__init__()
        # We store the huber_delta rather than a loss with fixed reduction.
        self.huber_delta = huber_delta

    def forward(
        self, pred, target, tag, num_atoms = None, ddp: Optional[bool] = None
    ) -> torch.Tensor:

        if tag == "energy":
            if ddp:
                loss_energy = torch.nn.functional.huber_loss(
                    pred / num_atoms,
                    target / num_atoms,
                    reduction="none",
                    delta=self.huber_delta,
                )
                # print(' pred["energy"]  : ', pred["energy"] )
                # print(' target["energy"] ', target["energy"] )
                loss_energy = reduce_loss(loss_energy, ddp)
            else:
                loss_energy = torch.nn.functional.huber_loss(
                    pred / num_atoms,
                    target / num_atoms,
                    reduction="mean",
                    delta=self.huber_delta,
                )
                # print(' pred["energy"]  : ', pred["energy"] )
                # print(' target["energy"] ', target["energy"] )
            total_loss = loss_energy

        if tag == "forces":
            if ddp:
                loss_forces = torch.nn.functional.huber_loss(
                    pred, target,
                    reduction="none",
                    delta=self.huber_delta
                )
                loss_forces = reduce_loss(loss_forces, ddp)
            else:
                loss_forces = torch.nn.functional.huber_loss(
                    pred, target,
                    reduction="mean",
                    delta=self.huber_delta
                )
                # print(' pred["forces"] : ', pred["forces"] )
                # print(' target["forces"] : ', target["forces"] )
            total_loss = loss_forces

        if tag == "stress":
            if ddp:
                loss_stress = torch.nn.functional.huber_loss(
                    pred, target,
                    reduction="none",
                    delta=self.huber_delta
                )
                loss_stress = reduce_loss(loss_stress, ddp)
            else:
                loss_stress = torch.nn.functional.huber_loss(
                    pred, target,
                    reduction="mean",
                    delta=self.huber_delta
                )
                # print(' pred["stress"] : ', pred["stress"] )
                # print(' target["stress"] : ', target["stress"] )
            total_loss = loss_stress

        return total_loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


MVE_UNITS = {"energy": "eV^2", "forces": "(eV/Å)^2", "stress": "(eV/Å^3)^2"}
MVE_FLOORS = {"energy": 1e-4, "forces": 1e-4, "stress": 1e-6}


def stress_voigt_to_irrep(dtype=torch.float32, device="cpu"):
    """ASE Voigt (xx, yy, zz, yz, xz, xy) -> orthonormal (0e, 2e).

    e3nn's CartesianTensor supplies the same normalized irrep basis used by
    the 6x0e+6x2e stress generator; symmetric off-diagonals occur twice.
    """
    basis = CartesianTensor("ij=ji").reduced_tensor_products().change_of_basis.to(
        dtype=dtype, device=device
    )
    tensors = torch.zeros(6, 3, 3, dtype=dtype, device=device)
    tensors[0, 0, 0] = tensors[1, 1, 1] = tensors[2, 2, 2] = 1
    for row, i, j in ((3, 1, 2), (4, 0, 2), (5, 0, 1)):
        tensors[row, i, j] = tensors[row, j, i] = 1
    return torch.einsum("aij,bij->ab", basis, tensors)


def assemble_effective_covariances(raw, floors, units=MVE_UNITS, adaptive=None, scales=None, atom_counts=None):
    """Validate raw E/F/S covariance and add fixed plus optional adaptive floor once."""
    if units != MVE_UNITS or set(floors) != set(MVE_UNITS):
        raise ValueError("MVE covariance units or E/F/S floor keys do not match v1")
    adaptive = adaptive or {key: 0.0 for key in MVE_UNITS}
    if set(adaptive) != set(MVE_UNITS):
        raise ValueError('MVE adaptive floor keys must be exactly E/F/S')
    for key, factor in adaptive.items():
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor < 0:
            raise ValueError(f'MVE {key} adaptive floor must be finite and nonnegative')
        if factor and (not isinstance(scales, dict) or key not in scales or
                       not math.isfinite(scales[key]) or scales[key] <= 0):
            raise ValueError(f'MVE {key} adaptive floor needs a positive train-only scale')
    if adaptive['energy'] and (atom_counts is None or atom_counts.shape != raw['energy'].shape or
                               not torch.isfinite(atom_counts).all() or (atom_counts <= 0).any()):
        raise ValueError('MVE energy adaptive floor needs positive graph atom counts')
    shapes = {"energy": (), "forces": (3, 3), "stress": (6, 6)}
    result = {}
    for key, trailing in shapes.items():
        value = raw[key]
        if not isinstance(value, torch.Tensor) or value.ndim != len(trailing) + 1 or value.shape[1:] != trailing:
            raise ValueError(f"MVE {key}_var must have shape [observations{',' if trailing else ''}{','.join(map(str, trailing))}]")
        if not torch.isfinite(value).all():
            raise ValueError(f"MVE {key}_var contains nonfinite values")
        floor = floors[key]
        if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor) or floor <= 0:
            raise ValueError(f"MVE {key} floor must be strictly positive and finite")
        if key == "energy":
            if (value < 0).any():
                raise ValueError("MVE raw energy variance must be nonnegative")
            addition = adaptive[key] * (atom_counts * scales[key]).square() if adaptive[key] else 0.0
            result[key] = value + floor + addition
        else:
            if not torch.allclose(value, value.transpose(-1, -2), atol=1e-6, rtol=1e-6):
                raise ValueError(f"MVE raw {key} covariance must be symmetric")
            if (torch.linalg.eigvalsh(value) < -1e-6).any():
                raise ValueError(f"MVE raw {key} covariance must be positive semidefinite")
            addition = adaptive[key] * scales[key] ** 2 if adaptive[key] else 0.0
            result[key] = value + (floor + addition) * torch.eye(trailing[0], device=value.device, dtype=value.dtype)
    return result


def detached_beta_weights(covariance, beta, atom_counts=None, min_floor=1e-12):
    """Per-task serial-global detached observation weights, mean exactly one."""
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta < 0:
        raise ValueError('MVE beta must be finite and nonnegative')
    if not math.isfinite(min_floor) or min_floor <= 0:
        raise ValueError('MVE beta minimum scale must be positive')
    if covariance.ndim == 1:
        if atom_counts is None or atom_counts.shape != covariance.shape or (atom_counts <= 0).any():
            raise ValueError('MVE energy beta needs positive graph atom counts')
        variance = covariance / atom_counts.square()
    elif covariance.ndim == 3:
        variance = torch.diagonal(covariance, dim1=-2, dim2=-1).mean(-1)
    else:
        raise ValueError('MVE beta expects energy[G] or force/stress[N,d,d]')
    if not len(variance) or beta == 0:
        return torch.ones_like(variance).detach()
    unnormalized = variance.detach().clamp(min=min_floor).pow(beta)
    return (unnormalized / unnormalized.mean()).detach()


def gaussian_observation_nll(residual, covariance):
    """One Gaussian NLL per observation, including d log(2π)."""
    if residual.ndim != 2 or covariance.shape != (residual.shape[0], residual.shape[1], residual.shape[1]):
        raise ValueError("MVE residual/covariance observation dimensions disagree")
    if not torch.isfinite(residual).all() or not torch.isfinite(covariance).all():
        raise ValueError("MVE Gaussian residual and effective covariance must be finite")
    chol = torch.linalg.cholesky(covariance)
    whitened = torch.linalg.solve_triangular(chol, residual.unsqueeze(-1), upper=False)
    quadratic = whitened.square().sum(dim=(-2, -1))
    logdet = 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
    return 0.5 * (quadratic + logdet + residual.shape[1] * math.log(2 * math.pi))
