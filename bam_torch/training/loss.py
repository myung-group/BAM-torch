import torch
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple


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