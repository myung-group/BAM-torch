import torch
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


class NLLLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, preds, targets, tag='stress', ddp: Optional[bool] = None
    ) -> torch.Tensor:

        # Energy
        if tag == 'energy':
            prd_enr = preds['energy'].flatten()
            tgt_enr = targets['energy'].flatten()
            enr_var = preds['energy_var']

            diff_enr = (tgt_enr - prd_enr)
            energy_term = (torch.einsum('i,i->i', diff_enr, diff_enr) / enr_var).mean()
            energy_log_term = torch.log(enr_var).mean()

            loss_dict = {
                'loss_e': energy_term,
                'log_e' : energy_log_term,
                'enr_var': enr_var.mean(),
            }

        # Forces
        if tag == 'forces' or tag == 'force':
            prd_frc = preds['forces']
            tgt_frc = targets['forces'] #.flatten()
            frc_var = preds['forces_var']

            def map_lower_triangle(l):
                return torch.tensor([
                    [l[0], 0.0, 0.0],
                    [l[5], l[1], 0.0],
                    [l[4], l[3], l[2]],
                ])
            eps = 1e-3
            identity = torch.eye(3).to(frc_var.device)
            frc_var_mapped = torch.stack([map_lower_triangle(l) for l in frc_var]).to(frc_var.device)
            frc_var_squared = torch.einsum('bij,bkj->bik', frc_var_mapped, frc_var_mapped)
            frc_var_squared += eps * identity

            diff_frc = (tgt_frc - prd_frc)
            inv_cov = torch.linalg.inv(frc_var_squared)
            force_term = torch.einsum('bi,bij,bj->b', diff_frc, inv_cov, diff_frc).mean()
            _, force_log_term = torch.linalg.slogdet(frc_var_squared)
            force_log_term = force_log_term.mean()

            loss_dict = {
                'loss_f': force_term,
                'log_f' : force_log_term,
                'frc_var': frc_var.mean(),
            }

        # Stress
        if tag == 'stress':
            if type(preds) == dict:
                prd_sts = preds['stress'].flatten()
                tgt_sts = targets['stress'].flatten()
            else:
                prd_sts = preds
                tgt_sts = targets

            diff_sts = (tgt_sts - prd_sts)
            stress_term = torch.einsum('i,i->i', diff_sts, diff_sts).mean()

            loss_dict = {'loss_s': stress_term}

        return loss_dict