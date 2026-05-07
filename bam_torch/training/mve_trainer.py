import torch
from bam_torch.utils.utils import date
from .base_trainer import BaseTrainer
from .loss import RMSELoss, l2_regularization, NLLLoss


class MVETrainer(BaseTrainer):
    """Trainer for mean-variance estimation (MVE) model
    """
    def __init__(self, json_data, rank, world_size):
        super().__init__(json_data, rank, world_size)
    
    def configure_loss(self, reduction='mean'):
        nn_config = self.json_data.get("NN")
        loss_config = nn_config.get("loss_config")
        if loss_config == None:
            if self.json_data["regress_forces"]:
                loss_config = {
                    'energy_loss': 'nll', 
                    'force_loss': 'nll', 
                    'stress_loss': 'mse'
                }
            else:
                loss_config = {'energy_loss': 'nll'}
        
        loss_fn = {}
        loss_fn['energy_loss'] = loss_config.get('energy_loss')
        loss_fn['force_loss'] = loss_config.get('force_loss')
        loss_fn['stress_loss'] = loss_config.get('stress_loss')
        
        for loss, loss_name in loss_fn.items():
            if loss_name in ['l1', 'L1', 'mae', 'MAE']:
                loss_fn[loss] = torch.nn.L1Loss(reduction=reduction)
            elif loss_name in ['mse', 'MSE']:
                loss_fn[loss] = torch.nn.MSELoss(reduction=reduction)
            elif loss_name in ['rmse', 'RMSE']:
                loss_fn[loss] = RMSELoss(reduction=reduction)
            elif loss_name in ['nll', 'NLL']:
                loss_fn[loss] = NLLLoss()

        return loss_fn, loss_config

    def compute_loss(self, preds, data):
        if 'nll' in self.loss_config.values() or 'NLL' in self.loss_config.values():
            lambda_config = self.json_data["NN"]
            e_lambda = lambda_config.get('enr_lambda', 1)
            f_lambda = lambda_config.get('frc_lambda', 1)
            s_lambda = lambda_config.get('str_lambda', 1)
            lambd = lambda_config.get('l2_lambda', 0)

            loss = {"loss": []}
            loss_enr = self.loss_fn["energy_loss"](preds, data, tag="energy")
            loss["loss"].append(e_lambda * loss_enr["loss_e"])
            loss['loss'].append(loss_enr['log_e'])
            loss['loss_e'] = loss_enr['loss_e']
            loss['log_e'] = loss_enr['log_e']
            loss['enr_var'] = loss_enr['enr_var']

            if "forces" in preds and self.loss_fn.get("force_loss") is not None:
                loss_frc = self.loss_fn["force_loss"](preds, data, tag="force")
                loss["loss"].append(f_lambda * loss_frc["loss_f"])
                loss['loss'].append(loss_frc['log_f'])
                loss['loss_f'] = loss_frc['loss_f']
                loss['log_f'] = loss_frc['log_f']
                loss['frc_var'] = loss_frc['frc_var']
            if "stress" in preds and self.loss_fn.get("stress_loss") is not None:
                loss['loss_s'] = self.loss_fn["stress_loss"](preds['stress'].flatten(), 
                                                             data['stress'].flatten())
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
                loss["loss"] = sum(loss["loss"])

        else:
            loss = self._compute_loss(preds, data)
        
        return loss

    def _compute_loss(self, preds, data):
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
            
        loss["loss"] = sum(loss["loss"])
        if "energy_var" in preds:
            loss["enr_var"] = preds["energy_var"].mean()
        if "forces_var" in preds:
            loss["frc_var"] = preds["forces_var"].mean()
        return loss

    def configure_logger_head(self):
        log_config = self.json_data.get("log_config")
        if log_config == None:
            if self.json_data["regress_forces"]:
                log_config = {
                    'step': ['date', 'epoch'],  
                    'train': ['loss', 'loss_e', 'loss_f', 'log_e', 'log_f', 'loss_s', 'enr_var', 'frc_var', 'loss_l2'], 
                    'valid': ['loss', 'loss_e', 'loss_f', 'log_e', 'log_f'],
                    'lr': ['lr'],
                    }  # loss_l2
            else:
                log_config = {
                    'step': ['date', 'epoch'],
                    'train': ['loss', 'loss_e', 'enr_var'],
                    'valid': ['loss', 'loss_e'],
                    'lr': ['lr'],
                    }
        return log_config
