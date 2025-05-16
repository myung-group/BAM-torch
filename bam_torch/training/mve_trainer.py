import torch
from bam_torch.utils.utils import date
from .base_trainer import BaseTrainer
from .loss import RMSELoss, l2_regularization, NLLLoss


class MVETrainer(BaseTrainer):
    """Trainer for mean-variance estimation (MVE) model
    """
    def __init__(self, json_data, rank, world_size):
        super().__init__(json_data, rank, world_size)
                
    def train(self):
        self.logger.print_logger_head()
        nepoch = self.json_data['NN']['nepoch']
        ## 11) Main loop
        for epoch in range(nepoch):
            epoch_loss_train = self.train_one_epoch(mode='train')
            if self.ddp:
                torch.distributed.barrier()

            if (epoch+1)%self.log_interval == 0:
                epoch_loss_valid = self.train_one_epoch(mode='valid')
                if self.ddp:
                    torch.distributed.barrier()

                ## 12) Save model to pckl file
                if self.rank == 0:
                    if epoch_loss_valid['loss'] < self.loss_test_min:
                        self.loss_test_min = epoch_loss_valid['loss']
                        self.loss_dict['epoch'] = epoch+1+self.start_epoch
                        self.loss_dict['train'] = epoch_loss_train['loss']
                        self.loss_dict['valid'] = epoch_loss_valid['loss']
                        state_dict = self.model.state_dict()
                        if self.ddp:
                            state_dict = self.model.module.state_dict()
                        self.ckpt['params'] = state_dict
                        self.ckpt['opt_state'] = self.optimizer.state_dict()
                        self.ckpt['scheduler'] = self.scheduler.state_dict()
                        self.ckpt['loss'] = self.loss_dict
                        self.l_ckpt_saved = False

                    if (epoch+1)%self.json_data['NN']['nsave'] == 0 and not self.l_ckpt_saved:
                        torch.save(self.ckpt, self.json_data['NN']['fname_pkl'])
                        #self.model.save('model.pt')
                        self.l_ckpt_saved = True
                
                    # Get the last learning rate
                    if self.json_data['scheduler'] != "Null":
                        lr = self.scheduler.get_lr()
                    
                    ## 13) Print out epoch loss
                    step_dict = {
                        "date": date(),
                        "epoch": epoch+1+self.start_epoch,
                    }
                    self.logger.print_epoch_loss(step_dict, 
                                                epoch_loss_train, 
                                                epoch_loss_valid,
                                                lr)
                
                ## 14) Update scheduler (learning rate)
                if self.json_data["scheduler"]["scheduler"] == "ReduceLROnPlateau":
                    metrics = epoch_loss_valid['loss']
                else:
                    metrics = None
                self.scheduler.step(metrics, epoch)
    
    def load_loss(self, reduction='mean'):
        nn_config = self.json_data.get("NN")
        loss_config = nn_config.get("loss_config")
        if loss_config == None:
            if self.json_data["regress_forces"]:
                loss_config = {'energy_loss': 'nll', 'force_loss': 'nll', 'stress_loss': 'mse'}
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
            loss_enr = self.loss_fn["energy_loss"](preds, data, tag="energy")
            loss["loss"].append(e_lambda * loss_enr["loss_e"])
            loss['loss'].append(loss_enr['log_e'])
            loss['loss_e'] = loss_enr['loss_e']
            loss['log_e'] = loss_enr['log_e']
            loss['enr_var'] = loss_enr['enr_var']

            if "forces" in preds:
                loss_frc = self.loss_fn["force_loss"](preds, data, tag="force")
                loss["loss"].append(f_lambda * loss_frc["loss_f"])
                loss['loss'].append(loss_frc['log_f'])
                loss['loss_f'] = loss_frc['loss_f']
                loss['log_f'] = loss_frc['log_f']
                loss['frc_var'] = loss_frc['frc_var']
            if "stress" in preds:
                loss['loss_s'] = self.loss_fn["stress_loss"](preds['stress'].flatten(), 
                                                             data['stress'].flatten())
                loss["loss"].append(s_lambda * loss["loss_s"])

            params = self.model.parameters()
            loss["loss_l2"] = l2_regularization(params)
            loss["loss"].append(lambd * loss["loss_l2"])
            loss["loss"] = sum(loss["loss"])

        else:
            loss = self._compute_loss(preds, data)
        
        return loss

    def _compute_loss(self, preds, data):
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
        cosine_sim = lambda_config.get('cosine_sim')
        energy_grad_mult = lambda_config.get('energy_grad_mult')
        energy_grad_loss = lambda_config.get('energy_grad_loss')

        loss = {"loss": []}
        energy_target = data["energy"].flatten()
        loss["loss_e"] = self.loss_fn["energy_loss"](preds["energy"].flatten(), energy_target)
        loss["loss"].append(e_lambda * loss["loss_e"])

        if "forces" in preds:
            force_target = data["forces"].flatten()
            loss["loss_f"] = self.loss_fn["force_loss"](preds["forces"].flatten(), force_target)
            loss["loss"].append(f_lambda * loss["loss_f"])
                
        # This is for frame-averaging or probabilistic-symmetrization
        if "forces_grad_target" in preds:
            grad_target = preds["forces_grad_target"]
            if cosine_sim:
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                loss["energy_grad_loss"] = -torch.mean(cos(preds["forces"], grad_target))
            else:
                loss["energy_grad_loss"] = self.loss_fn["force_loss"](preds["forces"], grad_target)
        
            if energy_grad_loss:
                loss["loss"].append(energy_grad_mult * loss["energy_grad_loss"])
        
        if "stress" in preds:
            stress_target = data["stress"].flatten()
            loss["loss_s"] = self.loss_fn["stress_loss"](preds["stress"].flatten(), stress_target)
            loss["loss"].append(s_lambda * loss["loss_s"])

        params = self.model.parameters()
        loss["loss_l2"] = l2_regularization(params)
        loss["loss"].append(lambd * loss["loss_l2"])
            
        ## Sanity check to make sure the compute graph is correct.
        #for lc in loss["loss"]:
        #    assert hasattr(lc, "grad_fn")
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
