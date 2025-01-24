import torch
import numpy as np
from e3nn import o3

import os
import random
import atexit
from copy import deepcopy

from bam_torch.utils.logger import Logger
from bam_torch.utils.scheduler import LRScheduler
from bam_torch.utils.utils import get_dataloader, date, on_exit


class RMSELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super(RMSELoss,self).__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)
        self.eps = 1e-7

    def forward(self,y,y_hat):
        return torch.sqrt(self.mse(y,y_hat) + self.eps)


def l2_regularization(params):
    wgt = torch.cat([p.view(-1) for p in params if p.requires_grad])
    #print(wgt)
    return (wgt * wgt).mean()


class BaseTrainer:
    def __init__(self, json_data):
        """ The json_data (dict.) should include the following information
        ...
        """
        self.json_data = json_data
        date1 = date()

        ## 1) Reproducibility
        self.set_random_seed()

        ## 2) Configure device
        self.configure_device()

        ## 3) Configure model
        self.configure_model()
        self.model.to(self.device)
        # Check the number of parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'\nnumber of parameters:\n -- model {n_params}\033[0m\n')

        ## 4) Configure optimizer
        self.configure_optimizer()
        
        ## 5) Configure data_loader
        self.train_loader, self.valid_loader, self.uniq_element, self.enr_avg_per_element = \
                                                        get_dataloader(
                                                            json_data['fname_traj'],
                                                            json_data['ntrain'],
                                                            json_data['ntest'],
                                                            json_data['nbatch'],
                                                            json_data['cutoff'],
                                                            json_data['NN']['data_seed'],
                                                            json_data['element'],
                                                            json_data['regress_forces']
                                                        )
        
        ## 6) Configure scheduler
        scheduler_config = json_data["scheduler"]
        scheduler_config["lr_init"] = json_data["NN"]["learning_rate"]
        self.scheduler = LRScheduler(self.optimizer, scheduler_config)

        ## 7) Configure loss function
        self.loss_fn = self.load_loss()

        ## 8) Configure logger
        self.log_config = json_data.get("log_config")
        if self.log_config == None:
            if json_data["regress_forces"]:
                self.log_config = {
                    'step': ['date', 'epoch'],
                    'train': ['loss', 'loss_e', 'loss_f'],
                    'valid': ['loss', 'loss_e', 'loss_f'],
                    'lr': ['lr'],
                    }  # loss_l2
            else:
                self.log_config = {
                    'step': ['date', 'epoch'],
                    'train': ['loss', 'loss_e'],
                    'valid': ['loss', 'loss_e'],
                    'lr': ['lr'],
                    }
        self.log_length = json_data.get("log_length")
        if self.log_length == None:
            self.log_length = 'simple'
        self.log_interval = json_data.get("log_interval")
        if self.log_interval == None:
            self.log_interval == 2

        train_config = json_data.get('train') 
        fname = train_config.get('fname_log') 
        if fname == None:
            fname = "loss_train.out"
        self.fout = open(fname, 'w')
        self.logger = Logger(self.log_config, self.loss_config, self.log_length)
        self.separator = self.logger.print_logger_head(self.fout)
        atexit.register(lambda: on_exit(
                                    self.fout, 
                                    self.separator, 
                                    n_params, 
                                    json_data,
                                    date1
                                )
                        )

        ## 9) Test train
        epoch_loss_test = self.train_one_epoch(mode='test')
        self.loss_test_min = epoch_loss_test['loss']

        ## 10) Configure check point dictionary
        self.loss_dict = {'train': [], 'valid': []}
        self.l_ckpt_saved = False
        self.ckpt = {
            'params': self.model.state_dict(),
            'opt_state': self.optimizer.param_groups,
            'uniq_element': self.uniq_element,
            'enr_avg_per_element': self.enr_avg_per_element,
            'loss': self.loss_dict
        }
    
    def train(self):
        nepoch = self.json_data['NN']['nepoch']
        ## 11) Main loop
        for epoch in range(nepoch):
            epoch_loss_train = self.train_one_epoch(mode='train')

            if (epoch+1)%self.log_interval == 0:
                epoch_loss_valid = self.train_one_epoch(mode='test')

                ## 12) Save model to pckl file
                if epoch_loss_valid['loss'] < self.loss_test_min:
                    self.loss_test_min = epoch_loss_valid['loss']
                    self.loss_dict['train'] = epoch_loss_train['loss']
                    self.loss_dict['valid'] = epoch_loss_valid['loss']
                    self.ckpt['params'] = self.model.state_dict()
                    self.ckpt['opt_state'] = self.optimizer.param_groups
                    self.ckpt['loss'] = self.loss_dict
                    self.l_ckpt_saved = False

                if (epoch+1)%self.json_data['NN']['nsave'] == 0 and not self.l_ckpt_saved:
                    torch.save(self.ckpt, self.json_data['NN']['fname_pkl'])
                    self.l_ckpt_saved = True
                
                # Get the last learning rate
                if self.json_data['scheduler'] != "Null":
                    lr = self.scheduler.get_lr()
                
                ## 13) Print out epoch loss
                #date = date()
                step_dict = {
                    "date": date(),
                    "epoch": epoch,
                }
                self.logger.print_epoch_loss(step_dict, 
                                             epoch_loss_train, 
                                             epoch_loss_valid,
                                             lr,
                                             self.fout)
                
                ## 14) Update scheduler (learning rate)
                if self.json_data["scheduler"]["scheduler"] == "ReduceLROnPlateau":
                    metrics = epoch_loss_valid['loss']
                else:
                    metrics = None
                self.scheduler.step(metrics, epoch)


    def train_one_epoch(self, mode='train', data_loader=None):
        if mode == 'train':
            self.model.train()
            backprop = True
            loss_log_config = self.log_config['train']
            if data_loader == None:
                data_loader = self.train_loader
        else:
            self.model.eval()
            backprop = False
            loss_log_config = self.log_config['valid']
            if data_loader == None:
                data_loader = self.valid_loader

        epoch_loss_dict = {key: [] for key in loss_log_config}
        for data in data_loader:
            data = data.to(self.device)
            preds = self.model(deepcopy(data), backprop)
            loss_dict = self.compute_loss(preds, data)
            for l in loss_log_config:
                epoch_loss_dict[l].append(loss_dict[l])
            
            loss = loss_dict['loss']
            if backprop:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        epoch_loss_dict = {key: torch.mean(torch.tensor(value)) \
                           for key, value in epoch_loss_dict.items()}      
        return epoch_loss_dict

    def load_loss(self, reduction='mean'):
        nn_config = self.json_data.get("NN")
        self.loss_config = nn_config.get("loss_config")
        if self.loss_config == None:
            if self.json_data["regress_forces"]:
                self.loss_config = {'energy_loss': 'mse', 'force_loss': 'mse'}
            else:
                self.loss_config = {'energy_loss': 'mse'}
        
        loss_fn = {}
        loss_fn['energy_loss'] = self.loss_config.get('energy_loss')
        loss_fn['force_loss'] = self.loss_config.get('force_loss')
        
        for loss, loss_name in loss_fn.items():
            if loss_name in ['l1', 'L1', 'mae', 'MAE']:
                loss_fn[loss] = torch.nn.L1Loss(reduction=reduction)
            elif loss_name in ['mse', 'MSE']:
                loss_fn[loss] = torch.nn.MSELoss(reduction=reduction)
            elif loss_name in ['rmse', 'RMSE']:
                loss_fn[loss] = RMSELoss(reduction=reduction)

        return loss_fn

    def compute_loss(self, preds, data):
        e_lambda = self.json_data["NN"]['enr_lambda']
        f_lambda = self.json_data["NN"]['frc_lambda']
        cosine_sim = self.json_data["NN"]['cosine_sim']
        energy_grad_mult = self.json_data["NN"]['energy_grad_mult']
        energy_grad_loss = self.json_data["NN"]['energy_grad_loss']
        lambd = self.json_data["NN"]['l2_lambda']

        loss = {"loss": []}
        energy_target = data.energy.flatten()
        loss["loss_e"] = self.loss_fn["energy_loss"](preds["energy"].flatten(), energy_target)
        loss["loss"].append(e_lambda * loss["loss_e"])

        if "forces" in preds:
            force_target = data.forces.flatten()
            loss["loss_f"] = self.loss_fn["force_loss"](preds["forces"].flatten(), force_target)
            loss["loss"].append(f_lambda * loss["loss_f"])
                
        if "forces_grad_target" in preds:
            grad_target = preds["forces_grad_target"]
            if cosine_sim:
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                loss["energy_grad_loss"] = -torch.mean(cos(preds["forces"], grad_target))
            else:
                loss["energy_grad_loss"] = self.loss_fn["force_loss"](preds["forces"], grad_target)
        
            if energy_grad_loss:
                loss["loss"].append(energy_grad_mult * loss["energy_grad_loss"])
        
        if lambd != 0:
            params = self.model.parameters()
            loss["loss_l2"] = l2_regularization(params)
            loss["loss"].append(lambd * loss["loss_l2"])
            
        # Sanity check to make sure the compute graph is correct.
        for lc in loss["loss"]:
            assert hasattr(lc, "grad_fn")

        loss["loss"] = sum(loss["loss"])
        return loss

    def set_random_seed(self):
        """ Initializes random seeds and settings to ensure reproducibility.
        
        Parameters:
            rng_seed (int): The random seed value.
            cublas_config (str): Configuration for CUBLAS workspace (default: ":16:8").
        """
        rng_seed = self.json_data['NN']['data_seed']
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        torch.manual_seed(rng_seed)
        torch.cuda.manual_seed_all(rng_seed)
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"]= ":16:8"  # :4096:8
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False    

    def configure_device(self):
        device_config = self.json_data['device']
        if device_config == 'cpu':
            self.device = 'cpu'
            print(f'\ndevice:\n\033[33m -- {self.device}\033[0m')
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f'\ndevice:\n\033[33m -- {self.device}\033[0m')
    
    def configure_model(self):
        """ Configure model using model configuration dictionary.
        """
        model_config = self.json_data
        cutoff = model_config['cutoff']
        avg_num_neighbors = model_config['avg_num_neighbors']
        num_species = model_config['num_species']
        hidden_irreps = o3.Irreps(model_config['hidden_channels'])
        features_dim = model_config['features_dim']
        num_basis_func = model_config['num_radial_basis']
        nlayers = model_config['nlayers']
        max_ell = model_config['max_ell']
        
        output_irreps = model_config['output_channels']
        if output_irreps == None:
            output_irreps = "1x0e"
        active_fn = model_config['active_fn']
        if active_fn == None:
            active_fn = "swish"
        regress_forces = model_config['regress_forces']
        if regress_forces:
            regress_forces = "direct"

        model = model_config["model"]
        if model in ["race", "RACE", "Race"]:
            from bam_torch.model.models import RACE
            self.model = RACE(
                cutoff=cutoff,
                avg_num_neighbors=avg_num_neighbors,
                num_species=num_species,
                max_ell=max_ell,
                num_basis_func=num_basis_func,
                hidden_irreps=hidden_irreps,
                nlayers=nlayers,
                features_dim=features_dim,
                output_irreps=output_irreps,
                active_fn=active_fn,
                regress_forces=regress_forces
                )
        elif model in ["mace", "MACE", "Mace"]:
            from bam_torch.model.models import MACE
            self.model = MACE(
                cutoff=cutoff,
                avg_num_neighbors=avg_num_neighbors,
                num_species=num_species,
                max_ell=max_ell,
                num_basis_func=num_basis_func,
                hidden_irreps=hidden_irreps,
                nlayers=nlayers,
                features_dim=features_dim,
                output_irreps=output_irreps,
                active_fn=active_fn,
                regress_forces=regress_forces
                ) 
        
    def configure_optimizer(self):
        optim_config = self.json_data["NN"]
        lr = optim_config['learning_rate']
        if lr == None:
            lr = 0.001
        weight_decay = optim_config['weight_decay']  
        if weight_decay == None:
            weight_decay = 1e-12
        amsgrad = optim_config.get('amsgrad')  
        if amsgrad == None:
            amsgrad = True

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=lr, 
                                          weight_decay=weight_decay,
                                          amsgrad=amsgrad)


