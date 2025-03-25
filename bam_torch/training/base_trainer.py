import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from e3nn import o3

import os
import sys
import random
import atexit
import pprint
from copy import deepcopy

from bam_torch.utils.logger import Logger
from bam_torch.utils.scheduler import LRScheduler
from bam_torch.utils.utils import get_dataloader, date, on_exit
from bam_torch.model.wrapper_ops import CuEquivarianceConfig
from .loss import RMSELoss, l2_regularization


class BaseTrainer:
    def __init__(self, json_data, rank=0, world_size=1):
        """ The json_data (dict.) should include the following information
        ...
        """
        self.json_data = json_data
        self.rank = rank
        self.world_size = world_size
        self.ddp = False
        if self.world_size > 1:
            self.ddp = True
        self.date1 = date()

        ## 1) Reproducibility
        self.set_random_seed()

        ## 2) Configure device
        self.device = self.configure_device()

        ## 3) Configure model
        self.model, self.n_params, _, self.start_epoch = self.configure_model()
        
        ## 4) Configure optimizer
        self.optimizer = self.configure_optimizer()
        
        ## 5) Configure data_loader
        self.train_loader, self.valid_loader, self.uniq_element, self.enr_avg_per_element = \
                                                        self.configure_dataloader()
        
        ## 6) Configure scheduler
        self.scheduler = self.configure_scheduler()

        ## 7) Configure loss function
        self.loss_fn, self.loss_config = self.load_loss()

        ## 8) Configure logger
        self.log_config, self.log_interval, self.logger = self.configure_logger()

        ## 9) Test train
        epoch_loss_test = self.train_one_epoch(mode='test')
        if self.ddp:
            torch.distributed.barrier()
        self.loss_test_min = epoch_loss_test['loss']

        ## 10) Configure check point dictionary
        self.loss_dict = {'epoch': 0, 'train': [], 'valid': []}
        self.l_ckpt_saved = False
        self.ckpt = {
            'params': self.model.state_dict(),
            'opt_state': self.optimizer.state_dict(), # param_groups,
            'uniq_element': self.uniq_element,
            'enr_avg_per_element': self.enr_avg_per_element,
            'loss': self.loss_dict,
            'input.json': self.json_data,
            'scheduler' : self.scheduler.state_dict()
        }

        # Save input parameters setting
        self.save_input_parameters(self.json_data)
     
    def train(self):
        nepoch = self.json_data['NN']['nepoch']
        ## 11) Main loop
        for epoch in range(nepoch):
            #self.train_loader.sampler.set_epoch(epoch)
            epoch_loss_train = self.train_one_epoch(mode='train')
            #check_parameter_sync(self.model, self.rank)
            if self.ddp:
                torch.distributed.barrier()

            if (epoch+1)%self.log_interval == 0:
                epoch_loss_valid = self.train_one_epoch(mode='test')
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
                        clean_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
                        self.ckpt['params'] = clean_state_dict
                        self.ckpt['opt_state'] = self.optimizer.state_dict()
                        self.ckpt['scheduler'] = self.scheduler.state_dict()
                        self.ckpt['loss'] = self.loss_dict
                        self.l_ckpt_saved = False

                    if (epoch+1)%self.json_data['NN']['nsave'] == 0 and not self.l_ckpt_saved:
                        torch.save(self.ckpt, self.json_data['NN']['fname_pkl'])
                        torch.save(deepcopy(self.model), 'model.pt')
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
            data.to(self.device)
            preds = self.model(deepcopy(data), backprop)
            loss_dict = self.compute_loss(preds, data)
            for l in loss_log_config:
                epoch_loss_dict[l].append(loss_dict.get(l, torch.nan))
            
            loss = loss_dict['loss']
            if backprop:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        epoch_loss_dict = {key: torch.mean(torch.tensor(value)) \
                           for key, value in epoch_loss_dict.items()}      
        return epoch_loss_dict
    
    def configure_dataloader(self):
        json_data = self.json_data
        train_loader, valid_loader, uniq_element, enr_avg_per_element = \
                                                        get_dataloader(
                                                            json_data['fname_traj'],
                                                            json_data['ntrain'],
                                                            json_data['ntest'],
                                                            json_data['nbatch'],
                                                            json_data['cutoff'],
                                                            json_data['NN']['data_seed'],
                                                            json_data['element'],
                                                            json_data['regress_forces'],
                                                            self.rank,
                                                            self.world_size
                                                        )
        return train_loader, valid_loader, uniq_element, enr_avg_per_element

    def configure_logger_head(self):
        log_config = self.json_data.get("log_config")
        if log_config == None:
            if self.json_data["regress_forces"]:
                log_config = {
                    'step': ['date', 'epoch'],
                    'train': ['loss', 'loss_e', 'loss_f'],
                    'valid': ['loss', 'loss_e', 'loss_f'],
                    'lr': ['lr'],
                    }  # loss_l2
            else:
                log_config = {
                    'step': ['date', 'epoch'],
                    'train': ['loss', 'loss_e'],
                    'valid': ['loss', 'loss_e'],
                    'lr': ['lr'],
                    }
        return log_config
    
    def configure_logger(self):
        log_config = self.configure_logger_head()

        log_length = self.json_data.get("log_length")
        if log_length == None:
            log_length = 'simple'
        log_interval = self.json_data.get("log_interval")
        if log_interval == None:
            log_interval == 2

        logger = None
        if self.rank == 0:
            fname = self.get_unique_filename()
            fout = open(fname, 'w')
            logger = Logger(log_config, self.loss_config, log_length, fout)
            logger.print_logger_head()
            separator = logger.get_seperator()
            atexit.register(lambda: on_exit(
                                        fout, 
                                        separator, 
                                        self.n_params, 
                                        self.json_data,
                                        self.date1
                                    )
                            )
        return log_config, log_interval, logger
    
    def get_unique_filename(self):
        train_config = self.json_data.get('train') 
        fname = train_config.get('fname_log') 
        if fname == None:
            fname = "loss_train.out"
        
        base, ext = os.path.splitext(fname) # "loss_train", ".out"
        count = 2
        # Make a unique filename
        while os.path.exists(fname): # if exist the file of ```fname``` in this directory
            old_name = fname
            fname = f"{base}-{count}{ext}"
            count += 1
        # Check this process if restart
        # if restart, make a unique filename
        model_config = self.json_data['NN']
        restart = model_config.get('restart')
        if restart:
            fname = old_name
            base = os.path.splitext(fname)[0]
            fname = f"{base}-re{ext}"
            re_count = 2
            while os.path.exists(fname):
                fname = f"{base}-re{re_count}{ext}"
                re_count += 1

        self.json_data['train']['fname_log'] = fname
        return fname

    def load_loss(self, reduction='mean'):
        nn_config = self.json_data.get("NN")
        loss_config = nn_config.get("loss_config")
        if loss_config == None:
            if self.json_data["regress_forces"]:
                loss_config = {'energy_loss': 'mse', 'force_loss': 'mse'}
            else:
                loss_config = {'energy_loss': 'mse'}
        
        loss_fn = {}
        loss_fn['energy_loss'] = loss_config.get('energy_loss')
        loss_fn['force_loss'] = loss_config.get('force_loss')
        
        for loss, loss_name in loss_fn.items():
            if loss_name in ['l1', 'L1', 'mae', 'MAE']:
                loss_fn[loss] = torch.nn.L1Loss(reduction=reduction)
            elif loss_name in ['mse', 'MSE']:
                loss_fn[loss] = torch.nn.MSELoss(reduction=reduction)
            elif loss_name in ['rmse', 'RMSE']:
                loss_fn[loss] = RMSELoss(reduction=reduction)

        return loss_fn, loss_config

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
        self.msg = ''
        if device_config == 'cpu':
            device = 'cpu'
            self.msg += f'\ndevice:\n\033[33m -- {device}\n'
            try:
                from mpi4py import MPI
                self.rank = MPI.COMM_WORLD.Get_rank()
                size = MPI.COMM_WORLD.Get_size()
                self.msg += f' -- number of cpu  {size}\033[0m\n'
            except:
                self.msg += f' -- number of cpu  {torch.get_num_threads()}\033[0m\n'
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.msg += f'\ndevice:\n\033[33m -- {device}\n'
                
            try:
                if self.ddp:
                    torch.cuda.set_device(self.rank)
                    self.msg += f' -- number of gpu  {self.world_size}\033[0m\n'
                else:
                    self.msg += f' -- number of gpu  {self.world_size}\033[0m\n'
            except:
                self.msg += f' -- number of gpu  1\033[0m\n'

        return device

    def save_input_parameters(self, input_json, fname=None):
        if fname == None:
            train_config = input_json.get('train') 
            if train_config:
                fname = train_config.get('fname_log') 
                if fname == None:
                    fname = "loss_train.out"
            else:
                fname = "loss_train.out"

            fname_ls = fname.rsplit('.', 1)
            fname = f'input_json_of_{fname_ls[0]}_{fname_ls[1]}.txt'
        
        fout = open(fname, 'w')
        pprint.pprint(input_json, stream=fout)

        return fname
    
    def configure_model(self):
        """ Configure model using model configuration dictionary.
        """
        model = self.set_model() # Set self.model
        model.to(self.device)

        model_config = self.json_data['NN']
        restart = model_config.get('restart')
        
        evaluate_config = self.json_data['predict']
        evaluate = evaluate_config.get('evaluate_tag')  # True or False(None)

        if restart:
            rank = self.rank
            evaluate = False
            self.json_data['predict']['evaluate_tag'] = False
            model_ckpt = torch.load(model_config["fname_pkl"])
            start_epoch = model_ckpt['loss']['epoch']
            try:
                model.load_state_dict(model_ckpt['params'])
                if self.ddp:
                    model = DDP(model, device_ids=[self.rank])
            except RuntimeError as e:
                input_json = model_ckpt['input.json']
                fname = self.save_input_parameters(input_json, 'input_json_from_trained_model.txt')
                print(e)
                print(f'\n\033[31mSome of the parameter dimensions in the trained model you are trying to use')
                print(f'do not match the current input parameters.')
                print(f' -- Please check the ```{fname}``` file\033[0m\n')
                sys.exit(1)
            self.msg += f'\n\033[32mrestarting training from the {model_config["fname_pkl"]}\033[0m\n'
            self.msg += f' -- restarting from the step where the loss was {model_ckpt["loss"]}\n'
        
        if evaluate:  # True or False(None)
            rank = 0
            start_epoch = 0
            model_ckpt = torch.load(evaluate_config["model"])
            model.load_state_dict(model_ckpt['params'])
            model.eval()
            self.msg += f'\n\033[32mevaluating the {evaluate_config["model"]}\033[0m\n'

        if not restart and not evaluate: # initial train case
            rank = self.rank
            start_epoch = 0
            model_ckpt = None
            if self.ddp:
                model = DDP(model, device_ids=[self.rank])
            self.msg += f'\n\033[32minitializing training, results will be saved in the {model_config["fname_pkl"]}\033[0m\n'
        
        # Check the number of parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.msg += f'\nnumber of parameters:\n\033[33m -- model ({self.json_data["model"]})  {n_params}\033[0m\n'
        
        if rank == 0:
            print(self.msg)

        return model, n_params, model_ckpt, start_epoch
        
    def set_model(self):
        model_config = self.json_data
        cutoff = model_config['cutoff']
        avg_num_neighbors = model_config['avg_num_neighbors']
        num_species = model_config['num_species']
        hidden_irreps = o3.Irreps(model_config['hidden_channels'])
        features_dim = model_config['features_dim']
        num_basis_func = model_config['num_radial_basis']
        nlayers = model_config['nlayers']
        max_ell = model_config['max_ell']
        
        output_irreps = model_config.get('output_channels')
        if output_irreps == None:
            output_irreps = "1x0e"
        active_fn = model_config.get('active_fn')
        if active_fn == None:
            active_fn = "identity"
        regress_forces = model_config.get('regress_forces')
        if regress_forces:
            regress_forces = "direct"
        
        cueq_config = model_config.get('cueq_config')  # true or false
        if cueq_config == None or cueq_config:
            try:
                import cuequivariance as cue
                import cuequivariance_torch as cuet
                CUET_AVAILABLE = True
            except ImportError:
                CUET_AVAILABLE = False
            if CUET_AVAILABLE:
                cueq_config = CuEquivarianceConfig(
                    enabled=True,
                    layout="ir_mul",
                    group="O3_e3nn",
                    optimize_all=True,
                )
                self.msg += f'\nequiv. lib.:\n\033[33m -- CuEquivariance\033[0m\n'
        else:
            cueq_config = None
            self.msg += f'\nequiv. lib.:\n\033[33m -- e3nn\033[0m\n'
        
        model = model_config["model"]
        if model in ["race", "RACE", "Race"]:
            from bam_torch.model.models import RACE
            model = RACE(
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
                regress_forces=regress_forces,
                cueq_config=cueq_config
                )
        elif model in ["mace", "MACE", "Mace"]:
            from bam_torch.model.models import MACE
            model = MACE(
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
                regress_forces=regress_forces,
                cueq_config=cueq_config
                ) 
        return model

    def configure_optimizer(self):
        """ Configure optimizer using model configuration dictionary.
        """
        optimizer = self.set_optimizer()
        model_config = self.json_data['NN']
        restart = model_config.get('restart')
        if restart:
            model_ckpt = torch.load(model_config["fname_pkl"])
            optimizer.load_state_dict(model_ckpt['opt_state'])
        return optimizer

    def set_optimizer(self):
        optim_config = self.json_data["NN"]
        lr = optim_config.get('learning_rate')
        if lr == None:
            lr = 0.001
        weight_decay = optim_config.get('weight_decay')
        if weight_decay == None:
            weight_decay = 1e-12
        amsgrad = optim_config.get('amsgrad')  
        if amsgrad == None:
            amsgrad = True

        optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=lr, 
                                          weight_decay=weight_decay,
                                          amsgrad=amsgrad)
        return optimizer

    def configure_scheduler(self):
        scheduler = self.set_scheduler()
        model_config = self.json_data['NN']
        restart = model_config.get('restart')
        if restart:
            model_ckpt = torch.load(model_config["fname_pkl"])
            scheduler.load_state_dict(model_ckpt['scheduler'])
        return scheduler

    def set_scheduler(self):
        scheduler_config = self.json_data["scheduler"]
        scheduler_config["lr_init"] = self.json_data["NN"]["learning_rate"]
        scheduler = LRScheduler(self.optimizer, scheduler_config)
        return scheduler

    def get_params(self):
        return self.n_params
    
    def check_parameter_sync(self):
        for name, param in self.model.named_parameters():
            print(f"Rank {self.rank}, Parameter name: {name}, Value: {param.data[0]}")
