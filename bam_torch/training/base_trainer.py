import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Batch as DataBatch
from torch_ema import ExponentialMovingAverage

from e3nn import o3
from e3nn.util import jit

import gc
import os
import sys
import random
import atexit
import inspect
import pprint
import numpy as np
from contextlib import nullcontext
from copy import deepcopy
from time import time

from bam_torch.utils.logger import Logger
from bam_torch.utils.scheduler import LRScheduler
from bam_torch.utils.utils import get_dataloader, date, on_exit
from bam_torch.model.wrapper_ops import (
    CuEquivarianceConfig,
    OEQConfig,
    CUET_AVAILABLE,
    OEQ_AVAILABLE,
)
from bam_torch.model import MODEL_REGISTRY
from .loss import RMSELoss, l2_regularization


class BaseTrainer:
    def __init__(self, json_data, rank=0, world_size=1):
        """ Base class for training models in BAM.

        Args:
            json_data (dict): Parsed configuration dictionary (from input.json).
            rank       (int): Process rank (for DDP).
            world_size (int): Total number of processes (for DDP).
        ...
        """
        self.date1 = date()
        self.json_data = json_data
        self.rank = rank
        self.world_size = world_size
        self.ddp = self.world_size > 1
        self.l_ckpt_saved = False
        self.msg = ''

        # Run setup routines
        self.setup()

    def setup(self):
        """Configure all core training components.

        Sets up device, model, optimizer, dataloader, scheduler,
        loss function, and logger.
        """
        self.set_random_seed() # Reproducibility
        self.device = self.configure_device()
        self.model, self.n_params, self.model_ckpt, self.start_epoch = self.configure_model()
        self.optimizer = self.configure_optimizer()
        self.train_loader, self.valid_loader, self.uniq_element, self.enr_avg_per_element \
                       = self.configure_dataloader()
        self.scheduler = self.configure_scheduler()
        self.loss_fn, self.loss_config = self.configure_loss()
        self.log_config, self.log_interval, self.logger = self.configure_logger()
        self.loss_dict, self.ckpt = self.configure_checkpoint()
        self.ema = self.configure_exponential_moving_average()

    def train(self):
        """Main training loop for BAM models.
        """
        # Initial test
        self.initial_test()

        # Print logger head
        if self.rank == 0:
            self.logger.print_logger_head()

        # Main training loop
        nepoch = self.json_data['NN']['nepoch']
        for epoch in range(nepoch):
            try:
                self.model.update_criterion_value(epoch+self.start_epoch+1)
            except:
                pass

            epoch_loss_train = self.train_one_epoch(mode='train')
            if self.ddp:
                torch.distributed.barrier()

            if (epoch+1)%self.log_interval == 0:
                # Evaluate with EMA-averaged parameters
                param_context = (
                    self.ema.average_parameters() if self.ema is not None else nullcontext()
                )
                with param_context:
                    epoch_loss_valid = self.train_one_epoch(mode='valid')
                    if self.ddp:
                        torch.distributed.barrier()

                # Snapshot/log with live (non-EMA) parameters
                if self.rank == 0:
                    # Update check point
                    if epoch_loss_valid['loss'] < self.loss_test_min:
                        self.update_check_point(epoch, epoch_loss_train, epoch_loss_valid)
                        self.loss_test_min = epoch_loss_valid['loss']
                        self.l_ckpt_saved = False

                    # Print epoch loss
                    self.print_logger(epoch, epoch_loss_train, epoch_loss_valid)

                    # Free GPU memory
                    torch.cuda.empty_cache()
                    gc.collect()

                # Update scheduler (learning rate)
                metrics = None
                scheduler_cfg = self.json_data.get("scheduler", {})
                if isinstance(scheduler_cfg, dict) \
                        and scheduler_cfg.get("scheduler") == "ReduceLROnPlateau":
                    metrics = epoch_loss_valid['loss']
                self.scheduler.step(metrics, epoch)

            # Save check point
            if (epoch+1)%self.json_data['NN']['nsave'] == 0 and not self.l_ckpt_saved:
                self.ckpt['train_scale_shift'] = {
                    k: (torch.stack(v).mean() if len(v) > 0 else torch.tensor(0.0, device=self.device))
                        for k, v in self.ckpt['train_scale_shift'].items()
                }
                self.ckpt['valid_scale_shift'] = {
                    k: (torch.stack(v).mean() if len(v) > 0 else torch.tensor(0.0, device=self.device))
                        for k, v in self.ckpt['valid_scale_shift'].items()
                }
                self.ckpt['valid_scale_shift_origin'] = torch.tensor(
                    self.ckpt['valid_scale_shift_origin']
                ).mean()
                torch.save(self.ckpt, self.json_data['NN']['fname_pkl'])
                self.l_ckpt_saved = True

    def initial_test(self):
        """Run a preliminary test epoch
        and record the initial reference loss (loss_test_min).
        """
        param_context = (
            self.ema.average_parameters() if self.ema is not None else nullcontext()
        )
        with param_context:
            epoch_loss_test = self.train_one_epoch(mode='test')
            if self.ddp:
                torch.distributed.barrier()
            self.loss_test_min = epoch_loss_test['loss']

    def train_one_epoch(self, mode='train', data_loader=None):
        if mode == 'train':
            self.model.train()
            backprop = True
            loss_log_config = self.log_config['train']
            if data_loader is None:
                data_loader = self.train_loader
            self.ckpt['train_scale_shift'] = {
                    k: [] for k in self.enr_avg_per_element.keys()
            }
        else:  # test or valid
            self.model.eval()
            backprop = False
            loss_log_config = self.log_config['valid']
            if data_loader is None:
                data_loader = self.valid_loader
            if mode == 'valid':
                self.ckpt['valid_scale_shift'] = {
                    k: [] for k in self.enr_avg_per_element.keys()
                }
                self.ckpt['valid_scale_shift_origin'] = []

        epoch_loss_dict = {key: [] for key in loss_log_config}
        for data in data_loader:
            data = self.move_to_device(data, self.device)
            # Predict energy, forces, and so on from model
            preds = self.model(data, backprop)
            preds = self.scale_shift(preds, data, mode)

            loss_dict = self.compute_loss(preds, data)
            loss = loss_dict['loss']
            if backprop:
                self.optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)
                self.optimizer.step()

                if self.ema is not None:
                    self.ema.update()

            for l in loss_log_config:
                val = loss_dict.get(l, torch.nan)
                epoch_loss_dict[l].append(val.detach().cpu() if isinstance(val, torch.Tensor) else val)

        torch.cuda.synchronize()
        epoch_loss_dict = {key: torch.mean(torch.tensor(value)) \
                           for key, value in epoch_loss_dict.items()}
        return epoch_loss_dict

    def scale_shift(self, preds, data, mode):
        energy_target = data["energy"].flatten()
        energy_predict = preds["energy"].flatten()
        shift_enr = energy_target.mean() - energy_predict.mean()
        preds["energy"] = energy_predict + shift_enr
        # Calculate element-wise scale-shift
        node_energy = preds["node_energy"].detach()
        energy = energy_predict.detach()
        batch = data["batch"]
        ratios = node_energy/energy[batch]
        shift_node_enr = shift_enr * ratios
        species = data["species"][:len(node_energy)]
        # Save scale-shift values
        if mode == 'train':
            for i in range(len(species)):
                self.ckpt['train_scale_shift'][species[i].item()].append(
                    shift_node_enr[i].detach().cpu())
        elif mode == 'valid':
            for i in range(len(species)):
                self.ckpt['valid_scale_shift'][species[i].item()].append(
                    shift_node_enr[i].detach().cpu())
                self.ckpt['valid_scale_shift_origin'].append(shift_enr.detach().cpu())
        return preds

    def update_check_point(self, epoch, epoch_loss_train, epoch_loss_valid):
        self.loss_dict.update({
            'epoch': epoch + 1 + self.start_epoch,
            'train': epoch_loss_train['loss'],
            'valid': epoch_loss_valid['loss']
        })
        state_dict = self.model.state_dict()
        if self.ddp:
            state_dict = self.model.module.state_dict()
        ema_state = None
        if self.ema is not None:
            ema_state = self.ema.state_dict()
        self.ckpt.update({
            'params': state_dict,
            'opt_state': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss': self.loss_dict,
            'ema_state': ema_state
        })

    def print_logger(self, epoch, epoch_loss_train, epoch_loss_valid):
        # Get the last learning rate
        lr = self.scheduler.get_lr() \
            if self.json_data['scheduler'] != "Null" \
            else self.json_data['NN']['learning_rate']
        # Print epoch loss
        step_dict = {
            "date": date(),
            "epoch": epoch+1+self.start_epoch,
        }
        self.logger.print_epoch_loss(step_dict,
                                    epoch_loss_train,
                                    epoch_loss_valid,
                                    lr)

    def configure_dataloader(self):
        train_loader, valid_loader, uniq_element, enr_avg_per_element = \
            get_dataloader(
                self.json_data['fname_traj'],
                self.json_data['ntrain'],
                self.json_data['nvalid'],
                self.json_data['nbatch'],
                self.json_data['cutoff'],
                self.json_data['NN']['data_seed'],
                self.json_data['element'],
                self.json_data['regress_forces'],
                self.json_data.get('max_neigh'),
                self.rank,
                self.world_size
            )
        return train_loader, valid_loader, uniq_element, enr_avg_per_element

    def configure_logger_head(self):
        log_config = self.json_data.get("log_config")

        # Set a default of log_config
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
        """Logging configuration specification.

        - log_length: 'simple' or 'precise'
            Controls the detail level of printed logs.
        - log_interval: int
            Print logs every N steps during training.
        """
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
        """Manage log file naming for training and restart (re-training) runs.

        Behavior:
        - If `fname = "loss_train.out"` and the file already exists in the current directory:
            - Automatically rename the new log to `loss_train-{n}.out` to avoid overwriting.
            - `{n}` starts from 2, 3, ... and increases based on the largest existing index.
        - On restart:
            - The log file with the largest `{n}` index is treated as the most recent one.
            - The new log is saved as `loss_train-{n}-re{m}.out`.
            - `{m}` follows the same increment rule as `{n}`, starting from 2, 3, ...
              based on existing restart logs

        This mechanism ensures previous logs are never accidentally overwritten.
        """
        train_config = self.json_data.get('train')
        fname = train_config.get('fname_log')
        if fname == None:
            fname = "loss_train.out"

        base, ext = os.path.splitext(fname) # "loss_train", ".out"
        count = 2
        old_name = fname
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

    def configure_loss(self, reduction='mean'):
        nn_config = self.json_data.get("NN")
        loss_config = nn_config.get("loss_config", {})

        loss_fn = {}
        loss_fn['energy_loss'] = loss_config.get('energy_loss', 'mse')
        loss_fn['force_loss'] = loss_config.get('force_loss', 'mse')
        loss_fn['stress_loss'] = loss_config.get('stress_loss', None)

        for loss, loss_name in loss_fn.items():
            if loss_name in ['l1', 'L1', 'mae', 'MAE']:
                loss_fn[loss] = torch.nn.L1Loss(reduction=reduction)
            elif loss_name in ['mse', 'MSE']:
                loss_fn[loss] = torch.nn.MSELoss(reduction=reduction)
            elif loss_name in ['rmse', 'RMSE']:
                loss_fn[loss] = RMSELoss(reduction=reduction)

        return loss_fn, loss_config

    def compute_loss(self, preds, data):
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
            stress_pred = preds["stress"].reshape(-1, 6)
            stress_target = data["stress"].reshape(-1, 6)
            stress_valid = data.get("stress_valid")
            if stress_valid is None:
                raise KeyError("stress_valid is required for stress loss")
            if not isinstance(stress_valid, torch.Tensor) or stress_valid.dtype != torch.bool:
                raise TypeError("stress_valid must be a torch.bool tensor")
            if stress_valid.ndim != 1 or stress_valid.numel() != stress_pred.shape[0]:
                raise ValueError(
                    "stress_valid must have one bool value per stress configuration"
                )
            if stress_pred.shape != stress_target.shape:
                raise ValueError(
                    "stress and stress target must have matching [-1, 6] shapes"
                )
            if stress_valid.any():
                valid_stress_pred = stress_pred[stress_valid].reshape(-1)
                valid_stress_target = stress_target[stress_valid].reshape(-1)
                loss["loss_s"] = self.loss_fn["stress_loss"](
                    valid_stress_pred, valid_stress_target
                )
            else:
                loss["loss_s"] = stress_pred[stress_valid].sum()
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

        # Get loss:
        # loss = (e_lambda * loss_e) + (f_lambda * loss_f)
        #        + (s_lambda * loss_s) + (lambd * loss_l2)
        loss["loss"] = sum(loss["loss"])
        return loss

    def set_random_seed(self):
        """ Initializes random seeds and settings to ensure reproducibility.

        Parameters:
            rng_seed (int): The random seed value.
            cublas_config (str): Configuration for CUBLAS workspace (default: ":16:8").
        """
        rng_seed = self.json_data['NN']['init_seed']
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
            device = 'cpu'
            self.msg += f'\ndevice:\n\033[33m -- {device}\n'
            try:
                from mpi4py import MPI
                self.rank = MPI.COMM_WORLD.Get_rank()
                size = MPI.COMM_WORLD.Get_size()
                self.msg += f' -- number of cpu  {size}\n\033[0m\n'
            except:
                self.msg += f' -- number of cpu  {torch.get_num_threads()}\033[0m\n'
        else:
            if torch.cuda.is_available():
                if self.ddp:
                    # Use local rank for device in multi-node setup
                    local_rank = self.rank % torch.cuda.device_count()
                    torch.cuda.set_device(local_rank)
                    device = torch.device(f"cuda:{local_rank}")
                    self.msg += f'\ndevice:\n\033[33m -- {device} (local_rank: {local_rank})\n'
                    self.msg += f' -- number of gpu  {self.world_size}\033[0m\n'
                else:
                    device = torch.device("cuda")
                    self.msg += f'\ndevice:\n\033[33m -- {device}\n'
                    self.msg += f' -- number of gpu  {self.world_size}\033[0m\n'
            else:
                device = torch.device("cpu")
                self.msg += f'\ndevice:\n -- {device}\n'

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
        #scripted_model = jit.compile(model)
        model.to(self.device)

        # Set criterion for direct force training
        criterion = self.json_data.get("criterion")
        criterion_value = self.json_data.get("criterion_value")
        try:
            model.set_criterion(criterion, criterion_value)
        except:
            pass

        model_config = self.json_data['NN']
        restart = model_config.get('restart')

        evaluate_config = self.json_data['predict']
        evaluate = evaluate_config.get('evaluate_tag')  # True or False(None)

        rank = self.rank
        start_epoch = 0
        model_ckpt = None

        if restart:
            rank = self.rank
            evaluate = False
            self.json_data['predict']['evaluate_tag'] = False
            model_ckpt = torch.load(
                model_config["fname_pkl"],
                map_location=self.device,
                weights_only=False
            )
            start_epoch = model_ckpt['loss']['epoch']
            try:
                model.load_state_dict(model_ckpt['params'])
                if self.ddp:
                    # Use local rank for device_ids in multi-node setup
                    local_rank = self.rank % torch.cuda.device_count()
                    model = DDP(model, device_ids=[local_rank])
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
            model_ckpt = torch.load(
                evaluate_config["model"],
                map_location=self.device,
                weights_only=False
            )
            model.load_state_dict(model_ckpt['params'])
            model.eval()
            self.msg += f'\n\033[32musing the {evaluate_config["model"]}\033[0m\n'

        if not restart and not evaluate: # initial train case
            rank = self.rank
            start_epoch = 0
            model_ckpt = None
            if self.ddp:
                # Use local rank for device_ids in multi-node setup
                local_rank = self.rank % torch.cuda.device_count()
                model = DDP(model, device_ids=[local_rank])
            self.msg += f'\n\033[32minitializing training, results will be saved in the {model_config["fname_pkl"]}\033[0m\n'

        # Check the number of parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.msg += f'\nnumber of parameters:\n\033[33m -- model ({self.json_data["model"].lower()})  {n_params}\033[0m\n'

        if rank == 0:
            print(self.msg)

        return model, n_params, model_ckpt, start_epoch

    def set_model(self):
        model_config = self.json_data

        cutoff = model_config.get('cutoff', 6.0)
        num_species = model_config.get('num_species', 4)
        avg_num_neighbors = model_config.get('avg_num_neighbors', 30)

        hidden_irreps = o3.Irreps(
            model_config.get('hidden_channels', "64x0e+64x1o+64x2e")
        )
        features_dim = model_config.get('features_dim', 64)
        num_basis_func = model_config.get('num_radial_basis', 8)
        nlayers = model_config.get('nlayers', 3)
        max_ell = model_config.get('max_ell', 3)

        output_irreps = model_config.get('output_channels', "1x0e")
        active_fn = model_config.get('active_fn', "identity")
        regress_forces = model_config.get('regress_forces', "auto")
        if regress_forces == True:
            regress_forces = "autograd"
        elif regress_forces == False:
            regress_forces = "false"

        # Equivariant backend selection: cueq -> oeq -> e3nn fallback.
        cueq_request = model_config.get('cueq_config')  # bool or None
        oeq_request = model_config.get('oeq_config')    # bool or None
        cueq_config = None
        oeq_config = None

        if (cueq_request is None or cueq_request) and CUET_AVAILABLE:
            cueq_config = CuEquivarianceConfig(
                enabled=True,
                layout="ir_mul",
                group="O3_e3nn",
                optimize_all=True,
            )
            self.msg += f'\nequiv. lib.:\n\033[33m -- CuEquivariance\033[0m\n'
        elif oeq_request and OEQ_AVAILABLE:
            oeq_config = OEQConfig(
                enabled=True,
                optimize_all=True,
            )
            if model_config.get('oeq_conv_fusion', False):
                oeq_config.conv_fusion = "atomic"
            self.msg += f'\nequiv. lib.:\n\033[33m -- OpenEquivariance\033[0m\n'
        else:
            self.msg += f'\nequiv. lib.:\n\033[33m -- e3nn\033[0m\n'

        model_name = model_config["model"].lower()
        model_cls = MODEL_REGISTRY.get(model_name)
        if model_cls is None:
            raise ValueError(f"Unknown model type: {model_name}")

        model_kwargs = dict(
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
            cueq_config=cueq_config,
        )
        model_params = inspect.signature(model_cls).parameters
        # Only forward if the chosen model implements separated layer-norm
        if 'l_separated_layer_norm' in model_params:
            model_kwargs['l_separated_layer_norm'] = model_config.get(
                'l_separated_layer_norm', False
            )
        # Only forward radial_polynomial_p when set; otherwise each model
        # falls back to its own default (2 for RACE/RACEUnified, 6 for MACE).
        if 'radial_polynomial_p' in model_params \
                and 'radial_polynomial_p' in model_config:
            model_kwargs['radial_polynomial_p'] = model_config[
                'radial_polynomial_p'
            ]
        # Only forward oeq_config when the model accepts it
        if 'oeq_config' in model_params:
            model_kwargs['oeq_config'] = oeq_config
        # Only forward interaction_block when the model accepts it.
        # Note: only "fast" supports OEQ; "slow" runs on e3nn.
        if 'interaction_block' in model_params:
            interaction_block = model_config.get(
                'interaction_block', 'slow'
            )
            model_kwargs['interaction_block'] = interaction_block
            self.msg += (
                f'\ninteraction block:\n'
                f'\033[33m -- {interaction_block}\033[0m\n'
            )
        model = model_cls(**model_kwargs)
        return model

    def configure_optimizer(self):
        """ Configure optimizer using model configuration dictionary.
        """
        optimizer = self.set_optimizer()
        model_config = self.json_data['NN']
        restart = model_config.get('restart')
        if restart and self.model_ckpt is not None:
            optimizer.load_state_dict(self.model_ckpt['opt_state'])
        return optimizer

    def set_optimizer(self):
        optim_config = self.json_data["NN"]
        lr = optim_config.get('learning_rate', 1.0e-3)
        weight_decay = optim_config.get('weight_decay', 0)
        amsgrad = optim_config.get('amsgrad', True)
        # We can modify this part
        optimizer_str = optim_config.get('optimizer', 'adam')
        if optimizer_str in ['adam', 'Adam']:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                foreach=None,
                fused=None,
                amsgrad=amsgrad
            )
        elif optimizer_str in ['adamw', 'AdamW']:
            decay_params, no_decay_params = [], []
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim <= 1 or name.endswith('.bias'):
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)

            optimizer = torch.optim.AdamW(
                [{'params': decay_params, 'weight_decay': weight_decay},
                 {'params': no_decay_params, 'weight_decay': 0.0}],
                lr=lr,
                foreach=None,
                fused=None,
                amsgrad=amsgrad
            )
        else:
            raise ValueError(
                f"Unknown optimizer '{optimizer_str}'. "
                f"Expected one of: 'adam', 'Adam', 'adamw', 'AdamW'."
            )

        return optimizer

    def configure_scheduler(self):
        scheduler = self.set_scheduler()
        model_config = self.json_data['NN']
        restart = model_config.get('restart')
        if restart and self.model_ckpt is not None:
            scheduler.load_state_dict(self.model_ckpt['scheduler'])
        return scheduler

    def set_scheduler(self):
        scheduler_config = self.json_data["scheduler"]
        scheduler_config["lr_init"] = self.json_data["NN"]["learning_rate"]
        scheduler = LRScheduler(self.optimizer, scheduler_config)
        return scheduler

    def configure_checkpoint(self):
        """Configure initial checkpoint dictionary for saving training states.
        """
        loss_dict = {'epoch': 0, 'train': [], 'valid': []}
        ckpt = {
            'loss': loss_dict,
            'params': self.model.state_dict(),
            'opt_state': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'uniq_element': self.uniq_element,
            'enr_avg_per_element': self.enr_avg_per_element,
            'input.json': self.json_data,
            'train_scale_shift': [],
            'valid_scale_shift': [],
            'valid_scale_shift_origin' : [],
            'ema_state': None
        }
        return loss_dict, ckpt

    def configure_exponential_moving_average(self):
        ema = self.json_data["NN"].get("ema", True)
        ema_decay = self.json_data["NN"].get("ema_decay", 0.999)
        if ema == True:
            ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            ema = None
        model_config = self.json_data['NN']
        restart = model_config.get('restart')
        evaluate_config = self.json_data['predict']
        evaluate = evaluate_config.get('evaluate_tag')  # True or False(None)
        if (restart or evaluate) \
                and ema is not None \
                and self.model_ckpt is not None \
                and self.model_ckpt.get('ema_state') is not None:
            ema.load_state_dict(self.model_ckpt['ema_state'])
            if evaluate:
                ema.copy_to(self.model.parameters())
        return ema

    def get_params(self):
        return self.n_params

    def check_parameter_sync(self):
        for name, param in self.model.named_parameters():
            #print(f"Rank {self.rank}, Parameter name: {name}, Value: {param.data[0]}, Shape: {param.shape}, Num.: {param.numel()}")
            print(f"Rank {self.rank}, Parameter name: {name}, Shape: {param.shape}, Num.: {param.numel()}")

    def move_to_device(self, data, device):
        if isinstance(data, dict):
            return {k: v.to(device) if hasattr(v, "to") else v for k, v in data.items()}
        elif hasattr(data, "to"):
            return data.to(device)
        else:
            return data
