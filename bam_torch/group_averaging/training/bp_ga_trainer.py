import os
import gc
import torch
import numpy as np
from e3nn import o3
from time import time

from .transforms import FrameAveraging
from .ga_forward import model_forward, pa_model_forward
from bam_torch.group_averaging.model.equiv_layer import EquivariantInterface
from bam_torch.group_averaging.training import FORWARD_REGISTRY
from bam_torch.group_averaging.model import MODEL_REGISTRY, ACTIVE_FN_REGISTRY
from bam_torch.utils.utils import get_dataloader
from bam_torch.training.loss import l2_regularization
from bam_torch.training.base_trainer import BaseTrainer
from bam_torch.model.wrapper_ops import CuEquivarianceConfig
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#torch.autograd.set_detect_anomaly(True)
from .ga_trainer import GATrainer


class BPGATrainer(GATrainer):
    """Trainer for group averaging 
    (eg. Frame averaging, Probablistic symmetrization)
    """
    def __init__(self, json_data, rank, world_size):
        super().__init__(json_data, rank, world_size)

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

        pbc = self.json_data.get('pbc') 
        if pbc == None:
            pbc = True

        epoch_loss_dict = {key: [] for key in loss_log_config}
        entropy_loss_list = []
        for i, data in enumerate(data_loader):
            data = self.move_to_device(data, self.device)
            #data.positions.requires_grad_(True)
            #data.cell.requires_grad_(True)
            batch, entropy_loss = self.transform(
                data=data, 
                equiv_model=self.equiv_model, # for the probabilistic symmetrization
                n_samples=self.json_data.get("nsamples") # for the probabilistic symmetrization
            )
            preds = self.model_forward_cls(
                batch=batch,  # transform the PyG graph data
                model=self.model,
                frame_averaging=self.group_averaging, 
                mode=mode,      
                crystal_task=pbc,
                edge_mask=None,
                permute=self.permute
            )
            preds = self.scale_shift(preds, data, mode)
            loss_dict = self.compute_loss(preds, data)

            for l in loss_log_config:
                val = loss_dict.get(l, torch.nan)
                epoch_loss_dict[l].append(val.detach().cpu() if isinstance(val, torch.Tensor) else val)

            loss = loss_dict['loss'] + 0.1*entropy_loss
            #entropy_loss_list.append(entropy_loss.detach().cpu())
            if backprop:
                self.optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5) 
                torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)
                self.optimizer.step()

                if self.ema is not None:
                    self.ema.update()

        torch.cuda.synchronize()
        epoch_loss_dict = {key: torch.mean(torch.tensor(value).detach().cpu()) \
                           for key, value in epoch_loss_dict.items()}
        #print(f" --> entropy_loss: {torch.tensor(entropy_loss).mean()}")
        torch.cuda.empty_cache()
        gc.collect()
        return epoch_loss_dict

    def configure_group_averaging(self):
        """
        The frame averaging method
        : ga_method = {"det", "all", "se3-stochastic", "se3-det", "se3-all", "stochastic"}
        
        The probabilistic symmetrization method
        : ga_method = "prob" or "probabilistic" 
        / "prob_rot": only rot, "prob": rot x permute
        """
        ga_method = self.json_data.get('ga_method')
        permute = True
        if ga_method == None:
            ga_method = "prob"
            permute = True
        elif ga_method == "prob_rot":
            ga_method = "prob"
            permute = False

        group_averaging = self.json_data.get('group_averaging')
        if group_averaging == None: # ["2D", "3D", "DA", ""]
            group_averaging = "3D"
        elif group_averaging == "no":
            ga_method = "no"

        transform = FrameAveraging(group_averaging, ga_method, permute)
        model_forward_cls = FORWARD_REGISTRY[ga_method]

        return transform, model_forward_cls, ga_method, group_averaging, permute


    def set_model(self):
        model_config = self.json_data
        model_name = model_config["model"].lower()
        model_cls = MODEL_REGISTRY.get(model_name)
        if model_cls is None:
            raise ValueError(f"Unknown model type: {cfg['model']}")

        regress_forces = model_config.get('regress_forces', "auto")
        if regress_forces == True:
            regress_forces = "autograd"
        elif regress_forces == False:
            regress_forces = "false"

        if model_name in ["bpnn"]:
            cutoff = model_config.get('cutoff', 6.0)
            num_species = model_config.get('num_species', 4)
            avg_num_neighbors = model_config.get('avg_num_neighbors', 30)
            max_neigh = model_config.get('max_neigh', 30)
            # default of hidden_channels = 128 and must be larger than 64 (not >=)
            hidden_channels = model_config.get('hidden_channels', 128)
            features_dim = model_config.get('features_dim', 128)
            num_radial_basis = model_config.get('num_radial_basis', 100)
            nlayers = model_config.get('nlayers', 4)
            # if tag_hidden_channels > 0 : for is2rs or s2ef

            model = model_cls(
                sf_config=sf_config, 
                symbols=symbols, 
                r_cutoff=r_cutoff
            )
        else:
            raise ValueError(f"Unknown model type: {cfg['model']}")

        if model_config.get("ga_method").lower() in ["prob", "probabilistic", "prob_rot"]: # Probabilistic symmetrization
            small_equiv_model_config = model_config.get('small_equiv', {})
            symmetry = small_equiv_model_config.get('symmetry', 'O3')
            interface = small_equiv_model_config.get('interface', 'prob')
            fixed_noise = small_equiv_model_config.get('fixed_noise', False)
            noise_scale = small_equiv_model_config.get('noise_scale', 1)
            tau = small_equiv_model_config.get('tau', 0.01)
            hard = small_equiv_model_config.get('hard', True)

            gate_name = small_equiv_model_config.get('active_fn', 'silu').lower()
            small_equiv_model_config_params = {
                'cutoff': model_config.get('cutoff', 6.0),
                'num_species': model_config.get('num_species', 4),
                'avg_num_neighbors': model_config.get('avg_num_neighbors', 30),
                'hidden_irreps': small_equiv_model_config.get(
                    'hidden_channels', "16x0e+8x1o+4x2e"
                    ),
                'features_dim': small_equiv_model_config.get('features_dim', 32),
                'num_basis_func': small_equiv_model_config.get('num_radial_basis', 8),
                'nlayers': small_equiv_model_config.get('nlayers', 1),
                'max_ell': small_equiv_model_config.get('max_ell', 3),
                'MLP_irreps': small_equiv_model_config.get('MLP_irreps', "16x0e"),
                'output_irreps': small_equiv_model_config.get('output_channels', "3x1o"),
                'gate': gate_name,
                'cueq_config': small_equiv_model_config.get('cueq_config', None),
                'radial_MLP': small_equiv_model_config.get('radial_MLP', [64, 64])
            }
            self.equiv_model = EquivariantInterface(
                symmetry=symmetry, 
                interface=interface,
                fixed_noise=fixed_noise,
                noise_scale=noise_scale,
                tau=tau,
                hard=hard,
                **small_equiv_model_config_params
            ).to(self.device)
            interface_n_params = sum(p.numel() for p in self.equiv_model.parameters() if p.requires_grad)
            print(f'\nnumber of parameters (small equiv. model):\n\033[36m -- interface (race) {interface_n_params}\033[0m')
        else:
            self.equiv_model = None

        return model
    
    def configure_dataloader(self):
        json_data = self.json_data
        train_loader, valid_loader, uniq_element, enr_avg_per_element = \
            get_dataloader(
                json_data['fname_traj'],
                json_data['ntrain'],
                json_data['nvalid'],
                json_data['nbatch'],
                json_data['cutoff'],
                json_data['NN']['data_seed'],
                json_data['element'],
                json_data['regress_forces'],
                json_data.get('max_neigh'),
                self.rank,
                self.world_size
            )
        return train_loader, valid_loader, uniq_element, enr_avg_per_element
    
