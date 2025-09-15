import torch

from bam_torch.ga.group_averaging.transforms import FrameAveraging
from bam_torch.ga.group_averaging.fa_forward import model_forward, pa_model_forward
from bam_torch.ga.model.equivariant_layer import EquivariantInterface
from bam_torch.utils.utils import get_dataloader
from .loss import l2_regularization
from .base_trainer import BaseTrainer

from time import time
import os
import gc
import numpy as np
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


#torch.autograd.set_detect_anomaly(True)

def check_nan(out):
    if isinstance(out, torch.Tensor):
        return torch.isnan(out).any()
    elif isinstance(out, (tuple, list)):
        return any(check_nan(o) for o in out if isinstance(o, torch.Tensor))
    elif isinstance(out, dict):
        return any(check_nan(o) for o in out.values() if isinstance(o, torch.Tensor))
    else:
        return False


def _has_nan(x):
    import torch
    if isinstance(x, torch.Tensor):
        return torch.isnan(x).any() or torch.isinf(x).any()
    if isinstance(x, (list, tuple)):
        return any(_has_nan(xx) for xx in x)
    if isinstance(x, dict):
        return any(_has_nan(v) for v in x.values())
    return False


class GATrainer(BaseTrainer):
    """Trainer for group averaging model 
    (eg. Frame averaging, Probablistic symmetrization)
    """
    def __init__(self, json_data, rank, world_size):
        super().__init__(json_data, rank, world_size)
       #torch.autograd.set_detect_anomaly(True)
        #for name, module in self.model.named_modules():
        #    module.register_forward_hook(
        #        lambda m, inp, out, n=name:
        #            print(f"[NaN/Inf in FORWARD] {n}") if _has_nan(out) else None
        #    )
        #    module.register_full_backward_hook(
        #        lambda m, grad_in, grad_out, n=name:
        #            print(f"[NaN/Inf in BACKWARD] {n}") if _has_nan(grad_out) else None
        #    )

    def train_one_epoch(self, mode='train', data_loader=None):
        if mode == 'train':
            self.model.train()
            backprop = True
            loss_log_config = self.log_config['train']
            if data_loader == None:
                data_loader = self.train_loader
            self.ckpt['train_scale_shift'] = []
        elif mode == 'compile':
            backprop = False
            loss_log_config = self.log_config['valid']
            if data_loader == None:
                data = next(iter(self.train_loader)).to(self.device)
                data = self.data_to_dict(data)
            self.model = torch.jit.trace(self.model, data)
        else:  # test or valid
            self.model.eval()
            backprop = False
            loss_log_config = self.log_config['valid']
            if data_loader == None:
                data_loader = self.valid_loader
            if mode == 'valid':
                self.ckpt['valid_scale_shift'] = []

        fa_method = self.json_data.get('ga_method')
        gs = [None] * len(data_loader)
        if fa_method == None:
            # The frame averaging method
            # : {"det", "all", "se3-stochastic", "se3-det", "se3-all", "stochastic"}
            fa_method = "stochastic"

        frame_averaging = self.json_data.get('frame_averaging')
        if frame_averaging == None:
            # ["2D", "3D", "DA", ""]
            frame_averaging = "3D"
        pbc = self.json_data.get('pbc') 
        if pbc == None:
            pbc = True
        transform = FrameAveraging(frame_averaging, fa_method)

        epoch_loss_dict = {key: [] for key in loss_log_config}
        for i, data in enumerate(data_loader):
            data = self.move_to_device(data, self.device)
            #data.to(self.device)
            #data = self.data_to_dict(data)  # This is for torch.jit compile
            if fa_method == "prob":
                preds = pa_model_forward(
                    batch=transform(data, self.equiv_model, self.json_data.get("nsamples")),  # transform the PyG graph data
                    model=self.model,
                    frame_averaging=frame_averaging, 
                    mode=mode,      
                    crystal_task=pbc, 
                )
            else:
                preds = model_forward(
                    batch=transform(data),  # transform the PyG graph data
                    model=self.model,
                    frame_averaging=frame_averaging, 
                    mode=mode,      
                    crystal_task=pbc, 
                )
            """
            data.pos = data.positions
            preds = self.model(data)
            """
            preds = self.scale_shift(preds, data, mode)
            loss_dict = self.compute_loss(preds, data)

            for l in loss_log_config:
                #epoch_loss_dict[l].append(loss_dict.get(l, torch.nan).detach().cpu())
                val = loss_dict.get(l, torch.nan)
                epoch_loss_dict[l].append(val.detach().cpu() if isinstance(val, torch.Tensor) else val)
            
            loss = loss_dict['loss']
            if backprop:
                self.optimizer.zero_grad()
                loss.backward()
                t1 = time()
                torch.cuda.synchronize()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5) 
                torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)
                self.optimizer.step()
            else:
                torch.cuda.synchronize()
        
        epoch_loss_dict = {key: torch.mean(torch.tensor(value).detach().cpu()) \
                           for key, value in epoch_loss_dict.items()}
        torch.cuda.empty_cache()
        gc.collect()
        return epoch_loss_dict
    
    def compute_loss(self, preds, data):
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
                loss["loss_grad"] = -torch.mean(cos(preds["forces"], grad_target))
            else:
                loss["loss_grad"] = self.loss_fn["force_loss"](preds["forces"], grad_target)
        
            if energy_grad_loss:
                loss["loss"].append(energy_grad_mult * loss["loss_grad"])
        
        if "stress" in preds and "stress" in data:
            stress_target = data["stress"].flatten()
            loss["loss_s"] = self.loss_fn["stress_loss"](preds["stress"].flatten(), stress_target)
            loss["loss"].append(s_lambda * loss["loss_s"])

        if lambd != 0:
            params = self.model.parameters()
            loss["loss_l2"] = l2_regularization(params)
            loss["loss"].append(lambd * loss["loss_l2"])
            
        ## Sanity check to make sure the compute graph is correct.
        #for lc in loss["loss"]:
        #    assert hasattr(lc, "grad_fn")
        loss["loss"] = sum(loss["loss"])
        return loss

    def set_model(self):
        model_config = self.json_data
        cutoff = model_config['cutoff']
        avg_num_neighbors = model_config['avg_num_neighbors']
        max_neigh = model_config['max_neigh']
        num_species = model_config['num_species']
        hidden_channels = model_config['hidden_channels']
        features_dim = model_config['features_dim']
        num_radial_basis = model_config['num_radial_basis']
        nlayers = model_config['nlayers']

        active_fn = model_config.get('active_fn')
        if active_fn == None:
            active_fn = "identity"
        regress_forces = model_config.get('regress_forces')
        if regress_forces == True:
            regress_forces = "direct"
        
        model = model_config["model"]
        if model in ["faenet", "feanet", "FAENet", "FEANet", "FAENET", "FEANET"]:
            from bam_torch.ga.model.model import FAENet
            force_decoder_model_config = {
                "simple":{
                    "hidden_channels": 128,
                    "norm": "batch1d"},
                "mlp":{
                    "hidden_channels": 256,
                    "norm": "batch1d"}, 
                "res":{
                    "hidden_channels": 128,
                    "norm": "batch1d"},
                "res_updown":{
                    "hidden_channels": 128,
                    "norm": "batch1d"}
            }
            model = FAENet(
                tag_hidden_channels=0,             # if > 0 : for is2rs or s2ef
                regress_forces=regress_forces,     # 'direct' for forces and grad(E) predictions
                force_decoder_type='mlp',          # default = mlp
                force_decoder_model_config=force_decoder_model_config,
                hidden_channels=hidden_channels,  # default = 128 and must be larger than 64 (not >=)
                num_filters=features_dim,         # default = 128
                max_num_neighbors=max_neigh,      # default = 40
                cutoff=cutoff,                    # default = 6.0
                num_interactions=nlayers,         # default = 4
                num_gaussians=num_radial_basis,   # default = 50 for Gaussian
            )
        if model_config.get("ga_method") == "prob": # Probabilistic symmetrization
            self.equiv_model = EquivariantInterface(
                symmetry='O3',
                interface='prob',
                fixed_noise=False,
                noise_scale=1,
                tau=0.01,
                hard=True,
                vnn_dropout=0.1,
                vnn_hidden_dim =96,
                vnn_k_nearest_neighbors=avg_num_neighbors
            ).to(self.device)
            interface_n_params = sum(p.numel() for p in self.equiv_model.parameters() if p.requires_grad)
            print(f'\nnumber of parameters:\n\033[36m -- interface (vnn) {interface_n_params}\033[0m\n')

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