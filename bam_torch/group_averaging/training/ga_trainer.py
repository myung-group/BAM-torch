import os
import gc
import torch
import numpy as np
from e3nn import o3
from time import time
import ast

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


class GATrainer(BaseTrainer):
    """Trainer for group averaging 
    (eg. Frame averaging, Probablistic symmetrization)
    """
    def __init__(self, json_data, rank, world_size):
        super().__init__(json_data, rank, world_size)

        self.transform, self.model_forward_cls, self.ga_method, self.group_averaging, self.permute \
            = self.configure_group_averaging()

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

    def compute_loss(self, preds, data):
        lambda_config = self.json_data["NN"]
        e_lambda = lambda_config.get('enr_lambda', 1)
        f_lambda = lambda_config.get('frc_lambda', 30)
        s_lambda = lambda_config.get('str_lambda', 1)
        lambd = lambda_config.get('l2_lambda', 0)

        cosine_sim = lambda_config.get('cosine_sim', False)
        energy_grad_mult = lambda_config.get('energy_grad_mult', 10)
        energy_grad_loss = lambda_config.get('energy_grad_loss', False)

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
                
        # This is for frame-averaging or probabilistic-symmetrization
        if "forces_grad_target" in preds:
            energy_grad_loss = True
            grad_target = preds["forces_grad_target"]
            if cosine_sim:
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                loss["loss_grad"] = -torch.mean(cos(preds["forces"], grad_target))
            else:
                loss["loss_grad"] = self.loss_fn["force_loss"](
                    preds["forces"], grad_target
                )
            if energy_grad_loss:
                loss["loss"].append(energy_grad_mult * loss["loss_grad"])
        
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
            
        # Get loss: 
        # loss = (e_lambda * loss_e) + (f_lambda * loss_f) 
        #        + (s_lambda * loss_s) + (lambd * loss_l2)
        loss["loss"] = sum(loss["loss"])
        return loss

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

        if model_name in ["faenet", "gnn"]:
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
            tag_hidden_channels = model_config.get('tag_hidden_channels', 0)
            force_decoder_type = model_config.get('force_decoder_type', 'mlp') # simple
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
            model = model_cls(
                tag_hidden_channels=tag_hidden_channels,
                regress_forces=regress_forces, 
                force_decoder_type=force_decoder_type, 
                force_decoder_model_config=force_decoder_model_config,
                hidden_channels=hidden_channels,  
                num_filters=features_dim,      
                max_num_neighbors=max_neigh,      
                cutoff=cutoff,            
                num_interactions=nlayers,      
                num_gaussians=num_radial_basis,   
            )
        elif model_name in ["transformer", "trans"]:
            num_species = model_config.get('num_species', 4)
            hidden_channels = model_config.get('hidden_channels', 200)
            features_dim = model_config.get('features_dim', 64)
            nlayers = model_config.get('nlayers', 5)
            dropout = model_config.get('dropout', 0.0)
            nhead = model_config.get('nhead', 4)
            active_fn_name = model_config.get('active_fn', 'gelu').lower()
            active_fn = ACTIVE_FN_REGISTRY.get(active_fn_name)
            model = model_cls(
                d_model=features_dim,
                dim_feedforward=hidden_channels,
                nhead=nhead,
                num_encoder_layers=nlayers,
                dropout=dropout,
                activation=active_fn,
                regress_forces=regress_forces,
                num_species=num_species,
            )
        elif model_name in ["equiformer", "equiformer_ga"]:
            cutoff = model_config.get('cutoff', 6.0)
            num_species = model_config.get('num_species', 4)
            avg_num_neighbors = model_config.get('avg_num_neighbors', 30)
            max_neigh = model_config.get('max_neigh', 30)
            # default of hidden_channels = 128 and must be larger than 64 (not >=)
            hidden_channels = model_config.get('hidden_channels', 128)
            features_dim = model_config.get('features_dim', 128)
            num_radial_basis = model_config.get('num_radial_basis', 500)
            nlayers = model_config.get('nlayers', 12)
            model = model_cls(
                use_pbc=True,
                regress_forces=regress_forces,
                otf_graph=True,
                max_neighbors=max_neigh,
                max_radius=cutoff,
                max_num_elements=num_species,
                num_layers=nlayers,
                sphere_channels=128,
                attn_hidden_channels=hidden_channels,
                num_heads=8,
                attn_alpha_channels=32,
                attn_value_channels=16,
                ffn_hidden_channels=hidden_channels*4,
                norm_type='rms_norm_sh',
                lmax_list=[6],
                mmax_list=[2],
                grid_resolution=None, 
                num_sphere_samples=128,
                edge_channels=features_dim,
                use_atom_edge_embedding=True, 
                share_atom_edge_embedding=False,
                use_m_share_rad=False,
                distance_function="gaussian",
                num_distance_basis=num_radial_basis, 
                attn_activation='scaled_silu',
                use_s2_act_attn=False, 
                use_attn_renorm=True,
                ffn_activation='scaled_silu',
                use_gate_act=False,
                use_grid_mlp=False, 
                use_sep_s2_act=True,
                alpha_drop=0.1,
                drop_path_rate=0.05, 
                proj_drop=0.0, 
                weight_init='normal'
            )
        elif model_name in ["race", "race_ga", "race_ga_b", "race_ga_r", "race_ga_g", "race_ga_r_df", "race_ga_g_b"]:
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
            
            cueq_config = model_config.get('cueq_config', False)  # true or false
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
            
            model_name = model_config["model"].lower()
            model_cls = MODEL_REGISTRY.get(model_name)
            if model_cls is None:
                raise ValueError(f"Unknown model type: {cfg['model']}")

            model = model_cls(
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
        elif model_name in ["schnet"]:
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
                hidden_channels=hidden_channels, 
                num_filters=features_dim, 
                num_interactions=nlayers, 
                num_gaussians=num_radial_basis, 
                cutoff=cutoff, 
                max_num_neighbors=max_neigh
            )
        elif model_name in ["dplr"]:
            cutoff = model_config.get('cutoff', 6.0)
            num_species = model_config.get('num_species', 4)
            max_neigh = model_config.get('max_neigh', 40)
            embedding_dim = model_config.get('embedding_dim', 32)
            descriptor_hidden = model_config.get('descriptor_hidden', [25, 50, 100])
            descriptor_axis_neurons = model_config.get('descriptor_axis_neurons', 16)
            fitting_hidden = model_config.get('fitting_hidden', [240, 240, 240])
            use_long_range = model_config.get('use_long_range', True)
            ewald_accuracy = model_config.get('ewald_accuracy', 1e-6)
            charge_fitting_hidden = model_config.get('charge_fitting_hidden', [240, 240, 240])
            max_sel = model_config.get('max_sel', 60)
            force_decoder_hidden = model_config.get('force_decoder_hidden', 128)
            use_type_embedding = model_config.get('use_type_embedding', True)
            preprocess = model_config.get('preprocess', 'pbc_preprocess')

            model = model_cls(
                cutoff=cutoff,
                num_species=num_species,
                embedding_dim=embedding_dim,
                descriptor_hidden_channels=descriptor_hidden,
                descriptor_axis_neurons=descriptor_axis_neurons,
                fitting_hidden_channels=fitting_hidden,
                regress_forces=regress_forces,
                use_long_range=use_long_range,
                ewald_accuracy=ewald_accuracy,
                charge_fitting_hidden=charge_fitting_hidden,
                max_num_neighbors=max_neigh,
                max_sel=max_sel,
                preprocess=preprocess,
                force_decoder_hidden=force_decoder_hidden,
                use_type_embedding=use_type_embedding,
            )
        elif model_name in ["bpnn", "v_bpnn"]:
            cutoff = model_config.get('cutoff', 6.0)
            num_species = model_config.get('num_species', 4)
            avg_num_neighbors = model_config.get('avg_num_neighbors', 30)
            max_neigh = model_config.get('max_neigh', 30)
            # default of hidden_channels = 128 and must be larger than 64 (not >=)
            hidden_channels = model_config.get('hidden_channels', 64)
            features_dim = model_config.get('features_dim', 128)
            num_radial_basis = model_config.get('num_radial_basis', 100)
            nlayers = model_config.get('nlayers', 4)
            # if tag_hidden_channels > 0 : for is2rs or s2ef

            def make_default_sf_from_uniq(uniq_element):
                from bam_torch.group_averaging.model.symmetryfunctions import (
                    G1, G2, G4
                )
                """
                uniq_element : dict[atomic_number -> element_index]
                return       : sf_config[ei][ej] -> list of SFs
                """
                sf_config = {}

                for Zi, ei in uniq_element.items():
                    sf_config[ei] = {}
                    for Zj, ej in uniq_element.items():
                        sf_config[ei][ej] = [
                            G1(),
                            G2(2.0, 1.0),
                            G2(4.0, 1.0),
                            G4(1.0, 1.0, -1.0),
                            G4(2.0, 2.0, 1.0),
                        ]

                return sf_config

            with open(self.json_data['enr_avg_per_element'], 'r', encoding='utf-8') as file:
                content = file.read()
            _, uniq_element = ast.literal_eval(content)
            
            sf_config = make_default_sf_from_uniq(uniq_element)

            model = model_cls(
                sf_config=sf_config, 
                uniq_element=uniq_element, 
                r_cutoff=cutoff,
                hidden=hidden_channels
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