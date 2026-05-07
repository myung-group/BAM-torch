import os
import gc
import re
import pickle
from pathlib import Path
from time import time
from copy import deepcopy

import torch
from torch_geometric.loader import DataLoader

from .transforms import FrameAveraging
from bam_torch.group_averaging.model.equiv_layer import EquivariantInterface
from bam_torch.group_averaging.training import FORWARD_REGISTRY
from bam_torch.group_averaging.model import MODEL_REGISTRY, ACTIVE_FN_REGISTRY
from bam_torch.training.base_trainer import BaseTrainer
from bam_torch.utils.sampler import DistributedBalancedAtomCountBatchSampler
from bam_torch.training.loss import RMSELoss, l2_regularization, HuberLoss
from bam_torch.training.mp_trainer import MPTrainer_V2, DataBatchDataset, collate_identity

import ast
import atexit
from tqdm import tqdm
from bam_torch.utils.utils import date, data_to_dict, get_graphset_with_pad
from torch.utils.data import Dataset, DataLoader as TorchLoader
from torch.utils.data.distributed import DistributedSampler


class GAMPTrainer(MPTrainer_V2):
    def __init__(self, json_data, rank=0, world_size=1):
        node_id = os.environ.get('SLURM_NODEID', 'unknown')
        local_rank = os.environ.get('SLURM_LOCALID', rank)

        # multi-node version
        log_filename = f'node{node_id}_gpu{local_rank}_global{rank}.log'
        self.gpu_test_log = open(log_filename, 'w')
        atexit.register(self.close_log_file)

        self.epoch = 0
        super().__init__(json_data, rank, world_size)

        self.transform, self.model_forward_cls, self.ga_method, self.group_averaging, self.permute \
            = self.configure_group_averaging()

    def configure_dataloader(self):
        with open(self.json_data['enr_avg_per_element'], 'r', encoding='utf-8') as file:
            content = file.read()
        enr_avg_per_element, uniq_element = ast.literal_eval(content)

        return None, None, uniq_element, enr_avg_per_element

    def load_pickle_files_with_progress(self, filename, folder_path):
        combined_list = []  
        #files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
        #for filename in tqdm(files, desc=f"Loading files from {folder_path}"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, list):  
                combined_list.extend(data)
            else: 
                combined_list.append(data)
        return combined_list

    def train_one_epoch(self, mode='train', data_loader=None):
        train_files, valid_files = self.get_pkl_data_path()
        if mode == 'train':
            self.model.train()
            backprop = True
            loss_log_config = self.log_config['train']
            self.ckpt['train_scale_shift'] = {
                    k: [] for k in self.enr_avg_per_element.keys()
            }
            # data_files = train_files
            data_files = train_files      
        else:
            self.model.eval()
            backprop = False
            loss_log_config = self.log_config['valid']
            folder_path = self.json_data["nvalid"]
            self.ckpt['valid_scale_shift'] = {
                    k: [] for k in self.enr_avg_per_element.keys()
                }
            self.ckpt['valid_scale_shift_origin'] = []
            data_files = valid_files

        self.gpu_test_log.flush()

        pbc = self.json_data.get('pbc') 
        if pbc == None:
            pbc = True

        epoch_loss_dict = {key: [] for key in loss_log_config}
        entropy_loss_list = []

        for filename in data_files:
            data_loader = self.configure_dataloader_from_pkl(filename, mode=mode)
     
            for i, data in enumerate(data_loader):
                data.to(self.device)
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

                data.clear()
                del data, preds, loss_dict
                torch.cuda.empty_cache()
                gc.collect()

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            del data_loader
            if 'data_sampler' in locals() and data_sampler is not None:
                del data_sampler

        """
        torch.cuda.synchronize()
        epoch_loss_dict = {key: torch.mean(torch.tensor(value).detach().cpu()) \
                           for key, value in epoch_loss_dict.items()}
        #print(f" --> entropy_loss: {torch.tensor(entropy_loss).mean()}")
        torch.cuda.empty_cache()
        gc.collect()
        """

        if self.world_size > 1:
            try:
                torch.distributed.barrier()
            except Exception as e:
                print(f"ERROR in barrier: {e}", file=self.gpu_test_log)
                self.gpu_test_log.flush()
       
        final_epoch_loss_dict = {}
        for key in epoch_loss_dict:
            tensor_list = epoch_loss_dict[key]

            if len(tensor_list) > 0:
                local_loss_sum = torch.sum(torch.stack([t.clone().detach().to(self.device) for t in tensor_list]))
                local_count = torch.tensor(len(tensor_list), device=self.device, dtype=torch.float)
            else:
                local_loss_sum = torch.tensor(0.0, device=self.device)
                local_count = torch.tensor(0.0, device=self.device)

            global_loss_sum = local_loss_sum.clone()
            global_count = local_count.clone()

            if self.world_size > 1:
                try:
                    torch.distributed.all_reduce(global_loss_sum, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(global_count, op=torch.distributed.ReduceOp.SUM)
                except Exception as e:
                    print(f"ERROR during all_reduce for key {key}: {e}", file=self.gpu_test_log)
                    self.gpu_test_log.flush()
                    global_loss_sum = torch.tensor(0.0, device=self.device)
                    global_count = torch.tensor(0.0, device=self.device)

            if global_count > 0:
                final_avg_loss = global_loss_sum / global_count
            else:
                final_avg_loss = torch.tensor(float('nan'), device=self.device)
                print(f"WARNING: No data for {key}!", file=self.gpu_test_log)

            final_epoch_loss_dict[key] = final_avg_loss
        self.gpu_test_log.flush()
    
        return final_epoch_loss_dict


    def get_pkl_data_path(self):
        dir_path = self.json_data.get('fname_traj')
        ntrain = self.json_data.get('ntrain')
        nvalid = self.json_data.get('nvalid')
        if type(ntrain) == str:
            train_dir_path = ntrain
            if os.path.isdir(train_dir_path):
                train_files = [
                    os.path.join(train_dir_path, f) 
                    for f in os.listdir(train_dir_path) 
                    if f.endswith(".pkl")
                ]
            else:
                train_files = [train_dir_path]
            
            valid_dir_path = nvalid
            if os.path.isdir(valid_dir_path):    
                valid_files = [
                    os.path.join(valid_dir_path, f) 
                    for f in os.listdir(valid_dir_path) 
                    if f.endswith(".pkl")
                ]
            else:
                valid_files = [valid_dir_path]
        else:
            train_dir_path = dir_path
            if os.path.isdir(train_dir_path):
                train_files = [
                    os.path.join(train_dir_path, f) 
                    for f in os.listdir(train_dir_path) 
                    if f.endswith(".pkl")
                ]
            else:
                train_files = [train_dir_path]
            valid_files = deepcopy(train_files)
        
        return train_files, valid_files

    def configure_dataloader_from_pkl(self, file_path, mode):
        file_number = 0
        match = re.search(r"_(\d+)\.pkl$", file_path)
        if match:
            file_number = int(match.group(1))
        
        sampled_dataset_save_folder = Path(f"./{mode}_datasets-{self.rank}")
        sampled_dataset_file_name = f"{mode}-{file_number}.pkl"
        sampled_dataset_file_path = sampled_dataset_save_folder / sampled_dataset_file_name

        if sampled_dataset_file_path.exists():
            t1 = time()
            file_path = sampled_dataset_file_path
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, list):
                data = [data]
            data_loader = self.get_dataloader_from_data(data)
        else:
            t1 = time()
            os.makedirs(sampled_dataset_save_folder, exist_ok=True)
    
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, list):
                data = [data]
            
            ntrain = self.json_data.get('ntrain')
            nvalid = self.json_data.get('nvalid')
            ntest = self.json_data.get('ntest')

            if type(ntrain) == float or ntrain < 1.0:
                ntrain = round(ntrain * len(data))
                nvalid = round(nvalid * len(data))
                if ntrain + nvalid > len(data):
                    nvalid = nvalid - (nvalid + ntrain - len(data))
                if type(ntest) == float:
                    ntest = round(ntest * len(data))
                    if ntrain + nvalid + ntest > len(data):
                        ntest = ntest - (ntrain + nvalid + ntest  - len(data))

            if type(ntrain) != str:
                if ntest == None:
                    ntest = 0
                    if ntrain + nvalid < len(data):
                        ntest = len(data) - (nvalid + ntrain)
                        assert ntrain + nvalid + ntest == len(data)

                idx = torch.arange(ntrain + nvalid + ntest)
                idx = idx[torch.randperm(ntrain + nvalid + ntest)]
                idx_train = idx[:ntrain]
                idx_valid = idx[ntrain:ntrain+nvalid]
                idx_test = idx[-ntest:]
                if mode == 'train':
                    test_data = [data[i] for i in idx_test]
                    data = [data[i] for i in idx_train]
                    sampled_test_dataset_save_folder = Path(f"./test_datasets-{self.rank}")
                    sampled_test_dataset_file_name = f"test-{file_number}.pkl"
                    sampled_test_dataset_file_path = sampled_test_dataset_save_folder / sampled_test_dataset_file_name
                    with open(sampled_test_dataset_file_path, "wb") as f:
                        pickle.dump(test_data, f)
                else:
                    data = [data[i] for i in idx_valid]
            
            if mode != 'test':
                with open(sampled_dataset_file_path, "wb") as f:
                    pickle.dump(data, f)

            data_loader = self.get_dataloader_from_data(data)
        return data_loader

    def get_dataloader_from_data(self, graphset):
        pad_nodes_to = 0 # nbatch * max_nodes 
        pad_edges_to = 0 # nbatch * max_edges
        for graph in graphset:
            pad_nodes_to = max(graph.num_nodes, pad_nodes_to)
            pad_edges_to = max(graph.num_edges, pad_edges_to)
        graphset = get_graphset_with_pad(deepcopy(graphset), pad_nodes_to, pad_edges_to)

        data_sampler = DistributedBalancedAtomCountBatchSampler(        
            dataset=graphset,
            batch_size=self.json_data['nbatch'],
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            seed=self.json_data['NN']['data_seed'],
            drop_last=False,
            reference='edges'
        )
        data_loader = DataLoader(
            graphset,
            self.json_data['nbatch'],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=0,
            collate_fn=None,
            sampler=data_sampler
        )
        return data_loader

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
                pg_hidden_channels=0,
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