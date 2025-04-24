from bam_torch.training.base_trainer import BaseTrainer
import os
import ast 
from tqdm import tqdm 
import pickle
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy
import gc
from bam_torch.utils.sampler import DistributedBalancedAtomCountBatchSampler
from bam_torch.training.loss import RMSELoss, l2_regularization, HuberLoss


def move_to_device(data, device):
    if isinstance(data, dict):
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in data.items()}
    elif hasattr(data, "to"):
        return data.to(device)
    else:
        return data


class MPTrainer(BaseTrainer):
    def __init__(self, json_data, rank=0, world_size=1):
        self.time_log = open(f'time_log-{rank}.txt', 'w')
        super().__init__(json_data, rank, world_size)

    def configure_dataloader(self):
        #with open(self.json_data['enr_avg_per_element'], 'r', encoding='utf-8') as file:
        #    content = file.read()
        #enr_avg_per_element, uniq_element = ast.literal_eval(content)
        
        #return None, None, enr_avg_per_element, uniq_element
        return None, None, None, None

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
            self.ckpt['scale_shift'] = []
            data_files = train_files
        else:  # test or valid
            self.model.eval()
            backprop = False
            loss_log_config = self.log_config['valid']
            data_files = valid_files

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        epoch_loss_dict = {key: [] for key in loss_log_config}
        for filename in data_files:
            data_loader = self.configure_dataloader_from_pkl(filename, mode=mode)

            for data in data_loader:
                data.to(self.device)
                data = self.data_to_dict(data)  # This is for torch.jit compile
                start.record()
                preds = self.model(deepcopy(data), backprop)
                preds = self.scale_shift(preds, data, mode)

                loss_dict = self.compute_loss(preds, data)
                for l in loss_log_config:
                    epoch_loss_dict[l].append(loss_dict.get(l, torch.nan))

                loss = loss_dict['loss']
                if backprop:
                    self.optimizer.zero_grad()
                    loss.backward()
                    end.record()
                    torch.cuda.synchronize()
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5) 
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)
                    self.optimizer.step()
                else:
                    end.record()
                    torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end)
                print(f"[Rank {self.rank}] Number of Atoms: {data['num_nodes']:<6} | Number of Edges: {sum(data['num_edges']):<6} | Elapsed Time: {elapsed_time/1000:.6f} sec ", file=self.time_log, flush=True)
        
                data.clear()
                del data, preds, loss_dict
                torch.cuda.empty_cache()
                gc.collect()

            torch.cuda.empty_cache()
            gc.collect()
            del data_loader

        epoch_loss_dict = {key: torch.mean(torch.tensor(value)) \
                           for key, value in epoch_loss_dict.items()}      
        return epoch_loss_dict

    def get_pkl_data_path(self):
        dir_path = self.json_data.get('fname_traj')
        ntrain = self.json_data.get('ntrain')
        nvalid = self.json_data.get('ntest')
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
        ntrain = self.json_data.get('ntrain')
        nvalid = self.json_data.get('ntest')

        with open(file_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            data = [data]

        if type(ntrain) == float or ntrain < 1.0:
            ntrain = round(ntrain * len(data))
            nvalid = round(nvalid * len(data))
            if ntrain + nvalid > len(data):
                nvalid = nvalid - (nvalid + ntrain - len(data))
            assert ntrain + nvalid == len(data)

        if type(ntrain) != str:
            idx = torch.arange(ntrain + nvalid)
            idx = idx[torch.randperm(ntrain + nvalid)]
            idx_train = idx[:ntrain]
            idx_valid = idx[-nvalid:]
            if mode == 'train':
                data = [data[i] for i in idx_train]
                print(f"[Rank {self.rank}] Number of Data: {len(data)} | Mode: {mode} ", file=self.time_log, flush=True)
            else: # mode == 'valid'
                data = [data[i] for i in idx_valid]
                print(f"[Rank {self.rank}] Number of Data: {len(data)} | Mode: {mode} ", file=self.time_log, flush=True)

        return self.get_dataloader_from_data(data) # => data_loader

    def get_dataloader_from_data(self, graphset):
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
    
    def load_loss(self, reduction='mean'):
        nn_config = self.json_data.get("NN")
        loss_config = nn_config.get("loss_config")
        if loss_config == None:
            if self.json_data["regress_forces"]:
                loss_config = {'energy_loss': 'huber', 
                               'force_loss': 'huber', 
                               'stress_loss' : 'huber'}
            else:
                loss_config = {'energy_loss': 'huber'}
        
        loss_fn = {}
        loss_fn['energy_loss'] = loss_config.get('energy_loss')
        loss_fn['force_loss'] = loss_config.get('force_loss')
        loss_fn['stress_loss'] = loss_config.get('stress_loss')
        huber_delta = loss_config.get('huber_delta')
        
        for loss, loss_name in loss_fn.items():
            if loss_name in ['l1', 'L1', 'mae', 'MAE']:
                loss_fn[loss] = torch.nn.L1Loss(reduction=reduction)
            elif loss_name in ['mse', 'MSE']:
                loss_fn[loss] = torch.nn.MSELoss(reduction=reduction)
            elif loss_name in ['rmse', 'RMSE']:
                loss_fn[loss] = RMSELoss(reduction=reduction)
            elif loss_name in ['huber', 'HUBER', 'h', 'H']:
                loss_fn[loss] = HuberLoss(huber_delta=huber_delta)

        return loss_fn, loss_config
    
    def compute_loss(self, preds, data):
        e_lambda = self.json_data["NN"]['enr_lambda']
        f_lambda = self.json_data["NN"]['frc_lambda']
        s_lambda = self.json_data["NN"]['str_lambda']
        lambd = self.json_data["NN"]['l2_lambda']

        loss = {"loss": []}
        energy_target = data["energy"].flatten()
        loss["loss_e"] = self.loss_fn["energy_loss"](preds["energy"].flatten(), 
                                                     energy_target,
                                                     tag="energy",
                                                     num_atoms=data["num_nodes"])
        loss["loss"].append(e_lambda * loss["loss_e"])

        if "forces" in preds:
            force_target = data["forces"].flatten()
            loss["loss_f"] = self.loss_fn["force_loss"](preds["forces"].flatten(), 
                                                        force_target,
                                                        tag="forces")
            loss["loss"].append(f_lambda * loss["loss_f"])
        if "stress" in preds:
            stress_target = data["stress"].flatten()
            loss["loss_s"] = self.loss_fn["stress_loss"](preds["stress"].flatten(), 
                                                         stress_target,
                                                         tag="stress")
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
