import os
import gc
import re
import pickle
from pathlib import Path
from time import time
from copy import deepcopy

import torch
from torch_geometric.loader import DataLoader

from bam_torch.training.base_trainer import BaseTrainer
from bam_torch.utils.sampler import DistributedBalancedAtomCountBatchSampler
from bam_torch.training.loss import RMSELoss, l2_regularization, HuberLoss

import ast
import atexit
from tqdm import tqdm
from bam_torch.utils.utils import date
from torch.utils.data import Dataset, DataLoader as TorchLoader
from torch.utils.data.distributed import DistributedSampler


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
                preds = self.model(data, backprop)
                preds = self.scale_shift(preds, data, mode)

                loss_dict = self.compute_loss(preds, data)
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
        
                for l in loss_log_config:
                    epoch_loss_dict[l].append(loss_dict.get(l, torch.nan).detach().cpu())

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
            print(f'[RANK {self.rank}] Dataset path: {sampled_dataset_file_path} | Elapsed time {(time()-t1)/1000:.6f} sec', file=self.time_log, flush=True)
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
                    print(f"[Rank {self.rank}] Number of Data: {len(data)} | Mode: {mode} ", file=self.time_log, flush=True)                    
                    print(f"[Rank {self.rank}] Number of Data: {len(test_data)} | Mode: test ", file=self.time_log, flush=True)
                    sampled_test_dataset_save_folder = Path(f"./test_datasets-{self.rank}")
                    sampled_test_dataset_file_name = f"test-{file_number}.pkl"
                    sampled_test_dataset_file_path = sampled_test_dataset_save_folder / sampled_test_dataset_file_name
                    with open(sampled_test_dataset_file_path, "wb") as f:
                        pickle.dump(test_data, f)
                else:
                    data = [data[i] for i in idx_valid]
                    print(f"[Rank {self.rank}] Number of Data: {len(data)} | Mode: {mode} ", file=self.time_log, flush=True)
            
            if mode != 'test':
                print(f'[RANK {self.rank}] Dataset path: {file_path} | Elapsed time {(time()-t1)/1000:.6f} sec', file=self.time_log, flush=True)
                with open(sampled_dataset_file_path, "wb") as f:
                    pickle.dump(data, f)

            data_loader = self.get_dataloader_from_data(data)
        return data_loader

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


class DataBatchDataset(Dataset):
    def __init__(self, list_of_data_batch):
        self.data = list_of_data_batch

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_identity(x):
    return x[0]

class MPTrainer_V2(BaseTrainer):
    def __init__(self, json_data, rank=0, world_size=1):
        # git version
        print('\n *** MPTrainer_V2 ***')
        self.time_log = open(f'time_log-{rank}.txt', 'w')
        
        # multi-node version
        # Get node and local rank info from SLURM env vars
        node_id = os.environ.get('SLURM_NODEID', 'unknown')
        local_rank = os.environ.get('SLURM_LOCALID', rank)

        # multi-node version
        log_filename = f'node{node_id}_gpu{local_rank}_global{rank}.log'
        self.gpu_test_log = open(log_filename, 'w')
        atexit.register(self.close_log_file)


        self.epoch = 0
        super().__init__(json_data, rank, world_size)

    # multi-node version
    def close_log_file(self):
        if self.gpu_test_log and not self.gpu_test_log.closed:
            print(f"[{date()}] Closing log file for Rank {self.rank} (Node: {os.environ.get('SLURM_NODEID', 'unknown')}, "
                    f"Local GPU: {os.environ.get('SLURM_LOCALID', self.rank)})", flush=True)
            self.gpu_test_log.close()

    # multi-node version
    def configure_dataloader(self):
        with open(self.json_data['enr_avg_per_element'], 'r', encoding='utf-8') as file:
            content = file.read()
        enr_avg_per_element, uniq_element = ast.literal_eval(content)

        return None, None, uniq_element, enr_avg_per_element
    
    # multi-node version
    def load_pickle_files(self, filename, folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data if isinstance(data, list) else [data]

    def configure_model(self):
        """ Configure model using model configuration dictionary.
        Override BaseTrainer to fix multi-node DDP device_ids issue.
        """
        from torch.nn.parallel import DistributedDataParallel as DDP
        import sys
        
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
                    # Use local rank for device_ids in multi-node setup
                    local_rank = int(os.environ.get('SLURM_LOCALID', self.rank % torch.cuda.device_count()))
                    model = DDP(model, device_ids=[local_rank])
            except RuntimeError as e:
                input_json = model_ckpt['input.json']
                fname = self.save_input_parameters(input_json, 'input_json_from_trained_model.txt')
                print(e)
                print(f'\n\033[31mSome of the parameter dimensions in the trained model you are trying to use')
                print(f'do not match the current input parameters.')
                print(f' -- Please check the ```{fname}``` file\033[0m\n')
                sys.exit(1)
            self.msg = f'\n\033[32mrestarting training from the {model_config["fname_pkl"]}\033[0m\n'
            self.msg += f' -- restarting from the step where the loss was {model_ckpt["loss"]}\n'
        
        if evaluate:  # True or False(None)
            rank = 0
            start_epoch = 0
            model_ckpt = torch.load(evaluate_config["model"], map_location=self.device)
            model.load_state_dict(model_ckpt['params'])
            model.eval()
            self.msg = f'\n\033[32mevaluating the {evaluate_config["model"]}\033[0m\n'

        if not restart and not evaluate: # initial train case
            rank = self.rank
            start_epoch = 0
            model_ckpt = None
            if self.ddp:
                # Use local rank for device_ids in multi-node setup
                local_rank = int(os.environ.get('SLURM_LOCALID', self.rank % torch.cuda.device_count()))
                model = DDP(model, device_ids=[local_rank])
            self.msg = f'\n\033[32minitializing training, results will be saved in the {model_config["fname_pkl"]}\033[0m\n'
        
        # Check the number of parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.msg += f'\nnumber of parameters:\n\033[33m -- model ({self.json_data["model"]})  {n_params}\033[0m\n'
        
        if rank == 0:
            print(self.msg)

        return model, n_params, model_ckpt, start_epoch

    def train_one_epoch(self, mode='train'):
        if mode == 'train':
            self.model.train()
            backprop = True
            loss_log_config = self.log_config['train']
            self.ckpt['scale_shift'] = []
            # data_files = train_files
            folder_path = self.json_data["ntrain"]
            print(f"\n ----------- train ------------", file=self.gpu_test_log)        
        else:
            self.model.eval()
            backprop = False
            loss_log_config = self.log_config['valid']
            folder_path = self.json_data["nvalid"]
            print(f"\n ----------- valid ------------", file=self.gpu_test_log)

        self.gpu_test_log.flush()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        epoch_loss_dict = {key: [] for key in loss_log_config}

        files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
        print(f"Found {len(files)} files in {folder_path}", file=self.gpu_test_log)

        if self.world_size > 1:
            num_files = len(files)
            if num_files < self.world_size:
                rank_files = files
                use_internal_sampler = True
            else:
                files_per_rank = max(1, num_files // self.world_size)
                rank_start = self.rank * files_per_rank
                rank_end = min(rank_start + files_per_rank, num_files)
                rank_files = files[rank_start:rank_end]
                use_internal_sampler = False
            files = rank_files
        else:
            use_internal_sampler = False

        if not files:
            print(f"WARNING: RANK {self.rank} has no files to process!", file=self.gpu_test_log)
            if self.world_size > 1:
                torch.distributed.barrier()
            return {key: torch.tensor(0.0, device=self.device) for key in loss_log_config}
        print(f"RANK {self.rank}: Processing {len(files)} files: {files}", file=self.gpu_test_log)
        self.gpu_test_log.flush()

        # with torch.set_grad_enabled(mode == 'train'):
        for filename in tqdm(files, desc="📂  Loading data" if mode == 'valid' else "🔄 Loading training data", dynamic_ncols=True):
            try:
                data_batch = self.load_pickle_files(filename, folder_path)
            except Exception as e:
                print(f"ERROR: Failed to load {filename}: {e}", file=self.gpu_test_log)
                self.gpu_test_log.flush()
                continue
            if not data_batch:
                print(f"WARNING: Empty data batch for {filename}", file=self.gpu_test_log)
                self.gpu_test_log.flush()
                continue

            dataset = DataBatchDataset(data_batch)

            if use_internal_sampler and self.world_size > 1:
                data_sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=(mode == 'train'))
                data_sampler.set_epoch(self.epoch)
                shuffle = False
            else:
                data_sampler = None
                shuffle = (mode == 'train')

            data_loader = TorchLoader(
                dataset, batch_size=1, shuffle=shuffle, drop_last=False,
                pin_memory=True, num_workers=0, sampler=data_sampler,
                collate_fn=collate_identity
            )

            for data in tqdm(data_loader, desc="🚀 Training" if mode == 'train' else "🔍 Evaluating", leave=False, dynamic_ncols=True):
                data.to(self.device)
                data = self.data_to_dict(data)  # This is for torch.jit compile
                print(f"RANK {self.rank} Epoch {self.epoch} | Data: {data['num_nodes']}\n", file=self.gpu_test_log)
                start.record()
                self.gpu_test_log.flush()

                preds = self.model(data, backprop)
                preds = self.scale_shift(preds, data, mode)

                loss_dict = self.compute_loss(preds, data)
                loss = loss_dict['loss']

                if backprop:
                    self.optimizer.zero_grad()
                    loss.backward()
                    end.record()
                    torch.cuda.synchronize()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)
                    self.optimizer.step()
                else:
                    end.record()
                    torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end)
                print(f"[Rank {self.rank}] Number of Atoms: {data['num_nodes']:<6} | Number of Edges: {sum(data['num_edges']):<6} | Elapsed Time: {elapsed_time/1000:.6f} sec ", file=self.time_log, flush=True)

                for l in loss_log_config:
                    epoch_loss_dict[l].append(loss_dict.get(l, torch.nan).detach().cpu())
                
                data.clear()
                del data, preds, loss_dict
                torch.cuda.empty_cache()
                gc.collect()
            data_batch.clear()
            del data_batch, dataset, data_loader
            if 'data_sampler' in locals() and data_sampler is not None:
                del data_sampler
            torch.cuda.empty_cache()
            gc.collect()

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

