import torch

import gc
import atexit
import pprint
from copy import deepcopy

from bam_torch.utils.logger import Logger
from bam_torch.training.base_trainer import BaseTrainer
from bam_torch.group_averaging.training.ga_trainer import GATrainer
from bam_torch.utils.utils import get_dataloader_to_predict, date, on_exit, get_dataloader
from bam_torch.group_averaging.training.transforms import FrameAveraging
from bam_torch.group_averaging.training.ga_forward import model_forward, pa_model_forward

from time import time
import numpy as np


class GAEvaluator(GATrainer):
    def __init__(self, json_data, rank=0, world_size=1):
        self.json_data = json_data
        self.json_data['NN']['restart'] = False
        self.json_data["predict"]["evaluate_tag"] = True
        self.json_data["nbatch"] = 1
        self.rank = 0
        self.world_size = 0
        super().__init__(self.json_data, self.rank, self.world_size)

    def setup(self):
        """Configure all core training components.

        Sets up device, model, optimizer, dataloader, scheduler,
        loss function, and logger.
        """
        self.set_random_seed() # Reproducibility
        self.device = self.configure_device()
        self.model, self.n_params, self.model_ckpt, self.start_epoch = self.configure_model()
        self.optimizer = self.configure_optimizer()
        self.data_loader, self.uniq_element, self.enr_avg_per_element = self.configure_dataloader()
        self.loss_fn, self.loss_config = self.configure_loss()
        self.log_config, self.log_interval, self.logger, self.fout = self.configure_logger()
        #self.ema = self.configure_exponential_moving_average()

    def evaluate(self, element_wise=True):
        self.logger.print_logger_head()
        target = {}
        total_loss_dict = {'loss':[],
                           'loss_e':[],
                           'loss_f':[],
                           'loss_grad':[],
                           } 

        e_corr_, element_wise = self.get_scale_shift_correction(element_wise)

        test_values = {
            'energy': [],
            'force_x': [],
            'force_y': [],
            'force_z': [],
            'exact_energy': [],
            'exact_force_x': [],
            'exact_force_y': [],
            'exact_force_z': [],
        }
        rotation_matrices = []
        permute_matrices = []
        elapsed_time = []
        pbc = self.json_data.get('pbc') 
        if pbc == None:
            pbc = True

        for i, data in enumerate(self.data_loader):
            data = data.to(self.device)
            data.positions.requires_grad_(True)
            data.cell.requires_grad_(True)
            t1 = time()
            batch, entropy_loss = self.transform(
                data=data, 
                equiv_model=self.equiv_model, # for the probabilistic symmetrization
                n_samples=self.json_data.get("nsamples") # for the probabilistic symmetrization
            )
            if "prob" in self.ga_method:
                rotation_matrices.append(batch.fa_rot[1].detach().cpu())
                permute_matrices.append(batch.fa_rot[0].detach().cpu())
            else:
                rotation_matrices.append(torch.ones(1, 1, 3, 3))
                permute_matrices.append(torch.ones(1, 1, 3, 3))

            #param_context = (
            #    self.ema.average_parameters() if self.ema is not None else nullcontext()
            #)
            #with param_context:
            preds = self.model_forward_cls(
                batch=batch,  # transform the PyG graph data
                model=self.model,
                frame_averaging=self.group_averaging, 
                mode='valid',      
                crystal_task=pbc,
                edge_mask=None,
                permute=self.permute
            )
            # print(f'Elapsed time of 1 epoch: {time()-t1}')
            elapsed_time.append(time()-t1)
            species = data['species']
            node_enr_avg = torch.tensor([self.enr_avg_per_element[int(iz)] for iz in species]).sum()
            if element_wise:
                e_corr = torch.tensor(
                    [e_corr_[int(iz)] for iz in species]
                ).sum()
            else:
                e_corr = e_corr_
            preds['energy'] = preds["energy"] + node_enr_avg + e_corr

            test_values['energy'].append(preds['energy'].detach().cpu())
            test_values['force_x'].append(preds['forces'][:,0].detach().cpu())
            test_values['force_y'].append(preds['forces'][:,1].detach().cpu())
            test_values['force_z'].append(preds['forces'][:,2].detach().cpu())
            test_values['exact_energy'].append(data['energy'].detach().cpu())
            test_values['exact_force_x'].append(data['forces'][:,0].detach().cpu())
            test_values['exact_force_y'].append(data['forces'][:,1].detach().cpu())
            test_values['exact_force_z'].append(data['forces'][:,2].detach().cpu())         

            loss_dict = self.compute_loss(preds, data)
            for l in total_loss_dict.keys(): # predict part
                val = loss_dict.get(l, torch.nan)
                total_loss_dict[l].append(val.detach().cpu() if isinstance(val, torch.Tensor) else val)

            del loss_dict['loss']
            if loss_dict.get('loss_grad') == None:
                loss_dict['loss_grad'] = torch.nan
            if not self.json_data['regress_forces']:
                del loss_dict['loss_grad']#, loss_dict['loss_f']
            step_dict = {
                    "date": date(),
                    "data": i,
                }

            for l in loss_dict.keys():
                loss_dict[l] = loss_dict.get(l)
            loss_dict['energy'] = float(preds['energy'][0].detach().cpu())
            target['energy'] = data['energy']
            
            self.logger.print_epoch_loss(step_dict, 
                                         loss_dict, 
                                         target,
                                         lr=None)
            if (i+1) % 50 == 0:                         
                data.clear()
                del data, preds, loss_dict
                torch.cuda.empty_cache()
            if (i+1) % 500 == 0:
                gc.collect()
        total_loss_dict = {key: torch.mean(torch.tensor(value).detach().cpu()) \
                        for key, value in total_loss_dict.items()}
        
        separator = self.logger.get_seperator()
        print(separator, file=self.fout)
        print(separator)
        print(f"MEAN_LOSS: {total_loss_dict['loss']:<11.5g}", file=self.fout)
        print(f"MEAN_LOSS(E): {total_loss_dict['loss_e']:<11.5g}", file=self.fout)
        print(f"MEAN_LOSS(F): {total_loss_dict['loss_f']:<11.5g}", file=self.fout)
        print(f"MEAN_LOSS: {total_loss_dict['loss']:<11.5g}")
        print(f"MEAN_LOSS(E): {total_loss_dict['loss_e']:<11.5g}")
        print(f"MEAN_LOSS(F): {total_loss_dict['loss_f']:<11.5g}\n")
        print(f"MEAN_ELAPSED_TIME(S): {torch.tensor(elapsed_time).mean():<11.5g}\n")
        torch.save(test_values, "test_values.pkl")
        np.save("group_reps_ks.npy", np.array(rotation_matrices))
        np.save("group_reps_hs.npy", np.array(permute_matrices))

    def get_scale_shift_correction(self, element_wise):
        if element_wise:
            try:
                e_corr = torch.tensor(
                    self.model_ckpt['valid_scale_shift']
                ).mean()
                element_wise = False
            except:
                e_corr = self.model_ckpt['valid_scale_shift'] 
                element_wise = True
        else:
            try:
                e_corr = torch.tensor(
                    self.model_ckpt['valid_scale_shift_origin']
                ).mean()
                element_wise = False
            except:
                e_corr = torch.tensor(
                    self.model_ckpt['valid_scale_shift']
                ).mean()
                element_wise = False   
        return e_corr, element_wise

    def save_input_parameters(self):
        train_config = self.json_data.get('predict') 
        fname = train_config.get('fname_plog') 
        if fname == None:
            fname = "predict.out"
        fname_ls = fname.rsplit('.', 1)
        fname = f'input_json_of_{fname_ls[0]}_{fname_ls[1]}.txt'
        fout = open(fname, 'w')
        pprint.pprint(self.json_data, stream=fout)

    def configure_dataloader(self):
        json_data = self.json_data
        data_loader, uniq_element, enr_avg_per_element = \
            get_dataloader_to_predict(
                json_data["predict"]['fname_traj'],
                json_data["predict"]['ndata'],
                1,  # nbatch
                json_data['cutoff'],
                self.model_ckpt,
                json_data['regress_forces']
        )
        return data_loader, uniq_element, enr_avg_per_element

    def configure_logger_head(self):
        log_config = self.json_data.get("plog_config")
        if log_config == None:
            if self.json_data["regress_forces"]:
                log_config = {
                    'step': ['date', 'data'],
                    'predict': ['energy', 'loss_e', 'loss_f', 'loss_grad'],
                    'exact': ['energy']
                    }  # loss_l2
            else:
                log_config = {
                    'step': ['date', 'data'],
                    'predict': ['energy', 'loss_e'],
                    'exact': ['energy']
                    }
        return log_config
    
    def configure_logger(self):
        log_config = self.configure_logger_head()

        log_length = self.json_data.get("plog_length") 
        if log_length == None:
            log_length = 'precise'
        log_interval = 1

        predict_config = self.json_data.get('predict') 
        fname = predict_config.get('fname_plog')
        if fname == None:
            fname = "predict.out"
        fout = open(fname, 'w')
        logger = Logger(log_config, self.loss_config, log_length, fout)
        #logger.print_logger_head()
        separator = logger.get_seperator()
        atexit.register(lambda: on_exit(
                                    fout, 
                                    separator, 
                                    self.n_params, 
                                    self.json_data,
                                    self.date1
                                )
                        )
        return log_config, log_interval, logger, fout
    
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
            force_target = data["forces"].flatten()
            if cosine_sim:
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                loss["loss_grad"] = -torch.mean(cos(preds["forces"], grad_target))
            else:
                loss["loss_grad"] = self.loss_fn["force_loss"](
                    grad_target.flatten(), force_target
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