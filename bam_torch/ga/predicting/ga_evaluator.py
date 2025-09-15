import torch

import gc
import atexit
import pprint
from copy import deepcopy

from bam_torch.utils.logger import Logger
from bam_torch.training.base_trainer import BaseTrainer
from bam_torch.training.ga_trainer import GATrainer
from bam_torch.utils.utils import get_dataloader_to_predict, date, on_exit, get_dataloader
from bam_torch.ga.group_averaging.transforms import FrameAveraging
from bam_torch.ga.group_averaging.fa_forward import model_forward, pa_model_forward
from bam_torch.ga.utils.eval import eval_model_symmetries
from time import time
import numpy as np


class GAEvaluator(GATrainer):
    def __init__(self, json_data):
        self.json_data = json_data
        self.json_data['NN']['restart'] = False
        self.json_data["predict"]["evaluate_tag"] = True
        self.json_data["nbatch"] = 1
        self.rank = 0
        self.world_size = 0
        self.date1 = date()

        ## 1) Reproducibility
        self.set_random_seed()

        ## 2) Configure device
        self.device = self.configure_device()

        ## 3) Configure model
        self.model, self.n_params, self.model_ckpt, self.start_epoch = self.configure_model()

        ## 4) Configure data_loader   
        self.data_loader, self.uniq_element, self.enr_avg_per_element = \
                                                        self.configure_dataloader()

        ## 5) Configure loss function
        self.loss_fn, self.loss_config = self.load_loss()

        ## 6) Configure logger
        self.log_config, self.log_interval, self.logger, self.fout = self.configure_logger()

        # Save input parameters setting
        self.save_input_parameters()

    def evaluate(self):
        self.logger.print_logger_head()
        target = {}
        total_loss_dict = {'loss':[],
                           'loss_e':[],
                           'loss_f':[],
                           'loss_grad':[],
                           } 
        fa_method = self.json_data.get('ga_method')
        gs = [None] * len(self.data_loader)
        if fa_method == None:
            # The frame averaging method
            # : {"det", "all", "se3-stochastic", "se3-det", "se3-all", "stochastic"}
            fa_method = "stochastic"
        elif fa_method == "prob":
            gs = np.load('test_gs.npy')

        frame_averaging = self.json_data.get('frame_averaging')
        if frame_averaging == None:
            # ["2D", "3D", "DA", ""]
            frame_averaging = "3D"
        pbc = self.json_data.get('pbc') # the frame averaging method: {"det", "all", "se3-stochastic", "se3-det", "se3-all", "stochastic"}:
        if pbc == None:
            pbc = True
        transform = FrameAveraging(frame_averaging, fa_method)
        e_corr = torch.tensor(self.model_ckpt['valid_scale_shift']) # or train_scale_shift?
        e_corr = e_corr.flatten().mean()
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
        for i, data in enumerate(self.data_loader):
            data = data.to(self.device)
            t1 = time()
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
                    batch=transform(data, gs[i]),  # transform the PyG graph data
                    model=self.model,
                    frame_averaging=frame_averaging, 
                    mode="test",      
                    crystal_task=pbc, 
                )
            """
            data.pos = data.positions
            preds = self.model(data)
            """
            print(f'Elapsed time of 1 epoch: {time()-t1}')
            
            species = data['species']
            node_enr_avg = torch.tensor([self.enr_avg_per_element[int(iz)] for iz in species]).sum()
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

            target['energy'] = data['energy']
            loss_dict['energy'] = float(preds['energy'][0].detach().cpu())
            del loss_dict['loss']
            if loss_dict.get('loss_grad') == None:
                loss_dict['loss_grad'] = torch.nan
            if not self.json_data['regress_forces']:
                del loss_dict['loss_grad']#, loss_dict['loss_f']
            step_dict = {
                    "date": date(),
                    "data": i,
                }
            self.logger.print_epoch_loss(step_dict, 
                                         loss_dict, 
                                         target,
                                         lr=None)
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
        torch.save(test_values, "test_values.pkl")

    def evaluate_model_symmetries(self):
        # Evaluate symmetry of model
        frame_averaging = "3D"  # symmetry preservation method used: {"3D", "2D", "DA", ""}:
        fa_method = "stochastic"  # the frame averaging method: {"det", "all", "se3-stochastic", "se3-det", "se3-all", ""}:
        #transform = FrameAveraging(frame_averaging, fa_method)
        symmetry = eval_model_symmetries(
            self.data_loader, 
            self.model, 
            model_forward,
            frame_averaging, 
            fa_method, 
            self.device
        )
        import pprint
        print("\nSYMMETRY_CHECK: ", file=fout)
        pprint.pprint(symmetry, stream=fout)
        pprint.pprint(symmetry)
        print(" ")

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
    