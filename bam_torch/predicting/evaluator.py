import torch

import gc
import atexit
import pprint
from copy import deepcopy

from bam_torch.utils.logger import Logger
from bam_torch.training.base_trainer import BaseTrainer
from bam_torch.utils.utils import get_dataloader_to_predict, date, on_exit, get_dataloader
import numpy as np

class Evaluator(BaseTrainer):
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

    def evaluate(self):
        self.logger.print_logger_head()
        target = {}
        eval_loss_dict = {'loss':[],
                           'loss_e':[],
                           'loss_f':[],
                           } 
        e_corr = torch.tensor(
            self.model_ckpt['valid_scale_shift']
        ).mean()
        for i, data in enumerate(self.data_loader):
            data = data.to(self.device)
            # Get node_enr_avg
            species = data['species']
            node_enr_avg = torch.tensor(
                [self.enr_avg_per_element[int(iz)] for iz in species],
            ).sum()
            # Predict energy, forces, and so on
            preds = self.model(data, backprop=False)
            # Correct the energy
            preds['energy'] = preds["energy"] + node_enr_avg + e_corr
            loss_dict = self.compute_loss(preds, data)

            for l in eval_loss_dict.keys():
                eval_loss_dict[l].append(loss_dict.get(l, torch.nan).detach().cpu())

            # Print evaluation loss and predicted vs. exact energies.
            step_dict = {
                    "date": date(),
                    "data": i,
                }
            loss_dict['energy'] = float(preds['energy'][0].detach().cpu())
            del loss_dict['loss']
            target['energy'] = data['energy']
            self.logger.print_epoch_loss(step_dict, 
                                         loss_dict, 
                                         target,
                                         lr=None)
            # Free memory
            del data, preds, loss_dict
            torch.cuda.empty_cache()
            if i % 100 == 0:
                gc.collect()

        eval_loss_dict = {key: torch.mean(torch.tensor(value)) \
                        for key, value in eval_loss_dict.items()}
        
        separator = self.logger.get_seperator()
        print(separator, file=self.fout)
        print(separator)
        print(f"MEAN_LOSS: {eval_loss_dict['loss']:<11.5g}", file=self.fout)
        print(f"MEAN_LOSS(E): {eval_loss_dict['loss_e']:<11.5g}", file=self.fout)
        print(f"MEAN_LOSS(F): {eval_loss_dict['loss_f']:<11.5g}", file=self.fout)
        print(f"MEAN_LOSS: {eval_loss_dict['loss']:<11.5g}")
        print(f"MEAN_LOSS(E): {eval_loss_dict['loss_e']:<11.5g}")
        print(f"MEAN_LOSS(F): {eval_loss_dict['loss_f']:<11.5g}\n")
    
    def save_input_parameters(self, input_json, fname=None):
        predict_config = self.json_data.get('predict') 
        if fname == None:
            fname = predict_config.get('fname_plog') 
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
                    'predict': ['energy', 'loss_e', 'loss_f'],
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
