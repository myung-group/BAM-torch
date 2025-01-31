import torch

import atexit
from copy import deepcopy

from bam_torch.utils.logger import Logger
from bam_torch.training.base_trainer import BaseTrainer
from bam_torch.utils.utils import get_dataloader_to_predict, date, on_exit


class Evaluator(BaseTrainer):
    def __init__(self, json_data):
        #super().__init__(json_data)
        self.json_data = json_data
        self.date1 = date()

        ## 1) Configure device
        self.configure_device()

        ## 3) Configure model
        self.configure_model()
        self.model.to(self.device)

        ## 4) Load trained-model
        model_ckpt = torch.load(json_data["predict"]["model"])
        self.model.load_state_dict(model_ckpt['params'])
        # Check the number of parameters
        self.n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'\nnumber of parameters:\n -- model {self.n_params}\033[0m\n')
        self.model.eval()

        ## 5) Configure data_loader
        self.data_loader, self.uniq_element, self.enr_avg_per_element = \
                                                get_dataloader_to_predict(
                                                    json_data["predict"]['fname_traj'],
                                                    json_data["predict"]['ndata'],
                                                    1,  # nbatch
                                                    json_data['cutoff'],
                                                    model_ckpt,
                                                    json_data['regress_forces']
                                                )

        ## 6) Configure loss function
        self.loss_fn = self.load_loss()

        ## 7) Configure logger
        self.configure_logger()
    
    def evaluate(self):
        target = {}
        total_loss_dict = {'loss':[],
                           'loss_e':[],
                           'loss_f':[],
                           } 
        keys = list(total_loss_dict.keys())
        for i, data in enumerate(self.data_loader):
            data = data.to(self.device)
            tgt_enr = data.energy
            target['energy'] = tgt_enr
            
            species = data.species
            node_enr_avg = torch.tensor([self.enr_avg_per_element[int(iz)] for iz in species]).sum()
            preds = self.model(deepcopy(data), backprop=False)
            energy = preds["energy"]
            energy += node_enr_avg
            preds['energy'] = energy
            loss_dict = self.compute_loss(preds, data)
            for l in total_loss_dict.keys(): # predict part
                total_loss_dict[l].append(loss_dict.get(l, torch.nan))
            loss_dict['energy'] = float(preds['energy'][0])
            del loss_dict['loss']

            step_dict = {
                    "date": date(),
                    "data": i,
                }
            self.logger.print_epoch_loss(step_dict, 
                                         loss_dict, 
                                         target,
                                         lr=None,
                                         fout=self.fout)
        total_loss_dict = {key: torch.mean(torch.tensor(value)) \
                        for key, value in total_loss_dict.items()}    

        print(self.separator, file=self.fout)
        print(self.separator)
        print(f"MEAN_LOSS: {total_loss_dict['loss']:<11.5g}", file=self.fout)
        print(f"MEAN_LOSS(E): {total_loss_dict['loss_e']:<11.5g}", file=self.fout)
        print(f"MEAN_LOSS(F): {total_loss_dict['loss_f']:<11.5g}", file=self.fout)
        print(f"MEAN_LOSS: {total_loss_dict['loss']:<11.5g}")
        print(f"MEAN_LOSS(E): {total_loss_dict['loss_e']:<11.5g}")
        print(f"MEAN_LOSS(F): {total_loss_dict['loss_f']:<11.5g}\n")

    def configure_logger_head(self):
        self.log_config = self.json_data.get("plog_config")
        if self.log_config == None:
            if self.json_data["regress_forces"]:
                self.log_config = {
                    'step': ['date', 'data'],
                    'predict': ['energy', 'loss_e', 'loss_f'],
                    'exact': ['energy']
                    }  # loss_l2
            else:
                self.log_config = {
                    'step': ['date', 'data'],
                    'predict': ['energy', 'loss_e'],
                    'exact': ['energy']
                    }
    
    def configure_logger(self):
        self.configure_logger_head()

        self.log_length = self.json_data.get("plog_length") 
        if self.log_length == None:
            self.log_length = 'precise'

        predict_config = self.json_data.get('predict') 
        fname = predict_config.get('fname_plog')
        if fname == None:
            fname = "predict.out"
        self.fout = open(fname, 'w')
        self.logger = Logger(self.log_config, self.loss_config, self.log_length)
        self.separator = self.logger.print_logger_head(self.fout)
        atexit.register(lambda: on_exit(
                                    self.fout, 
                                    self.separator, 
                                    self.n_params, 
                                    self.json_data,
                                    self.date1
                                )
                        )