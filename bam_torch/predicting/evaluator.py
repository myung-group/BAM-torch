import torch

import gc
import atexit
import pprint
from copy import deepcopy

from bam_torch.utils.logger import Logger
from bam_torch.training.base_trainer import BaseTrainer
from bam_torch.utils.utils import (
    get_dataloader_to_predict, 
    date, 
    on_exit, 
    get_dataloader
)
import numpy as np


class Evaluator(BaseTrainer):
    def __init__(self, json_data, rank=0, world_size=1):
        self.json_data = json_data
        self.json_data['NN']['restart'] = False
        self.json_data["predict"]["evaluate_tag"] = True
        self.json_data["nbatch"] = 1
        self.rank = 0
        self.world_size = 1
        
        pd_config = self.json_data.get("predict", {})
        if pd_config.get("loss_config") is not None:
            self.json_data["NN"]["loss_config"] = pd_config.get("loss_config")

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

    def evaluate(self, element_wise=True):
        self.logger.print_logger_head()
        target = {}
        eval_loss_dict = {'loss':[],
                           'loss_e':[],
                           'loss_f':[],
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
        mae_values = {
            'energy': [],
            'forces': [],
            'stress': [],
        }
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

            mae_values['energy'].append(
                torch.abs(preds['energy'].flatten() - data['energy'].flatten()).detach().cpu()
            )
            if 'forces' in preds:
                mae_values['forces'].append(
                    torch.abs(preds['forces'].flatten() - data['forces'].flatten()).detach().cpu()
                )
            if 'stress' in preds:
                try:
                    mae_values['stress'].append(
                        torch.abs(preds['stress'].flatten() - data['stress'].flatten()).detach().cpu()
                    )
                except Exception:
                    pass

            loss_dict = self.compute_loss(preds, data)

            for l in eval_loss_dict.keys():
                eval_loss_dict[l].append(loss_dict.get(l, torch.nan).detach().cpu())

            # Print evaluation loss and predicted vs. exact energies.
            step_dict = {
                    "date": date(),
                    "data": i,
                }

            for l in loss_dict.keys():
                loss_dict[l] = loss_dict.get(l).detach().cpu()
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
        model_metrics = self._build_model_metrics(eval_loss_dict, mae_values)
        
        separator = self.logger.get_seperator()
        print(separator, file=self.fout)
        print(separator)
        print(f"MEAN_LOSS(E): {eval_loss_dict['loss_e']:<11.5g}", file=self.fout)
        print(f"MEAN_LOSS(F): {eval_loss_dict['loss_f']:<11.5g}", file=self.fout)
        print(f"MAE(E):       {model_metrics['energy_mae']:<11.5g}", file=self.fout)
        if model_metrics.get('force_mae') is not None:
            print(f"MAE(F):       {model_metrics['force_mae']:<11.5g}", file=self.fout)
        if model_metrics.get('stress_mae') is not None:
            print(f"MAE(S):       {model_metrics['stress_mae']:<11.5g}", file=self.fout)
        print(f"MEAN_LOSS(E): {eval_loss_dict['loss_e']:<11.5g}")
        print(f"MEAN_LOSS(F): {eval_loss_dict['loss_f']:<11.5g}")
        print(f"MAE(E):       {model_metrics['energy_mae']:<11.5g}")
        if model_metrics.get('force_mae') is not None:
            print(f"MAE(F):       {model_metrics['force_mae']:<11.5g}")
        if model_metrics.get('stress_mae') is not None:
            print(f"MAE(S):       {model_metrics['stress_mae']:<11.5g}")
        print()
        self._run_optional_catalyst_economics(model_metrics)
        torch.save(test_values, "test_values.pkl")

    def _build_model_metrics(self, eval_loss_dict, mae_values):
        def scalar(value):
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu())
            return float(value)

        def mean_abs(values):
            if not values:
                return None
            return float(torch.cat([v.flatten() for v in values]).mean())

        metrics = {key: scalar(value) for key, value in eval_loss_dict.items()}
        metrics.update({
            'energy_mae': mean_abs(mae_values.get('energy', [])),
            'force_mae': mean_abs(mae_values.get('forces', [])),
            'stress_mae': mean_abs(mae_values.get('stress', [])),
        })
        return metrics

    def _run_optional_catalyst_economics(self, model_metrics):
        economics_config = self.json_data.get('economics') or {}
        if not economics_config.get('enabled', False):
            return
        if economics_config.get('type', 'catalyst') != 'catalyst':
            raise ValueError("Only economics.type=catalyst is currently supported")

        from pathlib import Path
        from bam_torch.economics.report import (
            merge_model_and_economics_report,
            run_catalyst_economics,
        )

        required = ['scc_data', 'catalyst_data']
        missing = [key for key in required if not economics_config.get(key)]
        if missing:
            raise ValueError(f"economics config missing required keys: {missing}")

        out_dir = Path(economics_config.get('out_dir', 'outputs/catalyst_economics'))
        summary = run_catalyst_economics(
            scc_csv=economics_config['scc_data'],
            catalyst_csv=economics_config['catalyst_data'],
            out_dir=out_dir,
            scc_draws_path=economics_config.get('scc_draws'),
            save_scc_draws_path=economics_config.get('save_scc_draws'),
            n_draws=int(economics_config.get('n_draws', 20_000)),
            verbose=True,
        )
        combined_report = Path(
            economics_config.get('combined_report', out_dir / 'bam_catalyst_economics_report.json')
        )
        merge_model_and_economics_report(
            model_metrics=model_metrics,
            economics_summary=summary,
            output_json=combined_report,
        )
        print(f"[ECONOMICS] Combined BAM + catalyst report -> {combined_report}", file=self.fout)
        print(f"[ECONOMICS] Combined BAM + catalyst report -> {combined_report}")

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

