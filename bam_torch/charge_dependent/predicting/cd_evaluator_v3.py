"""
Charge-dependent model evaluator Phase 3 (E).

Differences from Phase 2 CDEvaluator:
  - Inherits CDTrainerV3 (instead of CDTrainer)
  - set_model() creates ChargeRACEv3
  - Properly loads Phase 3 checkpoint attributes (use_cent_energy / charge_type)

Issues when running Phase 2 CDEvaluator with Phase 3 checkpoint:
  1) CDTrainer.set_model() -> creates ChargeRACE (Phase 2)
  2) ChargeRACEv3 state_dict key mismatch (use_cent_energy etc. missing)
  3) AttributeError or energy calculation error in forward()

CDEvaluatorV3 inherits CDTrainerV3 to resolve this.
"""

import torch
import gc
import atexit
import pprint
from copy import deepcopy

import numpy as np

from bam_torch.utils.logger import Logger
from bam_torch.charge_dependent.training.cd_trainer_v3 import CDTrainerV3
from bam_torch.charge_dependent.utils.cd_utils import (
    get_dataloader_charge_to_predict,
)
from bam_torch.utils.utils import date, on_exit


class CDEvaluatorV3(CDTrainerV3):
    """
    Phase 3 (E) Charge-dependent model evaluator.

    Inherits CDTrainerV3 and evaluates:
      - energy, forces prediction accuracy
      - CEP atomic charges (q_i) prediction accuracy
      - chi (electronegativity), U_CENT, E_SR collection
    """

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
        """Configure components required for evaluation."""
        self.set_random_seed()
        self.device = self.configure_device()
        self.model, self.n_params, self.model_ckpt, self.start_epoch = \
            self.configure_model()
        self.optimizer = self.configure_optimizer()
        self.data_loader, self.uniq_element, self.enr_avg_per_element = \
            self.configure_dataloader()
        self.loss_fn, self.loss_config = self.configure_loss()
        self.log_config, self.log_interval, self.logger, self.fout = \
            self.configure_logger()

    def evaluate(self, element_wise=True):
        """Run model evaluation. Collects Phase 3 CEP results."""
        self.logger.print_logger_head()
        eval_loss_dict = {
            'loss': [], 'loss_e': [], 'loss_f': [], 'loss_q': [],
        }

        e_corr_, element_wise = self.get_scale_shift_correction(element_wise)

        test_values = {
            'energy': [],
            'force_x': [], 'force_y': [], 'force_z': [],
            'exact_energy': [],
            'exact_force_x': [], 'exact_force_y': [], 'exact_force_z': [],
            'atomic_charges': [],
            'exact_atomic_charges': [],
            'total_charge': [],
            'total_multiplicity': [],
            'chi': [],
            'U_CENT': [],
            'E_SR': [],
        }

        target = {}

        for i, data in enumerate(self.data_loader):
            data = data.to(self.device)

            species = data['species']
            node_enr_avg = torch.tensor(
                [self.enr_avg_per_element[int(iz)] for iz in species],
            ).sum().to(self.device)

            preds = self.model(data, backprop=False)

            if element_wise:
                e_corr = torch.tensor(
                    [e_corr_[int(iz)] for iz in species]
                ).sum().to(self.device)
            else:
                e_corr = e_corr_
            preds['energy'] = preds["energy"] + node_enr_avg + e_corr

            # energy / forces
            test_values['energy'].append(preds['energy'].detach().cpu())
            test_values['force_x'].append(preds['forces'][:, 0].detach().cpu())
            test_values['force_y'].append(preds['forces'][:, 1].detach().cpu())
            test_values['force_z'].append(preds['forces'][:, 2].detach().cpu())
            test_values['exact_energy'].append(data['energy'].detach().cpu())
            test_values['exact_force_x'].append(data['forces'][:, 0].detach().cpu())
            test_values['exact_force_y'].append(data['forces'][:, 1].detach().cpu())
            test_values['exact_force_z'].append(data['forces'][:, 2].detach().cpu())

            # CEP results collection
            if "atomic_charges" in preds:
                test_values['atomic_charges'].append(
                    preds['atomic_charges'].detach().cpu()
                )
            if "atomic_charges" in data:
                test_values['exact_atomic_charges'].append(
                    data['atomic_charges'].detach().cpu()
                )
            if "total_charge" in preds:
                test_values['total_charge'].append(
                    preds['total_charge'].detach().cpu()
                )
            if "total_multiplicity" in data:
                test_values['total_multiplicity'].append(
                    data['total_multiplicity'].detach().cpu()
                )
            if "chi" in preds:
                test_values['chi'].append(preds['chi'].detach().cpu())
            if "U_CENT" in preds:
                test_values['U_CENT'].append(preds['U_CENT'].detach().cpu())
            if "E_SR" in preds:
                test_values['E_SR'].append(preds['E_SR'].detach().cpu())

            loss_dict = self.compute_loss(preds, data)

            for l in eval_loss_dict.keys():
                val = loss_dict.get(l)
                if val is not None:
                    eval_loss_dict[l].append(val.detach().cpu())
                else:
                    eval_loss_dict[l].append(torch.tensor(float('nan')))

            step_dict = {"date": date(), "data": i}
            for l in list(loss_dict.keys()):
                loss_dict[l] = loss_dict[l].detach().cpu()
            loss_dict['energy'] = float(preds['energy'][0].detach().cpu())
            del loss_dict['loss']
            target['energy'] = data['energy']
            self.logger.print_epoch_loss(step_dict, loss_dict, target, lr=None)

            del data, preds, loss_dict
            if str(self.device) != 'cpu':
                torch.cuda.empty_cache()
            if i % 100 == 0:
                gc.collect()

        # Results summary
        eval_loss_dict = {
            key: torch.mean(torch.tensor(value))
            for key, value in eval_loss_dict.items()
        }

        separator = self.logger.get_seperator()
        print(separator, file=self.fout)
        print(separator)
        for tag, key in [("E", "loss_e"), ("F", "loss_f"), ("Q", "loss_q")]:
            val = eval_loss_dict.get(key)
            if val is not None and not torch.isnan(val):
                line = f"MEAN_LOSS({tag}): {val:<11.5g}"
                print(line, file=self.fout)
                print(line)
        print(file=self.fout)

        torch.save(test_values, "test_values.pkl")
        print(f"\n\033[32mPrediction values saved to test_values.pkl\033[0m")

    def get_scale_shift_correction(self, element_wise):
        """Load scale-shift correction values saved during training."""
        if element_wise:
            try:
                e_corr = torch.tensor(
                    self.model_ckpt['valid_scale_shift']
                ).mean().to(self.device)
                element_wise = False
            except Exception:
                e_corr = self.model_ckpt['valid_scale_shift']
                element_wise = True
        else:
            try:
                e_corr = torch.tensor(
                    self.model_ckpt['valid_scale_shift_origin']
                ).mean().to(self.device)
                element_wise = False
            except Exception:
                e_corr = torch.tensor(
                    self.model_ckpt['valid_scale_shift']
                ).mean().to(self.device)
                element_wise = False
        return e_corr, element_wise

    def configure_dataloader(self):
        """Configure evaluation DataLoader (with charge data)."""
        jd = self.json_data
        charge_config = jd.get('charge', {})
        charge_key       = charge_config.get('charge_key', 'charges')
        total_charge_key = charge_config.get('total_charge_key', 'total_charge')

        data_loader, uniq_element, enr_avg_per_element = \
            get_dataloader_charge_to_predict(
                jd["predict"]['fname_traj'],
                jd["predict"]['ndata'],
                1,
                jd['cutoff'],
                self.model_ckpt,
                jd['regress_forces'],
                charge_key=charge_key,
                total_charge_key=total_charge_key,
            )
        return data_loader, uniq_element, enr_avg_per_element

    def save_input_parameters(self, input_json, fname=None):
        predict_config = self.json_data.get('predict')
        if fname is None:
            fname = predict_config.get('fname_plog')
            if fname is None:
                fname = "predict.out"
        fname_ls = fname.rsplit('.', 1)
        fname = f'input_json_of_{fname_ls[0]}_{fname_ls[1]}.txt'
        fout = open(fname, 'w')
        pprint.pprint(self.json_data, stream=fout)

    def configure_logger_head(self):
        log_config = self.json_data.get("plog_config")
        if log_config is None:
            predict_fields = ['energy', 'loss_e', 'loss_f', 'loss_q']
            if self.json_data["regress_forces"]:
                log_config = {
                    'step': ['date', 'data'],
                    'predict': predict_fields,
                    'exact': ['energy'],
                }
            else:
                log_config = {
                    'step': ['date', 'data'],
                    'predict': ['energy', 'loss_e', 'loss_q'],
                    'exact': ['energy'],
                }
        return log_config

    def configure_logger(self):
        log_config = self.configure_logger_head()
        log_length = self.json_data.get("plog_length") or 'precise'
        log_interval = 1

        predict_config = self.json_data.get('predict')
        fname = predict_config.get('fname_plog') or "predict.out"
        fout = open(fname, 'w')
        logger = Logger(log_config, self.loss_config, log_length, fout)
        separator = logger.get_seperator()
        atexit.register(
            lambda: on_exit(fout, separator, self.n_params,
                            self.json_data, self.date1)
        )
        return log_config, log_interval, logger, fout
