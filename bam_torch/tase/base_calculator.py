import torch
import numpy as np
from torch_geometric.loader import DataLoader
from ase.calculators.calculator import Calculator, all_changes
from copy import deepcopy

from bam_torch.training.base_trainer import BaseTrainer
from bam_torch.utils.utils import get_graphset_to_predict


class BaseCalculator(Calculator, BaseTrainer):
    implemented_properties = ['energy', 'forces']

    def __init__(self, json_data, model=None):
        """ Model is a trained-model's pckl file
        """
        Calculator.__init__(self)

        self.json_data = json_data
        self.json_data['NN']['restart'] = False
        self.json_data["predict"]["evaluate_tag"] = True

        ## 1) Reproducibility
        self.set_random_seed()

        ## 2) Configure device
        self.device = self.configure_device()
    
        ## 3) Configure model
        self.model, self.n_params, model_ckpt = self.configure_model()
        self.uniq_element = model_ckpt['uniq_element']
        self.enr_avg_per_element = model_ckpt['enr_avg_per_element']

    def calculate(self, atoms, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        data = get_graphset_to_predict(
                    [atoms.copy()],
                    self.json_data['cutoff'],
                    self.uniq_element,
                    self.json_data['regress_forces']
                )
        data = next(iter(DataLoader(data))).to(self.device)

        preds = self.model(deepcopy(data), backprop=False)
        species = np.array([self.uniq_element[iz] for iz in atoms.numbers])
        node_enr_avg = np.array([self.enr_avg_per_element[int(iz)] \
                        for iz in species]).sum()
        energy = preds["energy"] + node_enr_avg

        self.results['energy'] = float(energy)
        self.results['forces'] = np.array(preds['forces'].detach().cpu())

    
    
