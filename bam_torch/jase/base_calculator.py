import torch
import pickle
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from bam_torch.training.base_trainer import BaseTrainer


class BaseCalculator(Calculator, BaseTrainer):
    def __init__(self, json_data, model):
        """ Model is a trained-model's pckl file
        """
        Calculator.__init__(self)
        #BaseTrainer.__init__(self, json_data)
        self.json_data = json_data

        ## 1) Reproducibility
        self.set_random_seed()

        ## 2) Configure device
        self.configure_device()
    
        ## 3) Configure model
        self.configure_model()
        self.model.to(self.device)

        ## 4) Load trained-model
        #self.model_ckpt = torch.load(json_data["predict"]["model"])
        self.model_ckpt = model
        self.model.load_state_dict(self.model_ckpt['params'])
        # Check the number of parameters
        self.n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'\nnumber of parameters:\n -- model {self.n_params}\033[0m\n')
        self.model.eval()
    
    def calculate(self, atoms, properties=['energy', 'forces'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        uniq_element = model_ckpt['uniq_element']
        enr_avg_per_element = model_ckpt['enr_avg_per_element']

        species = np.array([uniq_element[iz] for iz in atoms.numbers])
        node_enr_avg = torch.tensor([enr_avg_per_element[int(iz)] for iz in species]).sum()
        preds = self.model(deepcopy(atoms), backprop=False)
        energy = preds["energy"]
        energy += node_enr_avg

        self.results['energy'] = float(energy)
        self.results['forces'] = np.array(preds['forces'])

    def get_params(self):
        return self.n_params
