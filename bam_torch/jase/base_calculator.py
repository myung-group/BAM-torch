import torch
import numpy as np
from torch_geometric.loader import DataLoader
from ase.calculators.calculator import Calculator, all_changes

from bam_torch.training.base_trainer import BaseTrainer
from bam_torch.utils.utils import get_graphset_to_predict


class BaseCalculator(Calculator, BaseTrainer):
    implemented_properties = ['energy', 'forces', 'free_energy']

    def __init__(self, json_data, model=None, element_wise=True):
        """ Model is a trained-model's pckl file
        """
        Calculator.__init__(self)

        # BaseTrainer's setup helpers read these; a calculator is always a
        # single, non-distributed process.
        self.ddp = False
        self.rank = 0
        self.world_size = 1
        self.msg = ''

        self.json_data = json_data
        self.json_data['NN']['restart'] = False
        self.json_data.setdefault("predict", {})
        self.json_data["predict"]["evaluate_tag"] = True
        if model is not None:
            self.json_data["predict"]["model"] = model

        ## 1) Reproducibility
        self.set_random_seed()

        ## 2) Configure device
        self.device = self.configure_device()
    
        ## 3) Configure model
        self.model, self.n_params, self.model_ckpt, _ = self.configure_model()
        self.uniq_element = self.model_ckpt['uniq_element']
        self.enr_avg_per_element = self.model_ckpt['enr_avg_per_element']
        self.e_corr, self.element_wise = self.get_scale_shift_correction(
            element_wise
        )

    def get_scale_shift_correction(self, element_wise):
        """Same correction the Evaluator and the ASE RACECalculator use.

        `valid_scale_shift` is a per-element mapping in newer checkpoints and a
        scalar in older ones; averaging fails on the mapping, which is how the
        element-wise case is detected.
        """
        if element_wise:
            try:
                e_corr = torch.tensor(
                    self.model_ckpt['valid_scale_shift']
                ).mean()
                element_wise = False
            except Exception:
                e_corr = self.model_ckpt['valid_scale_shift']
                element_wise = True
        else:
            try:
                e_corr = torch.tensor(
                    self.model_ckpt['valid_scale_shift_origin']
                ).mean()
            except Exception:
                e_corr = torch.tensor(
                    self.model_ckpt['valid_scale_shift']
                ).mean()
            element_wise = False
        return e_corr, element_wise

    def calculate(self, atoms, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        data = get_graphset_to_predict(
                    [atoms.copy()],
                    self.json_data['cutoff'],
                    self.uniq_element,
                    self.json_data['regress_forces']
                )
        data = next(iter(DataLoader(data))).to(self.device)

        # The batch carries non-leaf tensors (positions need grad for the
        # force autograd), which deepcopy cannot handle; the model does not
        # mutate it, so pass it through as the ASE RACECalculator does.
        preds = self.model(data, backprop=False)
        species = np.array([self.uniq_element[iz] for iz in atoms.numbers])
        node_enr_avg = np.array([self.enr_avg_per_element[int(iz)] \
                        for iz in species]).sum()
        # Same scale/shift correction the evaluator and the ASE RACECalculator
        # apply; without it the energy is offset from the training reference.
        if self.element_wise:
            e_corr = torch.tensor(
                [self.e_corr[int(iz)] for iz in species]
            ).sum()
        else:
            e_corr = self.e_corr
        energy = preds["energy"] + node_enr_avg + e_corr

        self.results['energy'] = float(energy.detach())
        self.results['free_energy'] = self.results['energy']
        self.results['forces'] = np.array(preds['forces'].detach().cpu())

    
    
