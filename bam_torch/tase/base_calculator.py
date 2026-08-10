import torch
import numpy as np
from torch_geometric.loader import DataLoader
from ase.calculators.calculator import Calculator, all_changes
from copy import deepcopy

from bam_torch.training.multihead_trainer import MultiheadTrainer
from bam_torch.predicting.mh_evaluator import MultiheadEvaluator
from bam_torch.utils.utils import get_graphset_to_predict


class RACECalculator(Calculator, MultiheadEvaluator):
    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']

    def __init__(
        self, 
        json_data=None, 
        model=None, 
        device=None, 
        element_wise=True, 
        multihead=None
    ):
        """ 
        To use the RACECalculator, specify one of the following inputs:
            - Load configuration from JSON:
                json_data = json.load(open('/path/to/your/input.json'))
                calc = RACECalculator(json_data)
            - Or provide directly a pre-trained model file with device:
                model = '/path/to/your/model.pkl'
                calc = RACECalculator(model=model, device='cuda')
        """
        Calculator.__init__(self)

        self.ddp = False
        self.rank = 0
        self.world_size = 1
        self.msg = ''
        self.json_data = json_data
        self.device = self.configure_device(device)

        if model is not None and json_data is None:
            model_ckpt = torch.load(
                model, 
                map_location=self.device, 
                weights_only=False
            )
            self.json_data = model_ckpt['input.json']
            self.json_data["predict"]["model"] = model
        elif model is None and json_data is None:
            raise ValueError(
                "\033[31mTo use the RACECalculator, specify one of the following inputs:\n"
                "\033[33m - Load configuration from JSON:\n"
                "       json_data = json.load(open('/path/to/your/input.json'))\n"
                "       calc = RACECalculator(json_data)\n"
                " - Or provide a pre-trained model file:\n"
                "       model = '/path/to/your/model.pkl'\n"
                "       calc = RACECalculator(model)\033[0m\n"
            )
            
        self.json_data['NN']['restart'] = False
        self.json_data["predict"]["evaluate_tag"] = True
    
        # Configure model
        self.configure_head(multihead)
        self.model, self.n_params, self.model_ckpt, _ = self.configure_model()
        self.uniq_element = self.model_ckpt['uniq_element']
        self.enr_avg_per_element = self.model_ckpt['enr_avg_per_element']
        self.e_corr, self.element_wise = self.get_scale_shift_correction(element_wise)

    def calculate(self, atoms, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        data = get_graphset_to_predict(
                    [atoms.copy()],
                    self.json_data['cutoff'],
                    self.uniq_element,
                    self.json_data['regress_forces']
                )
        data = next(iter(DataLoader(data))).to(self.device)

        preds = self.model(data, backprop=False)
        species = np.array([self.uniq_element[iz] for iz in atoms.numbers])
        node_enr_avg = np.array([self.enr_avg_per_element[int(iz)] \
                        for iz in species]).sum()
        if self.element_wise:
            e_corr = torch.tensor(
                [self.e_corr[int(iz)] for iz in species]
            ).sum()
        else:
            e_corr = self.e_corr
        energy = preds["energy"] + node_enr_avg + e_corr

        self.results['energy'] = float(energy.detach())
        self.results['forces'] = np.array(preds['forces'].detach().cpu())
        self.results['stress'] = np.array(preds['stress'][0].detach().cpu())
        # ASE optimizers / phonopy may request the force-consistent (free)
        # energy; BAM has no entropic or electronic-temperature term, so the
        # free energy is identical to the potential energy.
        self.results['free_energy'] = self.results['energy']

    def configure_device(self, device=None):
        if device is None and self.json_data is not None:
            device_config = self.json_data['device']
        elif device is not None:
            device_config = device
        else: 
            device_config = 'cpu'

        if device_config == 'cpu':
            device = 'cpu'
            self.msg += f'\ndevice:\n\033[33m -- {device}\n'
            try:
                from mpi4py import MPI
                self.rank = MPI.COMM_WORLD.Get_rank()
                size = MPI.COMM_WORLD.Get_size()
                self.msg += f' -- number of cpu  {size}\n\033[0m\n'
            except:
                self.msg += f' -- number of cpu  {torch.get_num_threads()}\033[0m\n'
        else:
            if torch.cuda.is_available():
                if self.ddp:
                    # Use local rank for device in multi-node setup
                    local_rank = self.rank % torch.cuda.device_count()
                    torch.cuda.set_device(local_rank)
                    device = torch.device(f"cuda:{local_rank}")
                    self.msg += f'\ndevice:\n\033[33m -- {device} (local_rank: {local_rank})\n'
                    self.msg += f' -- number of gpu  {self.world_size}\033[0m\n'
                else:
                    device = torch.device("cuda")
                    self.msg += f'\ndevice:\n\033[33m -- {device}\n'
                    self.msg += f' -- number of gpu  {self.world_size}\033[0m\n'
            else:
                device = torch.device("cpu")
                self.msg += f'\ndevice:\n -- {device}\n'

        print(self.msg)
        return device

    def configure_head(self, multihead):
        multihead_config = self.json_data.get("multihead", {})
        if multihead is None or multihead == False:
            multihead = multihead_config.get("enabled", False)
        
        # Update the multihead's configuration
        if multihead_config == {}:
            self.json_data["multihead"] = {}
            self.json_data["NN"]["foundation_model"] \
                        = self.json_data["NN"]["fname_pkl"]
        self.json_data["multihead"]["enabled"] = multihead

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