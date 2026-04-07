"""
Charge-Dependent RACE ASE Calculator (Phase 3.5).

Extends RACECalculator to support charge-dependent models (Phase 2/3).
Inherits device management, checkpoint loading, and scale-shift correction
from the base calculator, overriding only model building and prediction.

Includes ZBL repulsive prior for short-range nuclear repulsion, preventing
unphysical atom overlap during MD when the ML model was trained only on
equilibrium structures.

Usage:
    from bam_torch.charge_dependent.calculator import CDRACECalculator

    calc = CDRACECalculator(model='p3_model.pkl', device='cuda')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    charges = calc.results['charges']
"""

import torch
import numpy as np

from ase.calculators.calculator import Calculator, all_changes
from torch_geometric.loader import DataLoader

from bam_torch.tase.base_calculator import RACECalculator
from bam_torch.utils.utils import get_graphset_to_predict
from bam_torch.utils.model_config import (
    parse_model_config,
    parse_cueq_config,
    parse_charge_config,
)


class CDRACECalculator(RACECalculator):
    """ASE Calculator for charge-dependent RACE models (Phase 2/3/3.5).

    Extends RACECalculator with:
      - Charge-dependent model instantiation (ChargeRACE, ChargeRACEv3)
      - CEP hard charge conservation
      - Autograd force computation (torch.enable_grad)
      - Per-atom partial charges in results
      - ZBL repulsive prior for MD stability

    Properties:
        - energy (eV)
        - forces (eV/Angstrom)
        - charges (per-atom partial charges, e)
    """

    implemented_properties = ['energy', 'forces', 'charges']

    def __init__(
        self,
        model=None,
        json_data=None,
        device=None,
        total_charge=0.0,
        element_wise=True,
        use_zbl=True,
        zbl_inner=0.5,
        zbl_outer=0.9,
    ):
        """
        Args:
            model: Path to trained .pkl checkpoint file.
            json_data: JSON config dict (alternative to model path).
            device: 'cpu', 'cuda', or torch.device.
            total_charge: System total charge for CEP constraint.
            element_wise: Whether to use element-wise scale-shift correction.
            use_zbl: Enable ZBL repulsive prior for short-range stability.
            zbl_inner: Full ZBL repulsion below this distance (A).
            zbl_outer: No ZBL above this distance (A). Default 0.9 A is
                       below the shortest common bond (O-H ~0.96 A).
        """
        self.total_charge = total_charge
        self.use_zbl = use_zbl
        self.zbl_inner = zbl_inner
        self.zbl_outer = zbl_outer

        # Ensure json_data has required sections for RACECalculator
        if json_data is not None:
            json_data.setdefault('predict', {})
            json_data.setdefault('NN', {})

        super().__init__(
            json_data=json_data,
            model=model,
            device=device,
            element_wise=element_wise,
            multihead=False,
        )

    def configure_model(self):
        """Build charge-dependent model from checkpoint.

        Overrides MultiheadEvaluator.configure_model() to instantiate
        ChargeRACE/ChargeRACEv3 instead of RACEUnified, using the
        centralized config parser (parse_model_config).
        """
        import os
        from bam_torch.charge_dependent.model import MODEL_REGISTRY

        # Load checkpoint (same pattern as MultiheadEvaluator)
        predict_config = self.json_data.get('predict', {})
        model_path = (
            predict_config.get('model')
            or self.json_data.get('NN', {}).get('fname_pkl')
        )

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        ckpt = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        ckpt_config = ckpt.get('input.json', self.json_data)

        # Build model using centralized config parser (Issue #5 fix)
        model_kwargs = parse_model_config(ckpt_config)
        model_kwargs['cueq_config'] = parse_cueq_config(ckpt_config)

        # Add charge-dependent parameters
        cd_kwargs = parse_charge_config(ckpt_config)
        model_kwargs.update(cd_kwargs)

        model_name = ckpt_config.get("model", "").lower()
        model_cls = MODEL_REGISTRY.get(model_name)
        if model_cls is None:
            raise ValueError(
                f"Unknown charge-dependent model: {model_name}. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        model = model_cls(**model_kwargs)

        # Load weights
        state_dict = ckpt.get('ema_params', ckpt['params'])
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        n_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        return model, n_params, ckpt, 0

    def calculate(
        self, atoms=None, properties=['energy'], system_changes=all_changes
    ):
        """Calculate energy, forces, and charges.

        Overrides RACECalculator.calculate() to:
          - Add charge fields (atomic_charges, total_charge) to the graph
          - Use torch.enable_grad() for autograd force computation
          - Extract per-atom charges from model predictions
          - Add ZBL repulsive correction for short-range stability
        """
        Calculator.calculate(self, atoms, properties, system_changes)

        # Build graph using shared BAM-torch utility
        graphset = get_graphset_to_predict(
            [self.atoms.copy()],
            self.json_data['cutoff'],
            self.uniq_element,
            self.json_data.get('regress_forces', True),
        )
        graph = graphset[0]

        # Add charge-dependent fields
        n_atoms = graph['positions'].shape[0]
        graph['atomic_charges'] = torch.zeros(n_atoms, dtype=torch.float32)
        graph['total_charge'] = torch.tensor(
            self.total_charge, dtype=torch.float32
        )

        data = next(iter(DataLoader([graph]))).to(self.device)

        # Forward pass with autograd enabled for forces
        with torch.enable_grad():
            preds = self.model(data, backprop=False)

        # Energy correction (same pattern as RACECalculator)
        species = np.array(
            [self.uniq_element[iz] for iz in self.atoms.numbers]
        )
        node_enr_avg = torch.tensor(
            [self.enr_avg_per_element[int(iz)] for iz in species]
        ).sum().to(self.device)

        if self.element_wise:
            e_corr = torch.tensor(
                [self.e_corr[int(iz)] for iz in species]
            ).sum().to(self.device)
        else:
            e_corr = self.e_corr

        energy = float((preds["energy"] + node_enr_avg + e_corr).detach().cpu())
        forces = preds['forces'].detach().cpu().numpy()

        # ZBL repulsive correction
        if self.use_zbl:
            from bam_torch.utils.zbl import compute_zbl_correction
            e_zbl, f_zbl = compute_zbl_correction(
                self.atoms.get_positions(),
                self.atoms.numbers,
                r_inner=self.zbl_inner,
                r_outer=self.zbl_outer,
            )
            energy += e_zbl
            forces += f_zbl

        # Store results
        self.results['energy'] = energy
        self.results['forces'] = forces

        if 'atomic_charges' in preds:
            self.results['charges'] = (
                preds['atomic_charges'].detach().cpu().numpy()
            )
        if 'chi' in preds:
            self.results['chi'] = preds['chi'].detach().cpu().numpy()
        if 'U_CENT' in preds:
            self.results['U_CENT'] = float(preds['U_CENT'].detach().cpu())
        if 'stress' in preds:
            self.results['stress'] = (
                preds['stress'][0].detach().cpu().numpy()
            )
