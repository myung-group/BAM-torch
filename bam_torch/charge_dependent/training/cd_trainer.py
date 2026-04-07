"""
Charge-dependent trainer Phase 2 for BAM-torch.

Training support for CENT2-based CEP model (ChargeRACE Phase 2).

Differences from Phase 1:
  - charge_mode parameter removed (CEP always active)
  - chg_cons_lambda removed (unnecessary with hard conservation)
  - cep_hidden_dim parameter added
  - compute_loss: only loss_q retained (loss_q_cons removed)
"""

import torch
import numpy as np
from e3nn import o3

from bam_torch.training.base_trainer import BaseTrainer
from bam_torch.model.wrapper_ops import CuEquivarianceConfig
from bam_torch.charge_dependent.model import MODEL_REGISTRY
from bam_torch.charge_dependent.utils.cd_utils import get_dataloader_charge


class CDTrainer(BaseTrainer):
    """
    Phase 2 Charge-dependent model Trainer.

    Inherits BaseTrainer and overrides:
      - set_model()           : creates ChargeRACE (Phase 2, CEP) model
      - configure_dataloader(): DataLoader with charge information
      - compute_loss()        : energy/force loss + charge loss (CEP-based)

    Loss composition:
      total_loss = enr_lambda  * loss_E
                 + frc_lambda  * loss_F
                 + chg_lambda  * loss_Q   (q_pred vs NPA charges)
    """

    def __init__(self, json_data, rank=0, world_size=1):
        super().__init__(json_data, rank, world_size)

    def set_model(self):
        """Configure ChargeRACE Phase 2 (CEP) model."""
        mc = self.json_data

        cutoff            = mc.get('cutoff', 6.0)
        num_species       = mc.get('num_species', 4)
        avg_num_neighbors = mc.get('avg_num_neighbors', 30)
        hidden_irreps     = o3.Irreps(mc.get('hidden_channels', "64x0e+64x1o+64x2e"))
        features_dim      = mc.get('features_dim', 64)
        num_basis_func    = mc.get('num_radial_basis', 8)
        nlayers           = mc.get('nlayers', 3)
        max_ell           = mc.get('max_ell', 3)
        output_irreps     = mc.get('output_channels', "1x0e")
        active_fn         = mc.get('active_fn', "identity")

        regress_forces = mc.get('regress_forces', "auto")
        if regress_forces is True:
            regress_forces = "autograd"
        elif regress_forces is False:
            regress_forces = "false"

        # CEP config
        charge_config = mc.get('charge', {})
        cep_hidden_dim = charge_config.get('cep_hidden_dim', 64)

        # CuEquivariance config
        cueq_config = mc.get('cueq_config')
        if cueq_config is None or cueq_config:
            try:
                import cuequivariance as cue
                import cuequivariance_torch as cuet
                CUET_AVAILABLE = True
            except ImportError:
                CUET_AVAILABLE = False
            if CUET_AVAILABLE:
                cueq_config = CuEquivarianceConfig(
                    enabled=True,
                    layout="ir_mul",
                    group="O3_e3nn",
                    optimize_all=True,
                )
                self.msg += '\nequiv. lib.:\n\033[33m -- CuEquivariance\033[0m\n'
        else:
            cueq_config = None
            self.msg += '\nequiv. lib.:\n\033[33m -- e3nn\033[0m\n'

        model_name = mc["model"].lower()
        model_cls = MODEL_REGISTRY.get(model_name)
        if model_cls is None:
            raise ValueError(
                f"Unknown charge-dependent model: {mc['model']}"
            )

        model = model_cls(
            cutoff=cutoff,
            avg_num_neighbors=avg_num_neighbors,
            num_species=num_species,
            max_ell=max_ell,
            num_basis_func=num_basis_func,
            hidden_irreps=hidden_irreps,
            nlayers=nlayers,
            features_dim=features_dim,
            output_irreps=output_irreps,
            active_fn=active_fn,
            regress_forces=regress_forces,
            cueq_config=cueq_config,
            cep_hidden_dim=cep_hidden_dim,
        )

        self.msg += '\n\033[33m -- Phase 2: CEP (CENT2-based), hard charge conservation\033[0m\n'
        return model

    def configure_dataloader(self):
        """Configure DataLoader with charge information."""
        jd = self.json_data
        charge_config = jd.get('charge', {})
        charge_key       = charge_config.get('charge_key', 'charges')
        total_charge_key = charge_config.get('total_charge_key', 'total_charge')

        train_loader, valid_loader, uniq_element, enr_avg_per_element = \
            get_dataloader_charge(
                jd['fname_traj'],
                jd['ntrain'],
                jd['nvalid'],
                jd['nbatch'],
                jd['cutoff'],
                jd['NN']['data_seed'],
                jd['element'],
                jd['regress_forces'],
                jd.get('max_neigh'),
                charge_key=charge_key,
                total_charge_key=total_charge_key,
                rank=self.rank,
                world_size=self.world_size,
            )
        return train_loader, valid_loader, uniq_element, enr_avg_per_element

    def compute_loss(self, preds, data):
        """
        Phase 2 loss:
          total = enr_lambda * loss_E
                + frc_lambda * loss_F
                + chg_lambda * loss_Q   (CEP q_i vs NPA charges)

        Charge conservation loss removed since CEP guarantees it via hard constraint.
        """
        # Base energy / force / stress loss
        loss = super().compute_loss(preds, data)

        charge_config = self.json_data.get('charge', {})
        q_lambda = self.json_data['NN'].get('chg_lambda', 1.0)

        # Atomic charge supervision (NPA charges etc.)
        if ("atomic_charges" in preds and "atomic_charges" in data):
            charge_target = data["atomic_charges"].flatten()
            charge_pred   = preds["atomic_charges"].flatten()

            loss_fn_key = charge_config.get("charge_loss", "mse")
            if loss_fn_key in self.loss_fn:
                loss_q = self.loss_fn[loss_fn_key](charge_pred, charge_target)
            else:
                loss_q = torch.nn.functional.mse_loss(
                    charge_pred, charge_target
                )

            loss["loss_q"] = loss_q
            loss["loss"]   = loss["loss"] + q_lambda * loss_q

        return loss

    def train_one_epoch(self, mode='train', data_loader=None):
        """CPU compatibility: bypass cuda.synchronize as no-op."""
        if self.device == 'cpu' or str(self.device) == 'cpu':
            _orig_sync = torch.cuda.synchronize
            torch.cuda.synchronize = lambda *a, **kw: None
            try:
                return super().train_one_epoch(mode, data_loader)
            finally:
                torch.cuda.synchronize = _orig_sync
        else:
            return super().train_one_epoch(mode, data_loader)
