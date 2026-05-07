"""
Trainer for paper-faithful DPLR / DPSR / Deep Wannier models.

Supports three training phases:
  - "deep_wannier": Train DW model on atomic dipole data
  - "dplr": Train DPLR model with frozen DW (energy + force loss)
  - "dpsr": Train DPSR model (energy + force loss, no DW)

Extends GATrainer for group averaging (frame averaging) support.
"""

import torch
import gc
from pathlib import Path
from torch_geometric.data import Batch as DataBatch

from bam_torch.group_averaging.training.ga_trainer import GATrainer
from bam_torch.group_averaging.model import MODEL_REGISTRY
from bam_torch.utils.utils import get_dataloader_for_dipole


class DPLRTrainer(GATrainer):
    """Trainer for DPLR paper reproduction.

    Usage in input.json:
        "trainer": "dplr_paper"
        "training_phase": "deep_wannier" | "dplr" | "dpsr"
    """

    def __init__(self, json_data, rank, world_size):
        self.training_phase = json_data.get("training_phase", "dpsr")
        super().__init__(json_data, rank, world_size)

    def train_one_epoch(self, mode='train', data_loader=None):
        """Override for DW dipole training (no frame averaging needed)."""
        if self.training_phase != "deep_wannier":
            return super().train_one_epoch(mode, data_loader)

        # Simple forward loop for DW dipole training
        if mode == 'train':
            self.model.train()
            backprop = True
            loss_log_config = self.log_config['train']
            if data_loader is None:
                data_loader = self.train_loader
        else:
            self.model.eval()
            backprop = False
            loss_log_config = self.log_config['valid']
            if data_loader is None:
                data_loader = self.valid_loader

        epoch_loss_dict = {key: [] for key in loss_log_config}
        for i, data in enumerate(data_loader):
            data = self.move_to_device(data, self.device)
            # Direct forward (no frame averaging)
            data.pos = data.positions
            preds = self.model(data, mode=mode)

            # Convert data to dict for compute_loss
            data_dict = data.to_dict() if isinstance(data, DataBatch) else data
            loss_dict = self.compute_loss(preds, data_dict)

            for l in loss_log_config:
                val = loss_dict.get(l, torch.nan)
                epoch_loss_dict[l].append(
                    val.detach().cpu() if isinstance(val, torch.Tensor) else val
                )

            if backprop:
                self.optimizer.zero_grad()
                loss_dict['loss'].backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)
                self.optimizer.step()

                if self.ema is not None:
                    self.ema.update()

        torch.cuda.synchronize()
        epoch_loss_dict = {
            key: torch.mean(torch.tensor(value).detach().cpu())
            for key, value in epoch_loss_dict.items()
        }
        torch.cuda.empty_cache()
        gc.collect()
        return epoch_loss_dict

    def configure_dataloader(self):
        if self.training_phase == "deep_wannier":
            return self._configure_dipole_dataloader()
        else:
            return super().configure_dataloader()

    def _configure_dipole_dataloader(self):
        """Load DeePMD raw data with atomic_dipole for DW training."""
        cfg = self.json_data
        data_dirs = cfg.get("data_dirs", [])
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        # Resolve relative paths
        data_dirs = [str(Path(d).expanduser()) for d in data_dirs]

        sel_type = cfg.get("sel_type", [2, 3, 4])
        cutoff = cfg.get("cutoff", 6.0)
        nbatch = cfg.get("nbatch", 1)
        seed = cfg["NN"].get("data_seed", 42)

        train_loader, valid_loader, type_map = get_dataloader_for_dipole(
            data_dirs=data_dirs,
            cutoff=cutoff,
            sel_type=sel_type,
            nbatch=nbatch,
            train_ratio=0.9,
            seed=seed,
            rank=self.rank,
            world_size=self.world_size,
        )
        # Return tuple matching BaseTrainer.setup() expectations
        uniq_element = {}
        enr_avg_per_element = {}
        return train_loader, valid_loader, uniq_element, enr_avg_per_element

    def set_model(self):
        cfg = self.json_data
        model_name = cfg["model"].lower()

        # Paper models don't use the probabilistic symmetrization equiv_model
        self.equiv_model = None

        if model_name == "deep_wannier":
            return self._build_deep_wannier()
        elif model_name == "dplr_paper":
            return self._build_dplr_paper()
        elif model_name == "dpsr_paper":
            return self._build_dpsr_paper()
        else:
            return super().set_model()

    def _common_descriptor_args(self):
        """Extract common descriptor/model args from config."""
        cfg = self.json_data
        return dict(
            cutoff=cfg.get("cutoff", 6.0),
            rcut_smth=cfg.get("rcut_smth", 3.0),
            num_species=cfg.get("num_species", 5),
            sel=cfg.get("sel", [38, 120, 75, 14, 14]),
            embedding_dim=cfg.get("embedding_dim", 32),
            descriptor_hidden=cfg.get("descriptor_hidden", [25, 50, 100]),
            max_num_neighbors=cfg.get("max_neigh", 300),
            preprocess=cfg.get("preprocess", "pbc_preprocess"),
        )

    def _build_deep_wannier(self):
        from bam_torch.group_averaging.model.dplr_paper import DeepWannierModel

        cfg = self.json_data
        args = self._common_descriptor_args()

        model = DeepWannierModel(
            **args,
            axis_neurons=cfg.get("descriptor_axis_neurons", 8),
            fitting_hidden=cfg.get("fitting_hidden", [100, 100, 100]),
            sel_type=cfg.get("sel_type", [2, 3, 4]),
            regress_forces=cfg.get("regress_forces", "auto"),
        )
        self.model = model.to(self.device)
        return model

    def _build_dpsr_paper(self):
        from bam_torch.group_averaging.model.dplr_paper import DPSRPaper

        cfg = self.json_data
        args = self._common_descriptor_args()

        regress_forces = cfg.get("regress_forces", "auto")
        if regress_forces is True:
            regress_forces = "autograd"

        model = DPSRPaper(
            **args,
            axis_neurons=cfg.get("descriptor_axis_neurons", 16),
            fitting_hidden=cfg.get("fitting_hidden", [100, 100, 100]),
            regress_forces=regress_forces,
        )
        self.model = model.to(self.device)
        return model

    def _build_dplr_paper(self):
        from bam_torch.group_averaging.model.dplr_paper import DPLRPaper, DeepWannierModel

        cfg = self.json_data
        args = self._common_descriptor_args()

        regress_forces = cfg.get("regress_forces", "auto")
        if regress_forces is True:
            regress_forces = "autograd"

        # Load frozen DW model
        dw_checkpoint = cfg.get("dw_checkpoint")
        dw_model = None
        if dw_checkpoint:
            dw_state = torch.load(dw_checkpoint, map_location=self.device)
            # Build DW model with same architecture
            dw_model = DeepWannierModel(
                **args,
                axis_neurons=cfg.get("dw_axis_neurons", 8),
                fitting_hidden=cfg.get("dw_fitting_hidden",
                                       cfg.get("fitting_hidden", [100, 100, 100])),
                sel_type=cfg.get("sel_type", [2, 3, 4]),
            )
            if "params" in dw_state:
                dw_model.load_state_dict(dw_state["params"])
            elif "model_state_dict" in dw_state:
                dw_model.load_state_dict(dw_state["model_state_dict"])
            else:
                dw_model.load_state_dict(dw_state)
            dw_model = dw_model.to(self.device)

        model = DPLRPaper(
            **args,
            axis_neurons=cfg.get("descriptor_axis_neurons", 16),
            fitting_hidden=cfg.get("fitting_hidden", [100, 100, 100]),
            regress_forces=regress_forces,
            dw_model=dw_model,
            sys_charge_map=cfg.get("sys_charge_map", [4, 1, 6, 9, 7]),
            model_charge_map=cfg.get("model_charge_map", [-8, -8, -8]),
            sel_type=cfg.get("sel_type", [2, 3, 4]),
            ewald_alpha=cfg.get("ewald_beta", 0.1),
            ewald_accuracy=cfg.get("ewald_accuracy", 1e-6),
        )
        self.model = model.to(self.device)
        return model

    def configure_loss(self, reduction='mean'):
        if self.training_phase == "deep_wannier":
            loss_fn = {"dipole_loss": torch.nn.MSELoss(reduction=reduction)}
            return loss_fn, {}
        return super().configure_loss(reduction)

    def compute_loss(self, preds, data):
        if self.training_phase == "deep_wannier":
            return self._compute_dipole_loss(preds, data)
        return super().compute_loss(preds, data)

    def _compute_dipole_loss(self, preds, data):
        """Compute dipole RMSE loss for DW training."""
        dipole_pred = preds["dipole"]           # [N_sel, 3]
        dipole_target = data["atomic_dipole"]   # [N_sel, 3] (from DataBatch)

        # Flatten for MSE
        loss_d = self.loss_fn["dipole_loss"](
            dipole_pred.flatten(), dipole_target.flatten()
        )

        return {
            "loss": loss_d,
            "loss_d": loss_d,
        }

    def scale_shift(self, preds, data, mode):
        """Skip scale-shift for DW dipole training."""
        if self.training_phase == "deep_wannier":
            return preds
        return super().scale_shift(preds, data, mode)
