"""DistillTrainer — hybrid DFT + teacher distillation built on BaseTrainer.

Loss formula::

    L = lambda_dft       * (e_lambda * loss_e_dft  + f_lambda * loss_f_dft)
      + (1 - lambda_dft) * (e_lambda * loss_e_t    + f_lambda * loss_f_t)

Per-atom energy normalization (``pred_e/n_atoms`` vs ``target_e/n_atoms``) is
applied for ``huber`` and ``mse`` losses to keep the loss scale comparable
across cells of very different size — this matches ``MPTrainer`` behaviour.
"""
from __future__ import annotations

import torch

from bam_torch.training.base_trainer import BaseTrainer
from bam_torch.training.loss import RMSELoss, l2_regularization

from .dataset import get_distill_dataloader, teacher_baselines_from_ckpt


class DistillTrainer(BaseTrainer):
    def __init__(self, json_data, rank=0, world_size=1):
        # Load teacher baselines BEFORE super().__init__ since the parent
        # calls setup() -> configure_dataloader() which needs them.
        distill_cfg = json_data["NN"]["distill"]
        self._teacher_uniq_element, self._teacher_enr_avg = teacher_baselines_from_ckpt(
            distill_cfg["teacher_ckpt"]
        )
        self._teacher_pt_train = distill_cfg["teacher_pt_train"]
        self._teacher_pt_valid = distill_cfg["teacher_pt_valid"]
        super().__init__(json_data, rank, world_size)

    def configure_dataloader(self):
        cutoff = self.json_data["cutoff"]
        nbatch = self.json_data["nbatch"]
        regress_forces = self.json_data.get("regress_forces", True)
        max_neigh = self.json_data.get("max_neigh")

        train_loader = get_distill_dataloader(
            self.json_data["fname_traj"],
            self._teacher_pt_train,
            nbatch=nbatch,
            cutoff=cutoff,
            uniq_element=self._teacher_uniq_element,
            enr_avg_per_element=self._teacher_enr_avg,
            regress_forces=regress_forces,
            max_neigh=max_neigh,
            shuffle=True,
        )
        valid_loader = get_distill_dataloader(
            self.json_data["nvalid"],
            self._teacher_pt_valid,
            nbatch=nbatch,
            cutoff=cutoff,
            uniq_element=self._teacher_uniq_element,
            enr_avg_per_element=self._teacher_enr_avg,
            regress_forces=regress_forces,
            max_neigh=max_neigh,
            shuffle=False,
        )

        if self.rank == 0:
            print(f"[distill] train batches: {len(train_loader)}  valid batches: {len(valid_loader)}")
            print(f"[distill] using teacher's enr_avg_per_element ({len(self._teacher_enr_avg)} species)")
        return train_loader, valid_loader, self._teacher_uniq_element, self._teacher_enr_avg

    def configure_loss(self, reduction="mean"):
        nn_config = self.json_data["NN"]
        loss_config = nn_config.get("loss_config", {"energy_loss": "mse", "force_loss": "mse"})
        huber_delta = loss_config.get("huber_delta", 0.1)

        loss_fn: dict = {}
        for key, default in (("energy_loss", "mse"), ("force_loss", "mse")):
            name = (loss_config.get(key) or default).lower()
            if name in ("l1", "mae"):
                loss_fn[key] = torch.nn.L1Loss(reduction=reduction)
            elif name == "mse":
                loss_fn[key] = torch.nn.MSELoss(reduction=reduction)
            elif name == "rmse":
                loss_fn[key] = RMSELoss(reduction=reduction)
            elif name in ("huber", "h"):
                # SmoothL1 with beta=delta is equivalent to Huber with delta.
                loss_fn[key] = torch.nn.SmoothL1Loss(reduction=reduction, beta=huber_delta)
            else:
                raise ValueError(f"unknown loss '{name}' for {key}")
        return loss_fn, loss_config

    def compute_loss(self, preds, data):
        cfg = self.json_data["NN"]
        e_lambda = cfg.get("enr_lambda", 1)
        f_lambda = cfg.get("frc_lambda", 10)
        l2_lambda = cfg.get("l2_lambda", 0.0)
        lambda_dft = cfg["distill"].get("lambda_dft", 0.5)
        lambda_t = 1.0 - lambda_dft

        # Per-graph num_atoms from PyG ptr.
        ptr = data["ptr"]
        n_atoms_per_graph = (ptr[1:] - ptr[:-1]).to(torch.float32)

        # ---- DFT (hard) loss ---------------------------------------------------
        e_pred = preds["energy"].flatten()
        e_dft = data["energy"].flatten()
        loss_e_dft = self.loss_fn["energy_loss"](e_pred / n_atoms_per_graph,
                                                 e_dft / n_atoms_per_graph)
        loss_f_dft = self.loss_fn["force_loss"](preds["forces"].flatten(),
                                                data["forces"].flatten())
        l_dft = e_lambda * loss_e_dft + f_lambda * loss_f_dft

        # ---- Teacher (soft) loss ----------------------------------------------
        # `data["teacher_energy"]` is the raw teacher residual (no scale_shift).
        # The student's `preds["energy"]` *has* been scale-shifted by
        # BaseTrainer.scale_shift to centre on data["energy"].mean(); we apply
        # the same per-batch shift to the teacher target so both terms are
        # measured against a consistently-centred reference.
        teacher_e_raw = data["teacher_energy"].flatten()
        with torch.no_grad():
            teacher_shift = e_dft.mean() - teacher_e_raw.mean()
        teacher_e = teacher_e_raw + teacher_shift
        teacher_f = data["teacher_forces"]

        loss_e_t = self.loss_fn["energy_loss"](e_pred / n_atoms_per_graph,
                                               teacher_e / n_atoms_per_graph)
        loss_f_t = self.loss_fn["force_loss"](preds["forces"].flatten(),
                                              teacher_f.flatten())
        l_teacher = e_lambda * loss_e_t + f_lambda * loss_f_t

        # ---- Combined ---------------------------------------------------------
        total = lambda_dft * l_dft + lambda_t * l_teacher

        out = {
            "loss": total,
            "loss_e": loss_e_dft,
            "loss_f": loss_f_dft,
            "loss_e_t": loss_e_t,
            "loss_f_t": loss_f_t,
        }
        if l2_lambda != 0:
            out["loss_l2"] = l2_regularization(self.model.parameters())
            out["loss"] = out["loss"] + l2_lambda * out["loss_l2"]
        return out
