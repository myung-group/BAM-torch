"""Build a DataLoader that carries teacher predictions alongside DFT labels.

The student dataset is the standard ``bam_torch`` graphset (DFT energy/force
labels live in ``data.energy`` and ``data.forces``) augmented with two extra
attributes per graph:

  * ``data.teacher_energy`` — Tensor[1], raw teacher residual prediction.
  * ``data.teacher_forces`` — Tensor[n_atoms, 3], teacher per-atom forces.

``teacher_forces`` is registered as node-level via ``__cat_dim__`` override so
``torch_geometric`` concatenates it along dim 0 (like ``data.forces``) when
batches are collated.

The teacher's ``enr_avg_per_element`` / ``uniq_element`` are passed in
explicitly — we never recompute baselines from the subset, otherwise the DFT
residual labels stored in ``data.energy`` would land on a different reference
frame from the teacher predictions and the soft-loss term would be biased.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch
from ase.io import read
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from bam_torch.utils.utils import get_graphset


class DistillData(Data):
    """Custom Data so torch_geometric treats ``teacher_forces`` as node-level."""

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "teacher_forces":
            return 0  # concat along node axis when batching
        return super().__cat_dim__(key, value, *args, **kwargs)


def _to_distill_data(g: Data, teacher_e: torch.Tensor, teacher_f: torch.Tensor) -> DistillData:
    out = DistillData()
    for k, v in g:
        out[k] = v
    out.num_nodes = g.num_nodes
    out.teacher_energy = teacher_e.view(1).to(torch.float32)
    out.teacher_forces = teacher_f.to(torch.float32)
    return out


def get_distill_dataloader(
    traj_path: str,
    teacher_pt_path: str,
    nbatch: int,
    cutoff: float,
    uniq_element: dict,
    enr_avg_per_element: dict,
    *,
    regress_forces: bool = True,
    max_neigh: Optional[int] = None,
    shuffle: bool = False,
):
    """Ragged-batch DataLoader; RACE handles variable-sized batches natively
    via ``data["ptr"]`` / ``data["batch"]``, so no padding is needed.
    """
    traj = read(traj_path, index=":")
    teacher = torch.load(teacher_pt_path, map_location="cpu", weights_only=False)

    if len(traj) != len(teacher["forces"]):
        raise ValueError(
            f"traj has {len(traj)} frames but teacher pt has {len(teacher['forces'])}; "
            "they must align by index."
        )

    graphset = get_graphset(
        traj,
        cutoff,
        uniq_element,
        enr_avg_per_element,
        enr_var=None,
        regress_forces=regress_forces,
        max_neigh=max_neigh,
        show_progress=False,
    )

    distill_graphs: list[DistillData] = []
    for i, g in enumerate(graphset):
        n_g = int(g.num_nodes)
        n_t = int(teacher["forces"][i].shape[0])
        if n_g != n_t:
            raise ValueError(f"frame {i}: graph has {n_g} atoms but teacher_forces has {n_t}")
        distill_graphs.append(_to_distill_data(g, teacher["energy"][i], teacher["forces"][i]))

    loader = DataLoader(
        distill_graphs,
        batch_size=nbatch,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=True,
        num_workers=0,
    )
    return loader


def teacher_baselines_from_ckpt(ckpt_path: str) -> tuple[dict, dict]:
    """Read ``uniq_element`` and ``enr_avg_per_element`` directly from the teacher ckpt."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return deepcopy(ckpt["uniq_element"]), deepcopy(ckpt["enr_avg_per_element"])
