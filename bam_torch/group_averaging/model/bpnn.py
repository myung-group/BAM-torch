import torch
import torch.nn as nn
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import numpy as np
from bam_torch.utils.output_utils import (
    get_outputs, 
    get_symmetric_displacement,
    remove_net_torque
)

class BPNNModel(nn.Module):
    def __init__(self, sf_config, uniq_element=None, r_cutoff=6.0, hidden=64):
        """
        sf_config   : dict[ei][ej] -> list of symmetry functions
        uniq_element: dict[atomic_number -> element_index]
        """
        super().__init__()

        self.sf_config = sf_config
        self.uniq_element = uniq_element          # {Z: ei}
        self.inv_uniq = {v: k for k, v in uniq_element.items()}
        self.num_elem = len(uniq_element)
        self.r_cutoff = float(r_cutoff)

        # ---------- element-wise input dimension ----------
        input_sizes = {}
        for ei in range(self.num_elem):
            input_sizes[ei] = sum(len(v) for v in self.sf_config[ei].values())
            print(f"input_sizes: {input_sizes}")

        # ---------- element-wise networks ----------
        self.nets = nn.ModuleDict({
            str(ei): SpeciesMLP(input_sizes[ei], hidden=hidden)
            for ei in range(self.num_elem)
        })

    # --------------------------------------------------
    def cutoff(self, Rij):
        return 0.5 * (torch.cos(np.pi * Rij / self.r_cutoff) + 1.0)

    def forward(self, data):
        """
        data.pos: (sum_N,3)
        data.batch: (sum_N,)
        data.ptr: graph ptr
        """
        device = data.pos.device
        energies = []
        data["cell"].requires_grad_(True)
        data["pos"].requires_grad_(True)
        B, _, _ = data.cell.shape

        for g in range(data.ptr.numel() - 1):
            start, end = data.ptr[g], data.ptr[g+1]

            pos = data.pos[start:end]
            Z = data.species[start:end]
            cell = data.cell[g]
            pbc = torch.tensor([True, True, True], device=device)

            E = self.forward_one(pos, Z, cell, pbc)
            energies.append(E)

        energy = torch.stack(energies)
        E_per_atom = (energy / data.pos.shape[0]).unsqueeze(1).expand(B, data.pos.shape[0])
        num_graphs = data["ptr"].numel() - 1  # nbatch

        # forces, stress 등은 get_outputs 그대로 사용
        forces, virials, stress, hessian = get_outputs(
                energy=energy,
                positions=data["positions"],
                cell=data["cell"],
                batch_idx=data["batch"],
                num_graphs=num_graphs,
                training=True,
                compute_force=True,
                compute_virials=True,
                compute_stress=True,
                compute_hessian=False,
                displacement=None
            )

        return {
            "energy": energy,
            "forces": forces,
            "stress": stress,
            "virials": virials,
            "node_energy": E_per_atom
        }



    def forward_one(self, pos, atomic_numbers, cell, pbc):
        """
        pos: (N,3)
        atomic_numbers: (N,)
        cell: (3,3)
        return: energy (scalar)
        """
        device = pos.device
        N = pos.shape[0]

        dif = difference_matrix_pbc_cart(
            pos.unsqueeze(0), cell.unsqueeze(0), pbc
        )[0]                     # (N,N,3)

        dis = torch.norm(dif, dim=-1)     # (N,N)
        fc = self.cutoff(dis)
        fc.fill_diagonal_(0.0)

        dot = torch.einsum("ijn,ikn->ijk", dif, dif)
        denom = dis.unsqueeze(2) * dis.unsqueeze(1)
        cos = torch.zeros_like(dot)
        mask = denom > 0
        cos[mask] = dot[mask] / denom[mask]

        total_E = torch.tensor(0.0, device=device)

        for i in range(N):
            Zi = atomic_numbers[i].item()
            if Zi not in self.uniq_element:
                continue

            ei = self.uniq_element[Zi]
            sym_vals = []

            for Zj, ej in self.uniq_element.items():
                idx = (atomic_numbers == Zj).nonzero(as_tuple=True)[0]

                sfuncs = self.sf_config[ei][ej]
                if idx.numel() == 0:
                    for _ in sfuncs:
                        sym_vals.append(torch.zeros((1,), device=device))
                    continue

                Rij = dis[i, idx]
                fc_ij = fc[i, idx]
                cos_ijk = cos[i][idx][:, idx]

                for sf in sfuncs:
                    val = sf(Rij=Rij.unsqueeze(0),
                            fc=fc_ij.unsqueeze(0),
                            cos=cos_ijk.unsqueeze(0))
                    sym_vals.append(val)

            x = torch.cat(sym_vals).unsqueeze(0)
            Ei = self.nets[str(ei)](x).squeeze()
            total_E = total_E + Ei

        return total_E


def difference_matrix_pbc_cart(
    pos_cart: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """
    pos_cart: (B, N, 3)
    cell:     (3,3) or (B,3,3)
    pbc:      (3,)
    return:   (B, N, N, 3)  r_j - r_i with MIC
    """
    B, N, _ = pos_cart.shape
    device = pos_cart.device

    # ensure batched cell
    if cell.dim() == 2:
        cell = cell.unsqueeze(0).expand(B, 3, 3)

    # r_j - r_i
    dif_cart = pos_cart.unsqueeze(2) - pos_cart.unsqueeze(1)  # (B,N,N,3)

    # cart -> fractional
    cell_inv = torch.linalg.inv(cell)                          # (B,3,3)
    dif_frac = torch.einsum(
        "bijn,bnm->bijm", dif_cart, cell_inv
    )                                                          # (B,N,N,3)

    # minimum image
    for a in range(3):
        if pbc[a]:
            dif_frac[..., a] -= torch.round(dif_frac[..., a])

    # fractional -> cart
    dif_cart_mic = torch.einsum(
        "bijn,bnm->bijm", dif_frac, cell
    )

    return dif_cart_mic


def energy_and_forces(model: BPNNModel, atoms_batch, device="cpu"):
    """
    atoms_batch: list[ase.Atoms] 
    return: E_pred (B,), F_pred (B,N,3)
    """
    from ase import Atoms

    N = len(atoms_batch[0])
    for a in atoms_batch:
        assert len(a) == N

    B = len(atoms_batch)
    atom_symbols = [a.symbol for a in atoms_batch[0]]

    cell = torch.tensor(atoms_batch[0].get_cell().array, dtype=torch.float32, device=device)
    pbc = torch.tensor(atoms_batch[0].pbc, dtype=torch.bool, device=device)

    pos = torch.stack([
        torch.tensor(a.get_positions(), dtype=torch.float32)
        for a in atoms_batch
    ], dim=0).to(device)
    pos.requires_grad_(True)

    E = model.forward_energy(pos, atom_symbols, cell, pbc)  # (B,)

    # forces = - dE/dR
    grad_pos = torch.autograd.grad(
        outputs=E.sum(),
        inputs=pos,
        create_graph=True,
        retain_graph=True
    )[0]
    F = -grad_pos  # (B,N,3)

    return E, F


def get_edge_relative_vectors_with_pbc(data: Dict[str, torch.Tensor]):
    # iatoms ==> senders
    # jatoms ==> receivers
    R = data["positions"]
    cell = data["cell"]
    iatoms = data["edge_index"][0]  # shape = (b * n_edges)
    jatoms = data["edge_index"][1]  # shape = (b * n_edges) 
    Sij = data["edges"]   # shape = (b * n_edges, 3)
    n_edges: List[int] = data["num_edges"].tolist()
    
    Sij = torch.split(Sij, n_edges, dim=0)
    shift_v = torch.cat(
        [torch.einsum('ni,ij->nj', s, c)
            for s, c in zip(Sij, cell)], dim=0
    )
    _R = R[jatoms] - R[iatoms] 
    Rij = _R + shift_v

    return Rij # (num_edges, 3)


import torch.nn as nn

class SpeciesMLP_(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )
        # Xavier init (TF xavier_initializer)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)  # (B,1)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))

class SpeciesMLP(nn.Module):
    def __init__(self, input_dim, hidden=64, n_blocks=3):
        super().__init__()
        self.inp = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden) for _ in range(n_blocks)]
        )
        self.out = nn.Linear(hidden, 1)
        self.act = nn.SiLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.act(self.inp(x))
        for blk in self.blocks:
            x = blk(x)
        return self.out(x)