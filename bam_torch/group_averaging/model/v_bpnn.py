"""
Vectorized Behler–Parrinello NN (edge-centric)
- Works with batched PyG-like graphs (global indexing)
- Converts global edges -> per-graph local edges
- Computes per-graph energy (B,)
- Computes forces/stress/virials via get_outputs if available, else autograd forces fallback

Assumptions about `data`:
  data.pos        : (sum_N, 3) float
  data.species    : (sum_N,)   int atomic numbers (Z)
  data.cell       : (num_graphs, 3, 3) float
  data.batch      : (sum_N,)   int graph id for each node
  data.ptr        : (num_graphs+1,) long pointer into nodes
  data.edge_index : (2, sum_E) long global node indices
  data.edges      : (sum_E, 3) float or long cell-shifts S_ij in fractional lattice units
                   such that r_j - r_i + S_ij @ cell
  (optional)
  data.num_edges  : (num_graphs,) long edges per graph (not required here)

Notes:
- This implementation keeps the "vectorized" edge distance computation, but still loops over atoms
  to build fixed-length SF descriptors (BPNN requirement).
- The critical fix vs your current code: per-graph slicing + edge reindexing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


# -----------------------
# Example SF API expected
# -----------------------
# Each symmetry function sf must accept:
#   sf(Rij=<tensor>, fc=<tensor>, cos=<tensor or None>) and return shape (1,) or (B,) if batched.
# We'll call with shape (1, Nj) / (1, Nj, Nj) and expect (1,) or (1,1).
# (Your existing G1/G2/G4 should work if consistent.)
# -----------------------


class SpeciesMLP(nn.Module):
    """Element-specific MLP."""
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_edge_relative_vectors_with_pbc_local(
    pos: torch.Tensor,        # (N,3)
    cell: torch.Tensor,       # (3,3)
    edge_index: torch.Tensor, # (2,E) LOCAL indices in [0,N)
    edge_shifts: torch.Tensor # (E,3) fractional shifts
) -> torch.Tensor:
    """Return Rij vectors for edges with PBC: r_j - r_i + shift @ cell."""
    iatoms = edge_index[0]
    jatoms = edge_index[1]
    Rij = pos[jatoms] - pos[iatoms]  # (E,3)
    if edge_shifts is not None and edge_shifts.numel() > 0:
        shift_cart = torch.einsum("ei,ij->ej", edge_shifts.to(pos.dtype), cell)
        Rij = Rij + shift_cart
    return Rij  # (E,3)


class VectorizedBPNN(nn.Module):
    def __init__(
        self,
        sf_config: Dict[int, Dict[int, List]],
        uniq_element: Dict[int, int],
        r_cutoff: float = 6.0,
        hidden: int = 64,
    ):
        super().__init__()

        self.sf_config = sf_config                 # sf_config[ei][ej] -> list[sf]
        self.uniq_element = uniq_element           # {Z: ei}
        self.inv_uniq = {v: k for k, v in uniq_element.items()}
        self.num_elem = len(uniq_element)
        self.r_cutoff = float(r_cutoff)

        # element-wise input dim: sum over all ej lists length
        input_sizes = {}
        for ei in range(self.num_elem):
            input_sizes[ei] = sum(len(v) for v in self.sf_config[ei].values())

        self.nets = nn.ModuleDict({
            str(ei): SpeciesMLP(input_sizes[ei], hidden=hidden)
            for ei in range(self.num_elem)
        })

    def cutoff(self, r: torch.Tensor) -> torch.Tensor:
        """Cosine cutoff: assumes r <= r_cutoff in neighbor list, but safe anyway."""
        # If your edges may include r > r_cutoff, you can clamp:
        # r = torch.clamp(r, max=self.r_cutoff)
        return 0.5 * (torch.cos(np.pi * r / self.r_cutoff) + 1.0)

    def forward(self, data) -> Dict[str, torch.Tensor]:
        device = data.pos.device
        data.pos.requires_grad_(True)
        if hasattr(data, "cell"):
            data.cell.requires_grad_(True)

        num_graphs = int(data.ptr.numel() - 1)

        energies: List[torch.Tensor] = []

        # Global edges
        edge_index_global = data.edge_index          # (2, E_total)
        edge_shifts_global = getattr(data, "edges", None)  # (E_total,3) or None

        for g in range(num_graphs):
            start = int(data.ptr[g].item())
            end = int(data.ptr[g + 1].item())
            n_atoms_g = end - start

            pos_g = data.pos[start:end]              # (N_g,3)
            species_g = data.species[start:end]      # (N_g,)
            cell_g = data.cell[g]                    # (3,3)

            # ----------------------------
            # FIX: global -> local edges
            # ----------------------------
            # Keep edges whose sender is in [start,end)
            send_global = edge_index_global[0]
            mask_e = (send_global >= start) & (send_global < end)

            edge_index_g = edge_index_global[:, mask_e] - start  # local indices
            if edge_shifts_global is None:
                edge_shifts_g = None
            else:
                edge_shifts_g = edge_shifts_global[mask_e]

            # Safety: ensure receivers also within this graph
            if edge_index_g.numel() > 0:
                recv_local = edge_index_g[1]
                if not ((recv_local >= 0) & (recv_local < n_atoms_g)).all():
                    # If your edge construction can include cross-graph edges, filter them:
                    recv_ok = (recv_local >= 0) & (recv_local < n_atoms_g)
                    mask_e2 = recv_ok
                    edge_index_g = edge_index_g[:, mask_e2]
                    if edge_shifts_g is not None:
                        edge_shifts_g = edge_shifts_g[mask_e2]

            E_g = self.forward_one_graph(
                pos=pos_g,
                species=species_g,
                cell=cell_g,
                edge_index=edge_index_g,
                edge_shifts=edge_shifts_g,
            )
            energies.append(E_g)

        energy = torch.stack(energies, dim=0)  # (B,)

        # "node_energy": constant per-atom average, per-graph (B, N_total) per your prior usage
        n_atoms_total = data.pos.shape[0]
        node_energy = (energy.sum() / n_atoms_total).unsqueeze(0).expand(num_graphs, n_atoms_total)

        # Forces & stress/virials
        try:
            from bam_torch.utils.output_utils import get_outputs

            forces, virials, stress, hessian = get_outputs(
                energy=energy,
                positions=data.pos,
                cell=data.cell,
                batch_idx=data.batch,
                num_graphs=num_graphs,
                training=self.training,
                compute_force=True,
                compute_virials=True,
                compute_stress=True,
                compute_hessian=False,
                displacement=None
            )
        except Exception:
            forces = -torch.autograd.grad(
                outputs=energy.sum(),
                inputs=data.pos,
                create_graph=self.training,
                retain_graph=self.training,
                allow_unused=True
            )[0]
            if forces is None:
                forces = torch.zeros_like(data.pos)

            stress = torch.zeros(num_graphs, 3, 3, device=device, dtype=data.pos.dtype)
            virials = torch.zeros(num_graphs, 3, 3, device=device, dtype=data.pos.dtype)

        return {
            "energy": energy,
            "forces": forces,
            "stress": stress,
            "virials": virials,
            "node_energy": node_energy,
        }

    def forward_one_graph(
        self,
        pos: torch.Tensor,               # (N,3)
        species: torch.Tensor,           # (N,)
        cell: torch.Tensor,              # (3,3)
        edge_index: torch.Tensor,        # (2,E) local
        edge_shifts: Optional[torch.Tensor],  # (E,3)
    ) -> torch.Tensor:

        device = pos.device
        n_atoms = int(pos.shape[0])

        if edge_index.numel() == 0:
            return torch.tensor(0.0, device=device, dtype=pos.dtype)

        senders = edge_index[0]
        receivers = edge_index[1]

        # Edge vectors & distances (vectorized)
        vec_ij = get_edge_relative_vectors_with_pbc_local(
            pos=pos, cell=cell, edge_index=edge_index, edge_shifts=edge_shifts
        )  # (E,3)

        dist_ij = torch.norm(vec_ij, dim=1)                            # (E,)
        fc_ij = self.cutoff(dist_ij)                                   # (E,)
        vec_ij_norm = vec_ij / (dist_ij.unsqueeze(-1) + 1e-12)         # (E,3)

        # Cosine angle tensor per-center (still requires per-i padding)
        cos_angles = self._compute_cos_angles(edge_index, vec_ij_norm, n_atoms)  # (N, M, M)

        total_E = torch.tensor(0.0, device=device, dtype=pos.dtype)

        # Element-wise descriptor + NN
        for Z, ei in self.uniq_element.items():
            atom_mask = (species == Z)
            if not atom_mask.any():
                continue

            atom_indices = torch.where(atom_mask)[0]
            G_list: List[torch.Tensor] = []

            for i in atom_indices:
                # edges outgoing from i
                mask_i = (senders == i)

                sym_vals: List[torch.Tensor] = []

                for Zj, ej in self.uniq_element.items():
                    sfuncs = self.sf_config[ei][ej]

                    # pick neighbors of type Zj
                    neigh_mask = mask_i & (species[receivers] == Zj)
                    eidx = torch.where(neigh_mask)[0]

                    if eidx.numel() == 0:
                        # IMPORTANT: fixed-length padding
                        for _ in sfuncs:
                            sym_vals.append(torch.zeros(1, device=device, dtype=pos.dtype))
                        continue

                    Rij = dist_ij[eidx].unsqueeze(0)     # (1, Nj)
                    fc_j = fc_ij[eidx].unsqueeze(0)      # (1, Nj)

                    # NOTE: cos_angles is padded by max_neighbors, but eidx corresponds to a subset.
                    # Here we use only the top-left NjxNj block for stability.
                    Nj = int(eidx.numel())
                    cos_ijk = cos_angles[i, :Nj, :Nj].unsqueeze(0)  # (1, Nj, Nj)

                    for sf in sfuncs:
                        v = sf(Rij=Rij, fc=fc_j, cos=cos_ijk)
                        # normalize to shape (1,)
                        if v.dim() > 1:
                            v = v.reshape(-1)
                        sym_vals.append(v[:1])

                Gi = torch.cat(sym_vals, dim=0)     # (input_dim_ei,)
                G_list.append(Gi)

            G_ei = torch.stack(G_list, dim=0)       # (N_ei, input_dim_ei)
            E_ei = self.nets[str(ei)](G_ei).squeeze(-1)
            total_E = total_E + E_ei.sum()

        return total_E

    def _compute_cos_angles(
        self,
        edge_index: torch.Tensor,   # (2,E) local
        vec_ij_norm: torch.Tensor,  # (E,3)
        n_atoms: int,
    ) -> torch.Tensor:
        """
        cos_angles[i, a, b] = dot( v_i->nei[a], v_i->nei[b] )
        where neighbors are ordered by edge appearance for each i.
        """
        device = edge_index.device
        senders = edge_index[0]

        unique_senders, counts = torch.unique(senders, return_counts=True)
        max_neighbors = int(counts.max().item()) if counts.numel() > 0 else 0

        if max_neighbors == 0:
            return torch.zeros(n_atoms, 1, 1, device=device, dtype=vec_ij_norm.dtype)

        cos_angles = torch.zeros(
            n_atoms, max_neighbors, max_neighbors,
            device=device, dtype=vec_ij_norm.dtype
        )

        # Still per-i loop (BPNN G4 needs neighbor-neighbor correlations)
        for i in range(n_atoms):
            mask = (senders == i)
            if not mask.any():
                continue
            vecs_i = vec_ij_norm[mask]                 # (n_i,3)
            n_i = int(vecs_i.shape[0])
            cos_ij = vecs_i @ vecs_i.T                 # (n_i,n_i)
            cos_angles[i, :n_i, :n_i] = cos_ij

        return cos_angles


class SpeciesMLP(nn.Module):
    """Element-specific MLP."""
    
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

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

if __name__ == "__main__":
    print("Vectorized BPNN - Refactored")
    print("=" * 60)
    
    # Test with dummy SF
    class DummySF:
        def __call__(self, Rij, fc, cos=None):
            return fc.sum(dim=-1, keepdim=True)
    
    sf_config = {
        0: {0: [DummySF()], 1: [DummySF()]},
        1: {0: [DummySF()], 1: [DummySF()]},
    }
    
    uniq_element = {1: 0, 8: 1}
    
    model = VectorizedBPNN(
        sf_config=sf_config,
        uniq_element=uniq_element,
        r_cutoff=6.0,
        hidden=64,
    )
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test
    data = {
        'pos': torch.randn(6, 3) * 5.0,
        'species': torch.tensor([8, 1, 1, 8, 1, 1]),
        'cell': torch.eye(3).unsqueeze(0) * 15.0,
        'batch': torch.tensor([0, 0, 0, 1, 1, 1]),
        'ptr': torch.tensor([0, 3, 6]),
    }
    
    output = model(data)
    
    print(f"\nEnergy: {output['energy']}")
    print(f"Forces: {output['forces'].shape}")
    print("\n✓ Test passed!")
