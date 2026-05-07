"""
Paper-faithful DPLR / DPSR / Deep Wannier implementation.

Reproduces the architecture from:
  Zhang et al. "Molecular-scale insights into the electrical double layer
  at oxide-electrolyte interfaces", Nature Comm. (2024) 15:10270.

Models:
  - DPSRPaper:         Short-range DP (descriptor + fitting, autograd forces)
  - DeepWannierModel:  Deep Wannier model (descriptor + dipole fitting)
  - DPLRPaper:         Full DPLR (SR + frozen DW + fixed-charge Ewald)

Key differences from dplr.py:
  - Per-element neighbor selection (sel=[38,120,75,14,14])
  - ResNet fitting with resnet_dt (learnable skip gate)
  - Deep Wannier model for Wannier centroid prediction (replaces ChargeNet)
  - Fixed-charge Ewald summation on ionic + WC positions
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List, Union

from bam_torch.group_averaging.utils.ga_utils import pbc_preprocess, base_preprocess
from bam_torch.utils.output_utils import get_outputs
from bam_torch.utils.scatter import scatter_sum, scatter_mean


def _ensure_pos(data):
    """Ensure data.pos exists (GA forward sets it; standalone tests may not)."""
    if not hasattr(data, 'pos') or data.pos is None:
        data.pos = data.positions
    return data


# ---------------------------------------------------------------------------
# Building blocks (shared by all models)
# ---------------------------------------------------------------------------

class SmoothCutoff(nn.Module):
    """Smooth cosine cutoff from DeePMD: 1 for r<r_cs, cosine taper, 0 for r>r_cut."""

    def __init__(self, r_cut: float, r_cs: Optional[float] = None):
        super().__init__()
        self.r_cut = r_cut
        self.r_cs = r_cs if r_cs is not None else max(r_cut - 1.0, 0.0)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        u = torch.zeros_like(r)
        mask_inner = r < self.r_cs
        u[mask_inner] = 1.0
        mask_mid = (r >= self.r_cs) & (r < self.r_cut)
        x = (r[mask_mid] - self.r_cs) / (self.r_cut - self.r_cs)
        u[mask_mid] = 0.5 * torch.cos(math.pi * x) + 0.5
        return u


class EmbeddingNet(nn.Module):
    """Embedding network with tanh + ResNet skip connections (DeePMD style)."""

    def __init__(self, input_dim: int, hidden_channels: List[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_channels
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x_new = torch.tanh(layer(x))
            if x.shape[-1] == x_new.shape[-1]:
                x = x_new + x
            else:
                x = x_new
        return x


class ResNetFittingNet(nn.Module):
    """Fitting network with DeePMD resnet_dt: x_new = x + dt * activation(Linear(x)).

    When input/output dims differ, no skip connection (just activation(Linear(x))).
    The learnable scalar dt gates the residual.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: List[int],
        output_dim: int = 1,
        resnet_dt: bool = True,
    ):
        super().__init__()
        dims = [input_dim] + hidden_channels
        self.layers = nn.ModuleList()
        self.dt_params = nn.ParameterList()
        self.skip_flags = []

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            can_skip = (dims[i] == dims[i + 1])
            self.skip_flags.append(can_skip)
            if can_skip and resnet_dt:
                self.dt_params.append(nn.Parameter(torch.ones(1) * 0.1))
            else:
                self.dt_params.append(None)

        # Final linear to output_dim (no activation)
        self.output_layer = nn.Linear(dims[-1], output_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, dt, skip in zip(self.layers, self.dt_params, self.skip_flags):
            x_new = torch.tanh(layer(x))
            if skip and dt is not None:
                x = x + dt * x_new
            elif skip:
                x = x + x_new
            else:
                x = x_new
        return self.output_layer(x)


# ---------------------------------------------------------------------------
# Per-element Smooth Descriptor (se_e2_a)
# ---------------------------------------------------------------------------

class PerElementSmoothDescriptor(nn.Module):
    """Smooth edition descriptor (se_e2_a) with per-element neighbor selection.

    For each center atom, selects up to sel[s] nearest neighbors of species s,
    zero-pads, then computes: D_i = (G^T R)(R^T G2) / N_sel^2

    Args:
        num_species: Number of atom species
        sel: Per-element max neighbor count, e.g. [38, 120, 75, 14, 14]
        embedding_dim: Type embedding dimension
        hidden_channels: Embedding net hidden layers, e.g. [25, 50, 100]
        axis_neurons: M2 dimension (16 for energy, 8 for DW)
        cutoff: Radial cutoff
        rcut_smth: Smooth cutoff start distance
        use_type_embedding: Whether to use type embeddings
    """

    def __init__(
        self,
        num_species: int,
        sel: List[int],
        embedding_dim: int = 32,
        hidden_channels: List[int] = [25, 50, 100],
        axis_neurons: int = 16,
        cutoff: float = 6.0,
        rcut_smth: float = 3.0,
        use_type_embedding: bool = True,
    ):
        super().__init__()
        assert len(sel) == num_species
        self.num_species = num_species
        self.sel = sel
        self.total_sel = sum(sel)
        self.M1 = hidden_channels[-1]
        self.M2 = axis_neurons
        self.output_dim = self.M1 * self.M2
        self.use_type_embedding = use_type_embedding
        self.cutoff = cutoff

        self.cutoff_fn = SmoothCutoff(cutoff, rcut_smth)

        if use_type_embedding:
            self.type_embedding = nn.Embedding(num_species, embedding_dim)
            input_dim = 1 + 2 * embedding_dim
        else:
            input_dim = 1

        self.embedding_net = EmbeddingNet(input_dim, hidden_channels)

    def forward(self, species, edge_index, rel_pos, distances, batch):
        """
        Args:
            species: [N] atom species indices
            edge_index: [2, E] (sender=0, receiver=1 convention varies; we use center=1)
            rel_pos: [E, 3] relative position vectors
            distances: [E] edge distances
            batch: [N] graph index per atom

        Returns:
            descriptors: [N, M1*M2]
        """
        N = species.shape[0]
        E = distances.shape[0]
        device = distances.device

        center_idx = edge_index[1]
        neighbor_idx = edge_index[0]

        # --- Compute features for ALL edges ---
        envelope = self.cutoff_fn(distances)
        s = envelope / (distances + 1e-8)  # [E]

        unit = rel_pos / (distances.unsqueeze(-1) + 1e-8)
        R_all = torch.cat([s.unsqueeze(-1), s.unsqueeze(-1) * unit], dim=-1)  # [E, 4]

        if self.use_type_embedding:
            te_c = self.type_embedding(species[center_idx])
            te_n = self.type_embedding(species[neighbor_idx])
            emb_in = torch.cat([s.unsqueeze(-1), te_c, te_n], dim=-1)
        else:
            emb_in = s.unsqueeze(-1)

        G_all = self.embedding_net(emb_in)  # [E, M1]

        if E == 0:
            return torch.zeros(N, self.output_dim, device=device)

        # --- Per-element neighbor selection ---
        neighbor_species = species[neighbor_idx]  # [E]

        # Build padded R and G tensors: [N, total_sel, ...]
        R_padded = torch.zeros(N, self.total_sel, 4, device=device)
        G_padded = torch.zeros(N, self.total_sel, self.M1, device=device)
        valid = torch.zeros(N, self.total_sel, dtype=torch.bool, device=device)

        slot_offset = 0
        for s_type in range(self.num_species):
            max_k = self.sel[s_type]
            if max_k == 0:
                continue

            # Edges where neighbor is of type s_type
            type_mask = (neighbor_species == s_type)
            if not type_mask.any():
                slot_offset += max_k
                continue

            e_idx = type_mask.nonzero(as_tuple=True)[0]  # edge indices of this type
            e_center = center_idx[e_idx]   # [E_s]
            e_dist = distances[e_idx]      # [E_s]

            # Sort by (center_atom, distance)
            max_dist = e_dist.max().item() + 1.0
            sort_key = e_center.float() * max_dist + e_dist
            sorted_order = torch.argsort(sort_key)
            e_idx = e_idx[sorted_order]
            e_center_sorted = e_center[sorted_order]

            # Compute within-group position for each center atom
            counts = scatter_sum(
                torch.ones(e_idx.shape[0], device=device),
                e_center_sorted, dim=0, dim_size=N,
            )
            offsets = torch.zeros(N, dtype=torch.long, device=device)
            if N > 1:
                offsets[1:] = torch.cumsum(counts[:-1].long(), dim=0)

            global_pos = torch.arange(e_idx.shape[0], device=device)
            within_pos = (global_pos - offsets[e_center_sorted]).long()

            # Keep only top max_k per center atom
            keep = within_pos < max_k
            sel_edge_idx = e_idx[keep]
            sel_center = e_center_sorted[keep]
            sel_within = within_pos[keep]

            # Fill padded tensors
            slot_idx = slot_offset + sel_within
            R_padded[sel_center, slot_idx] = R_all[sel_edge_idx]
            G_padded[sel_center, slot_idx] = G_all[sel_edge_idx]
            valid[sel_center, slot_idx] = True

            slot_offset += max_k

        # --- Descriptor: D = (G^T @ R) @ (R^T @ G2) / n^2 ---
        G2_padded = G_padded[:, :, :self.M2]  # [N, total_sel, M2]

        A = torch.bmm(G_padded.transpose(1, 2), R_padded)    # [N, M1, 4]
        B = torch.bmm(R_padded.transpose(1, 2), G2_padded)   # [N, 4, M2]
        D = torch.bmm(A, B)                                   # [N, M1, M2]

        # Normalize by actual selected neighbor count
        actual_count = valid.sum(dim=1).float().clamp(min=1.0)  # [N]
        D = D / (actual_count.unsqueeze(-1).unsqueeze(-1) ** 2)
        D = D.reshape(N, -1)  # [N, M1*M2]

        return D


# ---------------------------------------------------------------------------
# Dipole fitting (for Deep Wannier model)
# ---------------------------------------------------------------------------

class DipoleFittingNet(nn.Module):
    """Fitting net that predicts 3D Wannier centroid offsets for selected atom types.

    Args:
        input_dim: Descriptor output dim (e.g. 800 for M1*M2=100*8)
        hidden_channels: e.g. [100, 100, 100]
        sel_type: Atom type indices to predict dipoles for, e.g. [2, 3, 4]
        resnet_dt: Use DeePMD resnet_dt
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: List[int] = [100, 100, 100],
        sel_type: List[int] = [2, 3, 4],
        resnet_dt: bool = True,
    ):
        super().__init__()
        self.sel_type = sel_type
        self.fitting = ResNetFittingNet(
            input_dim=input_dim,
            hidden_channels=hidden_channels,
            output_dim=3,
            resnet_dt=resnet_dt,
        )

    def forward(self, descriptors, species):
        """
        Args:
            descriptors: [N, D] all atoms' descriptors
            species: [N] atom type indices

        Returns:
            offsets: [N_sel, 3] WC offsets for selected atoms
            mask: [N] bool mask for selected atoms
        """
        mask = torch.zeros(species.shape[0], dtype=torch.bool, device=species.device)
        for t in self.sel_type:
            mask |= (species == t)
        selected_desc = descriptors[mask]  # [N_sel, D]
        offsets = self.fitting(selected_desc)  # [N_sel, 3]
        return offsets, mask


# ---------------------------------------------------------------------------
# Fixed-charge Ewald summation
# ---------------------------------------------------------------------------

class FixedChargeEwald(nn.Module):
    """Differentiable Ewald summation for fixed-charge systems.

    Computes Coulomb energy for a system of point charges (ionic + Wannier centroids)
    with periodic boundary conditions. All charges are fixed (not learnable).

    The positions ARE differentiable (for autograd forces).

    Args:
        alpha: Ewald splitting parameter (inverse length, Å^-1)
        k_max_factor: Factor for reciprocal cutoff: k_max = 2*alpha*sqrt(-ln(accuracy))
        accuracy: Target accuracy for Ewald convergence
        n_max_cap: Maximum k-vector index per direction
    """

    def __init__(
        self,
        alpha: float = 0.1,
        accuracy: float = 1e-6,
        n_max_cap: int = 10,
    ):
        super().__init__()
        self.alpha = alpha
        self.accuracy = accuracy
        self.n_max_cap = n_max_cap
        self.ke = 14.3996  # Coulomb constant: eV·Å/e^2

    def forward(self, positions, charges, cell, batch, num_graphs):
        """
        Args:
            positions: [M, 3] all particle positions (ionic + WC)
            charges: [M] charges in units of e
            cell: [B, 3, 3] unit cells
            batch: [M] graph index for each particle
            num_graphs: int

        Returns:
            energy: [B] Ewald energy per graph
        """
        device = positions.device
        B = num_graphs
        alpha = self.alpha
        q = charges

        # --- Self-energy correction ---
        e_self_atom = -(alpha / math.sqrt(math.pi)) * q ** 2
        e_self = scatter_sum(e_self_atom, batch, dim=0, dim_size=B)

        # --- Real-space sum (all pairs within cutoff via distance matrix per graph) ---
        # For small systems, compute all pairwise within each graph
        k_max = 2 * alpha * math.sqrt(-math.log(self.accuracy))
        r_cut_real = math.sqrt(-math.log(self.accuracy)) / alpha

        e_real = torch.zeros(B, device=device)
        e_recip = torch.zeros(B, device=device)

        for g in range(B):
            mask = (batch == g)
            pos_g = positions[mask]
            q_g = q[mask]
            n_g = pos_g.shape[0]
            cell_g = cell[g]

            # --- Real-space: pairwise within minimum image ---
            if n_g > 1:
                dr = pos_g.unsqueeze(0) - pos_g.unsqueeze(1)  # [n, n, 3]

                # Minimum image convention
                inv_cell = torch.linalg.inv(cell_g)
                frac = dr @ inv_cell.T
                frac = frac - torch.round(frac)
                dr = frac @ cell_g

                dist = torch.norm(dr, dim=-1)  # [n, n]

                # Mask: upper triangle, exclude self, within cutoff
                mask_pairs = (dist > 1e-8) & (dist < r_cut_real)
                triu_mask = torch.triu(torch.ones(n_g, n_g, device=device, dtype=torch.bool), diagonal=1)
                mask_pairs = mask_pairs & triu_mask

                qi_qj = q_g.unsqueeze(0) * q_g.unsqueeze(1)  # [n, n]
                e_real_pairs = qi_qj * torch.erfc(alpha * dist) / (dist + 1e-10)
                e_real[g] = e_real_pairs[mask_pairs].sum()

            # --- Reciprocal-space ---
            vol = torch.abs(torch.det(cell_g))
            recip_cell = 2 * math.pi * torch.linalg.inv(cell_g).T
            recip_norms = torch.norm(recip_cell, dim=1)
            n_max = torch.clamp(
                torch.ceil(k_max / recip_norms).long(), max=self.n_max_cap
            )

            ranges = [torch.arange(-n, n + 1, device=device) for n in n_max]
            n1, n2, n3 = torch.meshgrid(*ranges, indexing="ij")
            nvec = torch.stack([n1.flatten(), n2.flatten(), n3.flatten()], dim=-1).float()
            kvec = nvec @ recip_cell  # [K, 3]
            ksq = (kvec ** 2).sum(-1)

            valid_k = (ksq > 1e-10) & (ksq < k_max ** 2)
            kvec = kvec[valid_k]
            ksq = ksq[valid_k]

            if kvec.shape[0] > 0:
                kr = kvec @ pos_g.T  # [K, n_g]
                S_cos = (torch.cos(kr) * q_g.unsqueeze(0)).sum(-1)
                S_sin = (torch.sin(kr) * q_g.unsqueeze(0)).sum(-1)
                Ssq = S_cos ** 2 + S_sin ** 2

                prefactor = torch.exp(-ksq / (4 * alpha ** 2)) / ksq
                e_recip[g] = (2 * math.pi / vol) * (prefactor * Ssq).sum()

            # --- Charged system correction ---
            q_total = q_g.sum()
            e_charged = -math.pi / (2 * alpha ** 2) * q_total ** 2 / (vol + 1e-10)
            e_recip[g] = e_recip[g] + e_charged

        return self.ke * (e_real + e_recip + e_self)


# ---------------------------------------------------------------------------
# Deep Wannier Model
# ---------------------------------------------------------------------------

class DeepWannierModel(nn.Module):
    """Deep Wannier model: predicts Wannier centroid offsets for selected atom types.

    Architecture: PerElementSmoothDescriptor (axis_neurons=8) → DipoleFittingNet
    Trained on atomic_dipole.raw labels from DFT Wannier analysis.

    Args:
        cutoff, rcut_smth: Cutoff parameters
        num_species: Number of atom species (5 for TiO2-EDL)
        sel: Per-element neighbor selection
        embedding_dim: Type embedding dimension
        descriptor_hidden: Embedding net architecture
        axis_neurons: M2 for descriptor (default 8 for DW)
        fitting_hidden: Fitting net architecture
        sel_type: Atom types to predict dipoles for
        max_num_neighbors: Max total neighbors for preprocessing
        preprocess: Preprocessing function name
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        rcut_smth: float = 3.0,
        num_species: int = 5,
        sel: List[int] = [38, 120, 75, 14, 14],
        embedding_dim: int = 32,
        descriptor_hidden: List[int] = [25, 50, 100],
        axis_neurons: int = 8,
        fitting_hidden: List[int] = [100, 100, 100],
        sel_type: List[int] = [2, 3, 4],
        max_num_neighbors: int = 300,
        preprocess: Union[str, callable] = "pbc_preprocess",
        regress_forces: str = "auto",
    ):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.sel_type = sel_type
        self.regress_forces = regress_forces

        if isinstance(preprocess, str):
            self.preprocess = eval(preprocess)
        else:
            self.preprocess = preprocess

        self.descriptor = PerElementSmoothDescriptor(
            num_species=num_species,
            sel=sel,
            embedding_dim=embedding_dim,
            hidden_channels=descriptor_hidden,
            axis_neurons=axis_neurons,
            cutoff=cutoff,
            rcut_smth=rcut_smth,
        )

        self.dipole_fitting = DipoleFittingNet(
            input_dim=self.descriptor.output_dim,
            hidden_channels=fitting_hidden,
            sel_type=sel_type,
            resnet_dt=True,
        )

    def forward(self, data, mode="train"):
        _ensure_pos(data)
        z, batch, edge_index, rel_pos, distances = self.preprocess(
            data, self.cutoff, self.max_num_neighbors
        )

        descriptors = self.descriptor(z, edge_index, rel_pos, distances, batch)
        offsets, mask = self.dipole_fitting(descriptors, z)

        preds = {
            "dipole": offsets,       # [N_sel, 3]
            "dipole_mask": mask,     # [N] bool
        }
        return preds

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# DPSR (Short-Range only)
# ---------------------------------------------------------------------------

class DPSRPaper(nn.Module):
    """Deep Potential Short-Range model (no long-range correction).

    Architecture: PerElementSmoothDescriptor → ResNetFittingNet → autograd forces.
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        rcut_smth: float = 3.0,
        num_species: int = 5,
        sel: List[int] = [38, 120, 75, 14, 14],
        embedding_dim: int = 32,
        descriptor_hidden: List[int] = [25, 50, 100],
        axis_neurons: int = 16,
        fitting_hidden: List[int] = [100, 100, 100],
        max_num_neighbors: int = 300,
        preprocess: Union[str, callable] = "pbc_preprocess",
        regress_forces: str = "auto",
    ):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.regress_forces = regress_forces

        if isinstance(preprocess, str):
            self.preprocess = eval(preprocess)
        else:
            self.preprocess = preprocess

        self.descriptor = PerElementSmoothDescriptor(
            num_species=num_species,
            sel=sel,
            embedding_dim=embedding_dim,
            hidden_channels=descriptor_hidden,
            axis_neurons=axis_neurons,
            cutoff=cutoff,
            rcut_smth=rcut_smth,
        )

        self.fitting = ResNetFittingNet(
            input_dim=self.descriptor.output_dim,
            hidden_channels=fitting_hidden,
            output_dim=1,
            resnet_dt=True,
        )

    def forward(self, data, mode="train"):
        _ensure_pos(data)
        data.pos.requires_grad_(True)
        data.cell.requires_grad_(True)

        z, batch, edge_index, rel_pos, distances = self.preprocess(
            data, self.cutoff, self.max_num_neighbors
        )

        descriptors = self.descriptor(z, edge_index, rel_pos, distances, batch)
        node_energy = self.fitting(descriptors)  # [N, 1]

        B = data.ptr.numel() - 1
        energy = scatter_sum(
            node_energy.squeeze(-1), batch, dim=0, dim_size=B
        )

        preds = {
            "energy": energy,
            "node_energy": node_energy.squeeze(-1),
        }

        if "auto" in self.regress_forces:
            forces, virials, stress, _ = get_outputs(
                energy=energy,
                positions=data.pos,
                cell=data.cell,
                batch_idx=batch,
                num_graphs=B,
                training=True,
                compute_force=True,
                compute_virials=True,
                compute_stress=True,
                compute_hessian=False,
                displacement=None,
            )
            preds["forces"] = forces
            preds["stress"] = stress
            preds["virials"] = virials

        return preds

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# DPLR (Short-Range + Long-Range with frozen Deep Wannier)
# ---------------------------------------------------------------------------

class DPLRPaper(nn.Module):
    """Deep Potential Long-Range model following the paper.

    Short-range: PerElementSmoothDescriptor → ResNetFittingNet → E_SR
    Long-range:  Frozen DW → WC positions → Fixed-charge Ewald → E_LR
    Total:       E = E_SR + E_LR, forces via autograd

    Args:
        (descriptor/fitting params same as DPSRPaper)
        dw_model: Pre-trained DeepWannierModel instance (frozen)
        sys_charge_map: Nuclear charges per species [Ti=4,H=1,O=6,Na=9,Cl=7]
        model_charge_map: WC charges for sel_type atoms [-8,-8,-8]
        sel_type: Atom types with Wannier centroids [2,3,4]
        ewald_alpha: Ewald splitting parameter
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        rcut_smth: float = 3.0,
        num_species: int = 5,
        sel: List[int] = [38, 120, 75, 14, 14],
        embedding_dim: int = 32,
        descriptor_hidden: List[int] = [25, 50, 100],
        axis_neurons: int = 16,
        fitting_hidden: List[int] = [100, 100, 100],
        max_num_neighbors: int = 300,
        preprocess: Union[str, callable] = "pbc_preprocess",
        regress_forces: str = "auto",
        # Long-range parameters
        dw_model: Optional[nn.Module] = None,
        sys_charge_map: List[float] = [4, 1, 6, 9, 7],
        model_charge_map: List[float] = [-8, -8, -8],
        sel_type: List[int] = [2, 3, 4],
        ewald_alpha: float = 0.1,
        ewald_accuracy: float = 1e-6,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.regress_forces = regress_forces
        self.sel_type = sel_type

        if isinstance(preprocess, str):
            self.preprocess = eval(preprocess)
        else:
            self.preprocess = preprocess

        # Short-range descriptor + fitting
        self.descriptor = PerElementSmoothDescriptor(
            num_species=num_species,
            sel=sel,
            embedding_dim=embedding_dim,
            hidden_channels=descriptor_hidden,
            axis_neurons=axis_neurons,
            cutoff=cutoff,
            rcut_smth=rcut_smth,
        )

        self.fitting = ResNetFittingNet(
            input_dim=self.descriptor.output_dim,
            hidden_channels=fitting_hidden,
            output_dim=1,
            resnet_dt=True,
        )

        # Long-range: frozen DW + Ewald
        self.dw_model = dw_model
        if dw_model is not None:
            # Freeze DW parameters (but keep forward differentiable w.r.t. positions)
            for p in self.dw_model.parameters():
                p.requires_grad_(False)
            self.dw_model.eval()

        self.register_buffer(
            "sys_charges", torch.tensor(sys_charge_map, dtype=torch.float32)
        )
        self.register_buffer(
            "wc_charges", torch.tensor(model_charge_map, dtype=torch.float32)
        )

        self.ewald = FixedChargeEwald(
            alpha=ewald_alpha,
            accuracy=ewald_accuracy,
        )

    def _compute_long_range(self, data, z, batch, B):
        """Compute long-range Ewald energy using frozen DW model.

        Returns E_LR [B] - differentiable w.r.t. data.pos
        """
        # Run frozen DW to get WC offsets (differentiable w.r.t. positions)
        with torch.no_grad():
            # DW parameters frozen, but we need grad w.r.t. positions
            pass

        # Actually: DW params are frozen via requires_grad_(False),
        # but the forward computation IS in the autograd graph because
        # data.pos has requires_grad=True. So we just call forward normally.
        dw_preds = self.dw_model(data, mode="eval")
        wc_offsets = dw_preds["dipole"]     # [N_sel, 3]
        dw_mask = dw_preds["dipole_mask"]   # [N] bool

        # Compute WC absolute positions
        pos_sel = data.pos[dw_mask]         # [N_sel, 3]
        pos_wc = pos_sel + wc_offsets       # [N_sel, 3] - differentiable

        # Build extended system: ionic positions + WC positions
        ext_positions = torch.cat([data.pos, pos_wc], dim=0)

        # Ionic charges from species map
        ionic_charges = self.sys_charges[z]  # [N]

        # WC charges: map sel_type index to model_charge_map
        species_sel = z[dw_mask]             # [N_sel]
        wc_charges = torch.zeros(species_sel.shape[0], device=z.device)
        for i, st in enumerate(self.sel_type):
            wc_charges[species_sel == st] = self.wc_charges[i]

        ext_charges = torch.cat([ionic_charges, wc_charges], dim=0)

        # Extended batch indices
        batch_wc = batch[dw_mask]
        ext_batch = torch.cat([batch, batch_wc], dim=0)

        # Ewald on extended system
        e_lr = self.ewald(ext_positions, ext_charges, data.cell, ext_batch, B)
        return e_lr

    def forward(self, data, mode="train"):
        _ensure_pos(data)
        data.pos.requires_grad_(True)
        data.cell.requires_grad_(True)

        z, batch, edge_index, rel_pos, distances = self.preprocess(
            data, self.cutoff, self.max_num_neighbors
        )

        # Short-range energy
        descriptors = self.descriptor(z, edge_index, rel_pos, distances, batch)
        node_energy = self.fitting(descriptors)  # [N, 1]

        B = data.ptr.numel() - 1
        e_sr = scatter_sum(
            node_energy.squeeze(-1), batch, dim=0, dim_size=B
        )

        # Long-range energy
        if self.dw_model is not None:
            e_lr = self._compute_long_range(data, z, batch, B)
            energy = e_sr + e_lr
        else:
            energy = e_sr

        preds = {
            "energy": energy,
            "node_energy": node_energy.squeeze(-1),
        }

        if "auto" in self.regress_forces:
            forces, virials, stress, _ = get_outputs(
                energy=energy,
                positions=data.pos,
                cell=data.cell,
                batch_idx=batch,
                num_graphs=B,
                training=True,
                compute_force=True,
                compute_virials=True,
                compute_stress=True,
                compute_hessian=False,
                displacement=None,
            )
            preds["forces"] = forces
            preds["stress"] = stress
            preds["virials"] = virials

        return preds

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
