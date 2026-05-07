"""
Deep Potential Long Range (DPLR) model for BAM-torch group averaging framework.

Architecture:
- Smooth edition descriptor (se_e2_a style) for local environment
- Fitting network for atomic energy prediction
- Direct force decoder (non-equivariant, benefits from frame averaging)
- Optional long-range electrostatic correction via Ewald summation
- Autograd forces/stress as alternative or gradient target

Key design: Following DeePMD-kit, the descriptor uses distance-based
neighbor sorting, top-max_sel selection, and zero-padding. These operations
break rotational invariance, making energy NOT perfectly invariant and
autograd forces NOT perfectly equivariant. Frame averaging restores
equivariance for both autograd and direct force prediction modes.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List, Union

from bam_torch.group_averaging.utils.ga_utils import pbc_preprocess, base_preprocess
from bam_torch.utils.output_utils import get_outputs
from bam_torch.utils.scatter import scatter_sum, scatter_mean


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SmoothCutoff(nn.Module):
    """Smooth cosine cutoff envelope for the DeePMD descriptor."""

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
    """Embedding network with tanh activation and ResNet skip connections.
    Maps scalar distance features (+ optional type embeddings) to vectors.
    """

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


class SmoothDescriptor(nn.Module):
    """Smooth edition descriptor (se_e2_a) following DeePMD-kit.

    For each atom i, selects the closest max_sel neighbors (sorted by
    distance), zero-pads to a fixed size, then computes:

        D_i = (G_i^T @ R_i) @ (R_i^T @ G2_i) / N_sel^2

    Non-equivariance sources (same as DeePMD-kit):
        1. Distance-based sorting can reorder under rotation (float precision)
        2. Top-max_sel selection drops neighbors discontinuously at the boundary
        3. Zero-padding breaks rotational symmetry for under-coordinated atoms
    These make the energy NOT perfectly invariant, so autograd forces
    are NOT perfectly equivariant → frame averaging helps.
    """

    def __init__(
        self,
        num_species: int,
        embedding_dim: int,
        hidden_channels: List[int],
        axis_neurons: int,
        cutoff: float,
        max_sel: int = 60,
        use_type_embedding: bool = True,
    ):
        super().__init__()
        self.M1 = hidden_channels[-1]
        self.M2 = axis_neurons
        self.max_sel = max_sel
        self.output_dim = self.M1 * self.M2
        self.use_type_embedding = use_type_embedding

        self.cutoff_fn = SmoothCutoff(cutoff)

        if use_type_embedding:
            self.type_embedding = nn.Embedding(num_species, embedding_dim)
            input_dim = 1 + 2 * embedding_dim
        else:
            input_dim = 1

        self.embedding_net = EmbeddingNet(input_dim, hidden_channels)

    def forward(self, species, edge_index, rel_pos, distances, batch):
        """
        Returns:
            descriptors: [N_atoms, M1*M2]
            G_edge: [num_edges, M1]  (all edges, for force decoder)
        """
        N = species.shape[0]
        E = distances.shape[0]
        device = distances.device

        center_idx = edge_index[1]
        neighbor_idx = edge_index[0]

        # --- Compute features for ALL edges ---
        envelope = self.cutoff_fn(distances)
        s = envelope / (distances + 1e-8)

        unit = rel_pos / (distances.unsqueeze(-1) + 1e-8)
        R_all = torch.cat([s.unsqueeze(-1), s.unsqueeze(-1) * unit], dim=-1)  # [E, 4]

        if self.use_type_embedding:
            te_c = self.type_embedding(species[center_idx])
            te_n = self.type_embedding(species[neighbor_idx])
            emb_in = torch.cat([s.unsqueeze(-1), te_c, te_n], dim=-1)
        else:
            emb_in = s.unsqueeze(-1)

        G_all = self.embedding_net(emb_in)  # [E, M1]

        # --- Neighbor selection: sort by distance, keep top max_sel per atom ---
        if E == 0:
            return torch.zeros(N, self.output_dim, device=device), G_all

        # Sort edges by (center_atom, distance)
        max_dist = distances.max().item() + 1.0
        sort_key = center_idx.float() * max_dist + distances
        sorted_idx = torch.argsort(sort_key)
        sorted_center = center_idx[sorted_idx]

        # Compute within-group position for each sorted edge
        counts = scatter_sum(
            torch.ones(E, device=device), center_idx, dim=0, dim_size=N
        )
        offsets = torch.zeros(N, dtype=torch.long, device=device)
        if N > 1:
            offsets[1:] = torch.cumsum(counts[:-1].long(), dim=0)

        global_pos = torch.arange(E, device=device)
        within_pos = (global_pos - offsets[sorted_center]).long()

        # Keep only the closest max_sel neighbors per atom
        keep_mask = within_pos < self.max_sel
        selected_orig_idx = sorted_idx[keep_mask]
        selected_center = sorted_center[keep_mask]
        selected_within = within_pos[keep_mask]

        # --- Build fixed-size padded matrices [N, max_sel, ...] ---
        # Map each (atom, slot) to an edge index; invalid slots map to edge 0
        # and are zeroed out by the validity mask.
        edge_map = torch.zeros(N, self.max_sel, dtype=torch.long, device=device)
        valid = torch.zeros(N, self.max_sel, dtype=torch.bool, device=device)

        edge_map[selected_center, selected_within] = selected_orig_idx
        valid[selected_center, selected_within] = True

        # Gather into padded tensors (differentiable indexing)
        R_padded = R_all[edge_map] * valid.unsqueeze(-1).float()    # [N, max_sel, 4]
        G_padded = G_all[edge_map] * valid.unsqueeze(-1).float()    # [N, max_sel, M1]
        G2_padded = G_padded[:, :, :self.M2]                        # [N, max_sel, M2]

        # --- Descriptor: D = (G^T @ R) @ (R^T @ G2) per atom ---
        A = torch.bmm(G_padded.transpose(1, 2), R_padded)   # [N, M1, 4]
        B = torch.bmm(R_padded.transpose(1, 2), G2_padded)  # [N, 4, M2]
        D = torch.bmm(A, B)                                  # [N, M1, M2]

        # Normalize by actual selected neighbor count
        actual_count = valid.sum(dim=1).float().clamp(min=1.0)  # [N]
        D = D / (actual_count.unsqueeze(-1).unsqueeze(-1) ** 2)
        D = D.reshape(N, -1)  # [N, M1*M2]

        return D, G_all


class FittingNet(nn.Module):
    """MLP that maps descriptors to scalar atomic energies."""

    def __init__(self, input_dim: int, hidden_channels: List[int]):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_channels + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        return self.net(descriptors)  # [N, 1]


class DirectForceDecoder(nn.Module):
    """Non-equivariant force decoder from per-edge features + relative positions.

    Concatenates invariant embedding G with raw relative position vectors,
    breaking equivariance. Frame averaging restores it.
    """

    def __init__(self, emb_dim: int, hidden_channels: int = 128):
        super().__init__()
        self.edge_net = nn.Sequential(
            nn.Linear(emb_dim + 3, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
        )
        self.force_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, G, rel_pos, edge_index, num_atoms):
        """
        Args:
            G: [E, emb_dim] per-edge embedding features (invariant)
            rel_pos: [E, 3] raw relative position vectors (equivariant)
            edge_index: [2, E]
            num_atoms: int
        Returns:
            forces: [N, 3] (non-equivariant)
        """
        edge_feat = torch.cat([G, rel_pos], dim=-1)
        edge_feat = self.edge_net(edge_feat)
        atom_feat = scatter_sum(edge_feat, edge_index[1], dim=0, dim_size=num_atoms)
        return self.force_net(atom_feat)


class ChargeNet(nn.Module):
    """Predicts per-atom partial charges with charge neutrality per graph."""

    def __init__(self, input_dim: int, hidden_channels: List[int]):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_channels + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, descriptors: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        q_raw = self.net(descriptors)  # [N, 1]
        q_mean = scatter_mean(q_raw, batch, dim=0)  # [B, 1]
        return q_raw - q_mean[batch]  # neutralized [N, 1]


class EwaldSummation(nn.Module):
    """Differentiable Ewald summation for long-range Coulomb energy.

    Splits Coulomb sum into real-space (reuses neighbor list), reciprocal-space,
    and self-energy correction. All ops are differentiable for autograd forces.
    """

    def __init__(self, accuracy: float = 1e-6, cutoff: float = 6.0):
        super().__init__()
        self.accuracy = accuracy
        self.cutoff = cutoff
        self.alpha = math.sqrt(-math.log(accuracy)) / cutoff

    def forward(self, charges, positions, cell, batch, edge_index, distances):
        """
        Args:
            charges: [N, 1]
            positions: [N, 3]
            cell: [B, 3, 3]
            batch: [N]
            edge_index: [2, E]
            distances: [E]
        Returns:
            energy: [B]
        """
        device = positions.device
        B = cell.shape[0]
        alpha = self.alpha
        q = charges.squeeze(-1)

        # --- Real-space sum ---
        qi = q[edge_index[0]]
        qj = q[edge_index[1]]
        e_real_edge = qi * qj * torch.erfc(alpha * distances) / (distances + 1e-10)
        edge_batch = batch[edge_index[1]]
        e_real = 0.5 * scatter_sum(e_real_edge, edge_batch, dim=0, dim_size=B)

        # --- Self-energy correction ---
        e_self_atom = -(alpha / math.sqrt(math.pi)) * q ** 2
        e_self = scatter_sum(e_self_atom, batch, dim=0, dim_size=B)

        # --- Reciprocal-space sum (per graph) ---
        e_recip = torch.zeros(B, device=device)
        k_max = 2 * alpha * math.sqrt(-math.log(self.accuracy))

        for g in range(B):
            mask = batch == g
            pos_g = positions[mask]
            q_g = q[mask]
            cell_g = cell[g]
            vol = torch.abs(torch.det(cell_g))

            recip_cell = 2 * math.pi * torch.linalg.inv(cell_g).T
            recip_norms = torch.norm(recip_cell, dim=1)
            n_max = torch.clamp(torch.ceil(k_max / recip_norms).long(), max=10)

            ranges = [torch.arange(-n, n + 1, device=device) for n in n_max]
            n1, n2, n3 = torch.meshgrid(*ranges, indexing="ij")
            nvec = torch.stack(
                [n1.flatten(), n2.flatten(), n3.flatten()], dim=-1
            ).float()
            kvec = nvec @ recip_cell  # [K, 3]
            ksq = (kvec ** 2).sum(-1)

            # Exclude k=0 and |k| > k_max
            valid = (ksq > 1e-10) & (ksq < k_max ** 2)
            kvec = kvec[valid]
            ksq = ksq[valid]

            if kvec.shape[0] == 0:
                continue

            # Structure factor S(k) = sum_i q_i exp(-i k.r_i)
            kr = kvec @ pos_g.T  # [K, N_g]
            S_cos = (torch.cos(kr) * q_g.unsqueeze(0)).sum(-1)
            S_sin = (torch.sin(kr) * q_g.unsqueeze(0)).sum(-1)
            Ssq = S_cos ** 2 + S_sin ** 2

            prefactor = torch.exp(-ksq / (4 * alpha ** 2)) / ksq
            e_recip[g] = (2 * math.pi / vol) * (prefactor * Ssq).sum()

        # --- Charge neutrality correction ---
        q_total = scatter_sum(q, batch, dim=0, dim_size=B)
        volumes = torch.abs(torch.det(cell.view(-1, 3, 3)))
        e_charged = -math.pi / (2 * alpha ** 2) * q_total ** 2 / (volumes + 1e-10)

        # Coulomb constant: 14.3996 eV·Å/e^2
        ke = 14.3996
        return ke * (e_real + e_recip + e_self + e_charged)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DPLR(nn.Module):
    """Deep Potential Long Range model for the GA framework.

    Short-range: smooth descriptor (se_e2_a) with neighbor selection + fitting network
    Long-range:  learned atomic charges + Ewald summation
    Forces:      autograd (non-equivariant due to neighbor selection → GA improves)
                 or direct decoder (non-equivariant by architecture → GA improves)

    Both force modes benefit from frame averaging because the descriptor's
    neighbor sorting/selection/padding breaks rotational invariance.
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        num_species: int = 4,
        embedding_dim: int = 32,
        descriptor_hidden_channels: List[int] = [25, 50, 100],
        descriptor_axis_neurons: int = 16,
        fitting_hidden_channels: List[int] = [240, 240, 240],
        regress_forces: str = "direct_with_gradient_target",
        use_long_range: bool = True,
        ewald_accuracy: float = 1e-6,
        charge_fitting_hidden: List[int] = [240, 240, 240],
        max_num_neighbors: int = 40,
        max_sel: int = 60,
        preprocess: Union[str, callable] = "pbc_preprocess",
        force_decoder_hidden: int = 128,
        use_type_embedding: bool = True,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.regress_forces = regress_forces
        self.use_long_range = use_long_range
        self.max_num_neighbors = max_num_neighbors

        if isinstance(preprocess, str):
            self.preprocess = eval(preprocess)
        else:
            self.preprocess = preprocess

        # Descriptor (with DeePMD-style neighbor selection → non-equivariant)
        self.descriptor = SmoothDescriptor(
            num_species=num_species,
            embedding_dim=embedding_dim,
            hidden_channels=descriptor_hidden_channels,
            axis_neurons=descriptor_axis_neurons,
            cutoff=cutoff,
            max_sel=max_sel,
            use_type_embedding=use_type_embedding,
        )
        desc_dim = self.descriptor.output_dim

        # Energy fitting
        self.fitting = FittingNet(desc_dim, fitting_hidden_channels)

        # Direct force decoder (non-equivariant → GA helps)
        if "direct" in regress_forces:
            emb_dim = descriptor_hidden_channels[-1]
            self.force_decoder = DirectForceDecoder(emb_dim, force_decoder_hidden)

        # Long-range electrostatics
        if use_long_range:
            self.charge_net = ChargeNet(desc_dim, charge_fitting_hidden)
            self.ewald = EwaldSummation(ewald_accuracy, cutoff)

    def forward(self, data, mode="train"):
        """
        Args:
            data: PyG data with pos, cell, species, edge_index, edges, batch, ptr
            mode: "train" or "eval"
        Returns:
            dict: energy, forces, node_energy, [stress, virials, forces_grad_target]
        """
        data.pos.requires_grad_(True)
        data.cell.requires_grad_(True)

        # Preprocess: compute edges with PBC
        z, batch, edge_index, rel_pos, distances = self.preprocess(
            data, self.cutoff, self.max_num_neighbors
        )

        # Smooth descriptor
        descriptors, G_edge = self.descriptor(
            z, edge_index, rel_pos, distances, batch
        )

        # Short-range atomic energy
        node_energy = self.fitting(descriptors)  # [N, 1]
        B = data.ptr.numel() - 1
        energy = scatter_sum(
            node_energy.squeeze(-1), batch, dim=0, dim_size=B
        )

        # Long-range electrostatic energy
        if self.use_long_range:
            charges = self.charge_net(descriptors, batch)
            energy_lr = self.ewald(
                charges, data.pos, data.cell, batch, edge_index, distances
            )
            energy = energy + energy_lr

        preds = {
            "energy": energy,
            "node_energy": node_energy.squeeze(-1),
        }

        # --- Force computation ---
        if "auto" in self.regress_forces:
            # Autograd forces (equivariant by construction)
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

        elif "direct" in self.regress_forces:
            # Direct force prediction (non-equivariant → GA improves this)
            forces = self.force_decoder(
                G_edge, rel_pos, edge_index, z.shape[0]
            )
            preds["forces"] = forces

            # Autograd gradient target for energy-conservation loss
            if "gradient_target" in self.regress_forces:
                forces_gt, virials, stress, _ = get_outputs(
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
                preds["forces_grad_target"] = forces_gt
                preds["stress"] = stress
                preds["virials"] = virials

        return preds

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
