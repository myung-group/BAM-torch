"""
SimpleScalableGNN
-----------------
An invariant scalar-feature GNN designed to be paired with **probabilistic
symmetrization** (`ga_method="prob_rot"`). The model itself is fully invariant
to rotations of `data.pos`; O(3) equivariance of the predicted forces/stress is
guaranteed externally by the prob_rot averaging in `pa_model_forward`.

Design intent (universal potentials at MPtraj scale):
- Keep the architecture simple (scalar messages, no high-L tensor products) and
  spend the parameter budget on width and depth instead.
- Pre-LN residual stream so deep stacks (6-12 layers) train stably.
- Bessel basis with polynomial cutoff envelope -> smooth distance encoding,
  which matters for autograd-derived forces.
- Degree-normalized aggregation via `avg_num_neighbors` for cross-composition
  scale stability.

Hyperparameter keywords mirror the BAM-torch `gnn` / `faenet` convention so an
existing `input.json` mostly carries over.
"""
import math
from typing import Optional

import torch
from torch import nn

from bam_torch.model.blocks import RadialEmbeddingBlock
from bam_torch.utils.output_utils import get_outputs
from bam_torch.group_averaging.utils.ga_utils import pbc_preprocess, base_preprocess


class _ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, act: nn.Module):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            act,
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _InteractionBlock(nn.Module):
    """One scalar message-passing layer (pre-LN, residual, deg-normalized)."""

    def __init__(
        self,
        hidden_channels: int,
        message_dim: int,
        num_radial_basis: int,
        avg_num_neighbors: float,
        act: nn.Module,
    ):
        super().__init__()
        self.inv_sqrt_deg = 1.0 / math.sqrt(max(float(avg_num_neighbors), 1.0))

        self.node_norm = nn.LayerNorm(hidden_channels)
        self.radial_mlp = nn.Sequential(
            nn.Linear(num_radial_basis, message_dim),
            act,
            nn.Linear(message_dim, message_dim),
        )
        self.edge_in = nn.Linear(2 * hidden_channels, message_dim, bias=False)
        self.update = nn.Sequential(
            nn.Linear(message_dim, hidden_channels),
            act,
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.ffn = _ResidualMLP(hidden_channels, hidden_channels, act)

    def forward(
        self,
        h: torch.Tensor,            # (N, C)
        radial: torch.Tensor,       # (E, B)
        edge_index: torch.Tensor,   # (2, E)
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        h_n = self.node_norm(h)
        edge_h = torch.cat([h_n[src], h_n[dst]], dim=-1)         # (E, 2C)
        msg = self.edge_in(edge_h) * self.radial_mlp(radial)     # (E, M)

        agg = torch.zeros(
            h.size(0), msg.size(-1), device=h.device, dtype=h.dtype
        )
        agg.index_add_(0, dst, msg)
        agg = agg * self.inv_sqrt_deg

        h = h + self.update(agg)
        h = self.ffn(h)
        return h


class SimpleScalableGNN(nn.Module):
    """Invariant scalar GNN compatible with `pa_model_forward(prob_rot)`.

    The model returns the standard BAM-torch prediction dict
    {"energy", "node_energy", "forces", "stress", "virials"}.
    Forces are obtained via autograd of the energy w.r.t. positions
    (energy-conserving), which is the safe choice for MD downstream.
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        num_species: int = 89,
        hidden_channels: int = 256,
        features_dim: int = 256,            # message dim
        num_radial_basis: int = 8,
        nlayers: int = 6,
        max_num_neighbors: int = 30,
        avg_num_neighbors: float = 30.0,
        regress_forces: str = "auto",
        compute_stress: bool = True,
        compute_virials: bool = True,
        radial_type: str = "bessel",
        num_polynomial_cutoff: int = 6,
        pbc: bool = True,
        # Accepted-but-ignored kwargs for BAM convention parity:
        force_decoder_type: Optional[str] = None,
        force_decoder_model_config: Optional[dict] = None,
        tag_hidden_channels: int = 0,
        pg_hidden_channels: int = 0,
    ):
        super().__init__()

        self.cutoff = float(cutoff)
        self.num_species = int(num_species)
        self.hidden_channels = int(hidden_channels)
        self.features_dim = int(features_dim)
        self.num_radial_basis = int(num_radial_basis)
        self.nlayers = int(nlayers)
        self.max_num_neighbors = int(max_num_neighbors)
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.compute_stress = bool(compute_stress)
        self.compute_virials = bool(compute_virials)
        self.pbc = bool(pbc)

        if isinstance(regress_forces, bool):
            regress_forces = "autograd" if regress_forces else "false"
        self.regress_forces = (regress_forces or "auto").lower()

        act = nn.SiLU()

        # Element embedding (index 0 reserved for padding atoms).
        self.embedding = nn.Embedding(
            self.num_species + 1, self.hidden_channels, padding_idx=0
        )

        # Radial basis with smooth cutoff envelope.
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=self.cutoff,
            num_bessel=self.num_radial_basis,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=None,
        )

        self.layers = nn.ModuleList([
            _InteractionBlock(
                hidden_channels=self.hidden_channels,
                message_dim=self.features_dim,
                num_radial_basis=self.num_radial_basis,
                avg_num_neighbors=self.avg_num_neighbors,
                act=act,
            )
            for _ in range(self.nlayers)
        ])

        # Per-atom energy readout.
        self.energy_head = nn.Sequential(
            nn.LayerNorm(self.hidden_channels),
            nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            act,
            nn.Linear(self.hidden_channels // 2, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(
            self.embedding.weight, std=1.0 / math.sqrt(self.hidden_channels)
        )
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, data, mode: str = "train"):
        # Required for autograd forces and stress.
        data.pos.requires_grad_(True)
        data.cell.requires_grad_(True)

        preprocess = pbc_preprocess if self.pbc else base_preprocess
        z, batch, edge_index, _rel_pos, edge_weight = preprocess(
            data, self.cutoff, self.max_num_neighbors,
        )

        radial = self.radial_embedding(
            edge_lengths=edge_weight.unsqueeze(-1),
            node_attrs=None,
            edge_index=edge_index,
            atomic_numbers=z,
        )  # (E, B)

        h = self.embedding(z)
        for layer in self.layers:
            h = layer(h, radial, edge_index)

        per_atom_e = self.energy_head(h).view(-1)  # (N,)

        # Mask padding atoms (species==0 from get_graphset_with_pad). Without
        # this, LayerNorm/Linear bias makes padding atoms produce a non-zero
        # per-atom energy, which adds a graph-size-dependent offset noise.
        node_mask = getattr(data, "node_mask", None)
        if node_mask is not None:
            per_atom_e = per_atom_e * node_mask.to(per_atom_e.dtype)

        num_graphs = data["ptr"].numel() - 1
        energy = torch.zeros(num_graphs, device=h.device, dtype=h.dtype)
        energy.index_add_(0, batch, per_atom_e)

        preds = {"energy": energy, "node_energy": per_atom_e}

        if self.regress_forces in ("auto", "autograd", "true"):
            forces, virials, stress, _ = get_outputs(
                energy=preds["energy"],
                positions=data.pos,
                cell=data["cell"],
                batch_idx=data["batch"],
                num_graphs=num_graphs,
                training=True,
                compute_force=True,
                compute_virials=self.compute_virials,
                compute_stress=self.compute_stress,
                compute_hessian=False,
                displacement=None,
            )
            preds["forces"] = forces
            preds["stress"] = stress
            preds["virials"] = virials

        return preds
