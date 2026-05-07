"""
SparseAttentionTransformer
--------------------------
A non-equivariant 3D transformer designed to be **paired with probabilistic
symmetrization** (`ga_method="prob_rot"`).

Design intent:
- We deliberately let the model use the full 3D edge vector `rel_pos` as
  attention-bias input, **breaking SO(3) invariance on purpose**. The
  prob_rot averaging in `pa_model_forward` restores equivariance in
  expectation, so the model is free to use the richer non-invariant
  features. This trade-off buys angular expressiveness comparable to high-L
  equivariant models without any e3nn / spherical-harmonic machinery.
- Sparse attention over cutoff neighbors only — keeps memory at O(N * k_avg)
  like a GNN, but with full multi-head attention's parameter scaling.
- Energy is reduced over per-atom scalars; forces (and stress) are obtained
  by autograd through `data.pos` (energy-conserving, MD-friendly).

Recommended `nsamples` (K) in input.json: 4-8. Higher K -> lower variance
of the averaged equivariant prediction at the cost of K forward passes.
"""
import math
from typing import Optional

import torch
from torch import nn
from torch_geometric.utils import softmax as scatter_softmax

from bam_torch.model.blocks import RadialEmbeddingBlock
from bam_torch.utils.output_utils import get_outputs
from bam_torch.group_averaging.utils.ga_utils import pbc_preprocess, base_preprocess


class _EdgeBias(nn.Module):
    """Edge bias that depends on the full 3D relative-position vector
    (deliberately non-invariant) AND a radial basis of |r_ij|."""

    def __init__(self, num_radial: int, hidden: int, n_head: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 + num_radial, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_head),
        )

    def forward(self, rel_pos: torch.Tensor, radial: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([rel_pos, radial], dim=-1))  # (E, n_head)


class _AttentionLayer(nn.Module):
    """Sparse multi-head attention with edge bias + pre-LN residual + FFN."""

    def __init__(
        self,
        hidden: int,
        n_head: int,
        ffn_hidden: int,
        num_radial: int,
        edge_bias_hidden: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert hidden % n_head == 0, "hidden must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = hidden // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm_attn = nn.LayerNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.edge_bias = _EdgeBias(num_radial, edge_bias_hidden, n_head)

        # Radial gate on V: smoothes interaction strength with distance and
        # enforces zero contribution at the cutoff (via the polynomial envelope
        # already baked into `radial`).
        self.v_gate = nn.Sequential(
            nn.Linear(num_radial, n_head),
            nn.SiLU(),
            nn.Linear(n_head, n_head),
        )

        self.norm_ffn = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ffn_hidden, hidden),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        h: torch.Tensor,            # (N, C)
        rel_pos: torch.Tensor,      # (E, 3)
        radial: torch.Tensor,       # (E, B)
        edge_index: torch.Tensor,   # (2, E)
        num_nodes: int,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        N = num_nodes
        H, D = self.n_head, self.head_dim

        h_n = self.norm_attn(h)
        q = self.q_proj(h_n).view(N, H, D)
        k = self.k_proj(h_n).view(N, H, D)
        v = self.v_proj(h_n).view(N, H, D)

        bias = self.edge_bias(rel_pos, radial)                      # (E, H)
        v_gate = self.v_gate(radial)                                # (E, H)

        scores = (q[dst] * k[src]).sum(dim=-1) * self.scale + bias  # (E, H)
        attn = scatter_softmax(scores, dst, num_nodes=N)            # (E, H)
        attn = self.dropout(attn)

        msg = attn.unsqueeze(-1) * v[src] * v_gate.unsqueeze(-1)    # (E, H, D)
        out = torch.zeros(N, H, D, device=h.device, dtype=h.dtype)
        out.index_add_(0, dst, msg)
        out = out.reshape(N, H * D)
        out = self.o_proj(out)

        h = h + self.dropout(out)
        h = h + self.dropout(self.ffn(self.norm_ffn(h)))
        return h


class SparseAttentionTransformer(nn.Module):
    """3D sparse-attention transformer for atomic systems, intended for use
    with probabilistic symmetrization (prob_rot).

    Outputs the standard BAM prediction dict
    {"energy", "node_energy", "forces", "stress", "virials"}.
    Hyperparameter keywords mirror the BAM `transformer` / `gnn` convention.
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        num_species: int = 89,
        hidden_channels: int = 256,         # = d_model
        features_dim: int = 1024,           # FFN hidden dim (typically 4 * d_model)
        num_radial_basis: int = 8,
        nlayers: int = 6,
        nhead: int = 8,
        edge_bias_hidden: int = 64,
        dropout: float = 0.0,
        max_num_neighbors: int = 30,
        avg_num_neighbors: float = 30.0,    # unused; API parity
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
        self.nhead = int(nhead)
        self.max_num_neighbors = int(max_num_neighbors)
        self.compute_stress = bool(compute_stress)
        self.compute_virials = bool(compute_virials)
        self.pbc = bool(pbc)

        if isinstance(regress_forces, bool):
            regress_forces = "autograd" if regress_forces else "false"
        self.regress_forces = (regress_forces or "auto").lower()

        self.embedding = nn.Embedding(
            self.num_species + 1, self.hidden_channels, padding_idx=0
        )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=self.cutoff,
            num_bessel=self.num_radial_basis,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=None,
        )

        self.layers = nn.ModuleList([
            _AttentionLayer(
                hidden=self.hidden_channels,
                n_head=self.nhead,
                ffn_hidden=self.features_dim,
                num_radial=self.num_radial_basis,
                edge_bias_hidden=edge_bias_hidden,
                dropout=dropout,
            )
            for _ in range(self.nlayers)
        ])

        self.energy_head = nn.Sequential(
            nn.LayerNorm(self.hidden_channels),
            nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            nn.SiLU(),
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
        data.pos.requires_grad_(True)
        data.cell.requires_grad_(True)

        preprocess = pbc_preprocess if self.pbc else base_preprocess
        z, batch, edge_index, rel_pos, edge_weight = preprocess(
            data, self.cutoff, self.max_num_neighbors,
        )
        N = z.size(0)

        radial = self.radial_embedding(
            edge_lengths=edge_weight.unsqueeze(-1),
            node_attrs=None,
            edge_index=edge_index,
            atomic_numbers=z,
        )  # (E, B)

        h = self.embedding(z)
        for layer in self.layers:
            h = layer(h, rel_pos, radial, edge_index, num_nodes=N)

        per_atom_e = self.energy_head(h).view(-1)

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
