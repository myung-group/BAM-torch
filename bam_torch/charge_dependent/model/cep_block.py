"""
CENT2-based Charge Equilibration Process (CEP) Block.

Phase 2 implementation:
  chi_i = MLP(scalar node features)  — ANN-predicted environment-dependent electronegativity
  J_i = softplus(J_raw[species])     — per-element chemical hardness (learnable parameter)

  CEP analytical solution (Lagrange multiplier method):
    min sum_i [ chi_i q_i + 0.5 J_i q_i^2 ]   s.t.  sum_i q_i = Q_total

    -> lambda = (Q_total + sum_i chi_i/J_i) / sum_i (1/J_i)   [per graph]
    -> q_i = (lambda - chi_i) / J_i                            [per atom]

  Charge conservation is mathematically guaranteed (hard constraint).

  U_CENT = sum_i [ chi_i q_i + 0.5 J_i q_i^2 ]

Reference:
  Khajehpasha et al., Phys. Rev. B 105, 144106 (2022)  — CENT2
  Ghasemi & Goedecker, J. Chem. Phys. 154, 074107 (2021) — CENT1
"""

import torch
import torch.nn as tnn
from typing import Dict

from e3nn import o3

from bam_torch.utils.scatter import scatter_sum


class CEPBlock(tnn.Module):
    """
    CENT2-based Charge Equilibration Process Block.

    Predicts environment-dependent electronegativity chi_i from node features,
    then analytically determines atomic charges q_i via Lagrange multiplier method.
    Charge conservation (sum q_i = Q_total) is always mathematically guaranteed.

    Args:
        irreps_in   : irreps of input node features (used to extract scalar components)
        num_species : number of element types (size of J_i parameter table)
        hidden_dim  : hidden dimension of chi_i prediction MLP (default 64)
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        num_species: int,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # scalar (l=0, even parity) component dimension
        self.scalar_dim: int = irreps_in.count(o3.Irrep(0, 1))

        # chi_i prediction MLP (environment-dependent electronegativity)
        self.chi_mlp = tnn.Sequential(
            tnn.Linear(self.scalar_dim, hidden_dim),
            tnn.SiLU(),
            tnn.Linear(hidden_dim, hidden_dim),
            tnn.SiLU(),
            tnn.Linear(hidden_dim, 1),
        )

        # J_i : per-element chemical hardness (learnable, softplus ensures positive)
        self.J_raw = tnn.Parameter(torch.ones(num_species))

    def forward(
        self,
        node_feats: torch.Tensor,       # [num_nodes, irreps_dim]
        species: torch.Tensor,           # [num_nodes]  element index
        total_charge: torch.Tensor,      # [num_graphs] total charge (Q_total)
        batch: torch.Tensor,             # [num_nodes]  batch index
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            atomic_charges : [num_nodes]  CEP-determined atomic charges q_i
            chi            : [num_nodes]  predicted atomic electronegativity chi_i
            J              : [num_nodes]  per-element chemical hardness J_i
            U_CENT         : [num_graphs] CEP electrostatic energy
            total_charge   : [num_graphs] sum q_i (for conservation verification, ~ input Q_total)
        """
        # ── chi_i prediction ───────────────────────────────────────────────
        # Extract scalar (l=0) features only (located at the front)
        scalar_feats = node_feats[:, :self.scalar_dim]
        chi = self.chi_mlp(scalar_feats).squeeze(-1)          # [N]

        # ── J_i (softplus: always positive) ────────────────────────────────
        J = tnn.functional.softplus(self.J_raw)[species]      # [N]

        # ── CEP analytical solution (Lagrange multiplier) ──────────────────
        eps: float = 1e-8
        inv_J = 1.0 / (J + eps)                               # [N]

        # Per-graph aggregation
        sum_chi_over_J = scatter_sum(
            chi * inv_J, batch, dim=0, dim_size=num_graphs,
        )                                                      # [G]
        sum_inv_J = scatter_sum(
            inv_J, batch, dim=0, dim_size=num_graphs,
        )                                                      # [G]

        # lambda = (Q_total + sum chi/J) / sum (1/J)
        lam = (total_charge + sum_chi_over_J) / (sum_inv_J + eps)  # [G]
        lam_per_atom = lam[batch]                              # [N]

        # q_i = (lambda - chi_i) / J_i  ->  hard charge conservation
        q = (lam_per_atom - chi) * inv_J                      # [N]

        # ── U_CENT = sum_i [ chi_i q_i + 0.5 J_i q_i^2 ] ─────────────────
        U_CENT_per_atom = chi * q + 0.5 * J * q.pow(2)
        U_CENT = scatter_sum(
            U_CENT_per_atom, batch, dim=0, dim_size=num_graphs,
        )                                                      # [G]

        # For conservation verification
        q_total = scatter_sum(q, batch, dim=0, dim_size=num_graphs)

        return {
            "atomic_charges": q,
            "chi": chi,
            "J": J,
            "U_CENT": U_CENT,
            "total_charge": q_total,
        }
