"""
CENT2 기반 Charge Equilibration Process (CEP) Block.

Phase 2 구현:
  χ_i = MLP(scalar node features)  — ANN 예측 환경 의존 전기음성도
  J_i = softplus(J_raw[species])   — 원소별 화학적 경도 (학습 파라미터)

  CEP 해석해 (Lagrange 승수법):
    min Σ_i [ χ_i q_i + ½ J_i q_i² ]   s.t.  Σ_i q_i = Q_total

    → λ   = (Q_total + Σ_i χ_i/J_i) / Σ_i (1/J_i)   [그래프별]
    → q_i = (λ - χ_i) / J_i                           [원자별]

  전하 보존이 수학적으로 보장됨 (hard constraint).

  U_CENT = Σ_i [ χ_i q_i + ½ J_i q_i² ]

참고:
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
    CENT2 기반 Charge Equilibration Process Block.

    주어진 node features에서 환경 의존 전기음성도 χ_i를 예측하고,
    Lagrange 승수법으로 원자 전하 q_i를 해석적으로 결정한다.
    전하 보존 (Σ q_i = Q_total) 이 항상 수학적으로 보장된다.

    Args:
        irreps_in   : 입력 node features의 irreps (scalar 부분 추출에 사용)
        num_species : 원소 종류 수 (J_i 파라미터 테이블 크기)
        hidden_dim  : χ_i 예측 MLP의 hidden dimension (기본 64)
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        num_species: int,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # scalar(l=0, even parity) 성분 차원
        self.scalar_dim: int = irreps_in.count(o3.Irrep(0, 1))

        # χ_i 예측 MLP (환경 의존 전기음성도)
        self.chi_mlp = tnn.Sequential(
            tnn.Linear(self.scalar_dim, hidden_dim),
            tnn.SiLU(),
            tnn.Linear(hidden_dim, hidden_dim),
            tnn.SiLU(),
            tnn.Linear(hidden_dim, 1),
        )

        # J_i : 원소별 화학적 경도 (학습 파라미터, softplus 로 양수 보장)
        self.J_raw = tnn.Parameter(torch.ones(num_species))

    def forward(
        self,
        node_feats: torch.Tensor,       # [num_nodes, irreps_dim]
        species: torch.Tensor,           # [num_nodes]  원소 인덱스
        total_charge: torch.Tensor,      # [num_graphs] 총 전하 (Q_total)
        batch: torch.Tensor,             # [num_nodes]  배치 인덱스
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            atomic_charges : [num_nodes]  CEP로 결정된 원자 전하 q_i
            chi            : [num_nodes]  예측된 원자 전기음성도 χ_i
            J              : [num_nodes]  원소별 화학적 경도 J_i
            U_CENT         : [num_graphs] CEP 정전기 에너지
            total_charge   : [num_graphs] Σ q_i (보존 검증용 ≈ 입력 Q_total)
        """
        # ── χ_i 예측 ─────────────────────────────────────────────────────
        # scalar(l=0) features 만 추출 (앞쪽에 위치)
        scalar_feats = node_feats[:, :self.scalar_dim]
        chi = self.chi_mlp(scalar_feats).squeeze(-1)          # [N]

        # ── J_i (softplus: 항상 양수) ────────────────────────────────────
        J = tnn.functional.softplus(self.J_raw)[species]      # [N]

        # ── CEP 해석해 (Lagrange 승수) ────────────────────────────────────
        eps: float = 1e-8
        inv_J = 1.0 / (J + eps)                               # [N]

        # 그래프별 집계
        sum_chi_over_J = scatter_sum(
            chi * inv_J, batch, dim=0, dim_size=num_graphs,
        )                                                      # [G]
        sum_inv_J = scatter_sum(
            inv_J, batch, dim=0, dim_size=num_graphs,
        )                                                      # [G]

        # λ = (Q_total + Σ χ/J) / Σ (1/J)
        lam = (total_charge + sum_chi_over_J) / (sum_inv_J + eps)  # [G]
        lam_per_atom = lam[batch]                              # [N]

        # q_i = (λ - χ_i) / J_i  →  hard charge conservation
        q = (lam_per_atom - chi) * inv_J                      # [N]

        # ── U_CENT = Σ_i [ χ_i q_i + ½ J_i q_i² ] ───────────────────────
        U_CENT_per_atom = chi * q + 0.5 * J * q.pow(2)
        U_CENT = scatter_sum(
            U_CENT_per_atom, batch, dim=0, dim_size=num_graphs,
        )                                                      # [G]

        # 보존 검증용
        q_total = scatter_sum(q, batch, dim=0, dim_size=num_graphs)

        return {
            "atomic_charges": q,
            "chi": chi,
            "J": J,
            "U_CENT": U_CENT,
            "total_charge": q_total,
        }
