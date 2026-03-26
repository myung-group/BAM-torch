"""
Charge-dependent model Phase 2 for BAM-torch.

CENT2 기반 CEP (Charge Equilibration Process) 방식:
  - χ_i = MLP(node_feats): 환경 의존 전기음성도 (ANN 예측)
  - J_i = softplus(J_raw[species]): 원소별 화학적 경도 (학습 파라미터)
  - CEP 해석해: Lagrange 승수법으로 hard charge conservation 보장
  - E_total = E_SR (RACE 단거리) + U_CENT (CEP 정전기 에너지)

Phase 1 (MLP Readout, soft conservation) 과의 차이:
  - 전하 보존이 수학적으로 항상 보장 (hard constraint)
  - 전하가 총 에너지에 기여 (U_CENT 항 추가)
  - charge_mode 파라미터 제거 (CEP가 항상 활성화)

참고:
  Khajehpasha et al., Phys. Rev. B 105, 144106 (2022) — CENT2
"""

import torch
import torch.nn.init as init

import e3nn
from e3nn import o3, nn

from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from e3nn.util.jit import compile_mode

from bam_torch.model.blocks import (
    RadialEmbeddingBlock,
    LinearNodeEmbeddingBlock,
    ConcatenateRaceInteractionBlock,
    RaceEquivariantBlock,
    NonLinearReadoutBlock,
    LinearForceDecoderBlock,
    ScaleShiftBlock,
)
from bam_torch.model.wrapper_ops import Linear
from bam_torch.utils.scatter import scatter_sum, scatter_mean
from bam_torch.utils.output_utils import (
    get_outputs,
    get_symmetric_displacement,
    remove_net_torque,
)
from bam_torch.model.models import to_one_hot, get_edge_relative_vectors_with_pbc
from bam_torch.charge_dependent.model.cep_block import CEPBlock


@compile_mode("script")
class ChargeRACE(torch.nn.Module):
    """
    Phase 2 Charge-dependent RACE 모델 (CENT2 기반 CEP).

    총 에너지 분해:
        E_total = E_SR (RACE 단거리) + U_CENT (CEP 정전기)

    CEP Block:
        χ_i  = MLP(scalar node features)      ← 환경 의존 전기음성도
        J_i  = softplus(J_raw[species])        ← 원소별 경도 (학습)
        q_i  = (λ - χ_i) / J_i                ← Lagrange 해석해
        U_CENT = Σ_i [χ_i q_i + ½ J_i q_i²]  ← CEP 에너지

    Args:
        cep_hidden_dim : CEP χ_i MLP hidden dimension (기본 64)
        나머지 인자는 기존 RACE 와 동일.
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        avg_num_neighbors: int = 40,
        num_species: int = 1,
        max_ell: int = 3,
        num_basis_func: int = 8,
        hidden_irreps: e3nn.o3.Irreps = o3.Irreps("32x0e+32x1o+32x2e"),
        nlayers: int = 3,
        features_dim: int = 32,
        output_irreps: e3nn.o3.Irreps = o3.Irreps("1x0e"),
        active_fn: str = "swish",
        radial_MLP: Optional[List[int]] = [64, 64],
        MLP_irreps: e3nn.o3.Irreps = o3.Irreps("16x0e"),
        gate: Optional[Callable] = torch.nn.SiLU(),
        cueq_config: Optional[Dict[str, Any]] = None,
        regress_forces: str = "direct",
        compute_stress: bool = True,
        # CEP 파라미터
        cep_hidden_dim: int = 64,
    ):
        super().__init__()

        if active_fn in ["swish", "silu", "SiLU"]:
            self.act_fn = torch.nn.SiLU()
        elif active_fn in ["relu", "ReLU"]:
            self.act_fn = torch.nn.ReLU()
        elif active_fn in ["identity", None]:
            self.act_fn = torch.nn.Identity()

        self.cutoff = cutoff
        self.regress_forces = regress_forces
        self.compute_stress = compute_stress
        self.num_species = num_species
        self.output_irreps = o3.Irreps(output_irreps)
        hidden_irreps = hidden_irreps.sort().irreps
        self.hidden_irreps = hidden_irreps
        self.nlayers = nlayers

        # Criterion 관련 (RACE 호환성 유지)
        self.criterion = None
        self.criterion_tag = None
        self.criterion_value = 0

        # ── 1) Embedding ──────────────────────────────────────────────────
        node_attr_irreps = o3.Irreps([(num_species, (0, 1))])
        node_feats_irreps = o3.Irreps([(features_dim, (0, 1))])
        x_node_feats_irreps = node_feats_irreps

        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=1.0,
            num_bessel=num_basis_func,
            num_polynomial_cutoff=2,
            radial_type="bessel",
            distance_transform=None,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # ── 2) Interaction layers ─────────────────────────────────────────
        self.linear_x = Linear(
            x_node_feats_irreps,
            x_node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        )
        if radial_MLP is None:
            radial_MLP = [64, 64]

        self.interactions = torch.nn.ModuleList()
        self.products = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()
        self.force_decoders = torch.nn.ModuleList()
        self.stress_decoders = torch.nn.ModuleList()

        target_irreps = o3.Irreps(
            f"{hidden_irreps.count(o3.Irrep(0, 1))}x0e"
        )
        for i in range(nlayers):
            if i > 0:
                node_feats_irreps = hidden_irreps
                target_irreps = hidden_irreps

            inter = ConcatenateRaceInteractionBlock(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=target_irreps,
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
            )
            self.interactions.append(inter)

            prod = RaceEquivariantBlock(
                node_feats_irreps_1=x_node_feats_irreps,
                node_feats_irreps_2=hidden_irreps,
                output_irreps=hidden_irreps,
                use_sc=True,
                cueq_config=cueq_config,
            )
            self.products.append(prod)

            readout = NonLinearReadoutBlock(
                irreps_in=hidden_irreps,
                MLP_irreps="64x0e",
                gate=gate,
                irrep_out=output_irreps,
                cueq_config=cueq_config,
            )
            self.readouts.append(readout)

            if "direct" in self.regress_forces:
                force_decoder = LinearForceDecoderBlock(
                    irreps_in=hidden_irreps,
                    irrep_out="1x1o",
                    cueq_config=cueq_config,
                )
                stress_decoder = LinearForceDecoderBlock(
                    irreps_in=hidden_irreps,
                    irrep_out="6x0e",
                    cueq_config=cueq_config,
                )
            else:
                force_decoder = torch.nn.Identity()
                stress_decoder = torch.nn.Identity()
            self.force_decoders.append(force_decoder)
            self.stress_decoders.append(stress_decoder)

        # ── 3) CEP Block ──────────────────────────────────────────────────
        self.cep = CEPBlock(
            irreps_in=hidden_irreps,
            num_species=num_species,
            hidden_dim=cep_hidden_dim,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        backprop: bool = False,
        compute_displacement: bool = False,
    ):
        data["cell"].requires_grad_(True)
        data["positions"].requires_grad_(True)

        displacement: Optional[torch.Tensor] = None
        if compute_displacement:
            displacement = get_symmetric_displacement(data)

        Rij = get_edge_relative_vectors_with_pbc(data)
        Rij = Rij / self.cutoff
        num_graphs = data["ptr"].numel() - 1

        # ── Atom embedding ────────────────────────────────────────────────
        if "node_attrs" in data:
            node_attrs = data["node_attrs"]
            species = data["species"]
        else:
            species = data["species"]
            node_attrs = to_one_hot(species.unsqueeze(-1), self.num_species)
        node_feats = self.node_embedding(node_attrs)

        # ── Edge embedding ────────────────────────────────────────────────
        edge_index = data["edge_index"]
        lengths = torch.norm(Rij, dim=1)

        nonzero_idx = torch.arange(
            len(lengths), device=lengths.device
        )[lengths != 0]
        Rij = Rij[nonzero_idx]
        lengths = lengths[nonzero_idx]
        edge_index = edge_index[:, nonzero_idx]

        edge_attrs = self.spherical_harmonics(Rij)
        edge_feats = self.radial_embedding(
            lengths.unsqueeze(1),
            node_attrs,
            data["edge_index"],
            species,
        )

        x_node_feats = self.linear_x(node_feats)

        frc_out = []
        sts_out = []
        outputs = []
        node_feats_list: List[torch.Tensor] = []

        # ── Interaction loop ──────────────────────────────────────────────
        for (interaction, product, readout,
             force_decoder, stress_decoder) in zip(
            self.interactions, self.products, self.readouts,
            self.force_decoders, self.stress_decoders
        ):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )
            node_feats = product(
                x_node_feats=x_node_feats,
                node_feats=node_feats,
                sc=sc,
            )
            node_energies = readout(node_feats)

            if "direct" in self.regress_forces:
                node_force_dir = force_decoder(node_feats)
                frc_out.append(node_force_dir)
                node_stress_dir = stress_decoder(node_feats)
                sts_out.append(node_stress_dir)

            node_feats_list.append(node_feats)
            outputs.append(node_energies[:, 0])

        # ── E_SR: 단거리 에너지 합산 ──────────────────────────────────────
        node_energy = torch.stack(outputs, dim=-1)
        node_energy = self.act_fn(node_energy)
        node_energy = torch.sum(node_energy, dim=-1)

        E_SR = scatter_sum(
            src=node_energy,
            index=data["batch"],
            dim=-1,
            dim_size=num_graphs,
        )

        # ── CEP: 전하 결정 + U_CENT 계산 ──────────────────────────────────
        last_node_feats = node_feats_list[-1]

        # total_charge 없으면 0 으로 대체
        if "total_charge" in data:
            total_charge = data["total_charge"].float()
        else:
            total_charge = torch.zeros(
                num_graphs, dtype=torch.float32, device=E_SR.device
            )

        cep_out = self.cep(
            node_feats=last_node_feats,
            species=species,
            total_charge=total_charge,
            batch=data["batch"],
            num_graphs=num_graphs,
        )

        # ── E_total = E_SR + U_CENT ───────────────────────────────────────
        graph_energy = E_SR + cep_out["U_CENT"]

        preds: Dict[str, Optional[torch.Tensor]] = {}
        preds["energy"] = graph_energy
        preds["node_energy"] = node_energy
        preds["atomic_charges"] = cep_out["atomic_charges"]
        preds["total_charge"] = cep_out["total_charge"]
        preds["chi"] = cep_out["chi"]
        preds["U_CENT"] = cep_out["U_CENT"]
        preds["E_SR"] = E_SR

        # ── Forces ────────────────────────────────────────────────────────
        forces: Optional[torch.Tensor] = None
        stress: Optional[torch.Tensor] = None

        if self.criterion is not None:
            if self.criterion < self.criterion_value:
                self.regress_forces = "auto"
            else:
                self.regress_forces = "direct"

        if "auto" in self.regress_forces:
            forces, virials, stress, hessian = get_outputs(
                energy=graph_energy,
                positions=data["positions"],
                cell=data["cell"],
                batch_idx=data["batch"],
                num_graphs=num_graphs,
                training=backprop,
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
            node_force = torch.stack(frc_out, dim=-1)
            node_force = self.act_fn(node_force)
            forces = torch.sum(node_force, dim=-1)
            system_means = scatter_mean(forces, data["batch"], dim=0)
            node_broadcasted_means = system_means[data["batch"]]
            forces = forces - node_broadcasted_means
            forces = remove_net_torque(
                data["positions"], forces, data["batch"]
            )

            node_stress = torch.stack(sts_out, dim=-1)
            node_stress = self.act_fn(node_stress)
            stress = torch.sum(node_stress, dim=-1)
            stress = scatter_sum(
                src=stress,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )
            preds["forces"] = forces
            preds["stress"] = stress

        preds["displacement"] = displacement

        return preds

    def set_criterion(self, criterion_tag, criterion):
        self.criterion_tag = criterion_tag
        if "direct" in self.regress_forces:
            if criterion_tag is None:
                criterion_tag = "epoch"

        self.criterion = criterion
        if criterion_tag == "epoch":
            if criterion is None:
                self.criterion = 50
                self.criterion_value = 0
        elif criterion_tag == "loss":
            if criterion is None:
                self.criterion = 0.01
                self.criterion_value = 0.1

        self.criterion_value = 0

    def update_criterion_value(self, value):
        self.criterion_value = value
