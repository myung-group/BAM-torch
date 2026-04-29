import torch
import torch.nn.init as init

import e3nn
from e3nn import o3, nn

from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from e3nn.util.jit import compile_mode
from torch.jit import annotate
from .blocks import (
    RadialEmbeddingBlock,
    LinearNodeEmbeddingBlock,
    ConcatenateRaceInteractionBlock,
    RaceInteractionBlock,
    RaceEquivariantBlock,
    NonLinearReadoutBlock,
    LinearForceDecoderBlock,
    AgnosticResidualNonlinearInteractionBlock,
    EquivariantProductBasisBlock,
    LinearReadoutBlock,
    ScaleShiftBlock,
)


# Maps the `interaction_block` config string to a class. "slow" is the
# JAX-style FullTensorProduct path (e3nn-only); "fast" is the weighted-channel
# TensorProduct path (CUET / OEQ-capable).
_RACE_INTERACTION_BLOCKS = {
    "slow": ConcatenateRaceInteractionBlock,
    "fast": RaceInteractionBlock,
}


def _resolve_race_interaction_block(name: str):
    try:
        return _RACE_INTERACTION_BLOCKS[name]
    except KeyError:
        valid = ", ".join(sorted(_RACE_INTERACTION_BLOCKS))
        raise ValueError(
            f"Unknown interaction_block {name!r}. Valid options: {valid}"
        )
from .wrapper_ops import Linear
from bam_torch.utils.scatter import scatter_sum, scatter_mean
from bam_torch.utils.output_utils import (
    get_outputs, 
    get_symmetric_displacement,
    remove_net_torque
)


@compile_mode("script")
class RACE(torch.nn.Module):
    """Restratification Atomic Cluster Expansion (RACE) model
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
        oeq_config: Optional[Dict[str, Any]] = None,
        regress_forces: str = "direct",
        compute_stress: bool = True,
        l_separated_layer_norm: bool = False,
        interaction_block: str = "slow",
        radial_polynomial_p: int = 2,
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
        interaction_cls = _resolve_race_interaction_block(interaction_block)

        ## 1) Embedding
        # Node embedding
        node_attr_irreps = o3.Irreps([(num_species, (0, 1))])
        node_feats_irreps = o3.Irreps([(features_dim, (0, 1))])
        if interaction_block in ["slow"]:
            x_node_feats_irreps = node_feats_irreps
        else:
            x_node_feats_irreps = o3.Irreps([(8, (0, 1))])

        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        ) # [n_nodes, irreps]

        # Radial embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=1.0,
            num_bessel=num_basis_func,
            num_polynomial_cutoff=radial_polynomial_p,
            radial_type="bessel",
            distance_transform=None,
        )
        # Edge embedding
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell) # interaction_irreps in JAX
        #num_features = hidden_irreps.count(o3.Irrep(0, 1))
        #interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(sh_irreps, 
                                                         normalize=True,
                                                         normalization="component")
        
        ## 2) Interaction layer  # RealAgnosticInteractionBlock
        self.linear_x = Linear(
            node_feats_irreps,
            x_node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        ) # x_node_feats
        if radial_MLP is None:
            radial_MLP = [64, 64]

        self.interactions = torch.nn.ModuleList()
        self.products = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()
        self.force_decoders = torch.nn.ModuleList()
        self.stress_decoders = torch.nn.ModuleList()

        target_irreps = o3.Irreps(f"{hidden_irreps.count(o3.Irrep(0, 1))}x0e")
        for i in range(nlayers):
            if i > 0: 
                node_feats_irreps = hidden_irreps
                target_irreps = hidden_irreps

            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=target_irreps,  # interaction_irreps
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
                l_separated_layer_norm=l_separated_layer_norm,
            )
            self.interactions.append(inter)

            prod = RaceEquivariantBlock(
                node_feats_irreps_1=x_node_feats_irreps,  # x_node_feats
                node_feats_irreps_2=hidden_irreps,  # node_feats
                output_irreps=hidden_irreps,      # hidden_irreps
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
                biases=True
            )
            self.readouts.append(readout) # [n_nodes, output_irreps.count(o3.Irrep(0, 1))]

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

        #self.emb = torch.nn.Embedding(num_embeddings=num_species, embedding_dim=num_species)
    
    def forward(
            self, 
            data: Dict[str, torch.Tensor], 
            backprop: bool = False,
            compute_displacement: bool = False
    ):
        # assert Rij.ndim == 2 and Rij.shape[1] == 3
        # iatoms ==> senders     # edge_index[0]
        # jatoms ==> receivers   # edge_index[1]
        data["cell"].requires_grad_(True)
        data["positions"].requires_grad_(True)

        displacement: Optional[torch.Tensor] = None
        if compute_displacement:
            displacement = get_symmetric_displacement(data)

        Rij = get_edge_relative_vectors_with_pbc(data)
        #else:
        #    Rij = get_edge_relative_vectors_with_pbc_padding(R, cell, data)
        Rij = Rij / self.cutoff
        num_graphs = data["ptr"].numel() - 1  # nbatch

        # Embedding
        if "node_attrs" in data:
            node_attrs = data["node_attrs"]  # Pre-calculated in C++
            species = data["species"]
        else:
            species = data["species"]
            node_attrs = to_one_hot(species.unsqueeze(-1), self.num_species)
        node_feats = self.node_embedding(node_attrs)

        edge_index = data["edge_index"]
        lengths = torch.norm(Rij, dim=1)

        nonzero_idx = torch.arange(len(lengths), device=lengths.device)[lengths != 0]
        Rij = Rij[nonzero_idx]
        lengths = lengths[nonzero_idx]
        edge_index = edge_index[:, nonzero_idx]
        
        edge_attrs = self.spherical_harmonics(Rij)
        edge_feats = self.radial_embedding(lengths.unsqueeze(1), 
                                           node_attrs,
                                           data["edge_index"],
                                           species)
#        ###
#        i_sp = species[data["edge_index"][0]]
#        j_sp = species[data["edge_index"][1]]
#        sp = (i_sp + j_sp) / 2
#        edge_feats = edge_feats * sp[:, None]

        x_node_feats = self.linear_x(node_feats)

        frc_out = []
        sts_out = []                                 
        outputs = []
        node_logvar = [] 
        node_f_logvar = [] 
        node_feats_list = []
        for interaction, product, readout, force_decoder, stress_decoder in zip(
                self.interactions, self.products, self.readouts, self.force_decoders, self.stress_decoders
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
            node_energies = readout(node_feats) # [n_nodes, len(heads)]  == [nbatch*num_nodes, "1x0e" or "2x0e"]

            if "direct" in self.regress_forces:
                l_0_dim = 0
                l_1_dim = 0
                for mul, (l, p) in self.hidden_irreps:
                    if str(l) == "0":
                        l_0_dim += mul
                    elif str(l) == "1":
                        l_1_dim += mul

                node_force_dir = force_decoder(node_feats)
                node_forces = node_force_dir
                frc_out.append(node_forces)
                node_stress_dir = stress_decoder(node_feats) 
                node_stresses = node_stress_dir 
                sts_out.append(node_stresses)

            node_feats_list.append(node_feats)
            outputs.append(node_energies[:,0])
            if str(self.output_irreps) == "2x0e":
                node_logvar.append(node_energies[:,1])
            elif str(self.output_irreps) == "8x0e":
                node_logvar.append(node_energies[:,1])
                node_f_logvar.append(node_energies[:,2:])

        # Sum over energy contributions
        node_energy = torch.stack(outputs, dim=-1) # [nbatch*num_nodes, nlayers]
        node_energy = self.act_fn(node_energy)

        # Global pooling
        node_energy = torch.sum(node_energy, dim=-1) # [nbatch*num_nodes]  # total_energy
        graph_energy = scatter_sum(
                src=node_energy,
                index=data["batch"],
                dim=-1,
                dim_size=num_graphs,
            )

        node_logvar_ts = torch.zeros(node_feats.shape[0], device=node_energy.device)
        node_f_logvar_ts = torch.zeros((node_feats.shape[0], 6), device=node_energy.device)
        # Uncertainty quantification
        if str(self.output_irreps) == "8x0e":
            node_logvar_ts = torch.stack(node_logvar, dim=-1) # [nbatch*num_nodes, nlayers]
            node_logvar_ts = node_logvar_ts.mean(dim=-1) # [nbatch*num_nodes]
            # force variance L Voigt notation - xx, yy, zz, yz, xz, xy
            node_f_logvar_ts = torch.stack(node_f_logvar, dim=-1) # [nbatch*num_nodes, 6, nlayers]
            node_f_logvar_ts = node_f_logvar_ts.mean(dim=-1) # [nbatch*num_nodes, 6]
        elif str(self.output_irreps) == "2x0e":
            node_logvar_ts = torch.stack(node_logvar, dim=-1) # [nbatch*num_nodes, nlayers]
            node_logvar_ts = node_logvar_ts.mean(dim=-1) # [nbatch*num_nodes]
            node_f_logvar_ts = torch.zeros((node_feats.shape[0], 6), 
                                           device=node_energy.device)
        elif str(self.output_irreps) == "1x0e":
            node_logvar_ts = torch.zeros(node_feats.shape[0], 
                                         device=node_energy.device)
            node_f_logvar_ts = torch.zeros((node_feats.shape[0], 6), 
                                           device=node_energy.device)
        # Eenrgy variance
        graph_logvar = scatter_mean(
                src=node_logvar_ts,
                index=data["batch"],
                dim=-1,
                dim_size=num_graphs,
            )
        graph_energy_var = torch.exp(graph_logvar) 

        # Forces variance
        node_frc_var = torch.cat(
            [torch.exp(node_f_logvar_ts[:, :3]), node_f_logvar_ts[:, 3:]], 
            dim=1
        ).view(-1, 6)

        preds: Dict[str, Optional[torch.Tensor]] = {}
        preds["energy"] = graph_energy # total energy
        preds["energy_var"] = graph_energy_var
        preds["forces_var"] = node_frc_var
        preds["node_energy"] = node_energy

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
                displacement=None
            )
            preds["forces"] = forces
            preds["stress"] = stress
            preds["virials"] = virials
 
        elif "direct" in self.regress_forces:
            node_force = torch.stack(frc_out, dim=-1) # [nbatch*num_nodes, nlayers]
            node_force = self.act_fn(node_force)
            forces = torch.sum(node_force, dim=-1) # [nbatch*num_nodes]  # total_energy
            system_means = scatter_mean(forces, data["batch"], dim=0)
            node_boradcasteds_means = system_means[data["batch"]]
            forces = forces - node_boradcasteds_means
            forces = remove_net_torque(data["positions"], forces, data["batch"])

            node_stress = torch.stack(sts_out, dim=-1) # [nbatch*num_nodes, 6, nlayers]
            node_stress = self.act_fn(node_stress)
            stress = torch.sum(node_stress, dim=-1) # [nbatch*num_nodes, 6]  # total_energy
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
            if criterion_tag == None:
                criterion_tag = "epoch" 
        
        self.criterion = criterion
        if criterion_tag == "epoch":
            if criterion == None:
                self.criterion = 50
                self.criterion_value = 0
        elif criterion_tag == "loss":
            if criterion == None:
                self.criterion = 0.01
                self.criterion_value = 0.1
                
        self.criterion_value = 0
    
    def update_criterion_value(self, value):
        self.criterion_value = value


class RACEUnified(torch.nn.Module):
    """
    Unified RACE Model - Supports both single-head and multihead modes
    
    When heads=["default"] (single element): behaves like RACE
    When heads=["target", "replay", ...] (multiple elements): behaves like RACEMultihead
    
    This unified class eliminates code duplication between RACE and RACEMultihead.
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
        MLP_irreps: e3nn.o3.Irreps = o3.Irreps("64x0e"),
        gate: Optional[Callable] = torch.nn.SiLU(),
        cueq_config: Optional[Dict[str, Any]] = None,
        oeq_config: Optional[Dict[str, Any]] = None,
        regress_forces: str = "direct",
        compute_stress: bool = True,
        heads: Optional[List[str]] = None,  # ⭐ Multihead support
        l_separated_layer_norm: bool = False,
        interaction_block: str = "slow",
        radial_polynomial_p: int = 2,
    ):
        super().__init__()
    
        # ⭐ Determine single-head or multi-head mode based on the number of heads
        if heads is None:
            heads = ["default"]
        self.heads = heads
        self.num_heads = len(heads)
        self.is_multihead = self.num_heads > 1
        
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
        interaction_cls = _resolve_race_interaction_block(interaction_block)

        # Initialize criterion attributes (compatibility with RACE)
        self.criterion = None
        self.criterion_tag = None
        self.criterion_value = 0
        
        ## 1) Embedding
        node_attr_irreps = o3.Irreps([(num_species, (0, 1))])
        node_feats_irreps = o3.Irreps([(features_dim, (0, 1))])
        if interaction_block in ["slow"]:
            x_node_feats_irreps = node_feats_irreps
        else:
            x_node_feats_irreps = o3.Irreps([(8, (0, 1))])
        
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=1.0,
            num_bessel=num_basis_func,
            num_polynomial_cutoff=radial_polynomial_p,
            radial_type="bessel",
            distance_transform=None,
        )

        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        #num_features = hidden_irreps.count(o3.Irrep(0, 1))
        #interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, 
            normalize=True,
            normalization="component"
        )
        
        ## 2) Interaction layers
        self.linear_x = Linear(
            node_feats_irreps,
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

        target_irreps = o3.Irreps(f"{hidden_irreps.count(o3.Irrep(0, 1))}x0e")
        
        # ⭐ Multi-head-specific MLP and output irreps (enabled conditionally)
        if self.is_multihead:
            multihead_MLP_irreps = (self.num_heads * MLP_irreps).simplify()
            multihead_output_irreps = o3.Irreps(f"{self.num_heads}x0e")
        else:
            MLP_irreps = o3.Irreps("64x0e")
            multihead_MLP_irreps = MLP_irreps
            multihead_output_irreps = self.output_irreps
        
        for i in range(nlayers):
            if i > 0: 
                node_feats_irreps = hidden_irreps
                target_irreps = hidden_irreps

            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=target_irreps,
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
                l_separated_layer_norm=l_separated_layer_norm,
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

            # Readout: NonLinear on every layer for both single-head and
            # multihead so the parameter structure matches a RACE foundation
            # (NonLinearReadoutBlock at every layer).
            if self.is_multihead:
                readout = NonLinearReadoutBlock(
                    irreps_in=hidden_irreps,
                    MLP_irreps=multihead_MLP_irreps,
                    gate=gate,
                    irrep_out=multihead_output_irreps,
                    num_heads=self.num_heads,
                    cueq_config=cueq_config,
                    biases=True,
                )
            else:
                readout = NonLinearReadoutBlock(
                    irreps_in=hidden_irreps,
                    MLP_irreps=MLP_irreps,
                    gate=gate,
                    irrep_out=self.output_irreps,
                    cueq_config=cueq_config,
                    biases=True,
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

    def forward(
            self, 
            data: Dict[str, torch.Tensor], 
            backprop: bool = False,
            compute_displacement: bool = False
    ):
        data["cell"].requires_grad_(True)
        data["positions"].requires_grad_(True)

        displacement: Optional[torch.Tensor] = None
        if compute_displacement:
            displacement = get_symmetric_displacement(data)

        Rij = get_edge_relative_vectors_with_pbc(data)
        Rij = Rij / self.cutoff
        num_graphs = data["ptr"].numel() - 1

        # ⭐ Extract head information (used only in multi-head mode)
        if self.is_multihead:
            node_heads = (
                data["head"][data["batch"]]
                if "head" in data
                else torch.zeros_like(data["batch"])
            )
        else:
            node_heads = None

        # Embedding
        if "node_attrs" in data:
            node_attrs = data["node_attrs"]
            species = data["species"]
        else:
            species = data["species"]
            node_attrs = to_one_hot(species.unsqueeze(-1), self.num_species)
        node_feats = self.node_embedding(node_attrs)

        edge_index = data["edge_index"]
        lengths = torch.norm(Rij, dim=1)

        nonzero_idx = torch.arange(len(lengths), device=lengths.device)[lengths != 0]
        Rij = Rij[nonzero_idx]
        lengths = lengths[nonzero_idx]
        edge_index = edge_index[:, nonzero_idx]
        
        edge_attrs = self.spherical_harmonics(Rij)
        edge_feats = self.radial_embedding(
            lengths.unsqueeze(1), 
            node_attrs,
            data["edge_index"],
            species
        )

        x_node_feats = self.linear_x(node_feats)
        num_atoms_arange = torch.arange(node_attrs.shape[0], device=node_attrs.device)

        frc_out = []
        sts_out = []                                 
        outputs = []
        node_feats_list = []
        
        for interaction, product, readout, force_decoder, stress_decoder in zip(
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
            
            # ⭐ Readout: multihead vs singlehead branching
            if self.is_multihead:
                node_energies = readout(node_feats, node_heads)  # [N_atoms, num_heads]
                # Select only the energy corresponding to each atom's assigned head
                node_energies_selected = node_energies[num_atoms_arange, node_heads]
            else:
                node_energies = readout(node_feats)  # [N_atoms, output_dim]
                node_energies_selected = node_energies[:, 0]
            
            outputs.append(node_energies_selected)

            if "direct" in self.regress_forces:
                node_force_dir = force_decoder(node_feats)
                frc_out.append(node_force_dir)
                node_stress_dir = stress_decoder(node_feats) 
                sts_out.append(node_stress_dir)

            node_feats_list.append(node_feats)

        # Sum over energy contributions
        node_energy = torch.stack(outputs, dim=-1)
        node_energy = self.act_fn(node_energy)
        node_energy = torch.sum(node_energy, dim=-1)

        graph_energy = scatter_sum(
            src=node_energy,
            index=data["batch"],
            dim=-1,
            dim_size=num_graphs,
        )

        preds: Dict[str, Optional[torch.Tensor]] = {}
        preds["energy"] = graph_energy
        preds["node_energy"] = node_energy

        forces: Optional[torch.Tensor] = None
        stress: Optional[torch.Tensor] = None

        # Criterion check for force regression mode switching
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
                displacement=None
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
            forces = remove_net_torque(data["positions"], forces, data["batch"])

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
        """Set criterion for direct force training (compatibility with RACE)"""
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
        """Update criterion value (compatibility with RACE)"""
        self.criterion_value = value

    @classmethod
    def from_foundation(cls, foundation_model: "RACE", heads: List[str]):
        """
        Create RACEUnified from a trained RACE foundation model.
        
        Args:
            foundation_model: Trained single-head RACE model
            heads: List of head names (e.g., ["target", "replay"])
            
        Returns:
            RACEUnified with expanded readout weights
        """
        # Extract config from foundation model
        config = {
            "cutoff": foundation_model.cutoff,
            "num_species": foundation_model.num_species,
            "hidden_irreps": foundation_model.hidden_irreps,
            "nlayers": foundation_model.nlayers,
            "regress_forces": foundation_model.regress_forces,
            "compute_stress": foundation_model.compute_stress,
            "heads": heads,
        }
        
        # Create unified model
        unified_model = cls(**config)
        
        # Copy weights from foundation
        foundation_state = foundation_model.state_dict()
        unified_state = unified_model.state_dict()
        
        num_heads = len(heads)
        
        for name, param in foundation_state.items():
            if name in unified_state:
                if 'readout' in name:
                    # Expand readout weights for multihead
                    if 'linear_1.weight' in name:
                        # Replicate for each head
                        unified_state[name] = param.repeat(num_heads, 1)
                    elif 'linear_2.weight' in name:
                        # Block diagonal expansion
                        old_out, old_in = param.shape
                        new_weight = torch.zeros(
                            old_out * num_heads, 
                            old_in * num_heads,
                            dtype=param.dtype,
                            device=param.device
                        )
                        for h in range(num_heads):
                            new_weight[
                                h * old_out:(h + 1) * old_out,
                                h * old_in:(h + 1) * old_in
                            ] = param
                        unified_state[name] = new_weight
                    elif 'bias' in name:
                        unified_state[name] = param.repeat(num_heads)
                    else:
                        unified_state[name] = param
                else:
                    # Direct copy for non-readout params
                    if unified_state[name].shape == param.shape:
                        unified_state[name] = param
        
        unified_model.load_state_dict(unified_state)
        return unified_model

    
@compile_mode("script")
class MACE(torch.nn.Module):
    """Base model of E(3) Equivariant Graph Neural Network, based on e3nn 
    """
    def __init__(
        self,
        cutoff: float = 6.0, 
        avg_num_neighbors: int = 40, 
        num_species: int = 1, 
        max_ell: int = 3,
        num_basis_func: int = 8,
        hidden_irreps: e3nn.o3.Irreps = o3.Irreps("32x0e+8x1o+4x2e"),
        nlayers: int = 3,
        features_dim: int = 128,
        output_irreps: e3nn.o3.Irreps = o3.Irreps("1x0e"),
        active_fn: str = "swish",
        radial_MLP: Optional[List[int]] = [64, 64, 64],
        correlation: Union[int, List[int]] = 3,
        heads: Optional[List[str]] = None,
        MLP_irreps: e3nn.o3.Irreps = o3.Irreps("16x0e"),
        gate: Optional[Callable] = torch.nn.SiLU(),
        cueq_config: Optional[Dict[str, Any]] = None,
        oeq_config: Optional[Dict[str, Any]] = None,
        regress_forces: str = "direct",
        radial_polynomial_p: int = 6,
    ):
        super().__init__()

        if active_fn in ["swish", "silu", "SiLU"]:
            self.act_fn = torch.nn.SiLU()
        elif active_fn in ["relu", "ReLU"]:
            self.act_fn = torch.nn.ReLU()
        elif active_fn in ["identity", None]:
            self.act_fn = None   # Need to modify later
        
        self.cutoff = cutoff
        self.regress_forces = regress_forces
        self.num_species = num_species
        
        if heads is None:
            heads = ["default"]
        self.heads = heads
        atomic_inter_scale = [1.0] * len(heads)
        atomic_inter_shift = [0.0] * len(heads) # determine_atomic_inter_shift(args.mean, heads)
        # mean: Mean energy per atom of training set

        if isinstance(correlation, int):
            correlation = [correlation] * nlayers

        ## 1) Embedding
        # Node embedding
        node_attr_irreps = o3.Irreps([(num_species, (0, 1))])
        #node_feats_irreps = o3.Irreps([(features_dim, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        ) # [n_nodes, irreps]
 
        # Radial embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=1.0,
            num_bessel=num_basis_func,
            num_polynomial_cutoff=radial_polynomial_p,
            radial_type="bessel",
            distance_transform=None,
        )
        # Edge embedding
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell) # interaction_irreps in JAX
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(sh_irreps, 
                                                         normalize=True,
                                                         normalization="component")
        
        ## 2) Interaction layer
        inter = AgnosticResidualNonlinearInteractionBlock(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(inter):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_species,
            use_sc=use_sc_first,
            cueq_config=cueq_config,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(
            LinearReadoutBlock(
                hidden_irreps, output_irreps, cueq_config
            ) # hidden_irreps, o3.Irreps(f"{len(heads)}x0e"), cueq_config
        )  # if heads == ['default'].
           # o3.Irreps(f"{len(heads)}x0e") == e3nn.Irreps("1x0e") 
           # default of output_irreps in BAM-jax
           # [n_nodes, 1]

        self.interactions = torch.nn.ModuleList([inter])
        for i in range(nlayers-1):
            if i == nlayers - 2:  # if last
                hidden_irreps_out = str(hidden_irreps[0]) # o3.Irreps("1x0e") # 
            else:
                hidden_irreps_out = hidden_irreps 

            inter = AgnosticResidualNonlinearInteractionBlock(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_species,
                use_sc=True,
                cueq_config=cueq_config,
            )
            self.products.append(prod)
            if i == nlayers - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        output_irreps, # o3.Irreps(f"{len(heads)}x0e")
                        len(heads),
                        cueq_config,
                        biases=True,
                    )
                ) # [n_nodes, len(heads)]
            else:
                self.readouts.append(
                    LinearReadoutBlock(
                        hidden_irreps, output_irreps, cueq_config
                    ) # hidden_irreps, o3.Irreps(f"{len(heads)}x0e"), cueq_config
                )  # [n_nodes, 1]
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )
    
    def forward(self, data, backprop):

        #assert Rij.ndim == 2 and Rij.shape[1] == 3
        # iatoms ==> senders     # edge_index[0]
        # jatoms ==> receivers   # edge_index[1]
        R = data["positions"]
        R.requires_grad_(True)
        Rij = get_edge_relative_vectors_with_pbc(data)
        Rij = Rij / self.cutoff
        node_heads = (
            data["head"][data["batch"]]
            if "head" in data
            else torch.zeros_like(data["batch"])
        )
        num_graphs = data["ptr"].numel() - 1  # nbatch
        # num_atoms_arange = torch.arange(data.positions.shape[0])
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data.positions.dtype,
            device=data.positions.device,
        )
        # Embedding
        species = data.species.unsqueeze(-1)
        node_attrs = to_one_hot(species, self.num_species)
        node_feats = self.node_embedding(node_attrs)

        edge_index = data.edge_index
        lengths = torch.norm(Rij, dim=1)

        nonzero_idx = torch.arange(len(lengths), device=lengths.device)[lengths != 0]
        Rij = Rij[nonzero_idx]
        lengths = lengths[nonzero_idx]
        edge_index = edge_index[:, nonzero_idx]
        
        edge_attrs = self.spherical_harmonics(Rij)
        edge_feats = self.radial_embedding(lengths.unsqueeze(1), 
                                           node_attrs, 
                                           data.edge_index,
                                           species)
        outputs = []
        node_logvar = []
        node_feats_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs,
            )
            node_feats_list.append(node_feats)
            node_energies = readout(node_feats, node_heads)#[
            #    num_atoms_arange, node_heads
            #]  # [n_nodes, len(heads)]  == [nbatch*num_nodes, ]
            
            outputs.append(node_energies[:,0])
            if node_energies.shape[1] == 2:
                node_logvar.append(node_energies[:,1])
            
        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)
        """
        # Sum over energy contributions
        contributions = torch.stack(outputs, dim=0) # [nlayers, nbatch*num_nodes]
        node_energy = torch.sum(contributions, dim=0) # [nbatch*num_nodes]  # total_energy
        # node_energy = self.scale_shift(node_energy, node_heads) 
        """
        # Sum over energy contributions
        node_energy = torch.stack(outputs, dim=-1) # [nbatch*num_nodes, nlayers]
        node_energy = torch.sum(node_energy, dim=-1) # [nbatch*num_nodes]  # total_energy
        
        node_energy = scatter_sum(
                src=node_energy,   # node_energies
                index=data["batch"],
                dim=-1,
                dim_size=num_graphs,
            ) 
        
        if node_logvar != []:
            node_logvar = torch.stack(node_logvar, dim=-1) # [nbatch*num_nodes, nlayers]
            node_logvar = node_logvar.mean(dim=-1) # [nbatch*num_nodes]
        else:
            node_logvar = torch.zeros(node_feats.shape[0], device=node_energy.device)
        node_energy_var = torch.exp(node_logvar) 
        node_energy_var = scatter_sum(
                src=node_energy_var,
                index=data["batch"],
                dim=-1,
                dim_size=num_graphs,
            ) 
        n_node = int(data.num_nodes / num_graphs)
        n_nodes = torch.tensor([n_node]*num_graphs, device=node_energy_var.device)
        energy_var = node_energy_var/n_nodes

        preds = {}
        preds["energy"] = node_energy
        preds["energy_var"] = energy_var

        if self.regress_forces == 'direct' or self.regress_forces:
            forces, virials, stress, hessian = get_outputs(
                energy=node_energy,
                positions=R,
                displacement=displacement,
                batch_idx=data["batch"],
                num_graphs=num_graphs,
                cell=data.cell,
                training=backprop,
                compute_force=True,
                compute_virials=False,
                compute_stress=False,
                compute_hessian=False
            )
            preds["forces"] = forces

        return preds


def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Generates one-hot encoding with <num_classes> classes from <indices>
    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :param device: torch device
    :return: (N x num_classes) tensor
    """
    #shape = indices.shape[:-1] + (num_classes,)
    shape: List[int] = list(indices.shape[:-1]) + [num_classes]
    oh = torch.zeros(shape, device=indices.device) #.view(shape)

    # scatter_ is the in-place version of scatter
    #oh.scatter_(dim=-1, index=indices, value=1)
    return oh.scatter_(-1, indices, 1.0)
    #return oh.view(*shape)  ## similar with torch.nn.Embedding
    

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
    

def get_edge_relative_vectors_with_pbc_lammps(
    data: Dict[str, torch.Tensor],
) -> torch.Tensor:
    R = data["positions"]
    cell = data["cell"]
    cell = cell.unsqueeze(0)
    iatoms = data["edge_index"][0]  # shape = (b * n_edges)
    jatoms = data["edge_index"][1]  # shape = (b * n_edges) 
    Sij = data["unit_shifts"]   # shape = (b * n_edges, 3)
    n_edges: List[int] = [data["unit_shifts"].shape[0]]
    
    Sij = torch.split(Sij, n_edges, dim=0)
    shift_v = torch.cat(
        [torch.einsum('ni,ij->nj', s, c)
            for s, c in zip(Sij, cell)], dim=0
    )
    _R = R[jatoms] - R[iatoms] 
    Rij = _R + shift_v

    return Rij


