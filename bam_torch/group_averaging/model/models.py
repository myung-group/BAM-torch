import torch
import torch.nn.init as init

import e3nn
from e3nn import o3, nn

from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from e3nn.util.jit import compile_mode
from torch.jit import annotate
from bam_torch.model.blocks import (
    RadialEmbeddingBlock,
    LinearNodeEmbeddingBlock,
    ConcatenateRaceInteractionBlock,
    RaceEquivariantBlock,
    NonLinearReadoutBlock,
    LinearForceDecoderBlock,
    AgnosticResidualNonlinearInteractionBlock,
    EquivariantProductBasisBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    ScaleShiftBlock,
    NonLinearXReadoutBlock,
    NonLinearXForceReadoutBlock
)
from bam_torch.model.wrapper_ops import Linear
from bam_torch.utils.scatter import scatter_sum, scatter_mean
from bam_torch.utils.output_utils import (
    get_outputs, 
    get_symmetric_displacement,
    remove_net_torque
)


@compile_mode("script")
class GARACE_V2_R_DF(torch.nn.Module):
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
        regress_forces: str = "direct",
        compute_stress: bool = True
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
        
        ## 1) Embedding
        # Node embedding
        node_attr_irreps = o3.Irreps([(num_species, (0, 1))])
        node_feats_irreps = o3.Irreps([(features_dim, (0, 1))])
        x_node_feats_irreps = node_feats_irreps

        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        ) # [n_nodes, irreps]

        # Radial embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=1.0,
            num_bessel=num_basis_func,
            num_polynomial_cutoff=2,   # default of BAM-jax
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
        
        ## 2) Interaction layer  # RealAgnosticInteractionBlock
        self.linear_x = Linear(
            x_node_feats_irreps,
            x_node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        ) # x_node_feats
        self.x_readout = NonLinearReadoutBlock(
                irreps_in=x_node_feats_irreps,
                MLP_irreps="64x0e",
                gate=gate,
                irrep_out="1x0e",
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
        for i in range(nlayers):
            if i > 0: 
                node_feats_irreps = hidden_irreps
                target_irreps = hidden_irreps

            inter = ConcatenateRaceInteractionBlock(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=target_irreps,  # interaction_irreps
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
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
            )
            self.readouts.append(readout) # [n_nodes, output_irreps.count(o3.Irrep(0, 1))]

            if "direct" in self.regress_forces:
                force_decoder = NonLinearXForceReadoutBlock(
                    irreps_in=hidden_irreps,
                    MLP_irreps="64x0e+64x1o+64x2e",
                    gate=[gate,None,None],
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
        #data["cell"].requires_grad_(True)
        #data["positions"].requires_grad_(True)
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
        x_energy = self.x_readout(x_node_feats)

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
            outputs.append(node_energies[:,0] + x_energy.squeeze())
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
            preds["forces_grad_target"] = forces
            preds["stress_grad_target"] = stress

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

        preds["displacement"] = displacement

        return preds


@compile_mode("script")
class GARACE_V2_G_B(torch.nn.Module):
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
        regress_forces: str = "direct",
        compute_stress: bool = True
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
        
        ## 1) Embedding
        # Node embedding
        node_attr_irreps = o3.Irreps([(num_species, (0, 1))])
        node_feats_irreps = o3.Irreps([(features_dim, (0, 1))])
        x_node_feats_irreps = node_feats_irreps

        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        ) # [n_nodes, irreps]

        # Radial embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=1.0,
            num_bessel=num_basis_func,
            num_polynomial_cutoff=2,   # default of BAM-jax
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
        
        ## 2) Interaction layer  # RealAgnosticInteractionBlock
        self.linear_x = Linear(
            x_node_feats_irreps,
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
        self.x_readouts = torch.nn.ModuleList()

        target_irreps = o3.Irreps(f"{hidden_irreps.count(o3.Irrep(0, 1))}x0e")
        for i in range(nlayers):
            if i > 0: 
                node_feats_irreps = hidden_irreps
                target_irreps = hidden_irreps

            x_readout = NonLinearXReadoutBlock(
                            irreps_in=x_node_feats_irreps,
                            MLP_irreps="64x0e+64x1o+64x2e",
                            gate=[gate,None,None],
                            irrep_out="1x0e",
                            cueq_config=cueq_config,
                        )
            self.x_readouts.append(x_readout)
            
            inter = ConcatenateRaceInteractionBlock(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=target_irreps,  # interaction_irreps
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
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

            readout = NonLinearXReadoutBlock(
                irreps_in=hidden_irreps,
                MLP_irreps="64x0e+64x1o+64x2e",
                gate=[gate,None,None],
                irrep_out="1x0e",
                cueq_config=cueq_config,
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
        #data["cell"].requires_grad_(True)
        #data["positions"].requires_grad_(True)
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
        for interaction, product, readout, force_decoder, stress_decoder, x_readout in zip(
                self.interactions, self.products, self.readouts, self.force_decoders, self.stress_decoders, self.x_readouts
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

            x_energy = x_readout(x_node_feats)

            node_feats_list.append(node_feats)
            outputs.append(node_energies[:,0] + x_energy.squeeze())
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




@compile_mode("script")
class GARACE_V2_G(torch.nn.Module):
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
        regress_forces: str = "direct",
        compute_stress: bool = True
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
        
        ## 1) Embedding
        # Node embedding
        node_attr_irreps = o3.Irreps([(num_species, (0, 1))])
        node_feats_irreps = o3.Irreps([(features_dim, (0, 1))])
        x_node_feats_irreps = node_feats_irreps

        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        ) # [n_nodes, irreps]

        # Radial embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=1.0,
            num_bessel=num_basis_func,
            num_polynomial_cutoff=2,   # default of BAM-jax
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
        
        ## 2) Interaction layer  # RealAgnosticInteractionBlock
        self.linear_x = Linear(
            x_node_feats_irreps,
            x_node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        ) # x_node_feats
        
        self.x_readout = NonLinearXReadoutBlock(
                irreps_in=x_node_feats_irreps,
                MLP_irreps="64x0e+64x1o+64x2e",
                gate=[gate,None,None],
                irrep_out="1x0e",
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
        for i in range(nlayers):
            if i > 0: 
                node_feats_irreps = hidden_irreps
                target_irreps = hidden_irreps

            inter = ConcatenateRaceInteractionBlock(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=target_irreps,  # interaction_irreps
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
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

            readout = NonLinearXReadoutBlock(
                irreps_in=hidden_irreps,
                MLP_irreps="64x0e+64x1o+64x2e",
                gate=[gate,None,None],
                irrep_out="1x0e",
                cueq_config=cueq_config,
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
        #data["cell"].requires_grad_(True)
        #data["positions"].requires_grad_(True)
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
        x_energy = self.x_readout(x_node_feats)

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
            outputs.append(node_energies[:,0] + x_energy.squeeze())
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



@compile_mode("script")
class GARACE_V2_R(torch.nn.Module):
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
        regress_forces: str = "direct",
        compute_stress: bool = True
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
        
        ## 1) Embedding
        # Node embedding
        node_attr_irreps = o3.Irreps([(num_species, (0, 1))])
        node_feats_irreps = o3.Irreps([(features_dim, (0, 1))])
        x_node_feats_irreps = node_feats_irreps

        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        ) # [n_nodes, irreps]

        # Radial embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=1.0,
            num_bessel=num_basis_func,
            num_polynomial_cutoff=2,   # default of BAM-jax
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
        
        ## 2) Interaction layer  # RealAgnosticInteractionBlock
        self.linear_x = Linear(
            x_node_feats_irreps,
            x_node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        ) # x_node_feats
        self.x_readout = NonLinearReadoutBlock(
                irreps_in=x_node_feats_irreps,
                MLP_irreps="64x0e",
                gate=gate,
                irrep_out="1x0e",
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
        for i in range(nlayers):
            if i > 0: 
                node_feats_irreps = hidden_irreps
                target_irreps = hidden_irreps

            inter = ConcatenateRaceInteractionBlock(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=target_irreps,  # interaction_irreps
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
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
        #data["cell"].requires_grad_(True)
        #data["positions"].requires_grad_(True)
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
        x_energy = self.x_readout(x_node_feats)

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
            outputs.append(node_energies[:,0] + x_energy.squeeze())
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


@compile_mode("script")
class GARACE_V2_B(torch.nn.Module):
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
        regress_forces: str = "direct",
        compute_stress: bool = True
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
        
        ## 1) Embedding
        # Node embedding
        node_attr_irreps = o3.Irreps([(num_species, (0, 1))])
        node_feats_irreps = o3.Irreps([(features_dim, (0, 1))])
        x_node_feats_irreps = node_feats_irreps

        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        ) # [n_nodes, irreps]

        # Radial embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=1.0,
            num_bessel=num_basis_func,
            num_polynomial_cutoff=2,   # default of BAM-jax
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
        
        ## 2) Interaction layer  # RealAgnosticInteractionBlock
        self.linear_x = Linear(
            x_node_feats_irreps,
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
        self.x_readouts = torch.nn.ModuleList()

        target_irreps = o3.Irreps(f"{hidden_irreps.count(o3.Irrep(0, 1))}x0e")
        for i in range(nlayers):
            if i > 0: 
                node_feats_irreps = hidden_irreps
                target_irreps = hidden_irreps

            x_readout = NonLinearReadoutBlock(
                    irreps_in=x_node_feats_irreps,
                    MLP_irreps="64x0e",
                    gate=gate,
                    irrep_out="1x0e",
                    cueq_config=cueq_config,
                )
            self.x_readouts.append(x_readout)

            inter = ConcatenateRaceInteractionBlock(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=target_irreps,  # interaction_irreps
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
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
        #data["cell"].requires_grad_(True)
        #data["positions"].requires_grad_(True)
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
        for interaction, product, readout, force_decoder, stress_decoder, x_readout in zip(
                self.interactions, self.products, self.readouts, self.force_decoders, self.stress_decoders, self.x_readouts
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
            x_energy = x_readout(x_node_feats)
            outputs.append(node_energies[:,0] + x_energy.squeeze())
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

@compile_mode("script")
class GARACE(torch.nn.Module):
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
        regress_forces: str = "direct",
        compute_stress: bool = True
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
        
        ## 1) Embedding
        # Node embedding
        node_attr_irreps = o3.Irreps([(num_species, (0, 1))])
        node_feats_irreps = o3.Irreps([(features_dim, (0, 1))])
        x_node_feats_irreps = node_feats_irreps

        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        ) # [n_nodes, irreps]

        # Radial embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=1.0,
            num_bessel=num_basis_func,
            num_polynomial_cutoff=2,   # default of BAM-jax
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
        
        ## 2) Interaction layer  # RealAgnosticInteractionBlock
        self.linear_x = Linear(
            x_node_feats_irreps,
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
        self.x_readouts = torch.nn.ModuleList()

        target_irreps = o3.Irreps(f"{hidden_irreps.count(o3.Irrep(0, 1))}x0e")
        for i in range(nlayers):
            if i > 0: 
                node_feats_irreps = hidden_irreps
                target_irreps = hidden_irreps

            inter = ConcatenateRaceInteractionBlock(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=target_irreps,  # interaction_irreps
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
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
        #data["cell"].requires_grad_(True)
        #data["positions"].requires_grad_(True)
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


