import torch
import torch.nn.init as init

import e3nn
from e3nn import o3, nn

from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from e3nn.util.jit import compile_mode

from .blocks import (
    LinearNodeEmbeddingBlock,
    InteractionBlock,
    FullyConnectedTensorProduct,
    RealAgnosticInteractionBlock,
    AgnosticResidualNonlinearInteractionBlock,
    RaceInteractionBlock,
    ConcatenateRaceInteractionBlock,
    RaceInteractionBlockBasis,
    EquivariantProductBasisBlock,
    ReducedRaceEquivariantBlock,
    RaceEquivariantBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock
)
from .wrapper_ops import Linear
from bam_torch.utils.scatter import scatter_sum
from bam_torch.utils.output_utils import get_outputs, get_symmetric_displacement


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


def _to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Generates one-hot encoding with <num_classes> classes from <indices>
    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :param device: torch device
    :return: (N x num_classes) tensor
    """
    shape = indices.shape[:-1] + (num_classes,)
    oh = torch.zeros(shape, device=indices.device).view(shape)
    
    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)
    return oh.view(*shape)  ## similar with torch.nn.Embedding

    
@compile_mode("script")
class MACE(torch.nn.Module):
    """Base model of E(3) Equivariant Graph Neural Network, based on e3nn 
    """
    def __init__(
            self,
            interaction_cls_first: Optional[Type[InteractionBlock]] = AgnosticResidualNonlinearInteractionBlock,
            interaction_cls: Optional[Type[InteractionBlock]] = AgnosticResidualNonlinearInteractionBlock,
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
            radial_MLP: Optional[List[int]] = None,
            correlation: Union[int, List[int]] = 3,
            heads: Optional[List[str]] = None,
            MLP_irreps: e3nn.o3.Irreps = o3.Irreps("16x0e"),
            gate: Optional[Callable] = torch.nn.SiLU(),
            cueq_config: Optional[Dict[str, Any]] = None,
            regress_forces: str = "direct",
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
            num_polynomial_cutoff=6,   # default of BAM-jax
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
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]  # default is [64, 64] in BAM-jax ?

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
        )

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
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

            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
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
        Rij = get_edge_relative_vectors_with_pbc(R, data)
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
                cell=data.cell,
                training=backprop,
                compute_force=True,
                compute_virials=False,
                compute_stress=False,
                compute_hessian=False
            )
            preds["forces"] = forces

        return preds
    

@compile_mode("script")
class ReducedRACE(torch.nn.Module):
    """
    Parameter-reduced Race model
    """
    def __init__(
            self, 
            interaction_cls: Optional[Type[InteractionBlock]] = RaceInteractionBlockBasis,
            cutoff: float = 6.0, 
            avg_num_neighbors: int = 40, 
            num_species: int = 1, 
            max_ell: int = 3,
            num_basis_func: int = 8,
            hidden_irreps: e3nn.o3.Irreps = o3.Irreps("32x0e+32x1o+32x2e"),
            nlayers: int = 3,
            features_dim: int = 32,  # hidden_irreps.count(o3.Irrep(0, 1))
            output_irreps: e3nn.o3.Irreps = o3.Irreps("1x0e"),
            active_fn: str = "swish",
            radial_MLP: Optional[List[int]] = None,
            correlation: Union[int, List[int]] = 3,
            heads: Optional[List[str]] = None,
            MLP_irreps: e3nn.o3.Irreps = o3.Irreps("16x0e"),
            gate: Optional[Callable] = torch.nn.SiLU(),
            cueq_config: Optional[Dict[str, Any]] = None,
            regress_forces: str = "direct",
    ):
        super().__init__()
    
        if active_fn in ["swish", "silu", "SiLU"]:
            self.act_fn = torch.nn.SiLU()
        elif active_fn in ["relu", "ReLU"]:
            self.act_fn = torch.nn.ReLU()
        elif active_fn in ["identity", None]:
            self.act_fn = torch.nn.Identity()   # Need to modify later
        
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
        node_feats_irreps = o3.Irreps([(features_dim, (0, 1))])
        x_node_feats_irreps = o3.Irreps([(1,(0,1))]) #node_feats_irreps

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
        self.x_node_tp = FullyConnectedTensorProduct(
            node_feats_irreps,
            node_attr_irreps,
            x_node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        ) # x_node_feats
        if radial_MLP is None:
            radial_MLP = [64, 64]  # default is [64, 64] in BAM-jax ?

        self.interactions = torch.nn.ModuleList()
        self.products = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()
        for i in range(nlayers):
            if i > 0: 
                node_feats_irreps = hidden_irreps

            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
            )
            self.interactions.append(inter)

            prod = ReducedRaceEquivariantBlock(
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
                num_heads=len(heads),
                cueq_config=cueq_config,
            )
            self.readouts.append(readout) # [n_nodes, output_irreps.count(o3.Irrep(0, 1))]
            
        #self.scale_shift = ScaleShiftBlock(
        #    scale=atomic_inter_scale, shift=atomic_inter_shift
        #)
        #self.emb = torch.nn.Embedding(num_embeddings=num_species, embedding_dim=num_features)
        #self.reshape = reshape_irreps(self.hidden_irreps, cueq_config=cueq_config)
        #self.apply(self.initialize_weights)
    
    def initialize_weights(self, module):
        if hasattr(module, 'weight') and module.weight is not None:
            #if module.weight.dim() >= 2: 
            init.kaiming_normal_(module.weight)  
            #else:
               # init.uniform_(module.weight, a=-0.1, b=0.1)
        if hasattr(module, 'bias') and module.bias is not None:
            init.zeros_(module.bias) 
    
    def forward(
            self, 
            data: Dict[str, torch.Tensor], 
            backprop: bool = False
    ):
        # assert Rij.ndim == 2 and Rij.shape[1] == 3
        # iatoms ==> senders     # edge_index[0]
        # jatoms ==> receivers   # edge_index[1]
        R = data["positions"]
        R.requires_grad_(True)
        Rij = get_edge_relative_vectors_with_pbc(R, data)
        Rij = Rij / self.cutoff
        node_heads = (
            data["head"][data["batch"]]
            if "head" in data
            else torch.zeros_like(data["batch"])
        )
        num_graphs = data["ptr"].numel() - 1  # nbatch
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        # Embedding
        species = data["species"].unsqueeze(-1)

        node_attrs = to_one_hot(species, self.num_species)
        #node_feats = self.emb(data.species)
        node_feats = self.node_embedding(node_attrs)

        edge_index = data["edge_index"]
        lengths = torch.norm(Rij, dim=1)

        #nonzero_idx = torch.arange(len(lengths), device=lengths.device)[lengths != 0]
        #Rij = Rij[nonzero_idx]
        #lengths = lengths[nonzero_idx]
        #edge_index = edge_index[:, nonzero_idx]
        
        edge_attrs = self.spherical_harmonics(Rij)
        edge_feats = self.radial_embedding(lengths.unsqueeze(1), 
                                           node_attrs, 
                                           data["edge_index"],
                                           species)
        outputs = []
        node_logvar = []
        node_feats_list = []
        x_node_feats = self.x_node_tp(node_feats, node_attrs)
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
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
            node_feats_list.append(node_feats)
            node_energies = readout(node_feats, node_heads) # [n_nodes, len(heads)]  == [nbatch*num_nodes, "1x0e" or "2x0e"]
            
            outputs.append(node_energies[:,0])
            if node_energies.shape[1] == 2:
                node_logvar.append(node_energies[:,1])

        # Concatenate node features
        #node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        node_energy = torch.stack(outputs, dim=-1) # [nbatch*num_nodes, nlayers]
        if self.act_fn != None:
            node_energy = self.act_fn(node_energy)
        
        # Global pooling
        node_energy = torch.sum(node_energy, dim=-1) # [nbatch*num_nodes]  # total_energy
        # node_energy = self.scale_shift(node_energy, node_heads) 
        node_energy = scatter_sum(
                src=node_energy,
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
        n_node = int(data["num_nodes"] / num_graphs)
        n_nodes = torch.tensor([n_node]*num_graphs, device=node_energy_var.device)
        energy_var = node_energy_var/n_nodes

        preds: Dict[str, Optional[torch.Tensor]] = {}
        preds["energy"] = node_energy #node_energy
        preds["energy_var"] = energy_var

        forces: Optional[torch.Tensor] = None
        if self.regress_forces == 'direct' or self.regress_forces:
            forces, virials, stress, hessian = get_outputs(
                energy=node_energy,
                positions=R,
                displacement=displacement,
                cell=data["cell"],
                training=backprop,
                compute_force=True,
                compute_virials=False,
                compute_stress=False,
                compute_hessian=False
            )
            preds["forces"] = forces

        return preds


@compile_mode("script")
class RACE(torch.nn.Module):
    """Race model
    """
    def __init__(
            self, 
            interaction_cls: Optional[Type[InteractionBlock]] = ConcatenateRaceInteractionBlock,
            cutoff: float = 6.0, 
            avg_num_neighbors: int = 40, 
            num_species: int = 1, 
            max_ell: int = 3,
            num_basis_func: int = 8,
            hidden_irreps: e3nn.o3.Irreps = o3.Irreps("32x0e+32x1o+32x2e"),
            nlayers: int = 3,
            features_dim: int = 32,  # hidden_irreps.count(o3.Irrep(0, 1))
            output_irreps: e3nn.o3.Irreps = o3.Irreps("1x0e"),
            active_fn: str = "swish",
            radial_MLP: Optional[List[int]] = None,
            correlation: Union[int, List[int]] = 3,
            heads: Optional[List[str]] = None,
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
            self.act_fn = torch.nn.Identity()  # Need to modify later
        
        self.cutoff = cutoff
        self.regress_forces = regress_forces
        self.compute_stress = compute_stress
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
        for i in range(nlayers):
            if i > 0: 
                node_feats_irreps = hidden_irreps

            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
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
                num_heads=len(heads),
                cueq_config=cueq_config,
            )
            self.readouts.append(readout) # [n_nodes, output_irreps.count(o3.Irrep(0, 1))]

        #self.emb = torch.nn.Embedding(num_embeddings=num_species, embedding_dim=num_species)
    
    def forward(
            self, 
            data: Dict[str, torch.Tensor], 
            backprop: bool = False
    ):
        # assert Rij.ndim == 2 and Rij.shape[1] == 3
        # iatoms ==> senders     # edge_index[0]
        # jatoms ==> receivers   # edge_index[1]
        cell = data["cell"]
        cell.requires_grad_(True)
        R = data["positions"]
        R.requires_grad_(True)
        Rij = get_edge_relative_vectors_with_pbc(R, cell, data)
        Rij = Rij / self.cutoff
        node_heads = (
            data["head"][data["batch"]]
            if "head" in data
            else torch.zeros_like(data["batch"])
        )
        num_graphs = data["ptr"].numel() - 1  # nbatch
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if self.compute_stress:
            (
                data["positions"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )
        # Embedding
        species = data["species"]
        node_attrs = to_one_hot(species.unsqueeze(-1), self.num_species)
        #node_feats = self.emb(species)
        node_feats = self.node_embedding(node_attrs)

        edge_index = data["edge_index"]
        lengths = torch.norm(Rij, dim=1)

        #nonzero_idx = torch.arange(len(lengths), device=lengths.device)[lengths != 0]
        #Rij = Rij[nonzero_idx]
        #lengths = lengths[nonzero_idx]
        #edge_index = edge_index[:, nonzero_idx]
        # R = R[nonzero_idx]
        
        edge_attrs = self.spherical_harmonics(Rij)
        edge_feats = self.radial_embedding(lengths.unsqueeze(1), 
                                           node_attrs,
                                           data["edge_index"],
                                           species)
        outputs = []
        node_logvar = [] 
        node_feats_list = []
        x_node_feats = self.linear_x(node_feats)
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
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
            node_energies = readout(node_feats, node_heads) # [n_nodes, len(heads)]  == [nbatch*num_nodes, "1x0e" or "2x0e"]
            node_feats_list.append(node_feats)
            
            outputs.append(node_energies[:,0])
            if node_energies.shape[1] == 2:
                node_logvar.append(node_energies[:,1])

        # Concatenate node features
        #node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        node_energy = torch.stack(outputs, dim=-1) # [nbatch*num_nodes, nlayers]
        node_energy = self.act_fn(node_energy)

        # Global pooling
        node_energy = torch.sum(node_energy, dim=-1) # [nbatch*num_nodes]  # total_energy
        node_energy = scatter_sum(
                src=node_energy,
                index=data["batch"],
                dim=-1,
                dim_size=num_graphs,
            ) 

        if node_logvar != []:
            node_logvar_ts = torch.stack(node_logvar, dim=-1) # [nbatch*num_nodes, nlayers]
            node_logvar_ts = node_logvar_ts.mean(dim=-1) # [nbatch*num_nodes]
        else:
            node_logvar_ts = torch.zeros(node_feats.shape[0], device=node_energy.device)
        node_energy_var = torch.exp(node_logvar_ts) 
        node_energy_var = scatter_sum(
                src=node_energy_var,
                index=data["batch"],
                dim=-1,
                dim_size=num_graphs,
            ) 
        n_node = int(data["num_nodes"] / num_graphs)
        n_nodes = torch.tensor([n_node]*num_graphs, device=node_energy_var.device)
        energy_var = node_energy_var/n_nodes

        preds: Dict[str, Optional[torch.Tensor]] = {}
        preds["energy"] = node_energy #node_energy
        preds["energy_var"] = energy_var

        forces: Optional[torch.Tensor] = None
        if self.regress_forces == 'direct' or self.regress_forces:
            forces, virials, stress, hessian = get_outputs(
                energy=node_energy,
                positions=R,
                displacement=displacement,
                cell=cell,
                batch_idx=data["batch"],
                num_graphs=num_graphs,
                training=backprop,
                compute_force=True,
                compute_virials=True,
                compute_stress=True,
                compute_hessian=False
            )
            preds["forces"] = forces
            preds["stress"] = stress

        return preds
      

def get_edge_relative_vectors_with_pbc (
        R: torch.Tensor, 
        cell: torch.Tensor, 
        data_graph: Dict[str, torch.Tensor]):
    # iatoms ==> senders
    # jatoms ==> receivers
    iatoms = data_graph["edge_index"][0]  # shape = (b * n_edges)
    jatoms = data_graph["edge_index"][1]  # shape = (b * n_edges) 
    Sij = data_graph["edges"]   # shape = (b * n_edges, 3)
    #cell = data_graph["cell"]   # shape = (b, 3, 3)
    n_edges = data_graph["num_edges"] # [n_edge_1, n_edge_2, ..., n_edge_N]

    Sij = torch.split(Sij, n_edges.tolist(), dim=0)
    shift_v = torch.cat(
        [torch.einsum('ni,ij->nj', s, c)
            for s, c in zip(Sij, cell)], dim=0
    )
    _R = R[jatoms] - R[iatoms] 
    Rij = _R + shift_v

    return Rij # (num_edges, 3)


def get_edge_relative_vectors_with_pbc_padding (
        R: torch.Tensor, 
        data_graph: Dict[str, torch.Tensor]):
    # iatoms ==> senders
    # jatoms ==> receivers
    iatoms = data_graph["edge_index"][0]  # shape = (b * n_edges)
    jatoms = data_graph["edge_index"][1]  # shape = (b * n_edges) 
    Sij = data_graph["edges"]   # shape = (b * n_edges, 3)
    cell = data_graph["cell"]   # shape = (b, 3, 3)

    b, _, _ = cell.shape
    n_edges = data_graph["num_edges"][0]

    repeated_cell = torch.repeat_interleave(cell, repeats=n_edges, dim=0)
    repeated_cell =  repeated_cell.reshape(b, n_edges, 3, 3)
    Sij = Sij.view(b, n_edges, 3)
    shift_v = torch.einsum('bni,bnij->bnj', Sij, repeated_cell)

    _R = R[jatoms] - R[iatoms] 
    _R = _R.view(b, n_edges, 3)
    Rij = _R + shift_v
    Rij = Rij.view(b*n_edges, 3).to(R.device)

    return Rij # (num_edges, 3)


def determine_atomic_inter_shift(mean, heads):
    if isinstance(mean, np.ndarray):
        if mean.size == 1:
            return mean.item()
        if mean.size == len(heads):
            return mean.tolist()
        logging.info("Mean not in correct format, using default value of 0.0")
        return [0.0] * len(heads)
    if isinstance(mean, list) and len(mean) == len(heads):
        return mean
    if isinstance(mean, float):
        return [mean] * len(heads)
    logging.info("Mean not in correct format, using default value of 0.0")
    return [0.0] * len(heads)




        
