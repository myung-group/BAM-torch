import torch
import torch.nn.init as init

import e3nn
from e3nn import o3, nn

from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple, Sequence
from e3nn.util.jit import compile_mode
from torch.jit import annotate
from .blocks import (
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
)
from .wrapper_ops import Linear
from bam_torch.utils.scatter import scatter_sum, scatter_mean
from bam_torch.utils.output_utils import (
    get_outputs, 
    get_symmetric_displacement,
    remove_net_torque
)

import torch
import e3nn
from e3nn import o3, nn
from e3nn.util.jit import compile_mode

from abc import abstractmethod
from typing import Callable, List, Optional, Tuple

from .radial import (
    AgnesiTransform,
    BesselBasis,
    ChebychevBasis,
    GaussianBasis,
    PolynomialCutoff,
    SoftTransform,
    PolyEnvelope,
    BesselFunction
)
from bam_torch.model.wrapper_ops import (
    CuEquivarianceConfig,
    FullyConnectedTensorProduct,
    Linear,
    SymmetricContractionWrapper,
    TensorProduct,
    FullTensorProduct
)
from bam_torch.utils.irreps_tools import (
    linear_out_irreps,
    mask_head,
    reshape_irreps,
    tp_out_irreps_with_instructions,
)
from bam_torch.utils.scatter import scatter_sum
from bam_torch.model.concatenate import (
    TensorRegroupByIrreps,
    ConcatenateIrrepsTensor,
    TensorIrrepsArrayProduct
)
from .mlp import (
    MLP,
    SeparatedLayerNorm
)
from time import time
from torch.func import vmap



@compile_mode("script")
class RACEXReadout(torch.nn.Module):
    """RACE x_node_feature's readout layer with equivariant message passing."""

    def __init__(
        self,
        hidden_irreps: o3.Irreps,
        output_irreps: o3.Irreps,
        x_irreps: o3.Irreps,
        n_species: int,
        avg_n_neighbors: float,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.avg_n_neighbors = avg_n_neighbors

        self.x_readout1 = Linear(
            x_irreps,
            hidden_irreps,
            internal_weights=False,  # False
            shared_weights=True,
            cueq_config=cueq_config,
        )
        self.x_r_weight = torch.nn.Parameter(
            torch.zeros(n_species, self.x_readout1.weight_numel)
        )
        init.xavier_uniform_(self.x_r_weight)

        hidden_irreps_2 = o3.Irreps([(mul // 2, ir) for mul, ir in hidden_irreps])

        self.x_readout2 = Linear(
            hidden_irreps,
            hidden_irreps_2,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
            biases=True
        )
        self.x_readout3 = Linear(
            hidden_irreps_2,
            output_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
            biases=True
        )

        irreps_scalars, irreps_gates, irreps_gated = split_irreps_for_gate(hidden_irreps_2)
        """
        self.gate = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[torch.nn.SiLU()],
            irreps_gates=irreps_gates,
            act_gates=[torch.sigmoid],
            irreps_gated=irreps_gated,
        )
        """
        self.gate = nn.Activation(
            irreps_in=hidden_irreps_2,
            acts=[torch.nn.SiLU()]
        )

        self.separated_layer_norm = SeparatedLayerNorm(x_irreps, affine=False)


    def forward(
        self,
        x_feats: torch.Tensor,
        species: torch.Tensor,
    ) -> torch.Tensor:

        x_feats = self.separated_layer_norm (x_feats)
        x_feats_norm = x_feats / torch.sqrt(torch.tensor(self.avg_n_neighbors))

        w = self.x_r_weight[species].squeeze(1) 
        #skip = torch.stack([self.skip(features[i], w[i]) for i in range(features.shape[0])], dim=0)
        def linear_single(x, wi):
            return self.x_readout1(x, wi)

        x_features = vmap(linear_single)(x_feats_norm, w)
    
        x_features = self.x_readout2(x_features)
        x_features = self.gate(
            x_features,
        )
        #x_features = e3nn.scalar_activation(x_features, [jax.nn.silu])
        x_energy = self.x_readout3(x_features)

        return x_energy


@compile_mode("script")
class RACEConvolution(torch.nn.Module):
    """RACE convolution layer with equivariant message passing."""
    def __init__(
        self,
        input_irreps: o3.Irreps,
        output_irreps: o3.Irreps,
        x_irreps: o3.Irreps,
        sh_irreps: o3.Irreps,
        n_species: int,
        radial_basis_size: int,
        radial_mlp_size: int,
        radial_mlp_layers: int,
        mlp_init_scale: float,
        avg_n_neighbors: float,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.output_irreps = output_irreps
        self.avg_n_neighbors = avg_n_neighbors
        self.input_irreps = input_irreps

        tp_irreps, _ = tp_out_irreps_with_instructions(
            input_irreps,
            sh_irreps,
            self.output_irreps,
        )
        tp_irreps = tp_irreps.sort().irreps.simplify()

        self.linear_1 = Linear(
            input_irreps,
            input_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        )

        self.radial_mlp = MLP(
            sizes=[radial_basis_size]
            + [radial_mlp_size] * radial_mlp_layers
            + [tp_irreps.num_irreps],
            activation=torch.nn.functional.silu,
            use_bias=False,
            init_scale=mlp_init_scale,
        )

        #self.radial_mlp = nn.FullyConnectedNet(
        #    [radial_basis_size]
        #    + [radial_mlp_size] * radial_mlp_layers
        #    + [tp_irreps.num_irreps],
        #    torch.nn.functional.silu,
        #    out_act=False
        #)

        # add extra irreps to output to account for gate
        gate_irreps = o3.Irreps(
            f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e"
        )

        gated_output_irreps = (output_irreps + gate_irreps).sort().irreps.simplify()
        '''
        self.full_tp = FullyConnectedTensorProduct (
                irreps_in1=tp_irreps,
                irreps_in2=x_irreps,
                irreps_out=output_irreps,
                rngs=rngs)
        '''

        self.linear_2 = Linear(
            tp_irreps,
            gated_output_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        )

        # skip connection has per-species weights
        self.skip = Linear(
            input_irreps,
            gated_output_irreps,
            internal_weights=False,  # False
            shared_weights=True,
            cueq_config=cueq_config,
        )
        self.weight = torch.nn.Parameter(
            torch.zeros(n_species, self.skip.weight_numel)
        )
        init.xavier_uniform_(self.weight)

        output_irreps_2 = o3.Irreps([(mul // 2, ir) for mul, ir in output_irreps])

        self.readout1 = Linear(
            output_irreps,
            output_irreps_2,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
            biases=True
        )
        self.readout = Linear(
            output_irreps_2,
            o3.Irreps("0e"),
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
            biases=True
        )

        self.tensor_product = FullTensorProduct(
            input_irreps,
            sh_irreps,
            output_irreps
        )

        self.tensor_product2 = FullTensorProduct(
            tp_irreps,
            x_irreps,
            output_irreps
        )
        #self.tensor_regroup_by_irreps = TensorRegroupByIrreps(irreps_mid)
        self.tensor_irreps_array_product = TensorIrrepsArrayProduct(tp_irreps)
        #self.gate = nn.Gate()
        irreps_scalars, irreps_gates, irreps_gated = split_irreps_for_gate(output_irreps)

        self.gate = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[torch.nn.SiLU()],
            irreps_gates=irreps_gates,
            act_gates=[torch.sigmoid],
            irreps_gated=irreps_gated,
        )
        #self.gate = nn.Activation(
        #    irreps_in=output_irreps,
        #    acts=[torch.nn.SiLU(), None, None]
        #)
        self.separated_layer_norm = SeparatedLayerNorm(input_irreps, affine=False)

    def forward(
        self,
        features: torch.Tensor,
        x_feats: torch.Tensor,
        species: torch.Tensor,
        sh: torch.Tensor,
        radial_basis: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        features_norm = self.separated_layer_norm (features)
        #features_norm = features
        messages = self.linear_1(features_norm)[senders]
        messages = self.tensor_product(messages, sh)
        radial_message = self.radial_mlp(radial_basis)
        messages = self.tensor_irreps_array_product(radial_message, messages)

        messages_agg = scatter_sum(
            src=messages, index=receivers, dim=0, dim_size=features.shape[0]
        ) / torch.sqrt(torch.tensor(self.avg_n_neighbors))
        #messages_agg = self.full_tp (messages_agg, x_feats)

        messages_agg = self.tensor_product2(messages_agg, x_feats)

        w = self.weight[species].squeeze(1) 
        #skip = torch.stack([self.skip(features[i], w[i]) for i in range(features.shape[0])], dim=0)
        def linear_single(x, wi):
            return self.skip(x, wi)

        skip = vmap(linear_single)(features, w)
        features = self.linear_2(messages_agg) + skip #+ self.skip(features, w)

        features = self.gate(
            features
        )

        node_energies = self.readout1(features)
        node_energies = self.readout(node_energies)

        return node_energies, features


class RACE_V2_7(torch.nn.Module):
    """RACE model for predicting energies and forces.

    Neural Equivariant Interatomic Potential using E(3)-equivariant
    graph neural networks.

    Args:
        n_species: Number of atom species
        lmax: Maximum angular momentum for spherical harmonics
        cutoff: Radial cutoff distance
        hidden_size: Hidden feature dimension
        n_layers: Number of convolution layers
        radial_basis_size: Number of radial basis functions
        radial_mlp_size: Hidden size of radial MLP
        radial_mlp_layers: Number of radial MLP layers
        radial_polynomial_p: Polynomial cutoff order
        mlp_init_scale: Initialization scale for MLPs
        shift: Energy shift
        scale: Energy scale
        avg_n_neighbors: Average number of neighbors for normalization
        atom_energies: Isolated atom energies for each species
        rngs: Flax NNX random number generator
    """

    def __init__(
        self,
        cutoff: float = 6.0, 
        avg_num_neighbors: int = 25, 
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
        radial_mlp_size: int = 64,
        radial_mlp_layers: int = 3,
        radial_polynomial_p: float = 2.0,
        mlp_init_scale: float = 4.0,
        shift: float = 0.0,
        scale: float = 1.0,
        atom_energies: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        # Store static configuration
        self.lmax = max_ell
        self.cutoff = cutoff
        self.n_species = num_species
        self.radial_basis_size = num_basis_func
        self.radial_polynomial_p = radial_polynomial_p
        self.n_layers = nlayers

        self.regress_forces = regress_forces
        self.compute_stress = compute_stress

        self.output_irreps = o3.Irreps(output_irreps)
        hidden_irreps = hidden_irreps.sort().irreps
        self.hidden_irreps = hidden_irreps


        self.shift = torch.nn.Parameter(
            torch.tensor(shift, dtype=torch.float32)
        )
        self.scale = torch.nn.Parameter(
            torch.tensor(scale, dtype=torch.float32)
        )
        self.avg_n_neighbors = torch.nn.Parameter(
            torch.tensor(avg_num_neighbors, dtype=torch.float32)
        )

        if atom_energies is not None:
            self.atom_energies = torch.nn.Parameter(
                torch.tensor(atom_energies, dtype=torch.float32)
            )
        else:
            self.atom_energies = torch.nn.Parameter(
                torch.zeros(self.n_species, dtype=torch.float32)
            )

        self.input_irreps = o3.Irreps(f"{features_dim}x0e")
        self.x_irreps = o3.Irreps("1x0e")

        input_irreps = o3.Irreps(f"{features_dim}x0e")
        x_irreps = o3.Irreps("1x0e")
        self.x_linear = Linear(
            self.input_irreps,
            self.x_irreps,
            internal_weights=False,  # False
            shared_weights=True,
            cueq_config=cueq_config,
        )
        self.x_weight = torch.nn.Parameter(
            torch.zeros(self.n_species, self.x_linear.weight_numel)
        )
        init.xavier_uniform_(self.x_weight)

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)

        self.species_embedding = LinearNodeEmbeddingBlock(
            irreps_in=o3.Irreps(f"{self.n_species}x0e"),
            irreps_out=input_irreps,
            cueq_config=cueq_config,
        )

        # Radial embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=1.0,
            num_bessel=num_basis_func,
            num_polynomial_cutoff=2,   # default of BAM-jax
            radial_type="bessel",
            distance_transform=None,
        )
        self.spherical_harmonics = o3.SphericalHarmonics(sh_irreps, 
                                                         normalize=True,
                                                         normalization="component")

        self.layers = torch.nn.ModuleList()
        self.x_readouts = torch.nn.ModuleList()
        self.f_readouts = torch.nn.ModuleList()
        for i in range(self.n_layers):
            layer = RACEConvolution(
                input_irreps=input_irreps if i == 0 else hidden_irreps,
                output_irreps=hidden_irreps,
                x_irreps=x_irreps,
                sh_irreps=sh_irreps,
                n_species=num_species,
                radial_basis_size=num_basis_func,
                radial_mlp_size=radial_mlp_size,
                radial_mlp_layers=radial_mlp_layers,
                mlp_init_scale=mlp_init_scale,
                avg_n_neighbors=avg_num_neighbors,
                cueq_config=cueq_config,
            )
            self.layers.append(layer)

            x_readout = RACEXReadout(
                hidden_irreps=o3.Irreps("64x0e"), # 256
                output_irreps=o3.Irreps("0e"),
                x_irreps=x_irreps,
                n_species=num_species,
                avg_n_neighbors=avg_num_neighbors,
                cueq_config=cueq_config,
            )
            self.x_readouts.append(x_readout)

            f_readout = RACEXReadout(
                hidden_irreps=o3.Irreps("64x0e"),
                output_irreps=o3.Irreps("0e"),
                x_irreps=input_irreps if i == 0 else hidden_irreps,
                n_species=num_species,
                avg_n_neighbors=avg_num_neighbors,
                cueq_config=cueq_config,
            )
            self.f_readouts.append(f_readout)


    def node_energies(
            self, 
            data: Dict[str, torch.Tensor], 
    ):
        """Compute per-node energies.

        Args:
            positions: Atomic positions of shape (n_atoms, 3)
            data: Graph data structure

        Returns:
            Per-node energies of shape (n_atoms, 1)
        """
        # input features are one-hot encoded species

        species = data["species"]
        features = to_one_hot(species.unsqueeze(-1), self.n_species)
        features = self.species_embedding (features)
        w = self.x_weight[species].squeeze(1) 
        #print("linear.weight_numel =", self.x_linear.weight_numel)
        #print("w.shape =", w.shape)
        #x_feats = self.x_linear(features, w)\
        def linear_single(x, wi):
            return self.x_linear(x, wi)

        #x_feats = torch.stack([self.x_linear(features[i], w[i]) for i in range(features.shape[0])], dim=0)
        x_feats = vmap(linear_single)(features, w)
        r = get_edge_relative_vectors_with_pbc(data)
        r = r / self.cutoff

        # safe norm (avoids nan for r = 0)
        lengths = torch.norm(r, dim=1)

        edge_index = data["edge_index"]
        nonzero_idx = torch.arange(len(lengths), device=lengths.device)[lengths != 0]
        r = r[nonzero_idx]
        lengths = lengths[nonzero_idx]
        edge_index = edge_index[:, nonzero_idx]
        
        # compute spherical harmonics of edge displacements
        radial_basis = self.radial_embedding(lengths.unsqueeze(1), 
                                           features,
                                           edge_index,
                                           species)
        sh = self.spherical_harmonics(r)

        outputs = []
        for layer, x_readout, f_readout in zip(self.layers, self.x_readouts, self.f_readouts):
            f_energy = f_readout(
                features,
                species,
            )
            node_energies, features = layer(
                features,
                x_feats,
                species,
                sh,
                radial_basis,
                edge_index[0], # senders
                edge_index[1], # receivers
            )
            x_energy = x_readout(
                x_feats,
                species,
            )
            outputs.append(node_energies[:,0] + x_energy.squeeze() + f_energy.squeeze())

        node_energies = torch.sum(torch.stack(outputs, dim=-1), dim=-1)#, keepdims=True)
 
        # scale and shift energies
        #node_energies = node_energies * (
        #    self.scale[...]
        #) + (self.shift[...])

        # add isolated atom energies to each node as prior
        #node_energies = node_energies + (
        #    self.atom_energies[data["species"]]
        #    .unsqueeze(-1)
        #    .detach()
        #)

        return node_energies

    def forward(self,
            data: Dict[str, torch.Tensor], 
            backprop: bool = False,
            compute_displacement: bool = False
    ):
        data["cell"].requires_grad_(True)
        data["positions"].requires_grad_(True)

        """Compute energies and forces for a batch of graphs.

        Args:
            data: Graph data structure with nodes containing 'positions' and 'species'

        Returns:
            Tuple of (graph_energies, forces) where:
                - graph_energies: Total energy per graph, shape (n_graphs,)
                - forces: Forces on each atom, shape (n_atoms, 3)
        """
        # compute forces as gradient of total energy
        node_energies = self.node_energies(data)

        num_graphs = data["ptr"].numel() - 1  # nbatch

        graph_energy = scatter_sum(
                src=node_energies,
                index=data["batch"],
                dim=-1,
                dim_size=num_graphs,
            )
        
        preds: Dict[str, Optional[torch.Tensor]] = {}
        preds["energy"] = graph_energy # total energy
        preds["node_energy"] = node_energies

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
        #print(preds)
        #graph_energy.backward(torch.ones_like(graph_energy))

        return preds

def split_irreps_for_gate(tp_irreps: o3.Irreps):
    """
    Split irreps into (scalars, gates, gated) for e3nn.nn.Gate.

    Args:
        tp_irreps: Irreps (e.g. "192x0e + 256x1o + 256x2e")

    Returns:
        irreps_scalars: scalar irreps (0e)
        irreps_gates:   scalar gates (0e), one per non-scalar irrep
        irreps_gated:   non-scalar irreps (l > 0)
    """
    tp_irreps = o3.Irreps(tp_irreps)

    scalars = []
    gated = []
    n_gates = 0

    for mul, ir in tp_irreps:
        if ir.is_scalar():
            scalars.append((mul, ir))
        else:
            gated.append((mul, ir))
            n_gates += mul

    irreps_scalars = o3.Irreps(scalars)
    irreps_gated   = o3.Irreps(gated)
    irreps_gates   = o3.Irreps(f"{n_gates}x0e")

    return irreps_scalars, irreps_gates, irreps_gated

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