
# --------------------- 

import torch
import torch.nn.init as init

import e3nn
from e3nn import o3, nn

from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from e3nn.util.jit import compile_mode
from torch.jit import annotate
from bam_torch.model.blocks import (
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
from bam_torch.model.wrapper_ops import Linear
from bam_torch.utils.scatter import scatter_sum, scatter_mean
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
            output_irreps: e3nn.o3.Irreps = o3.Irreps("3x1o"),
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
        self.output_irreps = o3.Irreps(output_irreps)
        
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

            readout = Linear(
                hidden_irreps,
                output_irreps,
                internal_weights=True,
                shared_weights=True,
                cueq_config=cueq_config,
            ) 
            self.readouts.append(readout) # [n_nodes, output_irreps.count(o3.Irrep(0, 1))]

        #self.emb = torch.nn.Embedding(num_embeddings=num_species, embedding_dim=num_species)
    
    def forward(
            self, 
            data, 
            pos,
            backprop=False
    ):
        # assert Rij.ndim == 2 and Rij.shape[1] == 3
        # iatoms ==> senders     # edge_index[0]
        # jatoms ==> receivers   # edge_index[1]


        Rij = get_edge_relative_vectors_with_pbc(data, pos)
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
        if "node_attrs" in data:
            node_attrs = data["node_attrs"]  # Pre-calculated in C++
            species = data["species"]
        else:
            species = data["species"]
            node_attrs = to_one_hot(species.unsqueeze(-1), self.num_species)
        #node_feats = self.emb(species)
        node_feats = self.node_embedding(node_attrs)

        edge_index = data["edge_index"]
        lengths = torch.norm(Rij, dim=1)

        nonzero_idx = torch.arange(len(lengths), device=lengths.device)[lengths != 0]
        Rij = Rij[nonzero_idx]
        lengths = lengths[nonzero_idx]
        edge_index = edge_index[:, nonzero_idx]
        # R = R[nonzero_idx]
        
        edge_attrs = self.spherical_harmonics(Rij)
        edge_feats = self.radial_embedding(lengths.unsqueeze(1), 
                                           node_attrs,
                                           data["edge_index"],
                                           species)
        outputs = []
        node_logvar = [] 
        node_f_logvar = [] 
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
            node_energies = readout(node_feats) # [n_nodes, len(heads)]  == [nbatch*num_nodes, "1x0e" or "2x0e"]
            node_feats_list.append(node_feats)
            
            outputs.append(node_energies)

        # Concatenate node features
        #node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        node_energy = torch.stack(outputs, dim=-1) # [nbatch*num_nodes, nlayers]
        node_energy = self.act_fn(node_energy)

        # Global pooling
        node_energy = torch.sum(node_energy, dim=-1) # [nbatch*num_nodes]  # total_energy
        #print(f"node_energy = {node_energy}")
        graph_energy = scatter_sum(
                src=node_energy,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )
        out = graph_energy.view(num_graphs, 3, 3)

        return out
      

def get_edge_relative_vectors_with_pbc(data: Dict[str, torch.Tensor], R):
    # iatoms ==> senders
    # jatoms ==> receivers
    #R = data["positions"]
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



class AEquivariantInterface_1(torch.nn.Module):
    def __init__(
        self,
        symmetry='SO3',
        interface='prob',
        fixed_noise=False,
        noise_scale=0.1,
        tau=0.01,
        hard=True,
        vnn_hidden_dim=96,
        vnn_k_nearest_neighbors=4,
        vnn_dropout=0.1
    ):
        super().__init__()
        assert symmetry in ['SO3', 'O3']
        assert interface in ['prob', 'unif']
        self.symmetry = symmetry
        self.interface = interface
        self.fixed_noise = fixed_noise
        self.noise_scale = noise_scale
        self.tau = tau
        self.hard = hard
        self.vnn_interface = RACE(
            interaction_cls = ConcatenateRaceInteractionBlock,
            cutoff = 6.0, 
            avg_num_neighbors = 27, 
            num_species = 4, 
            max_ell = 3,
            num_basis_func = 8,
            hidden_irreps = o3.Irreps("16x0e+8x1o+4x2e"),
            nlayers= 3,
            features_dim = 32,  # hidden_irreps.count(o3.Irrep(0, 1))
            output_irreps = o3.Irreps("3x1o"),
            active_fn = "swish",
            radial_MLP = None,
            correlation = 3,
            heads = None,
            MLP_irreps = o3.Irreps("16x0e"),
            gate = torch.nn.SiLU(),
            cueq_config = None,
        )
        # self.compute_entropy_loss = PermutaionMatrixPenalty(n=5)

    def _postprocess_rotation(self, pseudo_ks, eps=1e-6):
        """Obtain rotation component (b, k, 3, 3) from hidden representation"""
        # note: this assumes left equivariance, i.e., pseudo_ks: (b, k, 3, C=3)
        # pseudo_ks: GL(N)
        device = pseudo_ks.device
        b, k, _, _ = pseudo_ks.shape
        assert pseudo_ks.shape == (b, k, 3, 3)
        # add small noise to prevent rank collapse
        pseudo_ks = pseudo_ks + eps * torch.randn_like(pseudo_ks, device=device)
        pseudo_ks = pseudo_ks.view(b*k, 3, 3)
        # use gram-schmidt to obtain orthogonal matrix
        ks = batched_gram_schmidt_3d(pseudo_ks)  # O(3)
        assert ks.shape == (b*k, 3, 3)

        if self.symmetry in ('SnxSO3', 'SO3'):
            # SO(3) equivariant map that maps O(3) matrix to SO(3) matrix
            # determinant are +- 1
            deter_ks = torch.linalg.det(ks)
            assert deter_ks.shape == (b*k,)
            # multiply the first column
            sign_arr = torch.ones(b*k, 3, device=device)
            #sign_arr = sign_arr.clone()
            sign_arr[:, 0] = deter_ks
            #sign_arr = torch.cat([deter_ks.unsqueeze(1), sign_arr[:, 1:]], dim=1)
            sign_arr = sign_arr[:, None, :].expand(b*k, 3, 3)
            # elementwise multiplication
            ks = ks * sign_arr

        ks = ks.reshape(b, k, 3, 3)
        return ks

    def sample_invariant_noise(self, x, idx):
        n, _ = x.shape
        if self.fixed_noise:
            zs = []
            for i in idx.tolist():
                seed = torch.seed()
                torch.manual_seed(i)
                z = torch.zeros(n, 3, device=x.device, dtype=x.dtype)
                z = z.normal_(0, self.noise_scale)
                zs.append(z)
                torch.manual_seed(seed)
            z = torch.stack(zs, dim=0)
        else:
            z = torch.zeros_like(x).normal_(0, self.noise_scale)
        return z

    def _forward_prob(self, data, k: int):
        # k is the number of interface samples
        #data["cell"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        x = data.positions
        x = x - x.mean(dim=0, keepdim=True)
        num_edges = data.num_edges
        b = num_edges.shape[0]
        b_n, _ = x.shape
        n = int(b_n / b)
        idx = torch.tensor([i for i in range(b)], device=x.device)
       
        pseudo_ks = []
        for i in range(k):
            # add noise
            x = x + self.sample_invariant_noise(x, idx)
            p_ks = self.vnn_interface(data, x)
            pseudo_ks.append(p_ks)
        
        pseudo_ks = torch.cat(pseudo_ks, dim=0)
        #print(pseudo_ks)
        assert pseudo_ks.shape == (b*k, 3, 3)  # [b*k, c=3, 3]
        pseudo_ks = pseudo_ks.transpose(1, 2)  # [b*k, 3, c=3]
        pseudo_ks = pseudo_ks.reshape(b, k, 3, 3)
        # post-processing for permutation matrix
        # hs, entropy_loss = self._postprocess_permutation(pseudo_hs)
        # assert hs.shape == (b, k, n, n)
        # post-processing for SO(3) or O(3) matrix
        ks = self._postprocess_rotation(pseudo_ks)
        assert ks.shape == (b, k, 3, 3)
        return ks #entropy_loss

    def _forward_unif(self, node_features, idx, k: int):
        b, n, _, _ = node_features.shape
        device = node_features.device
        # sample Sn representation
        assert self.hard
        if self.fixed_noise:
            raise NotImplementedError
        indices = torch.randn(b*k, n, device=device).argsort(dim=-1)
        # sample O(3) or SO(3) representation
        if self.symmetry in ('O3'):
            ks = torch.randn(b*k, 3, 3, device=device)
            ks = batched_gram_schmidt_3d(ks)
            ks = ks.reshape(b, k, 3, 3)
        elif self.symmetry in ('SO3'):
            ks = torch.randn(b*k, 3, 3, device=device)
            ks = batched_gram_schmidt_3d(ks)
            # SO(3) equivariant map that maps O(3) matrix to SO(3) matrix
            # determinant are +- 1
            deter_ks = torch.linalg.det(ks)
            assert deter_ks.shape == (b*k,)
            # multiply the first column
            sign_arr = torch.ones(b*k, 3, device=device)
            #sign_arr = sign_arr.clone()
            sign_arr[:, 0] = deter_ks
            #sign_arr = torch.cat([deter_ks.unsqueeze(1), sign_arr[:, 1:]], dim=1)
            sign_arr = sign_arr[:, None, :].expand(b*k, 3, 3)
            ks = ks * sign_arr
            ks = ks.reshape(b, k, 3, 3)
        else:
            raise NotImplementedError
        ks
        return ks

    
    def forward(self, data, k):
        # k is the number of interface samples


        gs = self._forward_prob(data, k)
        #if self.symmetry in ('O3', 'SO3'):
        #    # entropy loss is only for permutation involved groups
        #    entropy_loss = torch.tensor(0, device=node_features.device)
        return gs


