"""
FAENet: Frame Averaging Equivariant graph neural Network 
Simple, scalable and expressive model for property prediction on 3D atomic systems.
"""
from typing import Dict, Optional, Union

import torch
from torch import nn
from torch.nn import Linear

from .base_model import BaseModel
from .force_decoder import ForceDecoder
from .blocks import(
    EmbeddingBlock,
    InteractionBlock,
    OutputBlock
)
from bam_torch.ga.utils.fa_utils import (
    GaussianSmearing, 
    swish, 
    pbc_preprocess, 
    base_preprocess
)


class FAENet(BaseModel):
    r"""Non-symmetry preserving GNN model for 3D atomic systems,
    called FAENet: Frame Averaging Equivariant Network.

    Args:
        cutoff (float): Cutoff distance for interatomic interactions.
            (default: :obj:`6.0`)
        preprocess (callable): Pre-processing function for the data. This function
            should accept a data object as input and return a tuple containing the following:
            atomic numbers, batch indices, final adjacency, relative positions, pairwise distances.
            Examples of valid preprocessing functions include `pbc_preprocess`,
            `base_preprocess`, or custom functions.
        act (str): Activation function
            (default: `swish`)
        max_num_neighbors (int): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: `40`)
        hidden_channels (int): Hidden embedding size.
            (default: `128`)
        tag_hidden_channels (int): Hidden tag embedding size.
            (default: :obj:`32`)
        pg_hidden_channels (int): Hidden period and group embedding size.
            (default: :obj:`32`)
        phys_embeds (bool): Do we include fixed physics-aware embeddings.
            (default: :obj: `True`)
        phys_hidden_channels (int): Hidden size of learnable physics-aware embeddings.
            (default: :obj:`0`)
        num_interactions (int): The number of interaction (i.e. message passing) blocks.
            (default: :obj:`4`)
        num_gaussians (int): The number of gaussians :math:`\mu` to encode distance info.
            (default: :obj:`50`)
        num_filters (int): The size of convolutional filters.
            (default: :obj:`128`)
        second_layer_MLP (bool): Use 2-layers MLP at the end of the Embedding block.
            (default: :obj:`False`)
        skip_co (str): Add a skip connection between each interaction block and
            energy-head. (`False`, `"add"`, `"concat"`, `"concat_atom"`)
        mp_type (str): Specificies the Message Passing type of the interaction block.
            (`"base"`, `"updownscale_base"`, `"updownscale"`, `"updown_local_env"`, `"simple"`):
        graph_norm (bool): Whether to apply batch norm after every linear layer.
            (default: :obj:`True`)
        complex_mp (bool); Whether to add a second layer MLP at the end of each Interaction
            (default: :obj:`True`)
        energy_head (str): Method to compute energy prediction
            from atom representations.
            (`None`, `"weighted-av-initial-embeds"`, `"weighted-av-final-embeds"`)
        out_dim (int): size of the output tensor for graph-level predicted properties ("energy")
            Allows to predict multiple properties at the same time.
            (default: :obj:`1`)
        pred_as_dict (bool): Set to False to return a (property) prediction tensor.
            By default, predictions are returned as a dictionary with several keys (e.g. energy, forces)
            (default: :obj:`True`)
        regress_forces (str): Specifies if we predict forces or not, and how
            do we predict them. (`None` or `""`, `"direct"`, `"direct_with_gradient_target"`)
        force_decoder_type (str): Specifies the type of force decoder
            (`"simple"`, `"mlp"`, `"res"`, `"res_updown"`)
        force_decoder_model_config (dict): contains information about the
            for decoder architecture (e.g. number of layers, hidden size).
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        preprocess: Union[str, callable] = "pbc_preprocess",
        act: str = "swish",
        max_num_neighbors: int = 40,
        hidden_channels: int = 128,
        tag_hidden_channels: int = 0, # QM9
        pg_hidden_channels: int = 32,  # QM9
        phys_embeds: bool = False, # QM9
        phys_hidden_channels: int = 0,
        num_interactions: int = 4,
        num_gaussians: int = 50,
        num_filters: int = 128,
        second_layer_MLP: bool = True,  # QM9
        skip_co: bool = False, # QM9
        mp_type: str = "updownscale_base", # QM9
        graph_norm: bool = True,
        complex_mp: bool = False,
        energy_head: Optional[str] = None,
        out_dim: int = 1,
        pred_as_dict: bool = True,
        regress_forces: Optional[str] = None,
        force_decoder_type: Optional[str] = "mlp",
        force_decoder_model_config: Optional[dict] = {"hidden_channels": 128},
    ):
        super(FAENet, self).__init__()

        self.act = act
        self.complex_mp = complex_mp
        self.cutoff = cutoff
        self.energy_head = energy_head
        self.force_decoder_type = force_decoder_type
        self.force_decoder_model_config = force_decoder_model_config
        self.graph_norm = graph_norm
        self.hidden_channels = hidden_channels
        self.max_num_neighbors = max_num_neighbors
        self.mp_type = mp_type
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        self.num_interactions = num_interactions
        self.pg_hidden_channels = pg_hidden_channels
        self.phys_embeds = phys_embeds
        self.phys_hidden_channels = phys_hidden_channels
        self.regress_forces = regress_forces
        self.second_layer_MLP = second_layer_MLP
        self.skip_co = skip_co
        self.tag_hidden_channels = tag_hidden_channels
        self.preprocess = preprocess
        self.pred_as_dict = pred_as_dict

        if isinstance(self.preprocess, str):
            self.preprocess = eval(self.preprocess)

        if not isinstance(self.regress_forces, str):
            assert self.regress_forces is False or self.regress_forces is None, (
                "regress_forces must be a string "
                + "('', 'direct', 'direct_with_gradient_target') or False or None"
            )
            self.regress_forces = ""

        if self.mp_type == "simple":
            self.num_filters = self.hidden_channels

        self.act = (
            (getattr(nn.functional, self.act) if self.act != "swish" else swish)
            if isinstance(self.act, str)
            else self.act
        )
        assert callable(self.act), (
            "act must be a callable function or a string "
            + "describing that function in torch.nn.functional"
        )

        # Gaussian Basis
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, self.num_gaussians)

        # Radial basis based on "Soft envelope * bessel function"
        #self.distance_expansion = RadialBasis(self.num_gaussians)

        # Embedding block
        self.embed_block = EmbeddingBlock(
            self.num_gaussians,
            self.num_filters,
            self.hidden_channels,
            self.tag_hidden_channels,
            self.pg_hidden_channels,
            self.phys_hidden_channels,
            self.phys_embeds,
            self.act,
            self.second_layer_MLP,
        )

        # Interaction block
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    self.hidden_channels,
                    self.num_filters,
                    self.act,
                    self.mp_type,
                    self.complex_mp,
                    self.graph_norm,
                )
                for _ in range(self.num_interactions)
            ]
        )

        # Output block
        self.output_block = OutputBlock(
            self.energy_head, self.hidden_channels, self.act, out_dim
        )

        # Energy head
        if self.energy_head == "weighted-av-initial-embeds":
            self.w_lin = Linear(self.hidden_channels, 1)

        # Force head
        self.decoder = (
            ForceDecoder(
                self.force_decoder_type,
                self.hidden_channels,
                self.force_decoder_model_config,
                self.act,
            )
            if "direct" in self.regress_forces
            else None
        )

        # Skip co
        if self.skip_co == "concat":
            self.mlp_skip_co = Linear(out_dim * (self.num_interactions + 1), out_dim)
        elif self.skip_co == "concat_atom":
            self.mlp_skip_co = Linear(
                ((self.num_interactions + 1) * self.hidden_channels),
                self.hidden_channels,
            )

    # FAENet's forward pass in done in BaseModel, inherited here.
    # It uses forces_forward() and energy_forward() defined below.

    def forces_forward(self, preds):
        """Predicts forces for 3D atomic systems.
        Can be utilised to predict any atom-level property.

        Args:
            preds (dict): dictionnary with final atomic representations
                (hidden_state) and predicted properties (e.g. energy)
                for each graph

        Returns:
            (dict): additional predicted properties, at an atom-level (e.g. forces)
        """
        if self.decoder:
            return self.decoder(preds["hidden_state"])

    def energy_forward(self, data, preproc=True):
        """Predicts any graph-level property (e.g. energy) for 3D atomic systems.

        Args:
            data (data.Batch): Batch of graphs data objects.
            preproc (bool): Whether to apply (any given) preprocessing to the graph.
                Default to True.

        Returns:
            (dict): predicted properties for each graph (key: "energy")
                and final atomic representations (key: "hidden_state")
        """
        # Pre-process data (e.g. pbc, cutoff graph, etc.)
        # Should output all necessary attributes, in correct format.
        if preproc:
            z, batch, edge_index, rel_pos, edge_weight = self.preprocess(
                data, self.cutoff, self.max_num_neighbors
            )
        else:
            rel_pos = data.positions[data.edge_index[0]] - data.positions[data.edge_index[1]]
            z, batch, edge_index, rel_pos, edge_weight = (
                data.species.long(),
                data.batch,
                data.edge_index,
                rel_pos,
                rel_pos.norm(dim=-1),
            )
        edge_attr = self.distance_expansion(edge_weight)  # RBF of pairwise distances
        assert z.dim() == 1 and z.dtype == torch.long

        # Embedding block
        h, e = self.embed_block(
            z, rel_pos, edge_attr, data.tags if hasattr(data, "tags") else None
        )

        # Compute atom weights for late energy head
        if self.energy_head == "weighted-av-initial-embeds":
            alpha = self.w_lin(h)
        else:
            alpha = None

        # Interaction blocks
        energy_skip_co = []
        for interaction in self.interaction_blocks:
            if self.skip_co == "concat_atom":
                energy_skip_co.append(h)
            elif self.skip_co:
                energy_skip_co.append(
                    self.output_block(h, edge_index, edge_weight, batch, alpha)
                )
            #edge_mask = data.edge_mask
            edge_mask = None

            h = h + interaction(h, edge_index, e, edge_mask)

        # Atom skip-co
        if self.skip_co == "concat_atom":
            energy_skip_co.append(h)
            h = self.act(self.mlp_skip_co(torch.cat(energy_skip_co, dim=1)))

        energy = self.output_block(h, edge_index, edge_weight, batch, alpha)

        # Skip-connection
        energy_skip_co.append(energy)
        if self.skip_co == "concat":
            energy = self.mlp_skip_co(torch.cat(energy_skip_co, dim=1))
        elif self.skip_co == "add":
            energy = sum(energy_skip_co)

        preds = {"energy": energy, "hidden_state": h}

        return preds
