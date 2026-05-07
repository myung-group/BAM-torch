# pylint: disable=missing-module-docstring,missing-class-docstring,line-too-long
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from bam_torch.utils.scatter import scatter_sum
from bam_torch.group_averaging.utils.ga_utils import (
    pbc_preprocess, 
    base_preprocess, 
)


class Transformer_(nn.Module):
    def __init__(
            self,
            d_model=64,
            dim_feedforward=64,
            nhead=4,
            num_encoder_layers=4,
            dropout=0.5,
            activation=F.silu,
            regress_forces='auto',
            num_species=4,
        ):
        super().__init__()
        self.num_species = num_species
        self.d_model = d_model
        norm_first = True
        bias = True
        num_gaussians = 10

        self.node_embedding = nn.Linear(num_species, d_model, bias=True)
        self.pos_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.edge_embedder = nn.Sequential(
            nn.Linear(1, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
                bias=bias,
            ),
            norm=nn.LayerNorm(d_model),
            num_layers=num_encoder_layers,
        )

        self.preprocess = pbc_preprocess

        self.energy_decoder = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1)
        )
        self.force_decoder = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),
        )
        self.regress_forces = regress_forces

        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)

    def forward(self, data):
        """
        data: torch_geometric.data.Data
            - data.pos: (n, 3)
            - data.edge_index: (2, e)
            - data.edges: (e, d_edge)
        """
        
        #z, batch, edge_index, rel_pos, edge_weight = self.preprocess(
        #                data, 6.0, 30,
        #            )
        pos = data.pos
        try:
            num = int(data.node_mask.sum())
        except:
            num = len(pos)
        x = to_one_hot(data.species[:num].unsqueeze(-1), self.num_species)
        x_ = self.node_embedding(x)
        x = self.pos_embedder(pos) # (n, 3)
        x = x_ + self.pos_encoder(x) #+e # (n, n, d_model)
        x = self.transformer.forward(x) # (n, n, d_model)
        x_enr = self.energy_decoder(x) # (n, n, 1)
        n, n, _ = x_enr.shape

        num_graphs = data["ptr"].numel() - 1  # nbatch
        x_enr = (x_enr.sum(dim=0) + x_enr.sum(dim=1)) / n

        energy = scatter_sum(
            src=x_enr.squeeze(),
            index=data["batch"],
            dim=-1,
            dim_size=num_graphs
        )

        preds = {}
        preds["energy"] = energy
        preds["node_energy"] = x_enr.view(-1, 1)
        if self.regress_forces == "auto":
            # predicted forces are the energy gradient
            grad_forces = self.forces_as_energy_grad(pos, energy)
            preds["forces"] = grad_forces
        elif self.regress_forces in {"direct", "direct_with_gradient_target"}:
            # predicted forces are the model's direct forces
            x_frc = self.force_decoder(x)
            x_frc = x_frc[torch.arange(n), torch.arange(n), :].view(-1, 3)
            preds["forces"] = x_frc * x_enr.view(-1, 1)
            preds["forces_grad_target"] = self.forces_as_energy_grad(data.pos, energy)

        return preds

    def forces_as_energy_grad(self, pos, energy):
        """Computes forces from energy gradient

        Args:
            pos (tensor): 3D atom positions
            energy (tensor): system's predicted energy

        Returns:
            (tensor): forces as the energy gradient w.r.t. atom positions
        """
        return -1 * (
            torch.autograd.grad(
                energy,
                pos,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]
        )


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 729):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(max_len, 1, d_model))

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
            self,
            d_model=64,
            dim_feedforward=64,
            nhead=4,
            num_encoder_layers=4,
            dropout=0.5,
            activation=F.silu,
            regress_forces='auto',
            num_species=4,
        ):
        super().__init__()
        self.num_species = num_species
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.pos_linear = nn.Linear(3, d_model)
        self.embed = nn.Embedding(num_species, d_model)

        self.input = nn.Sequential(nn.Linear(4+4, d_model), nn.Dropout(dropout))
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=dropout)

        """
        self.pos_linear_x = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=nhead)
        self.transformer_encoder_pos_x = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.pos_linear_y = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=nhead)
        self.transformer_encoder_pos_y = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.pos_linear_z = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=nhead)
        self.transformer_encoder_pos_z = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        """
        self.regress_forces = regress_forces

        force_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=nhead)
        self.force_encoder = nn.TransformerEncoder(force_encoder_layer, num_layers=num_encoder_layers)

        self.energy_decoder = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 1)
        )
        self.force_decoder = nn.Sequential(
            nn.Linear(d_model, 3),
        )
        self.regress_forces = regress_forces

        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)

    def forward(self, data):
        """
        data: torch_geometric.data.Data
            - data.pos: (n, 3)
            - data.edge_index: (2, e)
            - data.edges: (e, d_edge)
        """
        
        z, batch, edge_index, rel_pos, distances = pbc_preprocess(
                        data, 6.0, 30,
                    )
        pos = data.pos
        try:
            num = int(data.node_mask.sum())
        except:
            num = len(pos)

        num_graphs = int(data["ptr"].numel() - 1)
        pos = pos.reshape(-1, num_graphs, 3)
        n, b, _ = pos.shape
        species = data.species[:num]
        species = species.reshape(-1, num_graphs)
        """
        species_feat = species.float().unsqueeze(-1)
        node_feat = torch.cat(
            [species_feat, pos], dim=-1
        )
        node_channels = 4
        edge_channels = 4
        edge_feat = torch.cat(
            [distances.unsqueeze(-1), rel_pos], dim=-1
        )
        edge_dense = torch.zeros(
            b, n, n, edge_channels, device=edge_feat.device
        )
        row, col = edge_index  # (E,), (E,)
        batch = data.batch    # (N_total,)
        edge_batch = batch[row]   # (E,)
        
        edge_dense[edge_batch, row % n, col % n, :] = edge_feat


        #pos_emb = self.pos_linear(pos)
        node_feat = node_feat.permute(1, 0, 2).contiguous() 
        idx = torch.arange(n, device=pos.device)
        x = torch.zeros(b, n, n, node_channels+edge_channels, device=pos.device)
        x[:, idx, idx, :node_channels] = node_feat
        x[:, :, :, node_channels:] = edge_dense
        x = x.reshape(b, n*n, node_channels+edge_channels)
        x = x.transpose(0, 1)
        x = self.input(x)
        x = self.pos_encoder(x)
        """

        """
        pos_x = pos[:, :, 0]
        pos_y = pos[:, :, 1]
        pos_z = pos[:, :, 2]
        #pos_emb = self.embed_pos(pos)
        pos_emb_x = self.pos_linear_x(pos_x.unsqueeze(-1))
        pos_emb_y = self.pos_linear_x(pos_y.unsqueeze(-1))
        pos_emb_z = self.pos_linear_x(pos_z.unsqueeze(-1))
        
        pos_out_x = self.transformer_encoder_pos_x(pos_emb_x)
        pos_out_y = self.transformer_encoder_pos_y(pos_emb_y)
        pos_out_z = self.transformer_encoder_pos_z(pos_emb_z)
        """
        # pos_out = self.transformer_encoder_pos(pos_emb)
        #x = self.embed(species)
        #x = x  #+ pos_emb
        out = self.transformer_encoder(x)
        out = out #+ pos_out_x + pos_out_y + pos_out_z
        node_energy = self.energy_decoder(out).squeeze(-1)

        force_out = self.force_encoder(x)
        
       # node_energy = out.sum(-1)
        n, b = node_energy.shape
        graph_energy = node_energy.sum(0)

        preds = {}
        preds["energy"] = graph_energy
        preds["node_energy"] = node_energy.view(-1, 1)
        if self.regress_forces == "auto":
            # predicted forces are the energy gradient
            grad_forces = self.forces_as_energy_grad(pos, graph_energy)
            preds["forces"] = grad_forces
        elif self.regress_forces in {"direct", "direct_with_gradient_target"}:
            # predicted forces are the model's direct forces
            x_frc = self.force_decoder(force_out)
            x_frc = x_frc.view(-1, 3)
            preds["forces"] = x_frc
            preds["forces_grad_target"] = self.forces_as_energy_grad(pos, graph_energy)

        return preds

    def forces_as_energy_grad(self, pos, energy):
        """Computes forces from energy gradient

        Args:
            pos (tensor): 3D atom positions
            energy (tensor): system's predicted energy

        Returns:
            (tensor): forces as the energy gradient w.r.t. atom positions
        """
        return -1 * (
            torch.autograd.grad(
                energy,
                pos,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]
        )

# ------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RadialEmbedding(nn.Module):
    """Gaussian/RBF expansion for distances."""
    def __init__(self, num_rbf=32, cutoff=6.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        self.cutoff = float(cutoff)
        # gamma: controls width; this is a reasonable default
        self.gamma = num_rbf / cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: (...,)
        # returns: (..., num_rbf)
        return torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centers) ** 2)


class DistanceAwareEncoderLayer(nn.Module):
    """Transformer encoder layer with additive distance bias on attention logits."""
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1, activation=F.silu, norm_first=True):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.norm_first = norm_first

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.dropout_attn = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=True),
            nn.SiLU() if activation in (F.silu, torch.nn.functional.silu) else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, bias=True),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def _sa_block(self, x, attn_bias, key_padding_mask=None):
        """
        x: (S, B, D)
        attn_bias: (B, S, S) additive bias (can include -inf for masking)
        key_padding_mask: (B, S) bool, True means "pad" (mask out)
        """
        S, B, D = x.shape
        qkv = self.qkv(x).view(S, B, 3, self.nhead, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each: (S, B, H, d_head)

        # (B, H, S, d_head)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # logits: (B, H, S, S)
        logits = torch.einsum("bhsd,bhtd->bhst", q, k) / math.sqrt(self.d_head)

        # add distance bias (broadcast over heads)
        logits = logits + attn_bias.unsqueeze(1)

        # apply key padding mask: mask keys (j dimension)
        if key_padding_mask is not None:
            # key_padding_mask: True for pad positions
            # expand to (B, 1, 1, S) and add -inf
            pad = key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,S)
            logits = logits.masked_fill(pad, float("-inf"))

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout_attn(attn)

        out = torch.einsum("bhst,bhtd->bhsd", attn, v)  # (B,H,S,d_head)
        out = out.permute(2, 0, 1, 3).contiguous().view(S, B, D)  # (S,B,D)
        return self.out_proj(out)

    def forward(self, x, attn_bias, key_padding_mask=None):
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_bias, key_padding_mask)
            x = x + self.ffn(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_bias, key_padding_mask))
            x = self.norm2(x + self.ffn(x))
        return x


class DistanceAwareEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout, activation, norm_first=True):
        super().__init__()
        self.layers = nn.ModuleList([
            DistanceAwareEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, attn_bias, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias, key_padding_mask=key_padding_mask)
        return x


class Transformer_(nn.Module):
    def __init__(
        self,
        d_model=64,
        dim_feedforward=256,
        nhead=4,
        num_encoder_layers=4,
        dropout=0.1,
        activation=F.silu,
        regress_forces="auto",  # "auto" or "direct" or "direct_with_gradient_target"
        num_species=4,
        num_rbf=32,
        cutoff=6.0,
    ):
        super().__init__()
        self.num_species = num_species
        self.d_model = d_model
        self.regress_forces = regress_forces

        # --- node embedding ---
        self.embed = nn.Embedding(num_species, d_model)

        # --- distance bias ---
        self.radial_emb = RadialEmbedding(num_rbf=num_rbf, cutoff=cutoff)
        # RBF -> scalar bias
        self.edge_bias = nn.Linear(num_rbf, 1, bias=False)
        self.cutoff = float(cutoff)

        # --- distance-aware encoders ---
        self.transformer_encoder = DistanceAwareEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=True,
        )
        self.force_encoder = DistanceAwareEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=True,
        )

        # --- decoders ---
        self.energy_decoder = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.force_decoder = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),
        )

        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)

    def _build_distance_bias(self, pos, node_mask=None):
        """
        pos: (S, B, 3)
        node_mask: (S, B) bool or 0/1, True/1 means valid node
        returns:
          attn_bias: (B, S, S)
          key_padding_mask: (B, S) bool (True means pad)
        """
        S, B, _ = pos.shape

        # pairwise distance: dist[b, i, j]
        pos_b = pos.permute(1, 0, 2).contiguous()

        # pairwise distance
        rij = pos_b.unsqueeze(2) - pos_b.unsqueeze(1)  # (B, S, S, 3)
        dist = torch.norm(rij, dim=-1)        

        # cutoff masking: far away edges get large negative bias
        far = dist > self.cutoff

        rbf = self.radial_emb(dist)                       # (B, S, S, R)
        bias = self.edge_bias(rbf).squeeze(-1)            # (B, S, S)

        # hard mask by cutoff (recommended)
        bias = bias.masked_fill(far, float("-inf"))

        # node padding mask (keys)
        key_padding_mask = None
        if node_mask is not None:
            # node_mask: (S,B) -> key_padding_mask: (B,S)
            valid = node_mask.bool().permute(1, 0).contiguous()
            key_padding_mask = ~valid  # True for pad

            # also: if query is pad, make its row all -inf (so it contributes nothing)
            # (optional but 안정적)
            q_pad = key_padding_mask.unsqueeze(-1)        # (B,S,1)
            bias = bias.masked_fill(q_pad, float("-inf"))

        return bias, key_padding_mask

    def forward(self, data):
        """
        Expected:
          data.pos: (N_total, 3)
          data.species: (N_total,) (Long)
          data.ptr: (num_graphs+1,)
          optionally data.node_mask: (N_total,) (0/1)
        """
        pos_flat = data.pos
        num_graphs = int(data["ptr"].numel() - 1)

        # --- infer S (nodes per graph) ---
        assert pos_flat.numel() % (num_graphs * 3) == 0, (
            "pos cannot be reshaped to (S, num_graphs, 3). "
            "You likely do NOT have fixed nodes per graph (need padding) or ptr batching differs."
        )
        S = pos_flat.shape[0] // num_graphs

        # reshape to (S, B, 3)
        pos = pos_flat.reshape(S, num_graphs, 3)

        # species reshape to (S, B)
        species = data.species.reshape(S, num_graphs)

        # node_mask reshape to (S,B) if present
        node_mask = None
        if hasattr(data, "node_mask") and data.node_mask is not None:
            node_mask = data.node_mask.reshape(S, num_graphs)

        # --- node embeddings ---
        x = self.embed(species)  # (S, B, d_model)

        # --- distance bias ---
        attn_bias, key_padding_mask = self._build_distance_bias(pos, node_mask=node_mask)

        # --- encoders ---
        out = self.transformer_encoder(x, attn_bias=attn_bias, key_padding_mask=key_padding_mask)
        force_out = self.force_encoder(x, attn_bias=attn_bias, key_padding_mask=key_padding_mask)

        # --- energy prediction ---
        node_energy = self.energy_decoder(out).squeeze(-1)   # (S, B)

        if node_mask is not None:
            node_energy = node_energy * node_mask.float()

        graph_energy = node_energy.sum(dim=0)                # (B,)

        preds = {
            "energy": graph_energy,
            "node_energy": node_energy.reshape(-1, 1),
        }

        # --- forces ---
        if self.regress_forces == "auto":
            # predicted forces are the energy gradient
            # NOTE: forces_as_energy_grad should use pos_flat with requires_grad=True internally or here.
            preds["forces"] = self.forces_as_energy_grad(pos_flat, graph_energy)
        elif self.regress_forces in {"direct", "direct_with_gradient_target"}:
            x_frc = self.force_decoder(force_out).reshape(-1, 3)  # (N_total, 3)
            preds["forces"] = x_frc
            preds["forces_grad_target"] = self.forces_as_energy_grad(pos_flat, graph_energy)

        return preds

    def forces_as_energy_grad(self, pos, energy):
        """Computes forces from energy gradient

        Args:
            pos (tensor): 3D atom positions
            energy (tensor): system's predicted energy

        Returns:
            (tensor): forces as the energy gradient w.r.t. atom positions
        """
        return -1 * (
            torch.autograd.grad(
                energy,
                pos,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]
        )


# ------------------------------

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 729):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(max_len, 1, d_model))

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


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