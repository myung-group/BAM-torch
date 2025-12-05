# pylint: disable=missing-module-docstring,missing-class-docstring,line-too-long
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from bam_torch.utils.scatter import scatter_sum
from bam_torch.ga.utils.fa_utils import pbc_preprocess, base_preprocess, GaussianSmearing
#from bam_torch.ga.group_averaging.frame_averaging import parse_batch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 25):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


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
            activation=F.gelu,
            regress_forces='from_energy',
            num_species=4,
            nlayers=4
        ):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(6+3, d_model), nn.Dropout(dropout))
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, norm_first=True)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.force_decoder = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.energy_decoder = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        self.regress_forces = regress_forces

    def forward_(self, node_features, edge_features):
        # get shapes
        b, n, c, d_node = node_features.shape
        _, _, _, d_edge = edge_features.shape
        assert n == 5 and c == 3
        assert edge_features.shape == (b, n, n, d_edge)
        node_channels = 3*d_node
        edge_channels = d_edge
        # embed input into a (b, n, n, c) tensor
        x = torch.zeros(b, n, n, node_channels + edge_channels, device=node_features.device)
        node_features = node_features.reshape(b, n, node_channels)
        x[:, torch.arange(n), torch.arange(n), :node_channels] = node_features
        x[:, :, :, node_channels:] = edge_features
        # reshape to (b, l=n*n, c)
        x = x.reshape(b, n*n, node_channels + edge_channels)
        # transpose to (l, b, c)
        x = x.transpose(0, 1)
        # push through transformer
        x = self.input(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        energy = self.energy_decoder(x)
        x_frc = self.force_decoder(x)
        # transpose to (b, l, 3)
        x_frc = x_frc.transpose(0, 1)
        # reshape into (b, n, n, 3)
        x_frc = x_frc.reshape(b, n, n, 3)
        # take diagonal, (b, n, 3)
        x_frc = x_frc[:, torch.arange(n), torch.arange(n)].reshape(-1, 3)
        preds["energy"] = energy
        preds["forces"] = x_frc
        return preds

    def forward(self, data):
        """
        data: torch_geometric.data.Data
            - data.pos: (n, 3)
            - data.edge_index: (2, e)
            - data.edges: (e, d_edge)
        """
        data.pos.requires_grad_(True)
        node_features, edge_features, edges, idx = parse_batch(data, data.pos.device)

        b, n, c, d_node = node_features.shape
        _, _, _, d_edge = edge_features.shape
        node_channels = 3 * d_node
        edge_channels = d_edge
        x = torch.zeros(b, n, n, node_channels + edge_channels, device=node_features.device)
        node_features = node_features.reshape(b, n, node_channels)
        x[:, torch.arange(n), torch.arange(n), :node_channels] = node_features
        x[:, :, :, node_channels:] = edge_features
        x = x.reshape(b, n * n, node_channels + edge_channels).transpose(0, 1)
        x = self.input(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x_enr = self.energy_decoder(x)
        # transpose to (b, l, 3)  
        x_enr = x_enr.transpose(0, 1)
        # reshape into (b, n, n, 3)
        x_enr = x_enr.reshape(b, n, n, 1)
        # take diagonal, (b, n, 3)
        x_enr = x_enr[:, torch.arange(n), torch.arange(n)]
        energy = x_enr.sum(dim=1).view(-1)
        preds = {}
        preds["energy"] = energy
        preds["node_energy"] = x_enr.view(-1, 1)
        if self.regress_forces == "from_energy":
            # predicted forces are the energy gradient
            grad_forces = self.forces_as_energy_grad(data.pos, energy)
            preds["forces"] = grad_forces
        elif self.regress_forces in {"direct", "direct_with_gradient_target"}:
            # predicted forces are the model's direct forces
            x_frc = self.force_decoder(x)
            x_frc = x_frc.transpose(0, 1)
            x_frc = x_frc.reshape(b, n, n, 3)
            x_frc = x_frc[:, torch.arange(n), torch.arange(n)].reshape(-1, 3)
            preds["forces"] = x_frc
        
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

def parse_batch(data, device):
    """
    pos = torch.tensor(data.x, dtype=torch.float32).to(device)
    forces = torch.tensor(data.x_forces, dtype=torch.float32).to(device)
    num_edges = torch.tensor(data.num_edges, dtype=int).to(device)
    b = num_edges.shape[0]
            
    edges = torch.tensor(data.edge_idx, dtype=torch.float32).to(device)
    """
    #torch.autograd.set_detect_anomaly(True)
    pos = data.pos
    num_edges = data.num_edges
    b = num_edges.shape[0]
    #num_edges = num_edges[0]
    b_n, _ = pos.shape
    n = int(b_n / b)
    
    #Rij = data.Rij 
    #loc_dist = data.distance 
    #Rij = Rij.view(b, num_edges, 3)
    #loc_dist = loc_dist.view(b, 1, num_edges).float()
    #iatoms = data.senders
    #jatoms = data.receivers
    #iatoms = iatoms.view(b, 1, num_edges).long()
    #jatoms = jatoms.view(b, 1, num_edges).long()

    #edges = torch.cat([iatoms, jatoms], dim=1).long()
    edges = data.edge_index
    cell = data.cell
    iatoms = edges[0]
    jatoms = edges[1]
    Sij = data.edges
    Sij = torch.split(Sij, num_edges.tolist(), dim=0)
    shift_v = torch.cat(
        [torch.einsum('ni,ij->nj', s, c)
         for s, c in zip(Sij, cell)], dim=0
    )
    _R = pos[jatoms] - pos[iatoms]
    Rij = _R + shift_v
    loc_dist = torch.norm(Rij, dim=1)
    loc_dist = loc_dist.view(b, 1, num_edges[0])

    iatoms = iatoms.view(b, 1, num_edges[0]).long()
    jatoms = jatoms.view(b, 1, num_edges[0]).long()
    offsets = torch.arange(b, device=iatoms.device).view(b, 1, 1) * n
    iatoms = iatoms - offsets
    jatoms = jatoms - offsets

    edges = torch.cat([iatoms, jatoms], dim=1).long()
    edge_attr = torch.cat([edges, loc_dist], dim=1)
    edge_attr = edge_attr.transpose(1, 2)


    species = data.species
    species = species.view(b, n, 1)
    species = species[:, :, :, None].expand(-1, -1, -1, 3).view(b, n, 3)
    #node_features = torch.cat([pos, forces], dim=-1)
    pos = pos.view(b, n, 3)
    node_features = torch.cat([pos, species], dim=-1) # pos, Rij..?
    # node_features = torch.cat([pos, forces, species.unsqueeze(-1)], dim=-1)
    node_features = node_features.view(b, n, 2, 3)    
    node_features = node_features.transpose(-1, -2)
    assert (pos - node_features[:, :, :, 0]).abs().sum().item() == 0

    idx = torch.tensor([i for i in range(b)], device=device)
    edge_features = torch.zeros(b, n, n, edge_attr.size(-1), device=device) 
    batch_idxs = torch.arange(b, device=device).repeat_interleave(num_edges[0]).long()
    edge_features[batch_idxs, edges[:, 0, :].flatten().long(), edges[:, 1, :].flatten().long(), :] \
        = edge_attr.reshape(-1, edge_attr.size(-1))

    return node_features, edge_features, edges, idx


def get_index_embedding(indices, emb_dim, max_len=2048):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., num_tokens] of type integer
        emb_dim: dimension of the embeddings to create
        max_len: maximum length

    Returns:
        positional embedding of shape [..., num_tokens, emb_dim]
    """
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding

class Transformer(nn.Module):
    def __init__(
            self,
            d_model=64,
            dim_feedforward=64,
            nhead=4,
            num_encoder_layers=4,
            dropout=0.5,
            activation=F.silu,
            regress_forces='from_energy',
            num_species=4,
            nlayers=4
        ):
        super().__init__()
        self.num_species = num_species
        self.d_model = d_model
        self.nlayers = nlayers
        norm_first = True
        bias = True
        num_gaussians = 10
        #self.distance_expansion = GaussianSmearing(0.0, 6.0, num_gaussians)

        self.atom_type_embedder = nn.Embedding(num_species, d_model)
        self.pos_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.edge_embedder = nn.Sequential(
            nn.Linear(1, d_model, bias=False),
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
            nn.Linear(1, d_model),
            #nn.ReLU(),
            #nn.Dropout(dropout),
            #nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.regress_forces = regress_forces

    def forward(self, data):
        """
        data: torch_geometric.data.Data
            - data.pos: (n, 3)
            - data.edge_index: (2, e)
            - data.edges: (e, d_edge)
        """
        data.pos.requires_grad_(True)
        z, batch, edge_index, rel_pos, edge_weight = self.preprocess(
                        data, 6.0, 30,
                    )
        x = self.atom_type_embedder(data.species.unsqueeze(-1)) # (n, 1)
        x = x + self.pos_embedder(data.pos) # (n, 3)
        # edge_attr = self.distance_expansion(edge_weight)
        #print(f"1) x: {x.shape}")
        e = self.edge_embedder(edge_weight.unsqueeze(-1))
        #print(f"2) e: {e.shape}")
        #x = e + x + self.pos_embedder(rel_pos)
        #print(f"3) x: {x.shape}")
        #a = self.pos_encoder(x)
        #print(f" -- self.pos_encoder(x): {a.shape}")
        x = x + self.pos_encoder(x) # (n, n, d_model)
        #print(f"4) x: {x.shape}")
        x = self.transformer.forward(x) # (n, n, d_model)
        #print(f"5) x: {x.shape}")
        x_enr = self.energy_decoder(x) # (n, n, 1)
        """
        import torch
        import matplotlib.pyplot as plt
        x2d = x_enr.squeeze(-1)  
        x_np = x2d.detach().cpu().numpy()
        plt.figure()
        im = plt.imshow(x_np, cmap='viridis')  
        plt.colorbar(im)                     
        plt.title("Value heatmap")
        plt.show()
        """
        # take diagonal, (n, 1)
        n, n, _ = x_enr.shape
        #print(x_enr.shape)
        #print(edge_index, '|', edge_index.shape)
        #print(edge_index[0].shape)
        #print(edge_index[:,0].shape)
        #x_enr = x_enr[torch.arange(n), torch.arange(n), :]
        num_graphs = data["ptr"].numel() - 1  # nbatch
        #x_enr = scatter_sum(
        #    src=x_enr,
        #    index=edge_index[:,0],
        #    dim=1,
        #    dim_size=num_graphs
        #)
        #print(f"x_enr: {x_enr.shape}")
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
        if self.regress_forces == "from_energy":
            # predicted forces are the energy gradient
            grad_forces = self.forces_as_energy_grad(data.pos, energy)
            preds["forces"] = grad_forces
        elif self.regress_forces in {"direct", "direct_with_gradient_target"}:
            # predicted forces are the model's direct forces
            x_frc = self.force_decoder(x_enr)
            #print(x_frc.shape)
            #x_frc = x_frc[torch.arange(n), torch.arange(n), :].reshape(-1, 3)
            preds["forces"] = x_frc
        
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