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
        norm_first = True
        bias = True
        num_gaussians = 10

        self.node_embedding = nn.Linear(num_species, d_model, bias=True)
        self.pos_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=True),
            #nn.SiLU(),
            #nn.Linear(d_model, d_model),
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
            #nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            #nn.SiLU(),
            nn.Linear(d_model // 2, 1)
        )
        self.force_decoder = nn.Sequential(
            #nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            #nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            #nn.SiLU(),
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
        data.pos.requires_grad_(True)
        z, batch, edge_index, rel_pos, edge_weight = self.preprocess(
                        data, 6.0, 30,
                    )
        x = to_one_hot(data.species.unsqueeze(-1), self.num_species)
        x = self.node_embedding(x)
        x = x + self.pos_embedder(data.pos) # (n, 3)
        x = x + self.pos_encoder(x) #+e # (n, n, d_model)
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
            grad_forces = self.forces_as_energy_grad(data.pos, energy)
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