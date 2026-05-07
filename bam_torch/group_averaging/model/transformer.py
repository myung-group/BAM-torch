import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.attention as attention
from torch_geometric.utils import to_dense_batch
import math

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        # input: (B, N, N) -> output: (B*N*N, num_gaussians)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class Transformer(nn.Module):
    def __init__(
            self,
            d_model=128,          
            dim_feedforward=128,
            nhead=4,
            num_encoder_layers=4,
            dropout=0.1,
            activation=F.silu,
            regress_forces='auto',
            num_species=4,
            cutoff=6.0,           
        ):
        super().__init__()
        self.num_species = num_species
        self.d_model = d_model
        self.regress_forces = regress_forces
        self.cutoff = cutoff

        # 1. Embedding Layers
        self.node_embedding = nn.Linear(num_species, d_model)
        
        # Distance-based embedding
        self.rbf = GaussianSmearing(start=0.0, stop=cutoff, num_gaussians=32)
        self.dist_embedding = nn.Linear(32, d_model)
        
        # 2. Information Mixing
        self.mix_dist = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # 3. Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation='gelu',
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )

        # 4. Output Decoders
        self.energy_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.force_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 3)
        )

    def forward(self, data, mode=None):
        if self.regress_forces == 'auto' or str(data.pos.device) != 'cpu':
             data.pos.requires_grad_(True)

        # 1. Batching & Masking
        x_one_hot = to_one_hot(data.species.unsqueeze(-1), self.num_species)
        x_dense, mask = to_dense_batch(x_one_hot, data.batch) # (B, N, num_species)
        
        pos_dense, _ = to_dense_batch(data.pos, data.batch) # (B, N, 3)
        mask_float = mask.float().unsqueeze(-1) # (B, N, 1)

        # 2. Compute Pairwise Distances
        # (B, N, 1, 3) - (B, 1, N, 3) -> (B, N, N, 3)
        rel_pos = pos_dense.unsqueeze(2) - pos_dense.unsqueeze(1)
        #dist = torch.norm(rel_pos, dim=-1) # (B, N, N)
        dist = (rel_pos.pow(2).sum(dim=-1) + 1e-8).sqrt()
       
        B, N, _ = dist.shape

        # 3. Distance Embedding
        dist_emb = self.rbf(dist) # (B*N*N, 32)
        dist_emb = self.dist_embedding(dist_emb) # (B*N*N, d_model)
        
        dist_emb = dist_emb.view(B, N, N, -1) 
        
        # 4. Feature Construction
        x = self.node_embedding(x_dense) # (B, N, d_model)
        
        # Distance Aggregation
        dist_mask = mask.unsqueeze(1) * mask.unsqueeze(2) # (B, N, N)
        
        dist_emb = dist_emb * dist_mask.unsqueeze(-1) 
        
        neighbor_dist_info = dist_emb.sum(dim=2)
        neighbor_dist_info = self.mix_dist(neighbor_dist_info)
        
        x = x + neighbor_dist_info

        # 5. Transformer Forward
        padding_mask = ~mask
        with attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            x = self.transformer(x, src_key_padding_mask=padding_mask)

        # 6. Energy Decoding
        out = self.energy_decoder(x)
        out = out * mask_float
        
        energy = out.sum(dim=1).view(-1)

        preds = {}
        preds["energy"] = energy
        preds["node_energy"] = out[mask]

        # 7. Force Calculation
        if self.regress_forces == "auto":
            grad_forces = -torch.autograd.grad(
                energy.sum(), 
                data.pos, 
                create_graph=True, 
                retain_graph=True
            )[0]
            preds["forces"] = grad_forces
            
        elif self.regress_forces == "direct":
            frc_out = self.force_decoder(x)
            frc_out = frc_out * mask_float
            preds["forces"] = frc_out[mask]
            if mode == 'train':
                 preds["forces_grad_target"] = -torch.autograd.grad(
                    energy.sum(), data.pos, create_graph=True, retain_graph=True)[0]

        return preds

def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    shape = list(indices.shape[:-1]) + [num_classes]
    oh = torch.zeros(shape, device=indices.device)
    return oh.scatter_(-1, indices, 1.0)
