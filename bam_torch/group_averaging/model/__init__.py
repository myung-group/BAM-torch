from .gnn_model import FAENet
from .transformer import Transformer
from .equiformer.equiformer_v2 import EquiformerV2_OC20
from .equiformer_ga.equiformer_v2 import EquiformerV2_OC20_GA
from bam_torch.model.models import RACE
from bam_torch.group_averaging.model.models import (
    GARACE, GARACE_V2_R, GARACE_V2_B, GARACE_V2_G, GARACE_V2_R_DF, GARACE_V2_G_B
)
from .schnet import SchNet
from .simple_gnn import SimpleScalableGNN
from .sparse_attn_transformer import SparseAttentionTransformer
from .bpnn import BPNNModel
from .v_bpnn import VectorizedBPNN
from .dplr import DPLR
from .dplr_paper import DeepWannierModel, DPLRPaper, DPSRPaper
import torch


MODEL_REGISTRY = {
    "faenet": FAENet,
    "gnn": FAENet,
    "transformer": Transformer,
    "equiformer": EquiformerV2_OC20,
    "equiformer_ga": EquiformerV2_OC20_GA,
    "race": RACE,
    "race_ga": GARACE,
    "race_ga_r": GARACE_V2_R,
    "race_ga_r_df": GARACE_V2_R_DF,
    "race_ga_g_b": GARACE_V2_G_B,
    "race_ga_b": GARACE_V2_B,
    "race_ga_g": GARACE_V2_G,
    "schnet": SchNet,
    "sgnn": SimpleScalableGNN,
    "simple_gnn": SimpleScalableGNN,
    "scalable_gnn": SimpleScalableGNN,
    "sat": SparseAttentionTransformer,
    "sparse_attn_transformer": SparseAttentionTransformer,
    "sat_3d": SparseAttentionTransformer,
    "bpnn": BPNNModel,
    "v_bpnn": VectorizedBPNN,
    "dplr": DPLR,
    "deep_wannier": DeepWannierModel,
    "dplr_paper": DPLRPaper,
    "dpsr_paper": DPSRPaper,
}

ACTIVE_FN_REGISTRY = {
    "silu": torch.nn.SiLU(),
    "swish": torch.nn.SiLU(),
    "relu": torch.nn.ReLU(),
    "identity": torch.nn.Identity(),
    "gelu": torch.nn.functional.gelu
}
