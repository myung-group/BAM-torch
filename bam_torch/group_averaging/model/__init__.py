from .gnn_model import FAENet
from .transformer import Transformer
import torch


MODEL_REGISTRY = {
    "faenet": FAENet,
    "gnn": FAENet,
    "transformer": Transformer,
}

ACTIVE_FN_REGISTRY = {
    "silu": torch.nn.SiLU(),
    "swish": torch.nn.SiLU(),
    "relu": torch.nn.ReLU(),
    "identity": torch.nn.Identity(),
    "gelu": torch.nn.functional.gelu
}